// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The acceptance suite for the **connectivity a plan can ask for**: `ops::fill`,
// `ops::regional` and `ops::detect` each take a `Connectivity`, and this is where
// that parameter is shown to reach the answer through the executor rather than
// only through the merge.
//
// What this file is for that `connectivity.rs` is not
// ----------------------------------------------------
// `connectivity.rs` establishes the machinery: the offset tables, the seam walk,
// the merge, and decomposition invariance driven by calling `walk_seams_with` by
// hand. That leaves precisely one gap, and it is the gap an op can fall into on
// its own — **the executor**. A block's flood, the fragment it writes, the
// gather its neighbour is given and the merge that reads it are four separate
// pieces of wiring, and a parameter that reaches three of them and not the fourth
// produces a plausible answer rather than an error. So every run below goes
// through `execute_phases` on a real `Decomposition`.
//
// Why the fixture is two diagonal chains and nothing else
// -------------------------------------------------------
// **A fixture that cannot fail proves nothing**, and for connectivity the fixture
// that cannot fail is anything whose parts touch face to face: a face join is
// made by every connectivity there is, so a scene of boxes and slabs answers the
// same under 6, 18 and 26 and would pass with the parameter dropped on the floor.
//
// So the fixture is `connectivity.rs`'s, copied rather than invented: two chains
// of single voxels, one stepping by a corner and one by an edge, **touching
// nothing face to face at all**. Each is twelve voxels, so:
//
// | | 6 | 18 | 26 |
// |---|---|---|---|
// | corner chain | 12 components | 12 | **1** |
// | edge chain | 12 components | **1** | 1 |
// | both | 24 | 13 | 2 |
//
// A chain rather than a pair because a chain crosses a seam *whatever the block
// size*, so the same fixture exercises a face seam, a lattice edge and a lattice
// corner without being built for one of them.
//
// Each op is asked its own question about that shape — `detect` labels it,
// `fill` labels its complement, `regional` labels it as a plateau with something
// higher a diagonal step away — and each of the three sections below states the
// counts its op must produce and why they differ.

use std::collections::BTreeMap;

use ndarray::Array3;

use blockflow::decomposition::Decomposition;
use blockflow::dtype::Dtype;
use blockflow::env::{ArrayEnvironment, Environment};
use blockflow::fragment::PhaseWork;
use blockflow::geometry::BlockGrid;
use blockflow::op::Chain;
use blockflow::ops::components::Connectivity;
use blockflow::ops::detect::{
    detect_phases, detect_regions, detect_regions_with, LabelRegionsOp, RegionPointsOp,
};
use blockflow::ops::fill::{
    fill_from_labels_into, fill_phases, label_background_into_with, outside_flags, FillHolesOp,
    LabelBackgroundOp,
};
use blockflow::ops::regional::{
    regional_maxima, regional_maxima_with, regional_phases, LabelPlateauxOp, RegionalMaximaOp,
};
use blockflow::points::{decode_points, Point};
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints, Workflow};
use blockflow::voxels::Voxels;

const VOLUME: [usize; 3] = [14, 14, 24];
const STREAM: &str = "connectivity.faces";
const POINTS: &str = "connectivity.points";

const EVERY: [Connectivity; 3] = [
    Connectivity::Faces,
    Connectivity::FacesAndEdges,
    Connectivity::FacesEdgesAndCorners,
];

// -------------------------------------------------------------- the shape --

/// The corner chain: twelve voxels at `(t, t, t)`, consecutive ones sharing only
/// a corner. One component under 26 and twelve under 18 and 6.
///
/// It starts at the volume's own corner deliberately — `fill` needs one end of
/// each chain to reach the outside, or every connectivity fills the whole chain
/// and the fixture stops discriminating. See [`fill_scene`].
fn corner_chain() -> Vec<[usize; 3]> {
    (0..12).map(|t| [t, t, t]).collect()
}

/// The edge chain: twelve voxels at `(t, t, 20)`, consecutive ones sharing only
/// an edge. One component under 18 and 26 and twelve under 6.
///
/// Nine slices away from the corner chain's far end on the last axis, which is
/// more than any connectivity here can span, so the two cannot leak into one
/// another and every count below is a statement about each separately.
fn edge_chain() -> Vec<[usize; 3]> {
    (0..12).map(|t| [t, t, 20]).collect()
}

fn chains() -> Vec<[usize; 3]> {
    let mut voxels = corner_chain();
    voxels.extend(edge_chain());
    voxels
}

fn mask_of(voxels: &[[usize; 3]]) -> Array3<bool> {
    let mut mask = Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), false);
    for &at in voxels {
        mask[at] = true;
    }
    mask
}

/// The decompositions every sweep below runs, chosen to cut one, two and three
/// axes, to divide the volume evenly and unevenly, and to include the single
/// block that has no seam at all — which is the case the merge cannot help with
/// and is therefore the reference the rest are measured against.
fn blockings() -> Vec<[usize; 3]> {
    vec![
        VOLUME,
        [7, 14, 24],
        [7, 7, 24],
        [7, 7, 12],
        [5, 5, 7],
        [4, 4, 8],
        [3, 3, 5],
    ]
}

/// The fixture discriminates all three connectivities, asserted before anything
/// is run against it. If it ever stops discriminating, every sweep below becomes
/// vacuous and this is the test that says so.
#[test]
fn the_chains_touch_nothing_face_to_face_and_separate_all_three_connectivities() {
    let mask = mask_of(&chains());

    // nothing in the fixture touches anything else by a face
    for &at in &chains() {
        for axis in 0..3 {
            for step in [-1isize, 1] {
                let moved = at[axis] as isize + step;
                if moved < 0 || moved >= VOLUME[axis] as isize {
                    continue;
                }
                let mut to = at;
                to[axis] = moved as usize;
                assert!(
                    !mask[to],
                    "{at:?} and {to:?} touch by a face, so this fixture cannot discriminate"
                );
            }
        }
    }

    // and the counts are the three the header states
    assert_eq!(regions_of(&mask, Connectivity::Faces).len(), 24);
    assert_eq!(regions_of(&mask, Connectivity::FacesAndEdges).len(), 13);
    assert_eq!(
        regions_of(&mask, Connectivity::FacesEdgesAndCorners).len(),
        2
    );
}

fn regions_of(mask: &Array3<bool>, connectivity: Connectivity) -> Vec<Point> {
    detect_regions_with(mask.view(), connectivity).expect("the reference runs")
}

// ------------------------------------------------------------ ops::detect --

/// `detect`'s question is the plainest of the three: the chains **are** the
/// regions, so the point count is the component count and the discrimination is
/// the table in the header — 24, 13 and 2.
fn detect_plan(
    block: [usize; 3],
    connectivity: Connectivity,
) -> (Decomposition, LabelRegionsOp, RegionPointsOp) {
    let grid = BlockGrid::new(VOLUME, block).expect("a lattice");
    let label =
        LabelRegionsOp::new("label", STREAM, Lifecycle::DeleteOnExit).connecting(connectivity);
    let points = RegionPointsOp::new("points", STREAM, 0, POINTS, Lifecycle::Persistent, &grid)
        .connecting(connectivity);
    let plan = detect_phases(grid, Dtype::Bool, &label, &points).expect("a plan");
    (plan, label, points)
}

/// Every point a run of `detect` left behind, in the canonical order.
fn detect_run(mask: &Array3<bool>, block: [usize; 3], connectivity: Connectivity) -> Vec<Point> {
    let (plan, label, points) = detect_plan(block, connectivity);
    let input: Voxels = mask.clone().into();
    let env = ArrayEnvironment::for_decomposition(input, &plan, [4, 4, 4]).expect("environment");
    let workflow = Workflow::new(Chain::sequence(Vec::new()), VOLUME, Dtype::Bool);
    execute_phases(
        "detect",
        &workflow,
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Fragments(&label), PhaseWork::Fragments(&points)],
    )
    .expect("a run");

    let counts = plan.phases[1].grid.blocks_per_axis();
    let mut found: Vec<Point> = Vec::new();
    for i in 0..counts[0] {
        for j in 0..counts[1] {
            for k in 0..counts[2] {
                let bytes = env
                    .read_sidecar(POINTS, 1, [i, j, k])
                    .expect("the store answers")
                    .unwrap_or_else(|| panic!("block {:?} wrote no blob", [i, j, k]));
                found.extend(decode_points(&bytes).expect("a point blob"));
            }
        }
    }
    canonical(found)
}

fn canonical(mut points: Vec<Point>) -> Vec<Point> {
    points.sort_by(|left, right| {
        left.at
            .cmp(&right.at)
            .then_with(|| left.weight.to_bits().cmp(&right.weight.to_bits()))
    });
    points
}

/// **The property this file exists for, for `detect`.** Every decomposition, at
/// every connectivity, byte-identical to the whole-volume reference — on a
/// fixture whose components touch only at edges and corners, so the wider
/// answers are reached across seams rather than inside blocks.
#[test]
fn detect_reproduces_the_whole_volume_reference_at_every_connectivity_and_blocking() {
    let mask = mask_of(&chains());
    for connectivity in EVERY {
        let want = regions_of(&mask, connectivity);
        for block in blockings() {
            assert_eq!(
                detect_run(&mask, block, connectivity),
                want,
                "{connectivity:?} at blocking {block:?} disagreed with the whole volume"
            );
        }
    }
}

/// **The parameter reaches the answer through a real plan.** The same volume,
/// the same lattice, the same executor — and three different component counts.
///
/// Asserted at a blocking that cuts all three axes, so every one of the counts
/// below is the *merge's* answer rather than a block's: at `[3, 3, 5]` no chain
/// lies inside one block, and a corner join between consecutive voxels can cross
/// a face seam, a lattice edge or a lattice corner depending on where it sits.
#[test]
fn the_same_volume_through_a_plan_gives_different_region_counts_at_six_and_twenty_six() {
    let mask = mask_of(&chains());
    let block = [3usize, 3, 5];

    let six = detect_run(&mask, block, Connectivity::Faces);
    let eighteen = detect_run(&mask, block, Connectivity::FacesAndEdges);
    let twenty_six = detect_run(&mask, block, Connectivity::FacesEdgesAndCorners);

    assert_eq!(
        six.len(),
        24,
        "6 joins nothing: twelve voxels in each chain"
    );
    assert_eq!(eighteen.len(), 13, "18 joins the edge chain only");
    assert_eq!(twenty_six.len(), 2, "26 joins both chains");

    // and the two components 26 finds are the two chains *whole*, which is what
    // says the accumulators were merged rather than eleven of twelve dropped
    for point in &twenty_six {
        assert_eq!(
            point.weight, 12.0,
            "a merged chain carries all twelve voxels"
        );
    }
    assert_eq!(
        twenty_six,
        canonical(vec![
            // the corner chain: coordinates 0..=11 on all three axes, mean 5.5,
            // half rounds up
            Point::weighted([6, 6, 6], 12.0),
            // the edge chain: the same on the first two axes, fixed at 20
            Point::weighted([6, 6, 20], 12.0),
        ])
    );
}

// -------------------------------------------------------------- ops::fill --

/// `fill`'s question is about the **complement**, so the chains are the
/// background and the mask is everything else.
///
/// Each chain starts at a voxel on the volume's own boundary — `(0, 0, 0)` and
/// `(0, 0, 20)` — and every other chain voxel is interior. That is what makes the
/// fixture discriminate: a background component drains if it reaches the outside,
/// so under `Faces` only the two starting voxels drain and the other twenty-two
/// fill, while under `FacesEdgesAndCorners` each chain is one component and the
/// whole of it drains. The clear voxels of the answer count 2, 13 and 24.
fn fill_scene() -> Array3<bool> {
    let mut mask = Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), true);
    for at in chains() {
        mask[at] = false;
    }
    mask
}

/// The whole-volume reference: the same kernels, called once, at the same
/// connectivity.
fn fill_reference(mask: &Array3<bool>, connectivity: Connectivity) -> Array3<bool> {
    let mut labels = Array3::<u32>::zeros(mask.raw_dim());
    let count =
        label_background_into_with(mask.view(), connectivity, labels.view_mut()).expect("a shape");
    let flags = outside_flags(labels.view(), count, [0, 0, 0], VOLUME, VOLUME);
    let mut out = Array3::from_elem(mask.raw_dim(), false);
    fill_from_labels_into(labels.view(), &flags, out.view_mut()).expect("a shape");
    out
}

fn fill_run(mask: &Array3<bool>, block: [usize; 3], connectivity: Connectivity) -> Array3<bool> {
    let grid = BlockGrid::new(VOLUME, block).expect("a lattice");
    let label =
        LabelBackgroundOp::new("label", STREAM, Lifecycle::DeleteOnExit).connecting(connectivity);
    let fill = FillHolesOp::new("fill", STREAM, 0, Dtype::Bool, &grid).connecting(connectivity);
    let plan = fill_phases(grid, Dtype::Bool, &label, &fill).expect("a plan");

    let input: Voxels = mask.clone().into();
    let env = ArrayEnvironment::for_decomposition(input, &plan, [4, 4, 4]).expect("environment");
    let workflow = Workflow::new(Chain::sequence(Vec::new()), VOLUME, Dtype::Bool);
    execute_phases(
        "fill",
        &workflow,
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Fragments(&label), PhaseWork::Fragments(&fill)],
    )
    .expect("a run");
    env.output().view::<bool>().unwrap().to_owned()
}

fn clear_voxels(mask: &Array3<bool>) -> usize {
    mask.iter().filter(|&&set| !set).count()
}

/// The scene discriminates, asserted before it is swept.
#[test]
fn the_background_chains_drain_by_a_different_amount_under_each_connectivity() {
    let mask = fill_scene();
    assert_eq!(
        clear_voxels(&fill_reference(&mask, Connectivity::Faces)),
        2,
        "6 drains only the two chain voxels that lie on the volume's boundary"
    );
    assert_eq!(
        clear_voxels(&fill_reference(&mask, Connectivity::FacesAndEdges)),
        13,
        "18 drains the whole edge chain and one voxel of the corner chain"
    );
    assert_eq!(
        clear_voxels(&fill_reference(&mask, Connectivity::FacesEdgesAndCorners)),
        24,
        "26 drains both chains entirely"
    );
}

/// **The property this file exists for, for `fill`.**
#[test]
fn fill_reproduces_the_whole_volume_reference_at_every_connectivity_and_blocking() {
    let mask = fill_scene();
    for connectivity in EVERY {
        let want = fill_reference(&mask, connectivity);
        for block in blockings() {
            assert_eq!(
                fill_run(&mask, block, connectivity),
                want,
                "{connectivity:?} at blocking {block:?} disagreed with the whole volume"
            );
        }
    }
}

/// The parameter reaches `fill`'s answer through a real plan, at a blocking that
/// cuts all three axes so that every join it makes is a merge's.
#[test]
fn the_same_volume_through_a_plan_fills_differently_at_six_and_twenty_six() {
    let mask = fill_scene();
    let block = [3usize, 3, 5];
    assert_eq!(
        clear_voxels(&fill_run(&mask, block, Connectivity::Faces)),
        2
    );
    assert_eq!(
        clear_voxels(&fill_run(&mask, block, Connectivity::FacesAndEdges)),
        13
    );
    assert_eq!(
        clear_voxels(&fill_run(&mask, block, Connectivity::FacesEdgesAndCorners)),
        24
    );
}

// ---------------------------------------------------------- ops::regional --

/// `regional`'s question needs one thing the other two do not: a **greater**
/// voxel placed where only a wider connectivity can see it.
///
/// The chains are one value, the surround another and lower, so under any
/// connectivity a chain voxel with nothing greater beside it is a maximum. Two
/// further voxels are set higher than the chains and placed so that:
///
/// * `(6, 6, 4)` is a **corner** step from the corner chain's `(5, 5, 5)` and
///   three steps from every other chain voxel — so it disqualifies nothing under
///   6 or 18, and under 26 it disqualifies the corner chain, which by then is one
///   plateau twelve voxels long;
/// * `(6, 7, 19)` is an **edge** step from the edge chain's `(6, 6, 20)` — so it
///   disqualifies nothing under 6, and under 18 and 26 it disqualifies the edge
///   chain, which by then is one plateau.
///
/// That is the load-bearing shape for this op and not for the other two: the
/// disqualification has to travel the whole length of a plateau that spans the
/// lattice, so a merge that joined the plateau but did not carry the ascent — or
/// carried the ascent at a different connectivity than it joined at — reports a
/// different mask. The set voxels count 26, 14 and 2.
fn regional_scene() -> Array3<f64> {
    let mut values = Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), 0.0);
    for at in chains() {
        values[at] = 5.0;
    }
    values[[6, 6, 4]] = 9.0;
    values[[6, 7, 19]] = 9.0;
    values
}

fn regional_run(
    values: &Array3<f64>,
    block: [usize; 3],
    connectivity: Connectivity,
) -> Array3<bool> {
    let grid = BlockGrid::new(VOLUME, block).expect("a lattice");
    let label =
        LabelPlateauxOp::new("label", STREAM, Lifecycle::DeleteOnExit).connecting(connectivity);
    let maxima =
        RegionalMaximaOp::new("maxima", STREAM, 0, Dtype::Bool, &grid).connecting(connectivity);
    let plan = regional_phases(grid, Dtype::F64, &label, &maxima).expect("a plan");

    let input: Voxels = values.clone().into();
    let env = ArrayEnvironment::for_decomposition(input, &plan, [4, 4, 4]).expect("environment");
    let workflow = Workflow::new(Chain::sequence(Vec::new()), VOLUME, Dtype::F64);
    execute_phases(
        "regional",
        &workflow,
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[PhaseWork::Fragments(&label), PhaseWork::Fragments(&maxima)],
    )
    .expect("a run");
    env.output().view::<bool>().unwrap().to_owned()
}

fn set_voxels(mask: &Array3<bool>) -> usize {
    mask.iter().filter(|&&set| set).count()
}

/// The scene discriminates, asserted before it is swept — and the two high
/// voxels are maxima under all three, which is what says the counts below move
/// because the *chains* changed and not because something was lost.
#[test]
fn the_diagonal_ascents_disqualify_the_chains_at_different_connectivities() {
    let values = regional_scene();
    let counts: Vec<usize> = EVERY
        .into_iter()
        .map(|connectivity| {
            let found = regional_maxima_with(values.view(), connectivity).expect("the reference");
            assert!(found[[6, 6, 4]] && found[[6, 7, 19]], "{connectivity:?}");
            set_voxels(&found)
        })
        .collect();
    assert_eq!(
        counts,
        vec![26, 14, 2],
        "6 leaves both chains maximal, 18 disqualifies the edge chain, 26 both"
    );
}

/// **The property this file exists for, for `regional`.**
#[test]
fn regional_reproduces_the_whole_volume_reference_at_every_connectivity_and_blocking() {
    let values = regional_scene();
    for connectivity in EVERY {
        let want = regional_maxima_with(values.view(), connectivity).expect("the reference");
        for block in blockings() {
            assert_eq!(
                regional_run(&values, block, connectivity),
                want,
                "{connectivity:?} at blocking {block:?} disagreed with the whole volume"
            );
        }
    }
}

/// The parameter reaches `regional`'s answer through a real plan, at a blocking
/// that cuts all three axes.
#[test]
fn the_same_volume_through_a_plan_finds_different_maxima_at_six_and_twenty_six() {
    let values = regional_scene();
    let block = [3usize, 3, 5];
    let counts: Vec<usize> = EVERY
        .into_iter()
        .map(|connectivity| set_voxels(&regional_run(&values, block, connectivity)))
        .collect();
    assert_eq!(counts, vec![26, 14, 2]);
}

// --------------------------------------------------- nothing existing moved --

/// **Every existing caller is byte-unchanged.** The bare constructors and the
/// ones that say [`Connectivity::Faces`] out loud produce the same bytes, for all
/// three ops, at every blocking — and the same bytes the whole-volume references
/// that predate the parameter produce.
///
/// This is the compatibility claim in the only form that can fail: a run, not a
/// signature.
#[test]
fn the_bare_ops_are_byte_identical_to_the_ones_that_state_faces() {
    let mask = mask_of(&chains());
    let complement = fill_scene();
    let values = regional_scene();

    // the whole-volume references, which are the pre-parameter functions
    assert_eq!(
        detect_regions(mask.view()).expect("a reference"),
        regions_of(&mask, Connectivity::Faces)
    );
    assert_eq!(
        regional_maxima(values.view()).expect("a reference"),
        regional_maxima_with(values.view(), Connectivity::Faces).expect("a reference")
    );

    for block in blockings() {
        // `detect`, bare
        let grid = BlockGrid::new(VOLUME, block).expect("a lattice");
        let label = LabelRegionsOp::new("label", STREAM, Lifecycle::DeleteOnExit);
        let points = RegionPointsOp::new("points", STREAM, 0, POINTS, Lifecycle::Persistent, &grid);
        assert_eq!(label.connectivity(), Connectivity::Faces);
        assert_eq!(points.connectivity(), Connectivity::Faces);
        assert!(detect_phases(grid, Dtype::Bool, &label, &points).is_ok());
        assert_eq!(
            detect_run(&mask, block, Connectivity::Faces),
            regions_of(&mask, Connectivity::Faces),
            "detect moved at blocking {block:?}"
        );

        // `fill`, bare
        let grid = BlockGrid::new(VOLUME, block).expect("a lattice");
        let background = LabelBackgroundOp::new("label", STREAM, Lifecycle::DeleteOnExit);
        let filling = FillHolesOp::new("fill", STREAM, 0, Dtype::Bool, &grid);
        assert_eq!(background.connectivity(), Connectivity::Faces);
        assert_eq!(filling.connectivity(), Connectivity::Faces);
        assert!(fill_phases(grid, Dtype::Bool, &background, &filling).is_ok());
        assert_eq!(
            fill_run(&complement, block, Connectivity::Faces),
            fill_reference(&complement, Connectivity::Faces),
            "fill moved at blocking {block:?}"
        );

        // `regional`, bare
        let grid = BlockGrid::new(VOLUME, block).expect("a lattice");
        let plateaux = LabelPlateauxOp::new("label", STREAM, Lifecycle::DeleteOnExit);
        let peaks = RegionalMaximaOp::new("maxima", STREAM, 0, Dtype::Bool, &grid);
        assert_eq!(plateaux.connectivity(), Connectivity::Faces);
        assert_eq!(peaks.connectivity(), Connectivity::Faces);
        assert!(regional_phases(grid, Dtype::F64, &plateaux, &peaks).is_ok());
        assert_eq!(
            regional_run(&values, block, Connectivity::Faces),
            regional_maxima(values.view()).expect("a reference"),
            "regional moved at blocking {block:?}"
        );
    }
}

/// The three plans still fingerprint as they did, and **a wider connectivity does
/// not move the number**.
///
/// Both halves are load-bearing and they say different things. The first is the
/// compatibility claim: a fingerprint is what a resumed run compares against, so
/// a plan that moved silently is a run that cannot be resumed, and nothing here
/// touches an op's name, reach, streams or element types.
///
/// The second is a **limit, pinned rather than left latent**. A `Decomposition`
/// records geometry — extents, halos, element widths — and a connectivity is none
/// of those: it is consumed inside `apply` and appears in no declaration. So two
/// plans that differ only in it are the same plan as far as the fingerprint is
/// concerned, exactly as two that differ only in `detect::Emission` are. A run
/// resumed against a checkpoint taken at a different connectivity would not be
/// refused by this number. That is a property of what a fingerprint covers rather
/// than a defect in these ops, and it is asserted here so that a change to either
/// is visible.
#[test]
fn the_plans_fingerprint_as_they_did_and_the_connectivity_is_not_in_the_number() {
    const PINNED: [usize; 3] = [24, 16, 12];
    const BLOCK: [usize; 3] = [8, 8, 6];

    let fingerprints = |connectivity: Connectivity| {
        let grid = BlockGrid::new(PINNED, BLOCK).expect("a lattice");
        let label = LabelBackgroundOp::new("label", "faces", Lifecycle::DeleteOnExit)
            .connecting(connectivity);
        let fill =
            FillHolesOp::new("fill", "faces", 0, Dtype::Bool, &grid).connecting(connectivity);
        let filling = fill_phases(grid, Dtype::Bool, &label, &fill).expect("a plan");

        let grid = BlockGrid::new(PINNED, BLOCK).expect("a lattice");
        let plateaux = LabelPlateauxOp::new("label", "faces", Lifecycle::DeleteOnExit)
            .connecting(connectivity);
        let maxima = RegionalMaximaOp::new("maxima", "faces", 0, Dtype::Bool, &grid)
            .connecting(connectivity);
        let regional = regional_phases(grid, Dtype::F64, &plateaux, &maxima).expect("a plan");

        let grid = BlockGrid::new(PINNED, BLOCK).expect("a lattice");
        let regions =
            LabelRegionsOp::new("label", "faces", Lifecycle::DeleteOnExit).connecting(connectivity);
        let points =
            RegionPointsOp::new("points", "faces", 0, "points", Lifecycle::Persistent, &grid)
                .connecting(connectivity);
        let detecting = detect_phases(grid, Dtype::Bool, &regions, &points).expect("a plan");

        (
            filling.fingerprint(),
            regional.fingerprint(),
            detecting.fingerprint(),
        )
    };

    // the numbers `connectivity.rs` pinned before any op took the parameter
    let pinned = (
        12_276_134_652_032_094_236u64,
        15_612_560_250_096_173_982u64,
        13_319_511_774_036_546_415u64,
    );
    for connectivity in EVERY {
        assert_eq!(
            fingerprints(connectivity),
            pinned,
            "{connectivity:?} moved a fingerprint"
        );
    }

    // and the three are genuinely different plans, so a constant that happened
    // to match one of them would not match all three
    let (a, b, c) = pinned;
    let mut all = vec![a, b, c];
    all.sort_unstable();
    all.dedup();
    assert_eq!(all.len(), 3);
}

/// A plan whose labelling and merge disagree is refused, for all three ops and
/// in both directions.
///
/// The two phases are two halves of one equivalence relation, so a mismatched
/// pair joins inside a block what it keeps apart across a seam — which is a
/// decomposition-dependent answer that looks perfectly plausible. It is refused
/// at planning time, before anything is scheduled.
#[test]
fn a_plan_whose_halves_disagree_about_connectivity_is_refused_by_every_op() {
    let mut refusals = 0usize;
    for labelling in EVERY {
        for merge in EVERY {
            if labelling == merge {
                continue;
            }
            let grid = BlockGrid::new(VOLUME, [7, 7, 12]).expect("a lattice");
            let label =
                LabelBackgroundOp::new("l", STREAM, Lifecycle::DeleteOnExit).connecting(labelling);
            let fill = FillHolesOp::new("f", STREAM, 0, Dtype::Bool, &grid).connecting(merge);
            let message = fill_phases(grid.clone(), Dtype::Bool, &label, &fill)
                .unwrap_err()
                .to_string();
            assert!(message.contains("same connectivity"), "{message}");

            let plateaux =
                LabelPlateauxOp::new("l", STREAM, Lifecycle::DeleteOnExit).connecting(labelling);
            let maxima =
                RegionalMaximaOp::new("m", STREAM, 0, Dtype::Bool, &grid).connecting(merge);
            assert!(regional_phases(grid.clone(), Dtype::F64, &plateaux, &maxima).is_err());

            let regions =
                LabelRegionsOp::new("l", STREAM, Lifecycle::DeleteOnExit).connecting(labelling);
            let points = RegionPointsOp::new("p", STREAM, 0, POINTS, Lifecycle::Persistent, &grid)
                .connecting(merge);
            assert!(detect_phases(grid, Dtype::Bool, &regions, &points).is_err());
            refusals += 1;
        }
    }
    assert_eq!(refusals, 6, "every ordered pair of unequal choices");
}

/// The chains' components are joined **across seams** rather than inside blocks,
/// which is what makes every sweep above a test of the merge.
///
/// Measured rather than asserted in prose: at each blocking, the number of
/// block-local components before the merge is counted, and it must exceed the
/// number after. At the single-block size there is nothing to merge and the two
/// are equal, which is the control.
#[test]
fn the_merge_is_load_bearing_at_every_blocking_but_the_single_block() {
    let mask = mask_of(&chains());
    let connectivity = Connectivity::FacesEdgesAndCorners;
    let after = regions_of(&mask, connectivity).len();
    assert_eq!(after, 2);

    for block in blockings() {
        let grid = BlockGrid::new(VOLUME, block).expect("a lattice");
        let mut before = 0usize;
        for core in grid.cores() {
            let start = core.core.start.clone();
            let shape = core.core.shape.clone();
            let mut inside = Array3::from_elem((shape[0], shape[1], shape[2]), false);
            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    for k in 0..shape[2] {
                        inside[[i, j, k]] = mask[[start[0] + i, start[1] + j, start[2] + k]];
                    }
                }
            }
            before += detect_regions_with(inside.view(), connectivity)
                .expect("a block")
                .len();
        }
        if block == VOLUME {
            assert_eq!(before, after, "one block has no seam to merge across");
        } else {
            assert!(
                before > after,
                "blocking {block:?} found {before} block-local components and the merge \
                 produced {after}; if these ever agree, this suite has stopped testing the \
                 merge"
            );
        }
    }
}

/// The three ops' choices are **independent**, which is the point of their being
/// three parameters rather than one.
///
/// `fill` states the *background*'s connectivity and `detect` the *foreground*'s,
/// and the complementary-pair convention in the literature deliberately pairs a
/// narrow one with a wide one. So a plan that fills at `Faces` and detects at
/// `FacesEdgesAndCorners` has to be expressible and has to mean what it says —
/// which it could not if the two ops shared one choice.
#[test]
fn the_foreground_and_the_background_connectivity_are_chosen_separately() {
    // the chains, filled at `Faces`: only the two boundary voxels drain, so the
    // answer is a volume set everywhere but those two
    let filled = fill_run(&fill_scene(), [3, 3, 5], Connectivity::Faces);
    assert_eq!(clear_voxels(&filled), 2);

    // and the *complement* of that, detected at `FacesEdgesAndCorners`: two
    // single voxels that are nine slices apart, so two regions whatever is asked
    let complement = filled.mapv(|set| !set);
    let regions = detect_run(&complement, [3, 3, 5], Connectivity::FacesEdgesAndCorners);
    assert_eq!(regions.len(), 2);
    assert_eq!(
        canonical(regions),
        canonical(vec![Point::unit([0, 0, 0]), Point::unit([0, 0, 20])])
    );

    // the pairing is the caller's to make, and neither op constrains the other:
    // the same two ops at the same two choices are two independent plans
    let grid = BlockGrid::new(VOLUME, [7, 7, 12]).expect("a lattice");
    let background = LabelBackgroundOp::new("l", STREAM, Lifecycle::DeleteOnExit)
        .connecting(Connectivity::Faces);
    let filling =
        FillHolesOp::new("f", STREAM, 0, Dtype::Bool, &grid).connecting(Connectivity::Faces);
    assert!(fill_phases(grid.clone(), Dtype::Bool, &background, &filling).is_ok());
    let regions = LabelRegionsOp::new("l", STREAM, Lifecycle::DeleteOnExit)
        .connecting(Connectivity::FacesEdgesAndCorners);
    let points = RegionPointsOp::new("p", STREAM, 0, POINTS, Lifecycle::Persistent, &grid)
        .connecting(Connectivity::FacesEdgesAndCorners);
    assert!(detect_phases(grid, Dtype::Bool, &regions, &points).is_ok());
}

/// A run of one decomposition at one connectivity is reproducible, and two runs
/// at *different* connectivities are not the same answer — asserted together so
/// that "the same" and "different" are measured by the same comparison.
#[test]
fn a_run_is_reproducible_and_two_connectivities_are_not_the_same_run() {
    let mask = mask_of(&chains());
    let block = [4usize, 4, 8];
    let six = detect_run(&mask, block, Connectivity::Faces);
    assert_eq!(six, detect_run(&mask, block, Connectivity::Faces));

    let mut seen: BTreeMap<usize, Connectivity> = BTreeMap::new();
    for connectivity in EVERY {
        let found = detect_run(&mask, block, connectivity);
        assert!(
            seen.insert(found.len(), connectivity).is_none(),
            "{connectivity:?} produced a count another connectivity already had, so this \
             comparison cannot tell them apart"
        );
    }
}
