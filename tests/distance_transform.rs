// SPDX-License-Identifier: MIT
//
// **`ops::distance` against the definition, not against another implementation.**
//
// An exact Euclidean distance transform has ground truth, and it is the
// definition: the distance to the nearest background voxel, found by looking at
// every one of them. That is the test the whole module rests on, because an
// "approximately correct" transform — a chamfer, a two-pass mask transform —
// agrees with the truth on smooth shapes and is wrong on thin ones. A comparison
// against another *implementation* can only say the two share an assumption; a
// comparison against the definition cannot.
//
// **Every parity test here has a liveness test beside it**, and the fixture is
// what makes it one. A mask whose every distance is 0 or 1 cannot tell an exact
// transform from a chamfer, so the fixture that ships is a one-voxel-thick
// **oblique background slab** with normal `(3, -5, 2)`: on it a chamfer is wrong
// at 9484 of 12 167 voxels. The centred ball is carried beside it as the control
// that *cannot* discriminate — a chamfer is wrong there at 1167 voxels and by
// under half a voxel, which looks right — and
// `the_chamfer_is_visible_on_the_slab_and_invisible_on_the_ball` measures both
// halves rather than describing them.
//
// The four claims, and what each is worth
// ----------------------------------------
//
// | claim | how |
// |---|---|
// | the field is exact | 0 differing from a brute-force nearest-background search over 90 904 voxels across five fixtures at two samplings |
// | the field is *SciPy's* | four recorded FNV-1a digests of `scipy.ndimage.distance_transform_edt` 1.15.2, reproduced bit for bit |
// | the block lattice is a synonym | eleven decompositions of a `29 x 23 x 19` volume, all byte-identical to the resident answer, with the negative control measured beside them |
// | the pass order is a synonym | all six orders bit-identical |
//
// What could not follow the op into this crate
// ---------------------------------------------
// The digests above are of masks this file *builds*, so they travel. What does
// not travel is evidence that depends on a stored asset: a `320 x 528 x 456`
// field on disk, against which this transform was shown to agree exactly on the
// squared values at all 77 045 760 voxels and to differ on the distance at
// 765 926 of them by exactly one ulp. That is a measurement of somebody's `sqrt`
// and of a particular file, it needs the file, and it belongs with the
// application that owns it. What crossed instead is the *consequence*:
// `the_squared_field_is_where_the_comparison_is_exact` asserts the property that
// made that comparison possible, on fixtures this file owns.

use blockflow::assemble::PlanBuilder;
use blockflow::decomposition::Decomposition;
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, BlockOp};
use blockflow::ops::distance::{
    self, brute_force_distance, chamfer_distance, distance_transform, seed,
    squared_distance_transform, sweep_axis, DistanceFinishOp, DistanceParams, DistanceSweepOp,
    Unbounded,
};
use blockflow::strategy::{execute_phases, Hints};
use blockflow::voxels::Voxels;
use blockflow::Dtype;
use ndarray::Array3;

// ------------------------------------------------------------- the fixtures --

/// A one-voxel-thick oblique sheet of **background** in a box of foreground.
///
/// The plane's normal is `(3, -5, 2)`: oblique to all three axes, to all three
/// face diagonals and to the body diagonal, with three different components so
/// that no two axes play the same role. That is what makes it discriminating —
/// see the test that measures it. The slab is `|3i - 5j + 2k - 20| <= 2`, which
/// is about four fifths of a voxel thick perpendicular to the plane and has no
/// holes, because the `i` coefficient is 3 and the slab is 5 wide in `t`.
fn oblique_sheet(shape: [usize; 3]) -> Array3<bool> {
    Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
        let t = 3 * i as i64 - 5 * j as i64 + 2 * k as i64 - 20;
        t.abs() > 2
    })
}

/// A ball of foreground centred in a cube. **The fixture that cannot
/// discriminate**, kept as the control rather than described.
fn centred_ball(edge: usize, radius: f64) -> Array3<bool> {
    let centre = (edge - 1) as f64 / 2.0;
    Array3::from_shape_fn((edge, edge, edge), |(i, j, k)| {
        let a = i as f64 - centre;
        let b = j as f64 - centre;
        let c = k as f64 - centre;
        a * a + b * b + c * c <= radius * radius
    })
}

/// A pseudo-random mask, from a linear congruential generator so that the same
/// mask can be built in Python and its SciPy transform recorded.
///
/// One voxel in seven is background, which leaves the field short-ranged and
/// dense with ties — the opposite regime from [`oblique_sheet`], and the one
/// where a lower envelope pops its stack most often.
fn lcg_mask(shape: [usize; 3]) -> Array3<bool> {
    let mut state: u64 = 1;
    Array3::from_shape_fn((shape[0], shape[1], shape[2]), |_| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) % 7 != 0
    })
}

/// The named fixtures, so that every test that wants "several shapes" gets the
/// same several and a failure names one.
fn fixtures() -> Vec<(&'static str, Array3<bool>)> {
    vec![
        ("oblique sheet, cube", oblique_sheet([23, 23, 23])),
        ("oblique sheet, non-cubic", oblique_sheet([29, 23, 19])),
        ("centred ball", centred_ball(23, 9.0)),
        ("pseudo-random", lcg_mask([24, 19, 17])),
        ("single foreground voxel", {
            let mut mask = Array3::from_elem((11, 9, 7), false);
            mask[[7, 2, 5]] = true;
            mask
        }),
    ]
}

// --------------------------------------------------------------- claim 1: truth --

/// **Ground truth.** An exact transform has one, and it is the definition.
#[test]
fn the_field_is_the_brute_force_answer() {
    let params = DistanceParams::default();
    let mut voxels_checked = 0usize;
    for (name, mask) in fixtures() {
        let fast = distance_transform(mask.view(), &params).expect("the transform must run");
        let slow = brute_force_distance(mask.view(), &params).expect("the search must run");
        let apart = differing(&fast, &slow);
        assert_eq!(
            apart, 0,
            "{name}: {apart} voxels differ from the definition"
        );
        voxels_checked += mask.len();

        // And anisotropically, where the squared distances are no longer
        // integers and the envelope's arithmetic is no longer exact.
        let stretched = params.with_sampling([1.0, 2.0, 5.0]);
        let fast = distance_transform(mask.view(), &stretched).expect("the transform must run");
        let slow = brute_force_distance(mask.view(), &stretched).expect("the search must run");
        assert_eq!(
            differing(&fast, &slow),
            0,
            "{name}, sampling [1, 2, 5]: the envelope and the definition part"
        );
        voxels_checked += mask.len();
    }
    println!(
        "{voxels_checked} voxels over {} fixtures at two samplings, 0 differing from a \
         brute-force nearest-background search",
        fixtures().len()
    );
    // The measured total, so that a fixture quietly dropped from the list is a
    // failure here rather than a smaller number nobody reads.
    assert_eq!(voxels_checked, 90_904, "a fixture left the list");
}

/// The reference's own numbers, recorded.
///
/// Brute force is the definition and agreeing with it is ground truth. Neither
/// that nor this module is *SciPy*, which is what one of the two reference call
/// sites reaches — and SciPy is not a dependency of this crate and must not
/// become one. So its answer is recorded: the FNV-1a hash of the field's
/// `float64` bytes in C order, computed by
/// `scipy.ndimage.distance_transform_edt` 1.15.2 on exactly the masks built
/// above, at unit sampling and off it.
///
/// A hash rather than an array because the whole point is that the comparison is
/// **bit for bit** — SciPy uses Maurer's exact dimensionality reduction and this
/// module uses Felzenszwalb–Huttenlocher's separable envelope, two different
/// algorithms, and at unit sampling both produce the correctly rounded square
/// root of the same exact integer. A tolerance would have hidden that.
#[test]
fn the_field_reproduces_scipys_own_numbers() {
    let recorded: [(&str, Array3<bool>, [f64; 3], u64); 4] = [
        (
            "oblique sheet, cube",
            oblique_sheet([23, 23, 23]),
            [1.0, 1.0, 1.0],
            0x63a5_67a5_9fa0_8192,
        ),
        (
            "oblique sheet, cube, sampling [1, 2, 5]",
            oblique_sheet([23, 23, 23]),
            [1.0, 2.0, 5.0],
            0x582c_3ae9_3d3f_89b0,
        ),
        (
            "pseudo-random",
            lcg_mask([24, 19, 17]),
            [1.0, 1.0, 1.0],
            0x8498_99a8_3868_5223,
        ),
        (
            "pseudo-random, sampling [1, 2, 5]",
            lcg_mask([24, 19, 17]),
            [1.0, 2.0, 5.0],
            0x2ab5_bad5_ce4c_e06d,
        ),
    ];
    for (name, mask, sampling, digest) in recorded {
        let params = DistanceParams::default().with_sampling(sampling);
        let field = distance_transform(mask.view(), &params).expect("the transform must run");
        assert_eq!(
            fnv1a(&field),
            digest,
            "{name}: this op and SciPy do not produce the same bits"
        );
    }
    println!("four SciPy 1.15.2 recordings reproduced bit for bit");
}

/// **The squared field is where the comparison is exact**, which is why it is
/// public.
///
/// At unit sampling every value is an exact integer, so a threshold against an
/// integer needs no square root at all, and squaring the distance back recovers
/// it exactly. That is the property that made a bit-exact comparison against a
/// stored field possible in the application this op came from, and it is the
/// half of that evidence which does not need the file.
#[test]
fn the_squared_field_is_where_the_comparison_is_exact() {
    let params = DistanceParams::default();
    let mut distinct = std::collections::BTreeSet::new();
    for (name, mask) in fixtures() {
        let squared = squared_distance_transform(mask.view(), &params).expect("the transform");
        let distance = distance_transform(mask.view(), &params).expect("the transform");
        for (squared, distance) in squared.iter().zip(distance.iter()) {
            if !squared.is_finite() {
                continue;
            }
            assert_eq!(
                *squared,
                squared.round(),
                "{name}: a squared distance at unit sampling must be an exact integer"
            );
            // The round trip: the distance squares back to the same integer,
            // which is the comparison a caller does against a stored field.
            assert_eq!(
                (distance * distance).round(),
                *squared,
                "{name}: the distance does not square back to its own integer"
            );
            distinct.insert(*squared as u64);
        }
        // And a threshold against an integer needs no root: the two predicates
        // agree at every voxel, for every radius the fixture reaches.
        for radius in 1..=6u64 {
            let by_square = squared
                .iter()
                .filter(|value| **value <= (radius * radius) as f64)
                .count();
            let by_root = distance
                .iter()
                .filter(|value| **value <= (radius as f64))
                .count();
            assert_eq!(by_square, by_root, "{name}: threshold at {radius}");
        }
    }
    println!(
        "{} distinct squared distances over the five fixtures, every one an exact integer, \
         every threshold identical with and without the square root",
        distinct.len()
    );
}

// ------------------------------------------------- claim 2: the approximation --

/// **The liveness test for every parity assertion above.**
///
/// A chamfer is exact along the directions its 26-offset set can represent and
/// wrong in between. On the oblique slab that is nearly everywhere; on a centred
/// ball it is a small error that looks like rounding. A test suite built on the
/// ball alone would pass with a chamfer in place of the transform, which is why
/// the slab is the fixture that ships and the ball is only ever the control.
#[test]
fn the_chamfer_is_visible_on_the_slab_and_invisible_on_the_ball() {
    let params = DistanceParams::default();

    let slab = oblique_sheet([23, 23, 23]);
    let exact = distance_transform(slab.view(), &params).expect("the transform must run");
    let approximate = chamfer_distance(slab.view());
    let wrong = differing(&exact, &approximate);
    let worst = worst_gap(&exact, &approximate);
    assert_eq!(
        wrong, 9484,
        "the discriminating fixture's recorded chamfer error has moved"
    );
    assert_eq!(exact.len(), 12_167);
    assert!(
        worst > 0.5,
        "on the slab the chamfer must be wrong by more than half a voxel, or it is not a \
         discriminating fixture; it is wrong by {worst}"
    );

    let ball = centred_ball(23, 9.0);
    let exact = distance_transform(ball.view(), &params).expect("the transform must run");
    let approximate = chamfer_distance(ball.view());
    let ball_wrong = differing(&exact, &approximate);
    let ball_worst = worst_gap(&exact, &approximate);
    assert_eq!(
        ball_wrong, 1167,
        "the control fixture's recorded error has moved"
    );
    assert!(
        ball_worst < 0.5,
        "on the ball the chamfer must look right, which is the whole point of the control; \
         it is wrong by {ball_worst}"
    );

    println!(
        "chamfer against the exact transform: oblique slab {wrong} of 12167 voxels wrong, worst \
         by {worst:.4}; centred ball {ball_wrong} of 12167 wrong, worst by {ball_worst:.4} — \
         under half a voxel, which is why a ball cannot be the parity fixture"
    );
}

// -------------------------------------------------- claim 3: the synonyms --

/// The three 1-D passes commute, and at unit sampling every intermediate is an
/// exact integer below `2^53`, so all six orders are *bit*-identical.
#[test]
fn the_pass_order_is_a_synonym() {
    for (name, mask) in fixtures() {
        for sampling in [[1.0, 1.0, 1.0], [1.0, 2.0, 5.0]] {
            let params = DistanceParams::default().with_sampling(sampling);
            let scales = params.squared_sampling().unwrap();
            let reference =
                squared_distance_transform(mask.view(), &params).expect("the transform");
            for order in permutations() {
                let mut field = seed(mask.view());
                for axis in order {
                    sweep_axis(&mut field, axis, scales[axis]);
                }
                assert_eq!(
                    differing(&field, &reference),
                    0,
                    "{name} at sampling {sampling:?}: pass order {order:?} changed the answer"
                );
            }
        }
    }
    println!("all six pass orders bit-identical on five fixtures at two samplings");
}

/// The transform is equivariant — `distance(mask.permuted())` is
/// `distance(mask).permuted()` — so the axis a pass sweeps is not observable
/// from the answer. What *is* observable is a sampling vector applied to the
/// wrong axes, and only anisotropically. So this is the test that makes the axes
/// checkable at all.
#[test]
fn the_axes_are_interchangeable_and_the_sampling_vectors_are_not() {
    let mask = oblique_sheet([23, 23, 23]);
    let params = DistanceParams::default();
    let field = distance_transform(mask.view(), &params).expect("the transform must run");

    for axes in permutations() {
        let permuted_mask = permute(&mask, axes);
        let of_permuted =
            distance_transform(permuted_mask.view(), &params).expect("the transform must run");
        assert_eq!(
            of_permuted,
            permute(&field, axes),
            "the transform is not equivariant under {axes:?}, which it must be"
        );
    }

    let straight = distance_transform(mask.view(), &params.with_sampling([1.0, 2.0, 5.0]))
        .expect("the transform must run");
    let swapped = distance_transform(mask.view(), &params.with_sampling([2.0, 1.0, 5.0]))
        .expect("the transform must run");
    let moved = differing(&straight, &swapped);
    assert!(
        moved > 0,
        "swapping two entries of the sampling vector has to move the answer, or `sampling` is \
         not a parameter"
    );
    println!(
        "axis permutations: 6 of 6 equivariant at unit sampling, so a permuted pass order is \
         undetectable; swapping two sampling entries moves {moved} of {} voxels",
        mask.len()
    );
}

// ---------------------------------------------- claim 4: the decomposition --

/// The same volume however it is cut.
///
/// The lattices are chosen to cut where it would hurt if the answer were
/// block-local: the sheet's longest run of foreground is more than half the box,
/// and edges of 5, 7 and 9 put a seam in the middle of it. The swept axis is
/// never cut by [`distance::plan`] — that is the plan's whole shape — so what is
/// varied is the other two, and one case cuts every axis at one voxel.
#[test]
fn the_field_is_decomposition_invariant() {
    let params = DistanceParams::default();
    let mask = oblique_sheet([29, 23, 19]);
    let input = Voxels::Bool(mask.clone());
    let whole = distance_transform(mask.view(), &params).expect("the transform must run");

    let mut cuts = 0;
    for block in [1usize, 2, 3, 5, 7, 8, 9, 11, 16, 23, 29] {
        let blocked = run_blocked(&params, &input, block).expect("the blocked run");
        assert_eq!(
            blocked,
            Voxels::F64(whole.clone()),
            "block edge {block} changed the answer"
        );
        cuts += 1;
    }
    assert_eq!(cuts, 11, "the sweep did not run");
    println!("{cuts} decompositions, all byte-identical to the whole-volume answer");
}

/// **The negative control.** If a block-local transform were the volume's, the
/// test above would say nothing.
///
/// The block-local answer is what an implementation that took the halo at face
/// value would produce: each block transformed on its own, so a voxel's distance
/// is to the nearest background *in its block*. It is a well-formed volume and it
/// is wrong, which is the failure this op's declaration exists to prevent.
#[test]
fn a_block_local_transform_is_not_the_volume_wide_one() {
    let params = DistanceParams::default();
    let mask = oblique_sheet([29, 23, 19]);
    let whole = distance_transform(mask.view(), &params).expect("the transform must run");

    let block = 8usize;
    let mut local = Array3::<f64>::zeros(whole.raw_dim());
    let shape = [mask.shape()[0], mask.shape()[1], mask.shape()[2]];
    let mut starts = Vec::new();
    for i in (0..shape[0]).step_by(block) {
        for j in (0..shape[1]).step_by(block) {
            for k in (0..shape[2]).step_by(block) {
                starts.push([i, j, k]);
            }
        }
    }
    for start in &starts {
        let extent = [
            (start[0] + block).min(shape[0]) - start[0],
            (start[1] + block).min(shape[1]) - start[1],
            (start[2] + block).min(shape[2]) - start[2],
        ];
        let piece = mask.slice(ndarray::s![
            start[0]..start[0] + extent[0],
            start[1]..start[1] + extent[1],
            start[2]..start[2] + extent[2]
        ]);
        let field = distance_transform(piece, &params).expect("the transform must run");
        for ((i, j, k), value) in field.indexed_iter() {
            local[[start[0] + i, start[1] + j, start[2] + k]] = *value;
        }
    }
    let wrong = differing(&whole, &local);
    assert_eq!(
        wrong, 7795,
        "the recorded block-local error has moved; if it fell to 0 the invariance test above is \
         vacuous"
    );
    assert_eq!(starts.len(), 36);
    assert_eq!(whole.len(), 12_673);
    println!(
        "block-local at edge {block}: {wrong} of {} voxels wrong over {} blocks",
        whole.len(),
        starts.len()
    );
}

/// The guard the op carries itself, and **the only way to reach it**.
///
/// A plan cannot hand a sweep a partial lane: the whole-axis declaration means
/// the halo is the axis, and a lattice that grants less is refused when the
/// decomposition is checked — `a_short_halo_against_the_whole_axis_reach_is_refused`
/// runs that on this op's own plan. What the planner cannot reach is a caller who
/// never built a plan, and `apply` is public. So the anchor here is built by
/// hand, because an assertion nothing can run is not an assertion.
#[test]
fn a_sweep_handed_a_partial_lane_refuses_rather_than_answering() {
    let op = DistanceSweepOp::along(0, 1.0, true).expect("axis 0 at unit pitch");
    let partial = Voxels::Bool(Array3::from_elem((8, 4, 4), true));
    let mut out = Voxels::F64(Array3::zeros((8, 4, 4)));
    let error = op
        .apply(&partial, &mut out, &Anchor::new([8, 0, 0], [29, 4, 4]))
        .expect_err("a partial lane must be refused");
    let text = format!("{error}");
    assert!(text.contains("sweeps axis 0"), "{text}");
    assert!(text.contains("block-local"), "{text}");

    // A buffer that starts at 0 but is short is refused for the same reason, and
    // is the case a reader is likelier to write by accident.
    let error = op
        .apply(&partial, &mut out, &Anchor::new([0, 0, 0], [29, 4, 4]))
        .expect_err("a short lane must be refused");
    assert!(format!("{error}").contains("block-local"), "{error}");

    // And the whole lane is accepted, so the refusal is about the lane and not
    // about the op.
    let whole = Voxels::Bool(Array3::from_elem((29, 4, 4), true));
    let mut out = Voxels::F64(Array3::zeros((29, 4, 4)));
    op.apply(&whole, &mut out, &Anchor::whole([29, 4, 4]))
        .expect("the whole lane must be accepted");
}

/// The plan's shape, which is what makes the sweep cheap rather than merely
/// correct.
#[test]
fn the_plan_leaves_each_swept_axis_whole_and_cuts_the_other_two() {
    let volume = [12usize, 10, 8];
    let assembly = distance::plan(&DistanceParams::default(), volume, 4).expect("the plan");
    let phases = &assembly.decomposition.phases;
    assert_eq!(phases.len(), 4, "three sweeps and a finish");
    for axis in 0..3usize {
        let grid = phases[axis].grid.block();
        assert_eq!(
            grid[axis], volume[axis],
            "the sweep on axis {axis} must not have its own axis cut"
        );
        for other in (0..3).filter(|&other| other != axis) {
            assert_eq!(grid[other], 4, "axis {other} of the sweep on {axis}");
        }
        assert!(
            phases[axis].reach.is_whole_axis(axis, volume[axis]),
            "the sweep on axis {axis} must declare the whole of it"
        );
        for other in (0..3).filter(|&other| other != axis) {
            assert!(!phases[axis].reach.is_whole_axis(other, volume[other]));
        }
        // Every block reads the whole of the swept axis, which is what the
        // declaration is a claim about.
        for block in &phases[axis].blocks {
            assert_eq!(block.read.start[axis], 0);
            assert_eq!(block.read.shape[axis], volume[axis]);
        }
    }
    // The finish is pointwise, so it is cut on every axis and is the only phase
    // that is not a planning barrier.
    assert_eq!(phases[3].grid.block(), [4, 4, 4]);
    assert!(!phases[3].reach.is_barrier(volume));
    assembly.decomposition.check().expect("the plan must check");
}

/// **The whole-axis mandate, both halves.**
///
/// A short halo against `AxisReach::All` is refused — no block of a cut axis has
/// a trustworthy voxel, so the valid regions do not tile — and a lattice that
/// cuts the swept axis *with* the whole-axis halo is accepted and is right,
/// merely redundant. Those two together are what makes the declaration a mandate
/// rather than a preference, and they are what the guard inside `apply` no longer
/// has to do for a caller who builds a plan.
#[test]
fn a_short_halo_against_the_whole_axis_reach_is_refused() {
    let params = DistanceParams::default();
    let volume = [29usize, 23, 19];
    let mask = oblique_sheet(volume);
    let resident = distance_transform(mask.view(), &params).expect("the transform must run");

    // The plan this module builds is **immune** to a forced short halo, and that
    // is worth recording rather than working around: `sweep_grid` leaves the
    // swept axis uncut, so every block spans it whatever the halo says, and the
    // halo on that axis buys nothing because there is nothing outside the block
    // to fetch. The guard has nothing to fire on because the lattice already
    // satisfies the declaration.
    let good = distance::plan(&params, volume, 8).expect("the plan");
    good.decomposition
        .check()
        .expect("the plan as built checks");
    good.decomposition
        .with_forced_halo([0usize, 0, 0])
        .check()
        .expect("a plan that leaves the swept axis whole needs no halo on it");

    // So the mandate is provoked on the lattice that *does* cut the swept axis:
    // a cube grid on every phase. With the whole-axis halo it is accepted — it
    // is redundant, not wrong, and the answer is the resident one.
    let cube = || BlockGrid::new(volume, [8, 8, 8]).unwrap();
    let scales = params.squared_sampling().unwrap();
    let mut builder = PlanBuilder::new(volume, Dtype::Bool, cube());
    builder
        .pixels(blockflow::op::Chain::op(
            DistanceSweepOp::along(0, scales[0].sqrt(), true).unwrap(),
        ))
        .expect("the first sweep");
    for axis in 1..3 {
        builder.regrid(cube());
        builder
            .pixels(blockflow::op::Chain::op(
                DistanceSweepOp::along(axis, scales[axis].sqrt(), false).unwrap(),
            ))
            .expect("a later sweep");
    }
    builder.regrid(cube());
    builder
        .pixels(blockflow::op::Chain::op(DistanceFinishOp::new(params)))
        .expect("the finish");
    let redundant = builder.finish().expect("the redundant plan");
    redundant
        .decomposition
        .check()
        .expect("cutting the swept axis with the full halo is legal");
    for axis in 0..3usize {
        for block in &redundant.decomposition.phases[axis].blocks {
            assert_eq!(
                (block.read.start[axis], block.read.shape[axis]),
                (0, volume[axis]),
                "every block of a cut swept axis must re-read the whole lane"
            );
        }
    }
    let answer = execute(
        &redundant.decomposition,
        &redundant,
        &Voxels::Bool(mask.clone()),
    )
    .expect("the redundant plan must run");
    assert_eq!(answer, Voxels::F64(resident));

    // And grant that same lattice anything less than the whole axis and no
    // interior block has a trustworthy voxel, so the valid regions do not tile
    // and `Decomposition::check` refuses by name. That is the half of the
    // mandate that makes it a mandate: whole, or a whole-axis halo, and there is
    // no third option a plan can express.
    // The last is the interesting one: it is one voxel short of covering
    // the axis from the *first* block, and a halo that covers it from every
    // block is a whole-axis halo by another name — `[28, 22, 18]` here would be
    // accepted, and rightly, because it fetches everything.
    for halo in [[4usize, 4, 4], [0, 0, 0], [8, 8, 8], [20, 14, 10]] {
        let short = redundant.decomposition.with_forced_halo(halo);
        let message = short
            .check()
            .expect_err(&format!("halo {halo:?} must not check"))
            .to_string();
        assert!(
            message.contains("do not tile the volume exactly"),
            "halo {halo:?}: {message}"
        );
    }
    println!(
        "the plan `sweep_grid` builds needs no halo at all on the swept axis; the cube lattice          that cuts it is accepted only with the whole-axis halo, every block then re-reads the          whole lane and the answer is byte-identical to the resident one, and all four short          halos tried are refused"
    );
}

/// **The residency argument, checked at two scales without running either.**
///
/// An op whose reach is the whole volume is only worth blocking if the block
/// budget is bounded, and the module header claims it is. The claim is
/// `decomposition::price_phase`'s own figure for this plan — this reads it back
/// rather than restating the multiplication.
#[test]
fn the_block_budget_is_bounded_at_both_scales() {
    let params = DistanceParams::default();
    let scales = [
        ("77.0 Mvoxel", [320usize, 528, 456], 616_366_080.0f64),
        ("1.775 Gvoxel", [404usize, 1304, 3369], 14_198_744_832.0f64),
    ];
    for (name, volume, whole_image_bytes) in scales {
        let voxels: f64 = volume.iter().map(|edge| *edge as f64).product();
        assert_eq!(
            voxels * 8.0,
            whole_image_bytes,
            "{name}: the image's own size"
        );
        for block in [32usize, 64] {
            let per_phase =
                distance::working_set_bytes(&params, volume, block).expect("the plan must price");
            assert_eq!(per_phase.len(), 4);
            let worst = per_phase.iter().copied().fold(0.0f64, |best, value| {
                if value.total_cmp(&best).is_gt() {
                    value
                } else {
                    best
                }
            });
            // The sweep on the longest axis is the worst, and it is that axis'
            // whole extent times the two free block edges, both buffers.
            let longest = *volume.iter().max().unwrap() as f64;
            assert_eq!(worst, longest * block as f64 * block as f64 * 8.0 * 2.0);
            assert!(
                worst < whole_image_bytes / 10.0,
                "{name} at block {block}: {worst} is not a budget against {whole_image_bytes}"
            );
            println!(
                "{name} at block {block}: worst phase holds {:.1} MB per block against {:.1} MB \
                 for one resident image",
                worst / 1e6,
                whole_image_bytes / 1e6
            );
        }
    }
}

// ------------------------------------------------- claim 5: the degenerates --

/// The inputs where a distance transform has no interior, pinned against the two
/// references' own recorded answers.
#[test]
fn the_degenerate_volumes_are_pinned() {
    let params = DistanceParams::default();

    // All background. Every voxel is its own nearest background voxel.
    let mask = Array3::from_elem((4, 5, 6), false);
    let field = distance_transform(mask.view(), &params).expect("the transform must run");
    assert!(field.iter().all(|value| *value == 0.0));
    assert_eq!(
        differing(&field, &brute_force_distance(mask.view(), &params).unwrap()),
        0
    );

    // All foreground: **no clipping, and the two references part.** SciPy's
    // recorded answer for a `2 x 3 x 4` volume is the distance to a phantom
    // background voxel at `(-1, 0, 0)`; the other reference's is `+inf`.
    let mask = Array3::from_elem((2, 3, 4), true);
    let phantom = distance_transform(mask.view(), &params).expect("the transform must run");
    assert_eq!(phantom[[0, 0, 0]], 1.0);
    assert_eq!(phantom[[1, 2, 3]], 17.0f64.sqrt());
    assert!(phantom.iter().all(|value| value.is_finite()));
    assert_eq!(
        differing(
            &phantom,
            &brute_force_distance(mask.view(), &params).unwrap()
        ),
        0,
        "the brute-force search must honour the same rule, or it is not the same function"
    );
    let infinite = distance_transform(mask.view(), &params.with_unbounded(Unbounded::Infinite))
        .expect("the transform must run");
    assert!(infinite.iter().all(|value| value.is_infinite()));
    assert_eq!(
        differing(
            &infinite,
            &brute_force_distance(mask.view(), &params.with_unbounded(Unbounded::Infinite))
                .unwrap()
        ),
        0
    );

    // The rule is inert wherever there is a background voxel, which is why the
    // sweep above needs a volume that has none.
    let sheet = oblique_sheet([17, 13, 11]);
    assert_eq!(
        differing(
            &distance_transform(sheet.view(), &params).unwrap(),
            &distance_transform(sheet.view(), &params.with_unbounded(Unbounded::Infinite)).unwrap()
        ),
        0,
        "the all-foreground rule cannot move a volume that has a background voxel"
    );

    // A single foreground voxel: one voxel from the surface, and nothing else
    // moved.
    let mut mask = Array3::from_elem((5, 5, 5), false);
    mask[[2, 2, 2]] = true;
    let field = distance_transform(mask.view(), &params).expect("the transform must run");
    assert_eq!(field[[2, 2, 2]], 1.0);
    assert_eq!(field.iter().filter(|value| **value != 0.0).count(), 1);

    // A volume of one voxel, both ways.
    assert_eq!(
        distance_transform(Array3::from_elem((1, 1, 1), false).view(), &params).unwrap()[[0, 0, 0]],
        0.0
    );
    assert_eq!(
        distance_transform(Array3::from_elem((1, 1, 1), true).view(), &params).unwrap()[[0, 0, 0]],
        1.0,
        "the phantom voxel sits one step off axis 0, so a lone foreground voxel of a lone voxel \
         volume is at distance 1"
    );

    // An empty axis: no voxels, no answer, and no panic.
    let empty = distance_transform(Array3::from_elem((0, 3, 3), true).view(), &params)
        .expect("an empty volume must not be an error");
    assert_eq!(empty.len(), 0);

    println!(
        "the degenerate volumes are pinned; the all-foreground one is where the two references \
         disagree and both readings are recorded"
    );
}

/// A pitch that is not a length is refused everywhere it can be stated, rather
/// than dividing by zero inside the envelope.
#[test]
fn a_pitch_that_is_not_a_length_is_refused() {
    let mask = Array3::from_elem((3, 3, 3), true);
    for pitch in [
        [0.0, 1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, f64::NAN],
        [f64::INFINITY, 1.0, 1.0],
    ] {
        let params = DistanceParams::default().with_sampling(pitch);
        assert!(
            distance_transform(mask.view(), &params).is_err(),
            "{pitch:?}"
        );
        assert!(params.squared_sampling().is_err(), "{pitch:?}");
        assert!(distance::plan(&params, [3, 3, 3], 2).is_err(), "{pitch:?}");
    }
    for bad in [0.0, -1.0, f64::NAN] {
        assert!(DistanceSweepOp::along(0, bad, true).is_err(), "{bad}");
    }
    assert!(
        DistanceSweepOp::along(3, 1.0, true).is_err(),
        "there is no axis 3"
    );
    assert!(
        distance::sweep_grid([4, 4, 4], 0, 0).is_err(),
        "a block edge of 0"
    );
}

/// The finish declares that it maps `+0.0` to `+0.0` and claims nothing else,
/// and the claim is checked against what it actually does.
#[test]
fn the_finishs_constant_declaration_is_true() {
    let params = DistanceParams::default();
    let finish = DistanceFinishOp::new(params);
    assert_eq!(finish.constant_maps_to(0.0), Some(0.0));
    assert_eq!(finish.constant_maps_to(-0.0), None);
    assert_eq!(finish.constant_maps_to(4.0), None);
    assert_eq!(finish.constant_maps_to(f64::INFINITY), None);

    // And the op agrees with its own declaration on a block that is entirely
    // background, which is the case the planner would use it for.
    let input = Voxels::F64(Array3::zeros((4, 4, 4)));
    let mut out = Voxels::F64(Array3::from_elem((4, 4, 4), 7.0));
    finish
        .apply(&input, &mut out, &Anchor::new([4, 0, 0], [12, 4, 4]))
        .expect("the finish must run");
    match &out {
        Voxels::F64(array) => assert!(array
            .iter()
            .all(|value| value.to_bits() == 0.0f64.to_bits())),
        other => panic!("the finish wrote {:?}", other.dtype()),
    }
}

// -------------------------------------------------------------------- helpers --

fn differing(ours: &Array3<f64>, theirs: &Array3<f64>) -> usize {
    assert_eq!(ours.shape(), theirs.shape());
    ours.iter()
        .zip(theirs.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count()
}

fn worst_gap(ours: &Array3<f64>, theirs: &Array3<f64>) -> f64 {
    ours.iter()
        .zip(theirs.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, |best, gap| {
            if gap.total_cmp(&best).is_gt() {
                gap
            } else {
                best
            }
        })
}

fn permutations() -> [[usize; 3]; 6] {
    [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ]
}

fn permute<T: Copy>(array: &Array3<T>, axes: [usize; 3]) -> Array3<T> {
    let shape = array.shape().to_vec();
    let target = (shape[axes[0]], shape[axes[1]], shape[axes[2]]);
    Array3::from_shape_fn(target, |(i, j, k)| {
        let from = [i, j, k];
        let mut source = [0usize; 3];
        for slot in 0..3 {
            source[axes[slot]] = from[slot];
        }
        array[[source[0], source[1], source[2]]]
    })
}

/// FNV-1a over the field's `float64` bytes in C order, which is what the Python
/// side of `the_field_reproduces_scipys_own_numbers` computed.
fn fnv1a(field: &Array3<f64>) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for value in field.iter() {
        for byte in value.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }
    hash
}

/// The transform through the executor, block by block — the path the whole
/// declaration exists for.
fn run_blocked(params: &DistanceParams, mask: &Voxels, block: usize) -> blockflow::Result<Voxels> {
    let assembly = distance::plan(params, mask.shape(), block)?;
    execute(&assembly.decomposition, &assembly, mask)
}

fn execute(
    decomposition: &Decomposition,
    assembly: &blockflow::assemble::Assembly,
    mask: &Voxels,
) -> blockflow::Result<Voxels> {
    let env = ArrayEnvironment::for_decomposition(mask.clone(), decomposition, [8, 8, 8])?;
    execute_phases(
        "distance",
        &assembly.workflow,
        decomposition,
        &Hints::default(),
        &env,
        &[],
        &assembly.work(),
    )?;
    Ok(env.output())
}
