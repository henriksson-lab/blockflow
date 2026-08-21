// SPDX-License-Identifier: MIT
//
// The **intermediate element type** a lattice-sampled statistic passes through,
// and the one thing about it that cannot be checked by comparing two
// implementations of it against each other.
//
// The parameter is `ops::local::LatticeNarrowing` and it has two sites: a value
// is narrowed once at the **sample grid**, before anything is interpolated, and
// once **after the interpolation**. The two use different rounding rules, they
// sit at different places, and — this is the whole reason this file exists — an
// implementation that gets one of them right and the other wrong produces an
// answer that is *close* to the correct one at every voxel. Close is what a
// comparison against a wrong reference looks like too, so nothing here compares
// this code against another implementation of the same idea. Every number below
// is either computed from the arithmetic by hand, or is a claim about the code
// against itself (the default is unchanged; every decomposition agrees; the
// split pair is the fused op).
//
// What each test settles
// ----------------------
// | test | what would break it |
// |---|---|
// | `the_default_narrows_at_neither_site_and_is_byte_unchanged` | any narrowing leaking into a caller that did not ask for one |
// | `the_two_sites_are_different_operations_at_different_places` | swapping the two rounding rules, or applying either site alone |
// | `every_decomposition_gives_the_whole_volume_answer_with_narrowing_on` | narrowing a block's *own* grid rather than the lattice's samples |
// | `the_reach_does_not_move_when_the_values_are_narrowed` | a narrowing that was mistaken for a read |
// | `the_split_pair_narrowed_is_the_fused_op_narrowed` | the two halves of the split disagreeing about which site each owns |
// | `a_uniform_block_is_declared_at_the_value_the_kernel_writes` | a short circuit filled with the un-narrowed value |
//
// The hand-computed case, and why it is the shape it is
// -----------------------------------------------------
// A test that pins "the narrowing happens" is easy and says nothing: on most
// data, truncating at the grid and rounding afterwards lands on the same number
// as rounding at the grid and truncating afterwards, so a test built on ordinary
// values passes under both. `the_two_sites_are_different_operations_at_different_places`
// is built on two sample values chosen so that **all four** arrangements — the
// correct pair, the swapped pair, either site alone — give different answers,
// and it asserts the whole table rather than only the right row.

use ndarray::Array3;

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, BlockOp, Chain};
use blockflow::ops::element::{ElementShape, Rank, StructuringElement, Total};
use blockflow::ops::lattice::{
    interpolate_block_edge, lattice_interpolate_phase, lattice_statistic_phase,
    statistic_block_edge, LatticeInterpolateOp, LatticeStatisticOp,
};
use blockflow::ops::local::{
    Alignment, LatticeNarrowing, LocalStatistic, LocalStatisticOp, Narrowing, Rounding,
    SampleLattice, Sampling, Statistic,
};
use blockflow::strategy::{execute, Hints, Workflow};
use blockflow::voxels::Voxels;
use blockflow::Dtype;

/// Round on no axis, so a spacing divides none of them unless it was chosen to.
const VOLUME: [usize; 3] = [30, 24, 18];

/// Structure at several scales, and values that are **not** whole numbers once a
/// mean has been taken over them — otherwise a narrowing would be the identity
/// and every assertion below would hold vacuously.
fn texture(shape: [usize; 3]) -> Array3<f64> {
    Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
        ((i * 7919 + j * 104729 + k * 1299709) % 1013) as f64 + 1.0
    })
}

fn element(size: [usize; 3]) -> StructuringElement {
    StructuringElement::from_size(ElementShape::Box, size).unwrap()
}

fn to_u16() -> LatticeNarrowing {
    LatticeNarrowing::through(Dtype::U16).unwrap()
}

fn statistic(size: [usize; 3], step: [usize; 3], alignment: Alignment) -> LocalStatistic {
    LocalStatistic::sampled(element(size), Sampling::every(step), Statistic::Mean)
        .unwrap()
        .with_alignment(alignment)
}

/// The whole-volume answer, taken by applying the chain to the entire array.
fn whole(chain: &Chain, input: &Voxels) -> Voxels {
    let shape = input.shape();
    let mut out = Voxels::zeros(
        chain.produces(input.dtype()).unwrap(),
        chain.output_shape(shape).unwrap(),
    )
    .unwrap();
    chain.apply(input, &mut out, &Anchor::whole(shape)).unwrap();
    out
}

fn values(voxels: &Voxels) -> Array3<f64> {
    voxels.view::<f64>().unwrap().to_owned()
}

// ------------------------------------------- 1. the default is unchanged --

/// **The acceptance criterion every existing caller depends on.**
///
/// `LatticeNarrowing::default()` narrows at neither site, so a statistic that
/// states it computes what a statistic that says nothing computes — bit for bit,
/// through the fused op, the masked path, the threshold shell and both halves of
/// the split.
///
/// Asserted on the **bits** rather than on `==`, because the failure this guards
/// against is a rounding that changes the last one. And asserted alongside its
/// own non-vacuity: the same statistic *with* narrowing on differs, so the
/// equality above is a fact about the default and not about the data.
#[test]
fn the_default_narrows_at_neither_site_and_is_byte_unchanged() {
    let input: Voxels = texture(VOLUME).into();
    let mut compared = 0;
    for size in [[3usize, 3, 3], [5, 5, 3]] {
        for step in [[7usize, 5, 4], [4, 4, 4]] {
            for alignment in [Alignment::SamplePositions, Alignment::PinnedEnds] {
                let plain = whole(
                    &Chain::op(LocalStatisticOp::new(
                        "plain",
                        statistic(size, step, alignment),
                    )),
                    &input,
                );
                let stated = whole(
                    &Chain::op(
                        LocalStatisticOp::new("stated", statistic(size, step, alignment))
                            .narrowed(LatticeNarrowing::none()),
                    ),
                    &input,
                );
                let narrowed = whole(
                    &Chain::op(
                        LocalStatisticOp::new("narrowed", statistic(size, step, alignment))
                            .narrowed(to_u16()),
                    ),
                    &input,
                );
                identical(
                    &values(&plain),
                    &values(&stated),
                    &format!("{size:?} {step:?} {alignment:?}"),
                );
                differs(
                    &values(&plain),
                    &values(&narrowed),
                    &format!("{size:?} {step:?} {alignment:?}"),
                );
                compared += 1;
            }
        }
    }
    assert_eq!(compared, 8);

    // The default is `none()` and not merely equal to it, which is what makes
    // "a caller who says nothing" and "a caller who says `none()`" the same
    // caller rather than two that happen to agree.
    assert_eq!(
        statistic([3, 3, 3], [4, 4, 4], Alignment::SamplePositions).narrowing(),
        LatticeNarrowing::none()
    );
    assert!(LatticeNarrowing::default().is_identity());
    assert_eq!(
        LatticeStatisticOp::new(
            "s",
            element([3, 3, 3]),
            &Sampling::every([4, 4, 4]),
            Statistic::Mean,
            VOLUME,
        )
        .unwrap()
        .narrowing(),
        None
    );
    assert_eq!(
        LatticeInterpolateOp::new(
            "i",
            SampleLattice::of(&Sampling::every([4, 4, 4]), VOLUME).unwrap(),
        )
        .unwrap()
        .narrowing(),
        None
    );
}

/// The other half of "unchanged": a narrowing that is never stated must not
/// change what an op *declares* either, since a declaration that moved would
/// change which blocks a plan skips.
#[test]
fn the_default_declares_what_it_always_declared() {
    let element = element([3, 3, 3]);
    let ranked = LocalStatistic::sampled(
        element.clone(),
        Sampling::every([4, 4, 4]),
        Statistic::Rank(Rank::median(&element)),
    )
    .unwrap();
    let op = LocalStatisticOp::new("ranked", ranked.clone());
    assert_eq!(op.constant_maps_to(0.25), Some(0.25));
    assert_eq!(
        LocalStatisticOp::new("ranked", ranked.clone())
            .narrowed(LatticeNarrowing::none())
            .constant_maps_to(0.25),
        Some(0.25)
    );
    // ...and with narrowing on it is the value the kernel would write, which is
    // a different number. Both sites are `u16`, so 0.25 truncates to 0 at the
    // grid and 0 rounds to 0 after it.
    assert_eq!(
        LocalStatisticOp::new("ranked", ranked)
            .narrowed(to_u16())
            .constant_maps_to(0.25),
        Some(0.0)
    );
}

// ------------------------------------- 2. where exactly the narrowing sits --

/// **The hand-computed case**, and it is arranged so that getting the two sites
/// the wrong way round gives a different answer rather than the same one.
///
/// The lattice is two samples on axis 0, at volume coordinates `0` and `4`, over
/// an axis of five voxels; the window is one voxel, so a sample's value is the
/// voxel under it exactly. The two sample values are **2.5** and **4.5**, and
/// they are half-integers on purpose: that is what makes `trunc` and `round`
/// disagree at both of them.
///
/// Worked out by hand, with `bracket` giving weights `0, 1/4, 1/2, 3/4, 0` along
/// the axis (the last voxel sits on the second sample and takes the degenerate
/// bracket):
///
/// ```text
/// site(s) narrowed        grid        the five voxels
/// neither                 2.5  4.5    2.5  3.0  3.5  4.0  4.5
/// both, trunc then round  2    4      2    3    3    4    4     <- what this asserts
/// both, round then trunc  3    5      3    3    4    4    5
/// after the blend only    2.5  4.5    3    3    4    4    5
/// at the grid only        2    4      2    2.5  3    3.5  4
/// ```
///
/// Four arrangements, four different rows — which is what a discriminating case
/// has to produce and what an arbitrary one does not. The correct row also pins
/// the *tie rule* of the second site: voxel 1 lands on exactly 2.5 and goes to
/// **3**, so the rounding is half away from zero and not half to even, which
/// would have sent it to 2.
///
/// **Confirmed against a widely used implementation of the same two steps**, in
/// process rather than from memory, so the table above is not this crate marking
/// its own homework. The lattice here puts a sample on the first and last voxel
/// of the axis, which is the arrangement an endpoint-pinned upsample assumes, so
/// the two constructions are comparable voxel for voxel:
///
/// ```python
/// import numpy as np
/// from scipy import ndimage as ndi
/// f = np.array([2.5, 4.5])
/// g = np.zeros(2, dtype=np.uint16); g[:] = f     # -> [2 4]   the first site
/// ndi.zoom(g, zoom=2.5, order=1)                 # -> [2 3 3 4 4]
/// ndi.zoom(g.astype(float), zoom=2.5, order=1)   # -> [2 2.5 3 3.5 4]
/// ndi.zoom(f, zoom=2.5, order=1)                 # -> [2.5 3 3.5 4 4.5]
/// ```
///
/// The second line is the correct row of the table, the third is "at the grid
/// only" and the fourth is "neither" — all three agreeing exactly with what is
/// asserted below. Truncating the third instead of rounding it gives
/// `[2 2 3 3 4]`, which is what makes the second site's rule *round* and not a
/// second truncation.
#[test]
fn the_two_sites_are_different_operations_at_different_places() {
    let volume = [5usize, 1, 1];
    let mut input = Array3::<f64>::from_elem((5, 1, 1), 100.0);
    input[[0, 0, 0]] = 2.5;
    input[[4, 0, 0]] = 4.5;

    let lattice = Sampling::At {
        positions: [vec![0, 4], vec![0], vec![0]],
    };
    let answer = |narrowing: LatticeNarrowing| -> Vec<f64> {
        let local = LocalStatistic::sampled(element([1, 1, 1]), lattice.clone(), Statistic::Mean)
            .unwrap()
            .narrowed(narrowing);
        let mut out = Array3::<f64>::zeros((5, 1, 1));
        local
            .evaluate_into(input.view(), &Anchor::whole(volume), out.view_mut())
            .unwrap();
        (0..5).map(|index| out[[index, 0, 0]]).collect()
    };

    let trunc = Narrowing::new(Dtype::U16, Rounding::TowardZero).unwrap();
    let round = Narrowing::new(Dtype::U16, Rounding::ToNearest).unwrap();

    // The pair the constructor names, and the row the arithmetic gives.
    let correct = answer(to_u16());
    assert_eq!(correct, vec![2.0, 3.0, 3.0, 4.0, 4.0]);
    assert_eq!(
        to_u16(),
        LatticeNarrowing {
            at_samples: Some(trunc),
            after_interpolation: Some(round),
        },
        "`through` must be trunc at the grid and round after it"
    );

    // The three arrangements that are wrong, each computed and each different.
    let swapped = answer(LatticeNarrowing {
        at_samples: Some(round),
        after_interpolation: Some(trunc),
    });
    let after_only = answer(LatticeNarrowing {
        at_samples: None,
        after_interpolation: Some(round),
    });
    let grid_only = answer(LatticeNarrowing {
        at_samples: Some(trunc),
        after_interpolation: None,
    });
    let neither = answer(LatticeNarrowing::none());

    assert_eq!(swapped, vec![3.0, 3.0, 4.0, 4.0, 5.0]);
    assert_eq!(after_only, vec![3.0, 3.0, 4.0, 4.0, 5.0]);
    assert_eq!(grid_only, vec![2.0, 2.5, 3.0, 3.5, 4.0]);
    assert_eq!(neither, vec![2.5, 3.0, 3.5, 4.0, 4.5]);

    for (name, wrong) in [
        ("the two rules swapped", &swapped),
        ("only the site after the blend", &after_only),
        ("only the site at the grid", &grid_only),
        ("no narrowing at all", &neither),
    ] {
        assert_ne!(
            &correct, wrong,
            "{name} must not reach the same answer, or this case discriminates nothing"
        );
    }
}

/// The two rounding rules, on their own, at the values where they part.
#[test]
fn the_rounding_rules_are_what_they_are_named() {
    let trunc = Narrowing::new(Dtype::I16, Rounding::TowardZero).unwrap();
    let round = Narrowing::new(Dtype::I16, Rounding::ToNearest).unwrap();
    for (value, toward_zero, to_nearest) in [
        (2.5f64, 2.0, 3.0),
        (-2.5, -2.0, -3.0),
        (3.5, 3.0, 4.0),
        (2.9, 2.0, 3.0),
        (-0.4, -0.0, -0.0),
        (7.0, 7.0, 7.0),
    ] {
        assert_eq!(trunc.apply(value), toward_zero, "trunc of {value}");
        assert_eq!(round.apply(value), to_nearest, "round of {value}");
    }
    // Half **away from zero**, not half to even: 2.5 and 3.5 both round up.
    assert_eq!(round.apply(2.5), 3.0);
    assert_eq!(round.apply(3.5), 4.0);
}

/// Out of range saturates and a `NaN` goes to zero, which is what every other
/// narrowing in this crate does and is therefore what a value passing through
/// here has to do as well.
#[test]
fn a_value_outside_the_range_saturates_and_a_nan_goes_to_zero() {
    let narrowing = Narrowing::to(Dtype::U16).unwrap();
    assert_eq!(narrowing.apply(70000.0), 65535.0);
    assert_eq!(narrowing.apply(-3.5), 0.0);
    assert_eq!(narrowing.apply(f64::NAN), 0.0);
    assert_eq!(narrowing.apply(f64::INFINITY), 65535.0);

    let signed = Narrowing::to(Dtype::I8).unwrap();
    assert_eq!(signed.apply(200.0), 127.0);
    assert_eq!(signed.apply(-200.0), -128.0);

    // A float element type narrows resolution rather than range, and the
    // rounding rule still applies before it — one rule for every type.
    let single = Narrowing::new(Dtype::F32, Rounding::ToNearest).unwrap();
    assert_eq!(single.apply(2.5), 3.0);
    assert_eq!(Narrowing::to(Dtype::F64).unwrap().apply(2.5), 2.0);
}

/// An element type with no value for a rounding rule to produce is refused where
/// it is **stated**, and the refusal says which one it is.
#[test]
fn an_element_type_with_no_rounding_rule_is_refused_by_name() {
    let two_valued = Narrowing::to(Dtype::Bool).unwrap_err().to_string();
    assert!(two_valued.contains("two-valued"), "got: {two_valued}");
    let half = Narrowing::to(Dtype::F16).unwrap_err().to_string();
    assert!(half.contains("half precision"), "got: {half}");
    assert!(LatticeNarrowing::through(Dtype::Bool).is_err());
    assert!(LatticeNarrowing::through(Dtype::U8).is_ok());
}

// --------------------------------- 3. decomposition, with narrowing on --

/// **Not a formality.** A narrowing at a sample grid interacts with *which
/// samples a block holds*: a block evaluates only the samples its own voxels
/// bracket, so if the narrowing were applied to a block's grid as a unit — a
/// scale derived from what that block saw, say, or a rounding relative to the
/// block's own extreme — every cut of the volume would give a different answer.
/// It is applied per sample, to a value already in hand, which is why it
/// survives; this is what says so.
///
/// Swept over both alignments and over the two sites separately as well as
/// together, because the site at the grid and the site after the blend are
/// reached by different code and a block seam could break either.
#[test]
fn every_decomposition_gives_the_whole_volume_answer_with_narrowing_on() {
    let input: Voxels = texture(VOLUME).into();
    let trunc = Narrowing::new(Dtype::U16, Rounding::TowardZero).unwrap();
    let round = Narrowing::new(Dtype::U16, Rounding::ToNearest).unwrap();
    let narrowings = [
        ("both sites", to_u16()),
        (
            "the grid alone",
            LatticeNarrowing {
                at_samples: Some(trunc),
                after_interpolation: None,
            },
        ),
        (
            "the blend alone",
            LatticeNarrowing {
                at_samples: None,
                after_interpolation: Some(round),
            },
        ),
    ];

    let mut checked = 0;
    for (label, narrowing) in narrowings {
        for step in [[7usize, 5, 4], [4, 4, 4]] {
            for alignment in [Alignment::SamplePositions, Alignment::PinnedEnds] {
                let op = || {
                    LocalStatisticOp::new("narrowed", statistic([3, 3, 3], step, alignment))
                        .narrowed(narrowing)
                };
                let reference = whole(&Chain::op(op()), &input);
                differs(
                    &values(&reference),
                    &values(&whole(
                        &Chain::op(LocalStatisticOp::new(
                            "plain",
                            statistic([3, 3, 3], step, alignment),
                        )),
                        &input,
                    )),
                    &format!("{label} {step:?} {alignment:?}"),
                );
                for grid in grids() {
                    let plan = plan(&Chain::op(op()), &grid);
                    let got = run(Chain::op(op()), &input, &plan);
                    assert_eq!(
                        got,
                        reference,
                        "{label}, spacing {step:?}, {alignment:?}, block {:?}",
                        grid.block()
                    );
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 3 * 2 * 2 * 8);
}

/// Block sizes that cut the lattice in every way available: not at all, on one
/// axis at three offsets relative to the samples, on two, and on all three.
fn grids() -> Vec<BlockGrid> {
    vec![
        BlockGrid::new(VOLUME, VOLUME).unwrap(),
        BlockGrid::along(VOLUME, &[0], 5).unwrap(),
        BlockGrid::along(VOLUME, &[0], 8).unwrap(),
        BlockGrid::along(VOLUME, &[0], 9).unwrap(),
        BlockGrid::along(VOLUME, &[1], 6).unwrap(),
        BlockGrid::along(VOLUME, &[2], 4).unwrap(),
        BlockGrid::along(VOLUME, &[0, 2], 6).unwrap(),
        BlockGrid::along(VOLUME, &[0, 1, 2], 6).unwrap(),
    ]
}

fn plan(chain: &Chain, grid: &BlockGrid) -> Decomposition {
    let slots = chain.slots();
    let reach = chain.reach3(&VOLUME);
    let phase = PhaseDecomposition::derive(
        (0..slots.len()).collect(),
        slots.iter().map(|slot| slot.display_name()).collect(),
        reach,
        reach,
        grid.clone(),
    );
    let mut plan = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases: vec![phase],
        chain_reach: reach,
    };
    plan.declare_dtypes(chain).unwrap();
    plan.declare_source_images(chain).unwrap();
    plan
}

fn run(chain: Chain, input: &Voxels, decomposition: &Decomposition) -> Voxels {
    let workflow = Workflow::new(chain, input.shape(), input.dtype());
    let env = ArrayEnvironment::for_decomposition(input.clone(), decomposition, [4, 4, 4]).unwrap();
    execute(
        "narrowed lattice",
        &workflow,
        decomposition,
        &Hints::default(),
        &env,
    )
    .unwrap();
    env.output()
}

// -------------------------------------------------- 4. the reach is still --

/// **Checked rather than assumed**, because the last parameter added to this
/// family did move the reach: `Alignment::PinnedEnds` maps a voxel near an end
/// past the lattice's unsampled margin and reads a sample further away than any
/// voxel reaches under the default.
///
/// A narrowing does not, and the reason it cannot is worth having in a test
/// rather than only in a comment: it is applied to a value the kernel already
/// holds. The samples evaluated, the windows gathered and the brackets blended
/// are the same ones. So every geometric declaration — the bound, the per-side
/// reach, the per-block halo — has to come out identical, on every axis and
/// under both alignments.
#[test]
fn the_reach_does_not_move_when_the_values_are_narrowed() {
    let grid = BlockGrid::along(VOLUME, &[0], 8).unwrap();
    let mut compared = 0;
    for size in [[3usize, 3, 3], [6, 5, 5]] {
        for step in [[7usize, 5, 4], [1, 1, 1]] {
            for alignment in [Alignment::SamplePositions, Alignment::PinnedEnds] {
                let plain = statistic(size, step, alignment);
                let narrowed = statistic(size, step, alignment).narrowed(to_u16());
                for axis in 0..3 {
                    assert_eq!(
                        plain.reach(axis, VOLUME[axis]),
                        narrowed.reach(axis, VOLUME[axis]),
                        "{size:?} {step:?} {alignment:?} axis {axis}"
                    );
                    assert_eq!(
                        plain.reach_sides(axis, VOLUME[axis]),
                        narrowed.reach_sides(axis, VOLUME[axis])
                    );
                }
                assert_eq!(plain.reach_spec(VOLUME), narrowed.reach_spec(VOLUME));
                assert_eq!(
                    plain.halo(&grid).unwrap(),
                    narrowed.halo(&grid).unwrap(),
                    "the granted halo is a table and it must not move either"
                );

                // And the op shells, which is where a plan reads them.
                let plain_op = LocalStatisticOp::new("plain", plain);
                let narrowed_op = LocalStatisticOp::new("narrowed", narrowed);
                assert_eq!(plain_op.reach_spec(VOLUME), narrowed_op.reach_spec(VOLUME));
                assert_eq!(
                    plain_op.cost_per_voxel(),
                    narrowed_op.cost_per_voxel(),
                    "the narrowing is one rounding per sample and is not priced"
                );
                compared += 1;
            }
        }
    }
    assert_eq!(compared, 8);

    // The alignment, for contrast: the parameter that *did* move the reach, on
    // a lattice with an unsampled margin. Without this the test above could
    // pass because nothing here can move a reach at all.
    let default = statistic([3, 3, 3], [7, 5, 4], Alignment::SamplePositions);
    let pinned = statistic([3, 3, 3], [7, 5, 4], Alignment::PinnedEnds);
    assert_ne!(
        default.reach(0, VOLUME[0]),
        pinned.reach(0, VOLUME[0]),
        "if these agreed the contrast would prove nothing"
    );
}

// ------------------------------------------------ 5. the split pair too --

/// The split pair, each half holding **its own site**, is the fused op with both
/// — exactly.
///
/// This is the claim that makes one mechanism serve both forms rather than two
/// that have to be kept in step. It is also where a half getting the *other*
/// half's site would show: `LatticeStatisticOp` narrowing with the rule meant
/// for after the blend, or the interpolation narrowing with the rule meant for
/// the grid, both produce the "swapped" row of the hand-computed table above.
///
/// Both halves run through their own plans and their own phases rather than
/// being applied by hand, because a cross-grid op's fetch geometry is part of
/// what has to survive the narrowing.
#[test]
fn the_split_pair_narrowed_is_the_fused_op_narrowed() {
    let fine = texture(VOLUME);
    let input: Voxels = fine.clone().into();
    let narrowing = to_u16();

    for (size, step) in [
        ([3usize, 3, 3], [2usize, 2, 2]),
        ([5, 5, 3], [4, 4, 2]),
        ([5, 5, 5], [7, 7, 7]),
    ] {
        let fused = || {
            LocalStatisticOp::new(
                "fused",
                LocalStatistic::sampled(element(size), Sampling::every(step), Statistic::Mean)
                    .unwrap()
                    .narrowed(narrowing),
            )
        };
        let reference = whole(&Chain::op(fused()), &input);
        differs(
            &values(&reference),
            &values(&whole(
                &Chain::op(LocalStatisticOp::new(
                    "plain",
                    LocalStatistic::sampled(element(size), Sampling::every(step), Statistic::Mean)
                        .unwrap(),
                )),
                &input,
            )),
            &format!("{size:?} {step:?}"),
        );

        // The coarse half, blocked on the lattice.
        let coarse_op = || {
            LatticeStatisticOp::new(
                "coarse",
                element(size),
                &Sampling::every(step),
                Statistic::Mean,
                VOLUME,
            )
            .unwrap()
            .narrowing_of(narrowing)
        };
        let op = coarse_op();
        assert_eq!(
            op.narrowing(),
            narrowing.at_samples,
            "the statistic half takes the site at the grid and no other"
        );
        let counts = op.lattice().lattice_volume();
        let mut block = counts;
        for axis in 0..3 {
            block[axis] = statistic_block_edge(&op, axis, 2).unwrap();
        }
        let coarse_grid = BlockGrid::new(counts, block).unwrap();
        let coarse_phase =
            lattice_statistic_phase(vec![0], vec!["coarse".to_string()], &op, coarse_grid).unwrap();
        let coarse_plan = Decomposition {
            volume: VOLUME,
            dtype: Dtype::F64,
            phases: vec![coarse_phase],
            chain_reach: [0, 0, 0],
        };
        let coarse = run(Chain::op(coarse_op()), &input, &coarse_plan);

        // The narrowed grid is the hand-written reference for it, truncated
        // sample by sample — so the split's coarse image is pinned by the
        // definition and not only by the fused op's agreement with it.
        let expected = {
            let mut grid = reference_statistic(&fine, op.element(), op.lattice());
            let trunc = narrowing.at_samples.unwrap();
            grid.map_inplace(|value| *value = trunc.apply(*value));
            grid
        };
        assert_eq!(values(&coarse), expected, "{size:?} {step:?} coarse");

        // The fine half, blocked on the fine grid.
        let fine_op = || {
            LatticeInterpolateOp::new(
                "fine",
                SampleLattice::of(&Sampling::every(step), VOLUME).unwrap(),
            )
            .unwrap()
            .narrowing_of(narrowing)
        };
        let interpolate = fine_op();
        assert_eq!(
            interpolate.narrowing(),
            narrowing.after_interpolation,
            "the interpolation half takes the site after the blend and no other"
        );
        let mut fine_block = VOLUME;
        for axis in 0..3 {
            fine_block[axis] = interpolate_block_edge(interpolate.lattice(), axis, 7).unwrap();
        }
        let fine_grid = BlockGrid::new(VOLUME, fine_block).unwrap();
        assert!(fine_grid.cores().len() > 1, "one block proves less");
        let fine_phase =
            lattice_interpolate_phase(vec![0], vec!["fine".to_string()], &interpolate, fine_grid)
                .unwrap();
        let fine_plan = Decomposition {
            volume: interpolate.lattice().lattice_volume(),
            dtype: Dtype::F64,
            phases: vec![fine_phase],
            chain_reach: [0, 0, 0],
        };
        let split = run(Chain::op(fine_op()), &coarse, &fine_plan);

        assert_eq!(
            split, reference,
            "element {size:?} spacing {step:?}: the split must be the fused op with the same \
             narrowing, not a variant of it"
        );
    }
}

/// The statistic at every lattice point, over the whole volume, written out.
///
/// Shares no code with the geometry under test, so a fetch mapping that is
/// subtly wrong cannot agree with this by agreeing with itself.
fn reference_statistic(
    input: &Array3<f64>,
    element: &StructuringElement,
    lattice: &SampleLattice,
) -> Array3<f64> {
    let counts = lattice.lattice_volume();
    let volume = lattice.volume();
    Array3::from_shape_fn((counts[0], counts[1], counts[2]), |(p, q, r)| {
        let centre = [
            lattice.centre(0, p) as isize,
            lattice.centre(1, q) as isize,
            lattice.centre(2, r) as isize,
        ];
        let mut window: Vec<Total> = Vec::new();
        for step in element.offsets() {
            let mut index = [0usize; 3];
            let mut inside = true;
            for axis in 0..3 {
                let at = centre[axis] + step[axis];
                if at < 0 || at >= volume[axis] as isize {
                    inside = false;
                    break;
                }
                index[axis] = at as usize;
            }
            if inside {
                window.push(Total(input[index]));
            }
        }
        Statistic::Mean.reduce(&mut window, element.len())
    })
}

// ------------------------------------------- 6. the short circuit agrees --

/// A block a plan may skip must be filled with the value a run would have
/// written, and with narrowing on that is the value **after both sites**.
///
/// Asserted by computing both: the declaration, and the kernel's own answer over
/// a uniform block. A rank statistic is used because it is the one whose
/// constant mapping is exact at every value, so the only thing that can differ
/// between the two numbers is the narrowing.
#[test]
fn a_uniform_block_is_declared_at_the_value_the_kernel_writes() {
    let element = element([3, 3, 3]);
    for value in [0.25f64, 7.75, 12.0, 3.5] {
        for narrowing in [LatticeNarrowing::none(), to_u16()] {
            let local = LocalStatistic::sampled(
                element.clone(),
                Sampling::every([3, 3, 3]),
                Statistic::Rank(Rank::median(&element)),
            )
            .unwrap()
            .narrowed(narrowing);
            let declared = LocalStatisticOp::new("ranked", local.clone())
                .constant_maps_to(value)
                .expect("a rank declares at every constant");

            let input = Array3::from_elem((9, 9, 9), value);
            let mut out = Array3::zeros(input.dim());
            local
                .evaluate_into(input.view(), &Anchor::whole([9, 9, 9]), out.view_mut())
                .unwrap();
            assert!(
                out.iter().all(|got| got.to_bits() == declared.to_bits()),
                "value {value} narrowing {narrowing:?}: declared {declared}, computed \
                 {:?}",
                out[[4, 4, 4]]
            );
        }
    }

    // And the two halves of the split declare their own site, which composed is
    // the same number.
    let statistic_half = LatticeStatisticOp::new(
        "coarse",
        element.clone(),
        &Sampling::every([3, 3, 3]),
        Statistic::Rank(Rank::median(&element)),
        VOLUME,
    )
    .unwrap()
    .narrowing_of(to_u16());
    let interpolate_half = LatticeInterpolateOp::new(
        "fine",
        SampleLattice::of(&Sampling::every([3, 3, 3]), VOLUME).unwrap(),
    )
    .unwrap()
    .narrowing_of(to_u16());
    let coarse = statistic_half.constant_maps_to(7.75).unwrap();
    assert_eq!(coarse, 7.0, "truncated at the grid");
    assert_eq!(
        interpolate_half.constant_maps_to(coarse),
        Some(7.0),
        "and a whole number survives the second site"
    );
    assert_eq!(interpolate_half.constant_maps_to(7.5), Some(8.0));
}

// ------------------------------------------------------------- helpers --

fn identical(got: &Array3<f64>, want: &Array3<f64>, what: &str) {
    assert_eq!(got.shape(), want.shape(), "{what}: different shapes");
    for ((index, one), other) in got.indexed_iter().zip(want.iter()) {
        assert!(
            one.to_bits() == other.to_bits(),
            "{what}: at {index:?} got {one} and wanted {other}"
        );
    }
}

fn differs(one: &Array3<f64>, other: &Array3<f64>, what: &str) {
    assert!(
        one.iter()
            .zip(other.iter())
            .any(|(one, other)| one.to_bits() != other.to_bits()),
        "{what}: the two must differ, or the comparison beside this one is vacuous"
    );
}
