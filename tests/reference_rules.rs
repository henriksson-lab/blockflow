// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// Three rules this crate reproduces from an outside implementation, each
// transcribed here **independently** and pinned against the numbers that
// implementation produces.
//
// Why the rules are here at all
// -----------------------------
// Each of the three answers a question the surrounding module could not
// previously express, and in each case the module already had something that
// looks adjacent and is not:
//
// | added | already there | why the difference is not a matter of tuning |
// |---|---|---|
// | `Statistic::Isodata` | `Mean`, `Deviation`, `Rank` | a mean says where the mass is, an isodata threshold says where two classes separate. On a bimodal window they are far apart, and no rank is either |
// | `LocalGainOp` | `LevelCorrectionOp::Divide`, `LocalContrastOp` | the gain is **capped** against a second estimate, and neither of the other two has anywhere to put a cap |
// | `RatioResponse` | `RidgeResponse` | the magnitude is carried rather than saturated, and the along-axis term is signed. No setting of the three sensitivities is this |
//
// Why this file transcribes rather than depends
// ---------------------------------------------
// The implementation these rules were read from is GPL and this crate is MIT, so
// a dependency on it — even a dev-dependency, even for a test — would be a
// licensing claim this crate is not in a position to make.
// `element_reference_rule.rs` set the precedent and this file follows it
// exactly: the **rule** is written out here, in the arithmetic it is stated in;
// the value-by-value agreement was established once, outside the crate, by
// building both and comparing; and what is kept here is that transcription plus
// the numbers the comparison produced, so that a drift in either direction fails
// loudly.
//
// What the outside comparison covered, since these tests cannot re-run it
// ----------------------------------------------------------------------
// * **isodata**: 415 windows — empty, constant, two-valued, tri-modal, negative,
//   spanning `1e-15` to `1e12`, and with non-finite values mixed in — all
//   bit-for-bit identical at 256 bins. Then the whole op, over a `24 x 20 x 18`
//   volume, at four window shapes: 8640 of 8640 voxels bit-identical.
// * **the bounded gain**: 75 scalar cases through the other implementation's own
//   kernel, and three whole-volume runs at three window shapes and three
//   ceilings — 4480 of 4480 voxels bit-identical each time.
// * **the ratio response**: five parameter sets over 1848 voxels, fed the
//   *other implementation's own eigenvalues* so that the comparison is of the
//   fold and not of the decomposition — bit-identical in every case.
//
// One disagreement was found and it is **not** in any of the three rules. Once a
// sample lattice coarser than every voxel is involved, this crate and that one
// part company, because that one lays its grid out relative to the array it is
// handed and upsamples by stretching the grid to the array's endpoints — the
// defect `src/ops/local.rs` is arranged against and
// `docs/design/XY_BLOCK_SPLITTING.md` records. The control run makes it plain:
// at spacing 4 the *existing, untouched* lowest-rank statistic disagrees at
// 8616 of 8640 voxels, the new isodata statistic at 8632. At spacing 1 both
// agree everywhere. So the sampling is where they differ and the rules are not,
// which is why the whole-op frozen numbers below are taken at spacing 1 and the
// lattice is exercised by the decomposition sweeps instead.

use ndarray::Array3;

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, BlockOp, Chain};
use blockflow::ops::{
    bounded_gain_value, AdaptiveThresholdOp, ElementShape, Isodata, LocalGainOp, LocalStatistic,
    LocalStatisticOp, Polarity, Rank, RatioResponse, RidgeFilterOp, RidgeResponse, ScaleSpace,
    Statistic, StructuringElement, Total,
};
use blockflow::strategy::{execute, Hints, Workflow};
use blockflow::synthetic::{Scene, SceneSpec};
use blockflow::voxels::Voxels;
use blockflow::Dtype;

const VOLUME: [usize; 3] = [32, 24, 20];

// ------------------------------------------------------------- fixtures --

/// Structure at several scales, so that every one of the three has something to
/// answer and a seam has something to get wrong.
fn intensities() -> Array3<f64> {
    let scene = Scene::new(
        SceneSpec::new(VOLUME, 20250903)
            .with_objects(30)
            .with_radius(1.2, 3.0)
            .with_gradient(0.4, 2.0)
            .with_noise(0.02),
    )
    .unwrap();
    let rendered = scene.render();
    let mut array = Array3::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    for i in 0..VOLUME[0] {
        for j in 0..VOLUME[1] {
            for k in 0..VOLUME[2] {
                // Lifted clear of zero and scaled up, so that a histogram has a
                // range to bin over and a gain has a level to divide by.
                array[[i, j, k]] = rendered.intensity[[i, j, k]] * 100.0 + 1.0;
            }
        }
    }
    array
}

fn box_element(radius: [usize; 3]) -> StructuringElement {
    StructuringElement::from_radius(ElementShape::Box, radius)
}

/// The oracle for every sweep below: the same kernel, called once, over the
/// whole array. Not a second implementation, so a disagreement is a
/// decomposition bug and nothing else.
fn whole_volume(chain: &Chain, input: &Array3<f64>) -> Array3<f64> {
    let source: Voxels = input.clone().into();
    let mut out = Voxels::zeros(Dtype::F64, VOLUME).unwrap();
    chain
        .apply(&source, &mut out, &Anchor::whole(VOLUME))
        .expect("the whole-volume reference must run");
    out.view::<f64>().unwrap().to_owned()
}

/// One phase holding the chain, at a given block edge and split axes, built
/// from the chain's **own** reach — nothing here supplies one, so nothing here
/// can hide one that is wrong.
///
/// The chain is taken as a factory rather than by value because a `Chain` owns
/// its ops and is not `Clone`, and every run here needs its own.
fn run(chain: Chain, input: &Array3<f64>, block: usize, split_axes: &[usize]) -> Array3<f64> {
    let workflow = Workflow::new(chain, VOLUME, Dtype::F64);
    let reach = workflow.chain.reach3(&VOLUME);
    let slots = workflow.chain.slots();
    let names: Vec<String> = slots.iter().map(|slot| slot.display_name()).collect();
    let grid = BlockGrid::along(VOLUME, split_axes, block).unwrap();
    let phase = PhaseDecomposition::derive((0..slots.len()).collect(), names, reach, reach, grid);
    let decomposition = Decomposition {
        volume: VOLUME,
        dtype: workflow.dtype,
        phases: vec![phase],
        chain_reach: reach,
    };
    decomposition
        .check()
        .expect("an honestly derived plan must tile");
    let env = ArrayEnvironment::new(input.clone().into(), 1, [4, 4, 4]).unwrap();
    execute(
        "reference rules",
        &workflow,
        &decomposition,
        &Hints::default(),
        &env,
    )
    .unwrap();
    env.output().view::<f64>().unwrap().to_owned()
}

/// **The standing bar**, applied to each of the three: byte-identical output
/// against the whole-volume answer, under several block sizes and several split
/// axes. A run that agrees at one block size is how a short halo passes.
fn agrees_under_every_decomposition(what: &str, chain: &dyn Fn() -> Chain, input: &Array3<f64>) {
    let want = whole_volume(&chain(), input);
    assert!(
        want.iter().any(|&value| value != want[[0, 0, 0]]),
        "{what} produced a constant volume, so byte-identity would prove nothing"
    );
    let mut ran = 0;
    for block in [4usize, 7, 13, 64] {
        for split_axes in [vec![0], vec![2], vec![0, 1], vec![0, 1, 2]] {
            let got = run(chain(), input, block, &split_axes);
            let differing = got.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
            assert_eq!(
                differing,
                0,
                "{what}: block {block}, axes {split_axes:?} disagreed with the whole-volume \
                 answer at {differing} of {} voxels",
                got.len()
            );
            ran += 1;
        }
    }
    assert_eq!(ran, 16, "{what}: the sweep did not run");
}

// ============================================================== isodata ==

/// **The isodata rule, transcribed.** Every bin centre with the fixed-point
/// property, lowest first.
///
/// Kept deliberately close to the form the rule is stated in rather than
/// simplified, for `element_reference_rule.rs`'s reason: the point of a
/// transcription is not to improve on what it transcribes. In particular the
/// mean of the upper class is `(all intensity - intensity below) / count above`
/// and not a fresh sum over the upper bins — the two are the same number in real
/// arithmetic and not in `f64`, and it is the first that is the rule.
///
/// The program is otherwise a different one from
/// [`Isodata::of`](blockflow::ops::Isodata::of): quadratic, re-walking the bins
/// below each candidate instead of carrying a running total, and returning every
/// threshold instead of one.
fn isodata_thresholds_by_rule(values: &[f64], bins: usize) -> Vec<f64> {
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return Vec::new();
    }
    let minimum = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let width = (maximum - minimum) / bins as f64;
    if !(width > 0.0) {
        // No histogram: every value is the same one, and the rule answers it.
        return vec![maximum];
    }

    let centre = |index: usize| minimum + (index as f64 + 0.5) * width;
    let mut counts = vec![0.0f64; bins];
    for value in &finite {
        let mut bin = ((value - minimum) / width) as usize;
        if bin >= bins {
            bin = bins - 1;
        }
        counts[bin] += 1.0;
    }
    if bins == 1 {
        return vec![centre(0)];
    }

    let mut all_count = 0.0;
    let mut all_intensity = 0.0;
    for (index, count) in counts.iter().enumerate() {
        all_count += count;
        all_intensity += count * centre(index);
    }

    let step = centre(1) - centre(0);
    let mut found = Vec::new();
    for candidate in 0..bins - 1 {
        let mut below_count = 0.0;
        let mut below_intensity = 0.0;
        for index in 0..=candidate {
            below_count += counts[index];
            below_intensity += counts[index] * centre(index);
        }
        let above_count = all_count - below_count;
        if below_count == 0.0 || above_count == 0.0 {
            continue;
        }
        let lower = below_intensity / below_count;
        let higher = (all_intensity - below_intensity) / above_count;
        let midpoint = (lower + higher) / 2.0;
        let distance = midpoint - centre(candidate);
        if distance >= 0.0 && distance < step {
            found.push(centre(candidate));
        }
    }
    found
}

/// A spread of windows that reaches every branch of the rule.
fn isodata_windows() -> Vec<(String, Vec<f64>)> {
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut cases: Vec<(String, Vec<f64>)> = vec![
        ("empty".into(), Vec::new()),
        ("single".into(), vec![7.25]),
        ("constant".into(), vec![3.5; 27]),
        ("constant zero".into(), vec![0.0; 27]),
        ("two values".into(), vec![0.0, 1.0]),
        (
            "negative range".into(),
            (0..60).map(|i| i as f64 - 30.0).collect(),
        ),
        ("huge range".into(), vec![-1e12, 0.0, 1e12, 3.0, -7.0]),
        ("tiny range".into(), vec![1.0, 1.0 + 1e-15, 1.0 + 2e-15]),
        (
            "with non-finite values".into(),
            vec![f64::NAN, 1.0, 2.0, f64::INFINITY, 3.0, f64::NEG_INFINITY],
        ),
        ("nothing finite".into(), vec![f64::NAN, f64::INFINITY]),
    ];
    for case in 0..120 {
        let n = 1 + (case * 7) % 200;
        let span = [0.0, 1e-9, 1.0, 1e3, 1e9][case % 5];
        let base = (case as f64) * 13.0 - 500.0;
        let values: Vec<f64> = (0..n).map(|_| base + next() * span).collect();
        cases.push((format!("random {case} n={n} span={span}"), values));
    }
    // Multi-modal, which is where several bins satisfy the rule at once and the
    // tie behaviour decides the answer.
    for modes in 2..5usize {
        let mut values = Vec::new();
        for mode in 0..modes {
            for step in 0..20 {
                values.push((mode as f64) * 300.0 + step as f64 * 0.5);
            }
        }
        cases.push((format!("{modes} modes"), values));
    }
    cases
}

/// **The bar for the rule**: what the crate computes is what the rule says, bit
/// for bit, over every window and every bin count.
#[test]
fn the_isodata_statistic_is_the_rule_it_transcribes() {
    let mut checked = 0;
    for bins in [1usize, 2, 7, 64, 256, 1024] {
        for fallback in [1.0f64, 0.0, -3.5] {
            let statistic = Isodata::new(bins, fallback).unwrap();
            for (name, values) in isodata_windows() {
                let want = isodata_thresholds_by_rule(&values, bins)
                    .last()
                    .copied()
                    .unwrap_or(fallback);
                let got = statistic.of(values.iter().copied());
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{name}, {bins} bins: got {got:?}, the rule says {want:?}"
                );
                checked += 1;
            }
        }
    }
    assert!(checked >= 2000, "the sweep ran only {checked} windows");
}

/// **The tie rule, which is where two implementations of "isodata" differ.**
///
/// The classical presentation iterates from a starting guess to a fixed point
/// and returns whichever fixed point that guess falls into. This is the direct
/// form: it tests every bin and takes the **last** that qualifies. So a window
/// that admits more than one threshold has an answer that is a function of the
/// data alone, and it is the highest — not the first, not the one nearest the
/// mean, and not whichever the sweep happened to reach first.
#[test]
fn where_several_thresholds_qualify_the_highest_is_the_answer() {
    // Five tight modes of unequal mass, which is the shape that admits more
    // than one split. Two modes do not: in the empty stretch between two of
    // them the class means are constant while the bin centre climbs by a bin
    // width per step, so the midpoint test can only be satisfied once. It takes
    // mass *between* the extremes to let it be satisfied again.
    let modes = [
        (254.0f64, 44usize),
        (412.0, 19),
        (430.0, 52),
        (520.0, 25),
        (705.0, 22),
    ];
    let mut values: Vec<f64> = Vec::new();
    for (position, mass) in modes {
        for step in 0..mass {
            values.push(position + step as f64 * 0.01);
        }
    }

    let all = isodata_thresholds_by_rule(&values, 256);
    assert_eq!(
        all.len(),
        3,
        "this window admits {} threshold(s), so it cannot show a tie rule at all: {all:?}",
        all.len()
    );
    let statistic = Isodata::new(256, 1.0).unwrap();
    let got = statistic.of(values.iter().copied());
    assert_eq!(got.to_bits(), all.last().unwrap().to_bits());
    assert_ne!(
        got.to_bits(),
        all[0].to_bits(),
        "the first and the last threshold coincide here, so this proves nothing"
    );
    // the three, frozen, so that a change to the rule that happened to keep the
    // *count* still fails here
    assert_eq!(
        all,
        vec![374.73392578125004, 482.24880859375, 545.70021484375]
    );
}

/// The numbers the outside comparison produced, frozen.
///
/// The sweep above proves the crate computes the transcribed rule; these say
/// what that rule *is*, in numbers that can be checked against the other
/// implementation by hand. Together they are what stops a change to both the
/// rule and its transcription passing quietly.
#[test]
fn the_frozen_isodata_values_are_the_ones_the_other_implementation_produces() {
    let statistic = Isodata::new(256, 1.0).unwrap();
    // (window, threshold)
    let cases: Vec<(Vec<f64>, f64)> = vec![
        (Vec::new(), 1.0),
        (vec![7.25], 7.25),
        (vec![3.5; 27], 3.5),
        (vec![f64::NAN, f64::INFINITY], 1.0),
        (vec![1.0, 2.0, 3.0, 4.0, 100.0], 51.080078125),
        (
            vec![0.0, 0.5, 1.0, 1.5, 2.0, 100.0, 100.5, 101.0],
            50.697265625,
        ),
        (
            {
                let mut v = vec![0.0; 20];
                v.extend(vec![255.0; 5]);
                v
            },
            127.001953125,
        ),
        ((0..100).map(|i| i as f64).collect(), 49.306640625),
        ((0..60).map(|i| i as f64 - 30.0).collect(), -0.615234375),
        (
            {
                let mut v: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
                v.extend((0..10).map(|i| 100.0 + i as f64 * 0.5));
                v.extend((0..5).map(|i| 900.0 + i as f64 * 20.0));
                v
            },
            488.0859375,
        ),
    ];
    for (values, want) in cases {
        let got = statistic.of(values.iter().copied());
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "window of {} values: got {got:?}, frozen {want:?}",
            values.len()
        );
    }
}

/// A constant window answers that constant **exactly**, which is what lets
/// `Statistic::constant_maps_to` declare it — and it is declared only where it
/// is exactly true, so a window with nothing finite in it declares nothing.
#[test]
fn an_isodata_threshold_of_a_constant_is_that_constant_bit_for_bit() {
    let statistic = Isodata::new(256, 1.0).unwrap();
    let wrapped = Statistic::Isodata(statistic);
    for constant in [0.0f64, 0.1, -7.5, 1e-300, 1e300, f64::MIN_POSITIVE] {
        let window = vec![constant; 33];
        assert_eq!(
            statistic.of(window.iter().copied()).to_bits(),
            constant.to_bits(),
            "constant {constant}"
        );
        assert_eq!(wrapped.constant_maps_to(constant), Some(constant));
    }
    for constant in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            wrapped.constant_maps_to(constant),
            None,
            "a window of {constant} has nothing finite to histogram, so nothing may be \
             declared for it"
        );
    }
}

/// It is a **different question** from the three statistics that were already
/// here, demonstrated rather than asserted: on a two-mode window the isodata
/// threshold sits between the modes and every one of the others sits in one of
/// them.
#[test]
fn the_isodata_threshold_is_not_a_mean_a_deviation_or_any_rank() {
    let mut window: Vec<f64> = vec![10.0; 95];
    window.extend(vec![200.0; 5]);
    let element = box_element([1, 1, 1]);
    let full = element.len();
    let isodata = Statistic::Isodata(Isodata::new(256, 1.0).unwrap()).reduce(
        &mut window.iter().map(|&v| Total(v)).collect::<Vec<_>>(),
        full,
    );
    assert!(
        isodata > 20.0 && isodata < 190.0,
        "the threshold must fall between the modes, got {isodata}"
    );
    // every rank of this window is one of the two mode values, whatever the
    // fraction, so no `Rank` reaches the gap between them
    for tenth in 0..=10 {
        let mut values: Vec<Total> = window.iter().map(|&v| Total(v)).collect();
        let rank = Rank::percentile(&element, tenth as f64 / 10.0);
        let got = Statistic::Rank(rank).reduce(&mut values, full);
        assert!(
            got == 10.0 || got == 200.0,
            "rank at {tenth}/10 gave {got}, which is not one of the two values present"
        );
    }
    // and the mean sits in the lower mode, because that is where the mass is
    let mut values: Vec<Total> = window.iter().map(|&v| Total(v)).collect();
    let mean = Statistic::Mean.reduce(&mut values, full);
    assert!(mean < 25.0, "the mean is {mean}, not in the lower mode");
}

/// **Decomposition invariance**, over several block sizes, for the statistic and
/// for the threshold built on it.
#[test]
fn the_isodata_statistic_is_the_same_under_every_decomposition() {
    let input = intensities();
    let isodata = || Statistic::Isodata(Isodata::new(256, 1.0).unwrap());
    let statistic = || LocalStatistic::new(box_element([1, 2, 1]), [5, 4, 3], isodata()).unwrap();

    assert_eq!(
        statistic().reach(0, VOLUME[0]),
        5,
        "the reach is the lattice term plus the window's, and nothing sets it"
    );
    let chain = || Chain::op(LocalStatisticOp::new("isodata", statistic()));
    assert_eq!(chain().reach3(&VOLUME), [5, 5, 3]);
    agrees_under_every_decomposition("the isodata statistic", &chain, &input);

    let chain = || {
        Chain::op(AdaptiveThresholdOp::new(
            "adaptive",
            LocalStatistic::new(box_element([1, 2, 1]), [6, 6, 6], isodata()).unwrap(),
            1.0,
            0.0,
        ))
    };
    assert_eq!(chain().reach3(&VOLUME), [6, 7, 6]);
    agrees_under_every_decomposition("a threshold against it", &chain, &input);
}

// ========================================================= bounded gain ==

/// **The bounded-gain rule, transcribed.**
///
/// Written as the test-and-replace it is rather than as a `min`, because that is
/// what the rule says and because the two are different functions — see the
/// frozen case with a negative upper estimate, and
/// `the_cap_is_a_test_and_not_a_minimum` below.
fn gain_by_rule(value: f64, low: f64, high: f64, floor: f64, ceiling: f64) -> f64 {
    let bounded_low = if low > floor { low } else { floor };
    let mut gain = 1.0 / bounded_low;
    if high * gain > ceiling {
        gain = ceiling / high;
    }
    value * gain
}

#[test]
fn the_bounded_gain_is_the_rule_it_transcribes() {
    let interesting = [
        -1e6f64, -100.0, -1.0, -1e-9, 0.0, 1e-9, 0.25, 0.5, 1.0, 1.5, 2.0, 10.0, 1e6,
    ];
    let mut checked = 0;
    for &floor in &[1e-6f64, 0.5, 1.0, 4.0] {
        for &ceiling in &[0.25f64, 1.0, 1.5, 10.0, 1e6] {
            for &value in &interesting {
                for &low in &interesting {
                    for &high in &interesting {
                        let want = gain_by_rule(value, low, high, floor, ceiling);
                        let got = bounded_gain_value(value, low, high, floor, ceiling);
                        assert_eq!(
                            got.to_bits(),
                            want.to_bits(),
                            "value {value}, low {low}, high {high}, floor {floor}, \
                             ceiling {ceiling}"
                        );
                        checked += 1;
                    }
                }
            }
        }
    }
    assert!(checked >= 20_000, "the sweep ran only {checked} cases");
}

/// The numbers the outside comparison produced, frozen. The floor is `1.0`
/// throughout, which is the value that implementation fixes and this crate makes
/// a parameter.
///
/// The fifth row is the one worth reading: a negative upper estimate leaves the
/// gain alone, because a negative product cannot exceed a positive ceiling. A
/// `min` would have taken `ceiling / high`, which is negative, and returned
/// `+2.0` for a voxel whose value is `-2.0`.
#[test]
fn the_frozen_gain_values_are_the_ones_the_other_implementation_produces() {
    // (value, low, high, ceiling, output), floor 1.0
    let cases = [
        (1.0f64, 0.5f64, 2.0f64, 0.5f64, 0.25f64),
        (3.0, 2.0, 4.0, 0.5, 0.375),
        (0.25, 0.25, 0.25, 0.5, 0.25),
        (1.0, 1.0, 1.0, 0.5, 0.5),
        (-2.0, -4.0, -1.0, 0.5, -2.0),
        (0.0, -4.0, 8.0, 0.5, 0.0),
        (0.0, 0.0, 0.0, 0.5, 0.0),
        (5.0, 1.0, 100.0, 0.5, 0.025),
        (20.0, 10.0, 30.0, 0.5, 0.3333333333333333),
        (1.0, 0.5, 2.0, 1.5, 0.75),
        (3.0, 2.0, 4.0, 1.5, 1.125),
        (1.0, 1.0, 1.0, 1.5, 1.0),
        (-2.0, -4.0, -1.0, 1.5, -2.0),
        (5.0, 1.0, 100.0, 1.5, 0.075),
        (20.0, 10.0, 30.0, 1.5, 1.0),
        (1.0, 0.5, 2.0, 10.0, 1.0),
        (3.0, 2.0, 4.0, 10.0, 1.5),
        (-2.0, -4.0, -1.0, 10.0, -2.0),
        (5.0, 1.0, 100.0, 10.0, 0.5),
        (20.0, 10.0, 30.0, 10.0, 2.0),
    ];
    for (value, low, high, ceiling, want) in cases {
        let got = bounded_gain_value(value, low, high, 1.0, ceiling);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "value {value}, low {low}, high {high}, ceiling {ceiling}: got {got:?}, \
             frozen {want:?}"
        );
    }
}

/// The cap is a **test**, not a minimum, and the difference is a sign rather
/// than a rounding.
#[test]
fn the_cap_is_a_test_and_not_a_minimum() {
    let (value, low, high, floor, ceiling) = (-2.0f64, -4.0f64, -1.0f64, 1.0f64, 0.5f64);
    let as_minimum = value * (1.0f64 / low.max(floor)).min(ceiling / high);
    let got = bounded_gain_value(value, low, high, floor, ceiling);
    assert_eq!(got, -2.0);
    assert_eq!(as_minimum, 1.0);
    assert_ne!(
        got, as_minimum,
        "if these ever agree this test is measuring nothing"
    );
}

/// **Decomposition invariance**, over several block sizes.
#[test]
fn the_bounded_gain_is_the_same_under_every_decomposition() {
    let input = intensities();
    let element = box_element([1, 1, 1]);
    let low =
        LocalStatistic::new(element.clone(), [4, 4, 4], Statistic::Rank(Rank::lowest())).unwrap();
    let high = LocalStatistic::new(
        element.clone(),
        [4, 4, 4],
        Statistic::Rank(Rank::highest(&element)),
    )
    .unwrap();
    // the maximum of the two statistics' reaches, per axis, and never their sum
    assert_eq!(
        LocalGainOp::new("gain", low, high, 1.0, 1.5)
            .unwrap()
            .reach(0, VOLUME[0]),
        4
    );
    let chain = || {
        let element = box_element([1, 1, 1]);
        Chain::op(
            LocalGainOp::new(
                "gain",
                LocalStatistic::new(element.clone(), [4, 4, 4], Statistic::Rank(Rank::lowest()))
                    .unwrap(),
                LocalStatistic::new(
                    element.clone(),
                    [4, 4, 4],
                    Statistic::Rank(Rank::highest(&element)),
                )
                .unwrap(),
                1.0,
                1.5,
            )
            .unwrap(),
        )
    };
    assert_eq!(chain().reach3(&VOLUME), [4, 4, 4]);
    agrees_under_every_decomposition("the bounded gain", &chain, &input);
}

/// The cap has to *fire* somewhere in that sweep, or the invariance above was
/// established on the uncapped half alone.
#[test]
fn the_cap_fires_on_the_data_the_sweep_uses() {
    let input = intensities();
    let element = box_element([1, 1, 1]);
    let statistic = |rank| LocalStatistic::new(element.clone(), [4, 4, 4], rank).unwrap();
    let capped = Chain::op(
        LocalGainOp::new(
            "capped",
            statistic(Statistic::Rank(Rank::lowest())),
            statistic(Statistic::Rank(Rank::highest(&element))),
            1.0,
            1.5,
        )
        .unwrap(),
    );
    // The same op with a ceiling far out of reach, which is the uncapped half.
    let free = Chain::op(
        LocalGainOp::new(
            "free",
            statistic(Statistic::Rank(Rank::lowest())),
            statistic(Statistic::Rank(Rank::highest(&element))),
            1.0,
            1e12,
        )
        .unwrap(),
    );
    let capped = whole_volume(&capped, &input);
    let free = whole_volume(&free, &input);
    let differing = capped
        .iter()
        .zip(free.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        differing > capped.len() / 20,
        "the ceiling changed only {differing} of {} voxels, so the sweep above barely \
         exercised it",
        capped.len()
    );
}

// ======================================================= ratio response ==

/// **The ratio response, transcribed.** Eigenvalues descending, ridge polarity.
///
/// Written in the form the rule is stated in, including the two `powf` calls
/// that are identities at the exponents most callers use — the point of a
/// transcription is not to improve on what it transcribes.
fn ratio_by_rule(eigenvalues: [f64; 3], cross: f64, along_power: f64, opposed: f64) -> f64 {
    let [e1, e2, e3] = eigenvalues;
    if !(e2 < 0.0 && e3 < 0.0) {
        return 0.0;
    }
    if e1 <= 0.0 {
        e3.abs() * (e2 / e3).powf(cross) * (1.0 + e1 / e2.abs()).powf(along_power)
    } else {
        let a = opposed * e1 / e2.abs();
        if a < 1.0 {
            e3.abs() * (e2 / e3).powf(cross) * (1.0 - a).powf(along_power)
        } else {
            0.0
        }
    }
}

#[test]
fn the_ratio_response_is_the_rule_it_transcribes() {
    let samples = [
        -1e6f64, -1000.0, -12.5, -1.0, -1e-9, 0.0, 1e-9, 0.5, 3.0, 40.0, 1e6,
    ];
    let mut checked = 0;
    for (cross, along_power, opposed) in [
        (0.0f64, 1.0f64, 0.25f64),
        (1.0, 1.0, 0.25),
        (0.5, 2.0, 0.75),
        (0.0, 0.0, 1.0),
        (2.0, 0.5, 0.0),
    ] {
        let response = RatioResponse::new(cross, along_power, opposed, Polarity::Ridge).unwrap();
        let mirrored = RatioResponse::new(cross, along_power, opposed, Polarity::Valley).unwrap();
        for &a in &samples {
            for &b in &samples {
                for &c in &samples {
                    let mut eigenvalues = [a, b, c];
                    eigenvalues.sort_by(|x, y| y.partial_cmp(x).unwrap());
                    let want = ratio_by_rule(eigenvalues, cross, along_power, opposed);
                    let got = response.evaluate(eigenvalues);
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "{eigenvalues:?} with {cross}/{along_power}/{opposed}"
                    );
                    assert!(!got.is_nan() && got >= 0.0, "{eigenvalues:?} -> {got}");
                    // the other polarity is this one on the negated triple,
                    // exactly
                    let negated = [-eigenvalues[2], -eigenvalues[1], -eigenvalues[0]];
                    assert_eq!(mirrored.evaluate(negated).to_bits(), got.to_bits());
                    checked += 1;
                }
            }
        }
    }
    assert!(checked >= 6000, "the sweep ran only {checked} triples");
}

/// The numbers the outside comparison produced, frozen — eigenvalues taken from
/// that implementation's own decomposition, so that what is pinned is the fold
/// and not a Hessian stencil.
///
/// One row per regime: the along-axis curvature with the structure's own sign,
/// against it but not dominating, against it and dominating (cut to zero), and
/// the wrong sign across the structure (not this structure at all).
#[test]
fn the_frozen_response_values_are_the_ones_the_other_implementation_produces() {
    let response = RatioResponse::new(0.5, 2.0, 0.75, Polarity::Ridge).unwrap();
    let cases: [([f64; 3], f64); 4] = [
        (
            [-1.0818133290668412, -3.1028187672674274, -5.684577150808424],
            1.7817609333481645,
        ),
        (
            [0.3274873111685874, -2.0709748233520053, -3.4851424206473225],
            2.0871073186778073,
        ),
        (
            [1.5366251896562835, -0.9962400158737301, -5.198971860885395],
            0.0,
        ),
        (
            [3.724165538458772, 1.7390969225483235, 1.3889420342857162],
            0.0,
        ),
    ];
    for (eigenvalues, want) in cases {
        let got = response.evaluate(eigenvalues);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{eigenvalues:?}: got {got:?}, frozen {want:?}"
        );
    }
}

/// **No setting of the existing response is this one**, shown rather than
/// claimed.
///
/// The existing response is a product of three terms each bounded by one, so it
/// can never exceed one for any parameters at all. This one carries the largest
/// curvature's magnitude as a factor, so on the same eigenvalues it is whatever
/// that magnitude is. A monotone rescaling could not reconcile them either: the
/// two also **order** a pair of structures differently, which is the property
/// that survives any rescaling.
#[test]
fn no_setting_of_the_existing_response_is_the_ratio_response() {
    let strong = [0.0f64, -100.0, -100.0];
    // a perfectly round but weak cross-section, and a flat but very strong one
    let round_weak = [0.0f64, -1.0, -1.0];
    let flat_strong = [0.0f64, -5.0, -1000.0];

    let ratio = RatioResponse::new(1.0, 1.0, 0.25, Polarity::Ridge).unwrap();
    assert!(
        ratio.evaluate(strong) > 1.0,
        "bounded by one after all? got {}",
        ratio.evaluate(strong)
    );
    assert!(
        ratio.evaluate(flat_strong) > ratio.evaluate(round_weak),
        "this response weighs the flat structure by its own curvature"
    );

    let mut disagreed = false;
    for line in [0.05f64, 0.5, 5.0] {
        for blob in [0.05f64, 0.5, 5.0] {
            for strength in [0.01f64, 1.0, 100.0] {
                let existing = RidgeResponse::new(line, blob, strength, Polarity::Ridge).unwrap();
                assert!(
                    existing.evaluate(strong) <= 1.0,
                    "the existing response is a product of three terms in [0, 1]"
                );
                if existing.evaluate(round_weak) > existing.evaluate(flat_strong) {
                    disagreed = true;
                }
            }
        }
    }
    assert!(
        disagreed,
        "no setting of the existing response ordered the pair the other way, so this test \
         is not showing the separation it claims"
    );
}

/// **Decomposition invariance**, over several block sizes, with the response
/// that was added — and the reach unchanged, because a fold contributes none.
#[test]
fn the_ratio_response_filter_is_the_same_under_every_decomposition() {
    let input = intensities();
    let build = || {
        RidgeFilterOp::new(
            "ratio",
            ScaleSpace::isotropic(&[1.0], 2.0, 1.0).unwrap(),
            RatioResponse::new(0.5, 1.0, 0.25, Polarity::Ridge).unwrap(),
        )
    };
    assert_eq!(
        build().reach(0, VOLUME[0]),
        3,
        "the gaussian radius plus the derivative stencil, and the fold adds nothing"
    );
    assert_eq!(
        build().constant_maps_to(3.0),
        Some(0.0),
        "a constant block has a Hessian of exactly zero, and this fold answers zero for it"
    );
    let chain = || Chain::op(build());
    assert_eq!(chain().reach3(&VOLUME), [3, 3, 3]);
    agrees_under_every_decomposition("the ratio response", &chain, &input);
}
