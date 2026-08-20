// SPDX-License-Identifier: MIT
//
// The acceptance bar for `ops::fft`, and the controls that say the bar is a bar.
//
// What is being checked, and why a tolerance rather than equality
// ---------------------------------------------------------------
// The transform route and the direct route compute the same quantity by
// arithmetic that shares no summation order, so they cannot agree to the bit and
// this file does not ask them to. What it asks is that they agree to **a stated
// bound**, and the bound is the whole point: agreement at `1e-15` says the
// padding, the conjugation, the lag convention and the three terms' supports are
// all right; agreement at `1e-3` says one of them is wrong in a way that looks
// plausible. Each parity test prints the figure it achieved, so the number in a
// report is a number this suite produced rather than one somebody remembered.
//
// The fixtures, and the single sharpest hazard here
// -------------------------------------------------
// **A symmetric fixture cannot distinguish an error that is a symmetry of the
// fixture.** A centred impulse, a square plane, a power-of-two extent or a lag
// window symmetric about zero would each hide a whole class of bug — a
// transposed axis, an off-by-one in the padding, a sign error in the lag
// convention — and a fixture with all four hides all of them at once while
// passing.
//
// So every fixture here is asymmetric on every axis and not a power of two:
// planes of `13 x 23` against `11 x 19`, and a lag window of `[-5, 6] x [-7, 4]`
// which is off-centre on both axes and in opposite directions. `the_fixture_
// discriminates` asserts that the fixture *is* able to tell these apart, rather
// than leaving it to be assumed — a landscape that equalled its own reflection
// or its own transpose would make every control below vacuous.
//
// The negative controls
// ---------------------
// Each is the same program with one thing changed. Each reproduces every count —
// the same lag window, the same overlap sizes, the same landscape shape — and
// each moves the answer.
//
// | control | what changes | what it would have hidden |
// |---|---|---|
// | `padding_below_the_minimum_wraps` | `Padding::Exact` shorter than [`minimal_wrap_free_length`] | a circular correlation passed off as a linear one |
// | `swapping_the_two_planes_reflects_the_landscape` | the operand order, which is conjugating the other spectrum | the wrong operand conjugated, and a sign error in the lag convention |
// | `transposing_both_planes_transposes_the_landscape` | the axis order | a transposed axis inside the two-pass transform |
// | `normalising_by_a_constant_count_moves_the_minimum` | the overlap count replaced by its largest value | the two energy terms summed over the whole plane instead of over the overlap |
//
// The liveness test beside the parity test
// ----------------------------------------
// A parity test that compares two constants passes forever. `the_parity_test_is_
// live` perturbs one element of one plane and asserts that **both** routes move,
// that they move by the same amount, and that the landscape's minimum is not
// simply flat. Without it, "the two agree to 4e-16" would be equally true of two
// implementations that both returned zero.
//
// Backends, and why the bar is a loop rather than a second copy of itself
// -----------------------------------------------------------------------
// `ops::fft` has a [`TransformBackend`], and the crate's optional `fftw` feature
// adds a second one. **The acceptance bar does not fork.** Every parity test
// here runs over `TransformBackend::available()` — one backend in the default
// build, where the loop is invisible and the printed figures are unchanged to
// the last digit, and both when the feature is on. A backend that could only
// meet a bar written for it is not a backend, it is a second program.
//
// Two tests exist only when there are two backends to compare, and one exists
// precisely because there might not be:
//
// | test | what it establishes |
// |---|---|
// | `the_feature_is_the_only_thing_that_chooses_the_backend` | runs **always**, and in the default build asserts the *absence*: one backend, and it is the portable one. The feature claim is inverted rather than dropped |
// | `the_two_backends_agree_on_the_fixture` | the agreement *between* backends, which is a sharper question than each agreeing with the direct walk — both could pass a `1e-13` bar against the oracle and still differ by `1e-13` from each other |
// | `the_two_backends_agree_at_the_consumers_geometry` | the same at `128 x 1350`, four orders more arithmetic for them to drift over |
//
// Both comparisons assert on **values**, and their liveness half is that the two
// landscapes must differ *somewhere* in the last places: two implementations of
// the same arithmetic that agreed to the bit would mean the test was comparing
// one backend with itself. Comparing argmins instead would be no test at all —
// `padding_below_the_minimum_wraps` already shows the argmin sitting still while
// the values it is chosen from move by 12-41%.
//
// `plans_built_at_the_same_time_agree_with_plans_built_alone` is the other thing
// a second backend brings: FFTW's planner is process-wide state its own
// documentation says is not re-entrant. That test builds six plans of six shapes
// from six threads. It is an exercise and not a proof, and `src/ops/fft.rs` says
// so where the lock that makes it safe is written.

use blockflow::ops::fft::{
    correlate_direct, minimal_wrap_free_length, squared_difference_direct, Correlation2, Landscape,
    Padding, RealTransform2, ShiftWindow, SquaredDifference, TransformBackend,
};
use ndarray::Array2;

/// The larger of two floats **by `f64::total_cmp`**, and `smaller` beside it.
///
/// `f64::max` and `f64::min` appear nowhere in this crate: `f64::max(-0.0, 0.0)`
/// may return either operand, and a rule with an exception for "it is only a
/// magnitude" is a rule that gets applied to the one place it mattered last.
fn larger(left: f64, right: f64) -> f64 {
    if left.total_cmp(&right).is_gt() {
        left
    } else {
        right
    }
}

fn smaller(left: f64, right: f64) -> f64 {
    if left.total_cmp(&right).is_lt() {
        left
    } else {
        right
    }
}

/// A deterministic plane with no symmetry on either axis.
///
/// An xorshift sequence with a different ramp along each axis on top. The ramps
/// are what make a reflection, a transpose and a one-element shift all visible:
/// pure noise is asymmetric too, but only in a way that a reader cannot check by
/// eye, and the ramps make the asymmetry a property of the *construction*.
fn plane(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
    let mut state = seed | 1;
    Array2::from_shape_fn((rows, cols), |(row, col)| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let noise = (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
        noise + 0.05 * row as f64 - 0.017 * col as f64
    })
}

/// The fixture geometry, in one place so every test below shares it.
///
/// `13 x 23` against `11 x 19`: four distinct extents, all odd, none a power of
/// two, and the two planes not the same shape as each other.
const SHAPE_A: [usize; 2] = [13, 23];
const SHAPE_B: [usize; 2] = [11, 19];

/// Off-centre on both axes, and in opposite directions, so a sign error in the
/// lag convention cannot be absorbed by the window.
fn window() -> ShiftWindow {
    ShiftWindow::new([-5, -7], [6, 4]).unwrap()
}

/// The worst absolute difference between two landscapes, and that difference
/// relative to the scale of the second.
fn agreement(left: &Landscape, right: &Landscape) -> (f64, f64) {
    let mut worst = 0.0f64;
    let mut scale = 0.0f64;
    let a = left.mean_squared();
    let b = right.mean_squared();
    assert_eq!(a.dim(), b.dim());
    for ((&got, &expected), &count) in a.iter().zip(b.iter()).zip(right.overlap().iter()) {
        if count == 0 {
            assert!(
                got.is_infinite() && expected.is_infinite(),
                "an empty overlap must be infinite on both sides"
            );
            continue;
        }
        worst = larger(worst, (got - expected).abs());
        scale = larger(scale, expected.abs());
    }
    (worst, worst / scale)
}

// ------------------------------------------------------- the fixture is sharp --

#[test]
fn the_fixture_discriminates() {
    let a = plane(SHAPE_A[0], SHAPE_A[1], 0x9E37_79B9_7F4A_7C15);
    let b = plane(SHAPE_B[0], SHAPE_B[1], 0x1234_5678_9ABC_DEF1);
    let landscape = squared_difference_direct(a.view(), b.view(), window());
    let values = landscape.mean_squared();
    let [rows, cols] = window().extent();

    // Not constant: a landscape that is flat makes every comparison below pass
    // for the wrong reason.
    let finite = values.iter().copied().filter(|v| v.is_finite());
    let low = finite.clone().fold(f64::INFINITY, smaller);
    let high = finite.fold(f64::NEG_INFINITY, larger);
    assert!(
        high > low * 1.05,
        "the landscape is nearly flat: {low} to {high}"
    );

    // Not equal to its own reflection, so a sign error in the lag convention is
    // visible in this fixture rather than absorbed by it.
    let mut reflected = 0usize;
    for row in 0..rows {
        for col in 0..cols {
            if values[[row, col]] != values[[rows - 1 - row, cols - 1 - col]] {
                reflected += 1;
            }
        }
    }
    assert!(
        reflected > rows * cols / 2,
        "only {reflected} of {} entries change under reflection",
        rows * cols
    );

    // The window is square (12 x 12) precisely so that this comparison is
    // well-formed: a landscape that equalled its own transpose could not detect
    // a transposed axis.
    assert_eq!(window().extent(), [12, 12]);
    let mut transposed = 0usize;
    for row in 0..rows {
        for col in 0..cols {
            if values[[row, col]] != values[[col, row]] {
                transposed += 1;
            }
        }
    }
    assert!(
        transposed > rows * cols / 2,
        "only {transposed} of {} entries change under transposition",
        rows * cols
    );

    // And the overlap count genuinely varies over the window, which is what
    // makes the normalisation load-bearing rather than a constant factor.
    let counts = landscape.overlap();
    let smallest = counts.iter().copied().min().unwrap();
    let largest = counts.iter().copied().max().unwrap();
    assert!(
        smallest * 2 < largest,
        "the overlap barely varies: {smallest} to {largest}"
    );
}

// ------------------------------------------------------------------- parity --

#[test]
fn the_correlation_agrees_with_the_direct_walk() {
    let a = plane(SHAPE_A[0], SHAPE_A[1], 0x9E37_79B9_7F4A_7C15);
    let b = plane(SHAPE_B[0], SHAPE_B[1], 0x1234_5678_9ABC_DEF1);
    let expected = correlate_direct(a.view(), b.view(), window());
    // The bar is the bar for every backend this build has, and it is the same
    // bar, on the same fixture, against the same oracle.
    for &backend in TransformBackend::available() {
        let mut plan =
            Correlation2::with_backend(SHAPE_A, SHAPE_B, window(), Padding::Smooth, backend)
                .unwrap();
        assert_eq!(plan.backend(), backend);
        assert!(!plan.wraps());
        let got = plan.correlate(a.view(), b.view()).unwrap();

        let mut worst = 0.0f64;
        let mut scale = 0.0f64;
        for (&left, &right) in got.iter().zip(expected.iter()) {
            worst = larger(worst, (left - right).abs());
            scale = larger(scale, right.abs());
        }
        println!(
            "correlation on {}: worst {worst:e} absolute, {:e} relative",
            backend.name(),
            worst / scale
        );
        assert!(
            worst / scale < 1.0e-13,
            "the {} correlation disagrees with the direct one by {:e} relative",
            backend.name(),
            worst / scale
        );
    }
}

#[test]
fn the_landscape_agrees_with_the_direct_walk_and_chooses_the_same_lag() {
    let a = plane(SHAPE_A[0], SHAPE_A[1], 0x9E37_79B9_7F4A_7C15);
    let b = plane(SHAPE_B[0], SHAPE_B[1], 0x1234_5678_9ABC_DEF1);
    let expected = squared_difference_direct(a.view(), b.view(), window());
    for &backend in TransformBackend::available() {
        let mut plan =
            SquaredDifference::with_backend(SHAPE_A, SHAPE_B, window(), Padding::Smooth, backend)
                .unwrap();
        let got = plan.landscape(a.view(), b.view()).unwrap();

        // The overlap counts are integers on both sides, so *these* compare exactly.
        assert_eq!(got.overlap(), expected.overlap());

        let (absolute, relative) = agreement(&got, &expected);
        println!(
            "landscape on {}: worst {absolute:e} absolute, {relative:e} relative",
            backend.name()
        );
        assert!(
            relative < 1.0e-13,
            "the {} landscape disagrees with the direct one by {relative:e} relative",
            backend.name()
        );
        // The decision the landscape exists to make is identical, which is the
        // strongest form of agreement available when the values cannot be.
        assert_eq!(got.argmin().unwrap().0, expected.argmin().unwrap().0);
    }
}

#[test]
fn the_parity_test_is_live() {
    let a = plane(SHAPE_A[0], SHAPE_A[1], 0x9E37_79B9_7F4A_7C15);
    let b = plane(SHAPE_B[0], SHAPE_B[1], 0x1234_5678_9ABC_DEF1);
    for &backend in TransformBackend::available() {
        let mut plan =
            SquaredDifference::with_backend(SHAPE_A, SHAPE_B, window(), Padding::Smooth, backend)
                .unwrap();

        let before_direct = squared_difference_direct(a.view(), b.view(), window());
        let before_fft = plan.landscape(a.view(), b.view()).unwrap();

        // One element, in an asymmetric position, changed by a large amount.
        let mut perturbed = a.clone();
        perturbed[[3, 17]] += 7.5;
        let after_direct = squared_difference_direct(perturbed.view(), b.view(), window());
        let after_fft = plan.landscape(perturbed.view(), b.view()).unwrap();

        // Both routes moved, so neither is a constant.
        let (direct_move, _) = agreement(&before_direct, &after_direct);
        let (fft_move, _) = agreement(&before_fft, &after_fft);
        // The move is ten orders of magnitude above the agreement bound, so "they
        // agree" is a statement about two things that both responded.
        assert!(
            direct_move > 1.0e-2,
            "the direct route did not move: {direct_move:e}"
        );
        assert!(
            fft_move > 1.0e-2,
            "the {} route did not move: {fft_move:e}",
            backend.name()
        );

        // And they still agree afterwards, to the same bound.
        let (_, relative) = agreement(&after_fft, &after_direct);
        println!(
            "after the perturbation on {}: {relative:e} relative",
            backend.name()
        );
        assert!(relative < 1.0e-13);
    }
}

#[test]
fn a_common_mean_costs_precision_and_the_cost_is_measured() {
    // `D = Ea + Eb - 2C`, and with a large common offset all three terms are
    // large and their combination is small. The module's header tells a caller
    // to centre the data; this is the measurement behind that sentence.
    let a = plane(SHAPE_A[0], SHAPE_A[1], 0x9E37_79B9_7F4A_7C15);
    let b = plane(SHAPE_B[0], SHAPE_B[1], 0x1234_5678_9ABC_DEF1);
    let mut plan = SquaredDifference::new(SHAPE_A, SHAPE_B, window(), Padding::Smooth).unwrap();

    let centred = agreement(
        &plan.landscape(a.view(), b.view()).unwrap(),
        &squared_difference_direct(a.view(), b.view(), window()),
    );
    let offset_a = a.mapv(|value| value + 50.0);
    let offset_b = b.mapv(|value| value + 50.0);
    let offset = agreement(
        &plan.landscape(offset_a.view(), offset_b.view()).unwrap(),
        &squared_difference_direct(offset_a.view(), offset_b.view(), window()),
    );
    println!(
        "centred {:e} relative, offset by 50 {:e} relative",
        centred.1, offset.1
    );
    assert!(centred.1 < 1.0e-14);
    // Still comfortably inside the bar, and visibly worse: the point is that the
    // loss is real and bounded, not that it does not exist.
    assert!(offset.1 < 1.0e-9);
    assert!(offset.1 > centred.1);
}

#[test]
fn the_landscape_agrees_at_the_consumers_geometry() {
    // The extents the consumer actually has — two `96 x 1304` planes with lags
    // over a bounded window — rather than a toy. The direct walk over the full
    // `61 x 61` window is 4.6e8 element pairs, so the window here is narrowed
    // to keep the test in the low seconds while the *planes* stay full size,
    // which is where the padding rule and the transform sizes come from.
    let shape_a = [96usize, 1304];
    let shape_b = [96usize, 1304];
    let lags = ShiftWindow::new([-7, -9], [8, 6]).unwrap();
    let a = plane(shape_a[0], shape_a[1], 0xDEAD_BEEF_CAFE_0001);
    let b = plane(shape_b[0], shape_b[1], 0x0BAD_C0DE_F00D_0002);

    let expected = squared_difference_direct(a.view(), b.view(), lags);
    for &backend in TransformBackend::available() {
        let mut plan =
            SquaredDifference::with_backend(shape_a, shape_b, lags, Padding::Smooth, backend)
                .unwrap();
        assert!(!plan.wraps());
        let got = plan.landscape(a.view(), b.view()).unwrap();
        assert_eq!(got.overlap(), expected.overlap());
        let (absolute, relative) = agreement(&got, &expected);
        println!(
            "consumer geometry on {}, padded to {:?}: worst {absolute:e} absolute, \
             {relative:e} relative",
            backend.name(),
            plan.padded_shape()
        );
        assert!(relative < 1.0e-12, "{relative:e}");
        assert_eq!(got.argmin().unwrap().0, expected.argmin().unwrap().0);
    }
}

#[test]
fn the_padding_rule_produces_the_geometry_the_header_claims() {
    // Two `96 x 1304` planes and lags in `[-30, 30]` on both axes — the centred
    // window the module's header works through. The **off-centre** case, which is
    // what the first consumer actually has, is the test below this one.
    let lags = ShiftWindow::symmetric([30, 30]);
    assert_eq!(
        minimal_wrap_free_length([96, 1304], [96, 1304], lags),
        [126, 1334]
    );
    let plan = SquaredDifference::new([96, 1304], [96, 1304], lags, Padding::Smooth).unwrap();
    assert_eq!(plan.padded_shape(), [128, 1350]);
    assert!(!plan.wraps());
    let minimal = SquaredDifference::new([96, 1304], [96, 1304], lags, Padding::Minimal).unwrap();
    assert_eq!(minimal.padded_shape(), [126, 1334]);
    assert!(!minimal.wraps());
}

#[test]
fn an_off_centre_window_is_the_case_the_rule_exists_for_and_it_is_not_the_centred_one() {
    // The first consumer's real geometry: its two cuts sit 30 apart in global
    // coordinates, so every row lag it wants is non-negative and its row window
    // is `[0, 60]` rather than `[-30, 30]`. Same width, different padding, and
    // the difference is the whole point of the rule.
    let off_centre = ShiftWindow::new([0, -30], [60, 30]).unwrap();
    let centred = ShiftWindow::symmetric([30, 30]);
    assert_eq!(off_centre.extent(), centred.extent());

    assert_eq!(
        minimal_wrap_free_length([96, 1304], [96, 1304], off_centre),
        [156, 1334]
    );
    let plan = SquaredDifference::new([96, 1304], [96, 1304], off_centre, Padding::Smooth).unwrap();
    assert_eq!(plan.padded_shape(), [160, 1350]);
    assert!(!plan.wraps());

    // And it is right as well as different: a landscape over the off-centre
    // window still agrees with the direct walk. The planes are cut down here so
    // the direct side stays cheap; the *window* is the consumer's.
    let narrow = ShiftWindow::new([0, -4], [9, 5]).unwrap();
    let shape = [24usize, 61];
    let a = plane(shape[0], shape[1], 0x9E37_79B9_7F4A_7C15);
    let b = plane(shape[0], shape[1], 0x1234_5678_9ABC_DEF1);
    let mut narrow_plan = SquaredDifference::new(shape, shape, narrow, Padding::Smooth).unwrap();
    assert!(!narrow_plan.wraps());
    // `max(24, 24, 24 - 0, 24 + 9) = 33` on the first axis, not `24 + 24 - 1`.
    assert_eq!(minimal_wrap_free_length(shape, shape, narrow)[0], 33);
    let got = narrow_plan.landscape(a.view(), b.view()).unwrap();
    let expected = squared_difference_direct(a.view(), b.view(), narrow);
    assert_eq!(got.overlap(), expected.overlap());
    let (_, relative) = agreement(&got, &expected);
    println!("off-centre window: {relative:e} relative");
    assert!(relative < 1.0e-13, "{relative:e}");
    assert_eq!(got.argmin().unwrap().0, expected.argmin().unwrap().0);
}

// ------------------------------------------------------------- the buffers --

#[test]
fn landscape_into_gives_the_same_answer_as_the_allocating_form_and_survives_reuse() {
    let mut plan = SquaredDifference::new(SHAPE_A, SHAPE_B, window(), Padding::Smooth).unwrap();
    let mut buffer = plan.empty_landscape();
    // An empty buffer is not a landscape of zeros: it reads as "no overlap
    // anywhere", which is the only honest thing an uncomputed buffer can say.
    assert!(buffer
        .mean_squared()
        .iter()
        .all(|value| value.is_infinite()));
    assert!(buffer.overlap().iter().all(|&count| count == 0));
    assert_eq!(buffer.argmin(), None);

    for round in 0..4u64 {
        let a = plane(
            SHAPE_A[0],
            SHAPE_A[1],
            0x9E37_79B9_7F4A_7C15 ^ (round * 0x1234_5),
        );
        let b = plane(
            SHAPE_B[0],
            SHAPE_B[1],
            0x1234_5678_9ABC_DEF1 ^ (round * 0x9_ABCD),
        );
        let allocated = plan.landscape(a.view(), b.view()).unwrap();
        plan.landscape_into(a.view(), b.view(), &mut buffer)
            .unwrap();
        // Bit-identical, and a reused buffer must not carry anything of the
        // previous round — which is what makes running this in a loop the test.
        assert_eq!(buffer, allocated, "round {round}");
    }
}

#[test]
fn a_landscape_buffer_of_the_wrong_window_is_refused_rather_than_reshaped() {
    // A landscape carries the lag convention its indices are read against, so a
    // stale buffer of the right *shape* and the wrong window would be a wrong
    // answer that looked right. It is an error instead.
    let mut plan = SquaredDifference::new(SHAPE_A, SHAPE_B, window(), Padding::Smooth).unwrap();
    let a = plane(SHAPE_A[0], SHAPE_A[1], 0x9E37_79B9_7F4A_7C15);
    let b = plane(SHAPE_B[0], SHAPE_B[1], 0x1234_5678_9ABC_DEF1);

    let shifted = ShiftWindow::new([-4, -6], [7, 5]).unwrap();
    assert_eq!(
        shifted.extent(),
        window().extent(),
        "same shape, different lags"
    );
    let mut wrong = Landscape::empty(shifted);
    assert!(plan.landscape_into(a.view(), b.view(), &mut wrong).is_err());

    let mut right = Landscape::empty(window());
    assert!(plan.landscape_into(a.view(), b.view(), &mut right).is_ok());
}

// -------------------------------------------------------- negative controls --

#[test]
fn padding_below_the_minimum_wraps() {
    let a = plane(SHAPE_A[0], SHAPE_A[1], 0x9E37_79B9_7F4A_7C15);
    let b = plane(SHAPE_B[0], SHAPE_B[1], 0x1234_5678_9ABC_DEF1);
    let expected = squared_difference_direct(a.view(), b.view(), window());

    let minimal = minimal_wrap_free_length(SHAPE_A, SHAPE_B, window());
    assert_eq!(minimal, [18, 30]);

    // Walked from the smallest possible version of the mistake upwards, because
    // *where* it starts to bite is the interesting part.
    let mut moved_the_minimum_at = None::<usize>;
    for shortfall in 1..=5usize {
        let short = [minimal[0] - shortfall, minimal[1] - shortfall];
        let mut plan =
            SquaredDifference::new(SHAPE_A, SHAPE_B, window(), Padding::Exact(short)).unwrap();
        assert!(
            plan.wraps(),
            "the control must actually be the broken thing"
        );
        let got = plan.landscape(a.view(), b.view()).unwrap();

        // Every count is reproduced: the same window, the same overlaps, the
        // same shape. Only the values move.
        assert_eq!(got.overlap(), expected.overlap());
        assert_eq!(got.mean_squared().dim(), expected.mean_squared().dim());

        let (absolute, relative) = agreement(&got, &expected);
        let chosen = got.argmin().unwrap().0;
        println!(
            "{shortfall} short on each axis, padded {short:?}: {absolute:e} absolute, \
             {relative:e} relative, minimum at {chosen:?}"
        );
        // Twelve orders of magnitude above the bound the correct plan clears.
        // That is the gap the acceptance test measures, and it is why "agrees to
        // 1e-3" is a failure rather than a rounding.
        assert!(
            relative > 1.0e-3,
            "a wrapped correlation should be visibly wrong, and this one is only \
             {relative:e} away from the right answer — the control is not controlling"
        );
        // And it is a *plausible* landscape rather than a broken one, which is
        // the whole hazard: finite everywhere, and it still has a minimum.
        assert!(got.mean_squared().iter().all(|value| value.is_finite()));
        if chosen != expected.argmin().unwrap().0 && moved_the_minimum_at.is_none() {
            moved_the_minimum_at = Some(shortfall);
        }
    }

    // **The part of this control worth keeping, and it is not the part that was
    // expected.** Every shortfall from one element to five moves the landscape —
    // by 12% of its own scale at one and 40% at four — and **none of them moves
    // the chosen lag**. The wrap-around contamination is smooth enough that the
    // well the minimum sits in survives it.
    //
    // So "both routes pick the same minimum" is *not* an acceptance test for
    // this arithmetic. It is a necessary condition that a badly wrapped
    // correlation satisfies, and the only thing that separates a correct plan
    // from this one is comparing the values, which is what
    // `the_landscape_agrees_with_the_direct_walk_and_chooses_the_same_lag` does
    // and why it asserts the relative agreement first and the lag second.
    //
    // This is an **absence**, so it is asserted rather than left unsaid: if a
    // future fixture or a future padding rule makes the wrap move the selection,
    // this fails and the paragraph above is rewritten rather than quietly
    // becoming false. A control that does move the selection exists — see
    // `normalising_by_a_constant_count_moves_the_minimum`.
    assert_eq!(
        moved_the_minimum_at, None,
        "the wrap now moves the chosen lag as well as the landscape, at a \
         shortfall of {moved_the_minimum_at:?}; that is a stronger control than \
         this test was written for and the comment above it is now wrong"
    );
}

#[test]
fn swapping_the_two_planes_reflects_the_landscape() {
    // `D_ba(k) = D_ab(-k)`: substituting `x = y - k` turns one sum into the
    // other. So swapping the operands — which is exactly conjugating the other
    // spectrum in the product — reflects the landscape through the origin, and
    // the reflected window is `[-hi, -lo]`.
    let a = plane(SHAPE_A[0], SHAPE_A[1], 0x9E37_79B9_7F4A_7C15);
    let b = plane(SHAPE_B[0], SHAPE_B[1], 0x1234_5678_9ABC_DEF1);
    let forward_window = window();
    let reversed = ShiftWindow::new(
        [-forward_window.upper()[0], -forward_window.upper()[1]],
        [-forward_window.lower()[0], -forward_window.lower()[1]],
    )
    .unwrap();

    let mut forward_plan =
        SquaredDifference::new(SHAPE_A, SHAPE_B, forward_window, Padding::Smooth).unwrap();
    let mut swapped_plan =
        SquaredDifference::new(SHAPE_B, SHAPE_A, reversed, Padding::Smooth).unwrap();
    let forward = forward_plan.landscape(a.view(), b.view()).unwrap();
    let swapped = swapped_plan.landscape(b.view(), a.view()).unwrap();

    let [rows, cols] = forward_window.extent();
    let mut worst = 0.0f64;
    let mut differing = 0usize;
    for row in 0..rows {
        for col in 0..cols {
            let here = forward.mean_squared()[[row, col]];
            let there = swapped.mean_squared()[[rows - 1 - row, cols - 1 - col]];
            worst = larger(worst, (here - there).abs());
            if here != swapped.mean_squared()[[row, col]] {
                differing += 1;
            }
            assert_eq!(
                forward.overlap()[[row, col]],
                swapped.overlap()[[rows - 1 - row, cols - 1 - col]]
            );
        }
    }
    println!("swap: reflection identity holds to {worst:e}");
    assert!(
        worst < 1.0e-12,
        "the reflection identity failed by {worst:e}"
    );
    // And the control moves: read *without* reflecting, the swapped landscape is
    // a different answer almost everywhere, which is what conjugating the wrong
    // operand would silently produce.
    assert!(
        differing > rows * cols / 2,
        "only {differing} of {} entries move under the swap",
        rows * cols
    );
}

#[test]
fn transposing_both_planes_transposes_the_landscape() {
    let a = plane(SHAPE_A[0], SHAPE_A[1], 0x9E37_79B9_7F4A_7C15);
    let b = plane(SHAPE_B[0], SHAPE_B[1], 0x1234_5678_9ABC_DEF1);
    let straight = window();
    let swapped_axes = ShiftWindow::new(
        [straight.lower()[1], straight.lower()[0]],
        [straight.upper()[1], straight.upper()[0]],
    )
    .unwrap();

    let mut plan = SquaredDifference::new(SHAPE_A, SHAPE_B, straight, Padding::Smooth).unwrap();
    let mut transposed_plan = SquaredDifference::new(
        [SHAPE_A[1], SHAPE_A[0]],
        [SHAPE_B[1], SHAPE_B[0]],
        swapped_axes,
        Padding::Smooth,
    )
    .unwrap();
    let straight_landscape = plan.landscape(a.view(), b.view()).unwrap();
    let at = a.t().to_owned();
    let bt = b.t().to_owned();
    let transposed_landscape = transposed_plan.landscape(at.view(), bt.view()).unwrap();

    let [rows, cols] = straight.extent();
    let mut worst = 0.0f64;
    let mut differing = 0usize;
    for row in 0..rows {
        for col in 0..cols {
            let here = straight_landscape.mean_squared()[[row, col]];
            let there = transposed_landscape.mean_squared()[[col, row]];
            worst = larger(worst, (here - there).abs());
            if here != transposed_landscape.mean_squared()[[row, col]] {
                differing += 1;
            }
        }
    }
    println!("transpose: identity holds to {worst:e}");
    assert!(
        worst < 1.0e-12,
        "the transpose identity failed by {worst:e}"
    );
    assert!(
        differing > rows * cols / 2,
        "only {differing} of {} entries move under transposition",
        rows * cols
    );
}

#[test]
fn normalising_by_a_constant_count_moves_the_minimum() {
    // The two energy terms are summed over the *overlap*, and the mean is taken
    // over the overlap's size. Both are easy to state and easy to get wrong in
    // the same direction — summing over the whole plane and dividing by a
    // constant. This is what that costs.
    let a = plane(SHAPE_A[0], SHAPE_A[1], 0x9E37_79B9_7F4A_7C15);
    let b = plane(SHAPE_B[0], SHAPE_B[1], 0x1234_5678_9ABC_DEF1);
    let landscape = squared_difference_direct(a.view(), b.view(), window());

    let largest = landscape.overlap().iter().copied().max().unwrap() as f64;
    let [rows, cols] = window().extent();
    let mut best = None::<([isize; 2], f64)>;
    for row in 0..rows {
        for col in 0..cols {
            let count = landscape.overlap()[[row, col]];
            if count == 0 {
                continue;
            }
            // The same landscape with the normalisation replaced by a constant:
            // every total is the same, only the divisor changes.
            let total = landscape.mean_squared()[[row, col]] * count as f64;
            let value = total / largest;
            if best.map_or(true, |(_, incumbent)| value.total_cmp(&incumbent).is_lt()) {
                best = Some((window().shift_at([row, col]), value));
            }
        }
    }
    let normalised = landscape.argmin().unwrap().0;
    let constant = best.unwrap().0;
    println!("overlap-normalised minimum at {normalised:?}, constant-normalised at {constant:?}");
    assert_ne!(
        normalised, constant,
        "normalising by a constant should move the minimum on this fixture; if it \
         does not, the fixture's overlap barely varies and every claim about the \
         normalisation is untested"
    );
}

// ---------------------------------------------------------------- the plans --

#[test]
fn a_cloned_plan_computes_the_same_answer_from_another_thread() {
    // The claim `ops::fft`'s header makes about parallelism across planes:
    // clone the plan, keep the twiddles, get a fresh working set.
    //
    // For the FFTW backend this is also the thread-safety test. A clone there
    // shares one plan through an `Arc` and executes it from four threads at
    // once, which is the one thing FFTW documents as re-entrant — and bit
    // identity is what says the shared plan was not being mutated underneath.
    let a = plane(SHAPE_A[0], SHAPE_A[1], 0x9E37_79B9_7F4A_7C15);
    let b = plane(SHAPE_B[0], SHAPE_B[1], 0x1234_5678_9ABC_DEF1);
    for &backend in TransformBackend::available() {
        let plan =
            SquaredDifference::with_backend(SHAPE_A, SHAPE_B, window(), Padding::Smooth, backend)
                .unwrap();
        let expected = plan.clone().landscape(a.view(), b.view()).unwrap();

        let handles = (0..4)
            .map(|_| {
                let mut mine = plan.clone();
                let a = a.clone();
                let b = b.clone();
                std::thread::spawn(move || mine.landscape(a.view(), b.view()).unwrap())
            })
            .collect::<Vec<_>>();
        for handle in handles {
            // Bit-identical: the same plan and the same data on another thread is
            // the same arithmetic in the same order.
            assert_eq!(handle.join().unwrap(), expected, "on {}", backend.name());
        }
    }
}

#[test]
fn plans_built_at_the_same_time_agree_with_plans_built_alone() {
    // The other half of the plan question, and the half a shared-plan clone
    // does not reach: **building** several plans at once.
    //
    // `rustfft`'s planner is an ordinary owned value, so this is uninteresting
    // for the portable backend and runs anyway. FFTW's planner is process-wide
    // state that its own documentation says is not re-entrant, so this is the
    // shape of program that would find an unlocked one. It is an exercise
    // rather than a proof — a race that does not happen to fire still passes —
    // which is why the lock is argued in `src/ops/fft.rs` rather than left to
    // this test to establish.
    //
    // The shapes are all different, so no two threads are asking for a plan the
    // planner has already made.
    let shapes: [[usize; 2]; 6] = [[13, 23], [17, 29], [11, 19], [21, 31], [15, 27], [19, 25]];
    for &backend in TransformBackend::available() {
        let alone = shapes
            .iter()
            .map(|&shape| {
                let source = plane(shape[0], shape[1], 0x5EED_0000 ^ shape[1] as u64);
                let mut transform = RealTransform2::with_backend(shape, backend).unwrap();
                let mut spectrum = transform.spectrum();
                transform.forward(source.view(), &mut spectrum).unwrap();
                spectrum
            })
            .collect::<Vec<_>>();

        let handles = shapes
            .iter()
            .map(|&shape| {
                std::thread::spawn(move || {
                    let source = plane(shape[0], shape[1], 0x5EED_0000 ^ shape[1] as u64);
                    let mut transform = RealTransform2::with_backend(shape, backend).unwrap();
                    let mut spectrum = transform.spectrum();
                    transform.forward(source.view(), &mut spectrum).unwrap();
                    spectrum
                })
            })
            .collect::<Vec<_>>();
        for (handle, expected) in handles.into_iter().zip(alone.iter()) {
            assert_eq!(&handle.join().unwrap(), expected, "on {}", backend.name());
        }
    }
}

#[test]
fn a_plan_reused_over_many_pairs_gives_the_same_answers_as_fresh_ones() {
    // Plan reuse is the largest single speed lever here, and a reused plan that
    // carried state between calls would be a silent wrong answer rather than a
    // slow one.
    for &backend in TransformBackend::available() {
        let mut plan =
            SquaredDifference::with_backend(SHAPE_A, SHAPE_B, window(), Padding::Smooth, backend)
                .unwrap();
        for round in 0..5u64 {
            let a = plane(
                SHAPE_A[0],
                SHAPE_A[1],
                0x9E37_79B9_7F4A_7C15 ^ (round * 0x1234_5),
            );
            let b = plane(
                SHAPE_B[0],
                SHAPE_B[1],
                0x1234_5678_9ABC_DEF1 ^ (round * 0x9_ABCD),
            );
            let reused = plan.landscape(a.view(), b.view()).unwrap();
            let fresh = SquaredDifference::with_backend(
                SHAPE_A,
                SHAPE_B,
                window(),
                Padding::Smooth,
                backend,
            )
            .unwrap()
            .landscape(a.view(), b.view())
            .unwrap();
            assert_eq!(reused, fresh, "round {round} on {}", backend.name());
        }
    }
}

#[test]
fn a_transform_refuses_what_it_cannot_hold() {
    // The shape checks live above the backend, so they are the same checks
    // whichever one is underneath — which is a claim, and this is where it is
    // tested rather than assumed.
    for &backend in TransformBackend::available() {
        assert!(RealTransform2::with_backend([0, 4], backend).is_err());
        assert!(RealTransform2::with_backend([4, 0], backend).is_err());
        let mut transform = RealTransform2::with_backend([6, 9], backend).unwrap();
        let mut spectrum = transform.spectrum();
        assert_eq!(spectrum.dim(), (6, 5));
        let too_big = Array2::<f64>::zeros((7, 9));
        assert!(transform
            .forward_zero_padded(too_big.view(), &mut spectrum)
            .is_err());
        let wrong_shape = Array2::<f64>::zeros((5, 9));
        assert!(transform
            .forward(wrong_shape.view(), &mut spectrum)
            .is_err());
        // A padded length that cannot even hold the two planes is refused rather
        // than silently truncating them.
        assert!(Correlation2::with_backend(
            SHAPE_A,
            SHAPE_B,
            window(),
            Padding::Exact([4, 4]),
            backend
        )
        .is_err());
    }
}

// ------------------------------------------------------------ the backends --

/// What the `fftw` feature does, asserted from both sides of it.
///
/// The interesting half is the one that runs by default: **with the feature off
/// there is one backend and it is the portable one**. That is the assertion the
/// package count and the byte-for-byte claim rest on, and it is written as an
/// inverted form of the same test rather than left out of the default build.
#[test]
fn the_feature_is_the_only_thing_that_chooses_the_backend() {
    let available = TransformBackend::available();
    assert_eq!(available[0], TransformBackend::Portable);
    assert!(available.contains(&TransformBackend::default()));
    // Every plan that is not asked for a backend is on the default one, at each
    // of the three levels that can build one.
    assert_eq!(
        RealTransform2::new([8, 12]).unwrap().backend(),
        TransformBackend::default()
    );
    assert_eq!(
        Correlation2::new(SHAPE_A, SHAPE_B, window(), Padding::Smooth)
            .unwrap()
            .backend(),
        TransformBackend::default()
    );
    assert_eq!(
        SquaredDifference::new(SHAPE_A, SHAPE_B, window(), Padding::Smooth)
            .unwrap()
            .backend(),
        TransformBackend::default()
    );

    #[cfg(not(feature = "fftw"))]
    {
        assert_eq!(
            available,
            &[TransformBackend::Portable],
            "without the `fftw` feature there is one backend and it is the pure-Rust one"
        );
        assert_eq!(TransformBackend::default(), TransformBackend::Portable);
    }
    #[cfg(feature = "fftw")]
    {
        assert_eq!(
            available,
            &[TransformBackend::Portable, TransformBackend::Fftw],
            "with the `fftw` feature both backends are compiled, which is what lets \
             the parity test below compare them in one process"
        );
        assert_eq!(
            TransformBackend::default(),
            TransformBackend::Fftw,
            "the feature's whole job is to swap the default"
        );
    }
}

/// The feature must not change the *shape* of the API either, and the auto
/// traits are the part of that shape nothing writes down.
///
/// `Send` is what `a_cloned_plan_computes_the_same_answer_from_another_thread`
/// needs and would catch. `Sync` is the one that could have gone missing in
/// silence: the portable plans are `Sync` because `rustfft`'s and `realfft`'s
/// trait objects are, and an FFTW working set holding a raw pointer is `Sync`
/// only because this test says it must be. A feature that took `Sync` off three
/// public types would be changing the API rather than the transform behind it.
#[test]
fn the_plans_are_send_and_sync_whichever_backend_they_are_on() {
    fn assert_send_and_sync<T: Send + Sync>() {}
    assert_send_and_sync::<RealTransform2>();
    assert_send_and_sync::<Correlation2>();
    assert_send_and_sync::<SquaredDifference>();

    // And `Sync` used rather than only asserted: two threads sharing one plan
    // by reference, reading the geometry it computed before any data arrived.
    for &backend in TransformBackend::available() {
        let plan =
            SquaredDifference::with_backend(SHAPE_A, SHAPE_B, window(), Padding::Smooth, backend)
                .unwrap();
        let shared = &plan;
        std::thread::scope(|scope| {
            let handles = (0..2)
                .map(|_| scope.spawn(move || (shared.padded_shape(), shared.overlap().sum())))
                .collect::<Vec<_>>();
            for handle in handles {
                assert_eq!(
                    handle.join().unwrap(),
                    (plan.padded_shape(), plan.overlap().sum())
                );
            }
        });
    }
}

/// The two backends against each other, on the fixture the bar already uses.
///
/// This is a different question from "each agrees with the direct walk", and a
/// sharper one: the direct walk is compared against at `1e-13`, so two backends
/// could each pass it and still differ from one another by `1e-13`. The number
/// this prints is the one to quote for "the backends agree".
///
/// **Values, not the argmin.** The short-padding control in this file already
/// established that an argmin can sit still while the values it is chosen from
/// move by tens of percent, so "both backends pick the same lag" would be a test
/// that passes for a backend that is wrong.
#[cfg(feature = "fftw")]
#[test]
fn the_two_backends_agree_on_the_fixture() {
    let a = plane(SHAPE_A[0], SHAPE_A[1], 0x9E37_79B9_7F4A_7C15);
    let b = plane(SHAPE_B[0], SHAPE_B[1], 0x1234_5678_9ABC_DEF1);
    let landscape = |backend| {
        SquaredDifference::with_backend(SHAPE_A, SHAPE_B, window(), Padding::Smooth, backend)
            .unwrap()
            .landscape(a.view(), b.view())
            .unwrap()
    };
    let portable = landscape(TransformBackend::Portable);
    let fftw = landscape(TransformBackend::Fftw);

    // Geometry is geometry: these are integers on both sides and comparing them
    // exactly is the whole point of computing them without a transform.
    assert_eq!(portable.overlap(), fftw.overlap());

    let (absolute, relative) = agreement(&fftw, &portable);
    println!("the two backends: worst {absolute:e} absolute, {relative:e} relative");
    assert!(
        relative < 1.0e-13,
        "the two backends disagree by {relative:e} relative, which is more than \
         either is allowed to disagree with the direct walk"
    );
    assert_eq!(fftw.argmin().unwrap().0, portable.argmin().unwrap().0);

    // And the liveness half: a comparison of two identical things would pass the
    // assertion above forever. These are two different implementations of the
    // same arithmetic, so they must differ *somewhere* in the last few places —
    // if they do not, this test is comparing one backend with itself.
    assert!(
        absolute > 0.0,
        "the two backends produced a bit-identical landscape, which means the \
         feature did not swap anything"
    );
}

/// The same question at the geometry the consumer actually has, where the
/// padded transform is `128 x 1350` rather than `128 x 32` and there are four
/// orders of magnitude more arithmetic for the two to drift apart over.
#[cfg(feature = "fftw")]
#[test]
fn the_two_backends_agree_at_the_consumers_geometry() {
    let shape = [96usize, 1304];
    let lags = ShiftWindow::new([-7, -9], [8, 6]).unwrap();
    let a = plane(shape[0], shape[1], 0xDEAD_BEEF_CAFE_0001);
    let b = plane(shape[0], shape[1], 0x0BAD_C0DE_F00D_0002);
    let landscape = |backend| {
        SquaredDifference::with_backend(shape, shape, lags, Padding::Smooth, backend)
            .unwrap()
            .landscape(a.view(), b.view())
            .unwrap()
    };
    let portable = landscape(TransformBackend::Portable);
    let fftw = landscape(TransformBackend::Fftw);

    assert_eq!(portable.overlap(), fftw.overlap());
    let (absolute, relative) = agreement(&fftw, &portable);
    println!(
        "the two backends at the consumer's geometry: worst {absolute:e} absolute, \
         {relative:e} relative"
    );
    assert!(relative < 1.0e-12, "{relative:e}");
    assert_eq!(fftw.argmin().unwrap().0, portable.argmin().unwrap().0);
    assert!(
        absolute > 0.0,
        "the two backends produced a bit-identical landscape, which means the \
         feature did not swap anything"
    );
}

// -------------------------------------------------- the full-scale numbers --

/// The consumer's whole geometry, against the direct walk it exists to replace.
///
/// Two `96 x 1304` planes and the full `61 x 61` lag window: `3721` lags over a
/// `125_184`-element plane is **4.7e8 element pairs** for the direct route, which
/// is why it is `#[ignore]`d rather than run on every `cargo test` — and why the
/// transform route exists at all. Run it with
/// `cargo test --release --test fft_correlation -- --ignored --nocapture`.
#[test]
#[ignore = "4.7e8 element pairs on the direct side; run it deliberately"]
fn the_full_consumer_window_agrees_with_the_direct_walk() {
    let shape = [96usize, 1304];
    let lags = ShiftWindow::symmetric([30, 30]);
    let a = plane(shape[0], shape[1], 0xDEAD_BEEF_CAFE_0001);
    let b = plane(shape[0], shape[1], 0x0BAD_C0DE_F00D_0002);

    // The oracle is the expensive half and does not depend on the backend, so it
    // is walked once and every backend is held against it.
    let started = std::time::Instant::now();
    let expected = squared_difference_direct(a.view(), b.view(), lags);
    let slow = started.elapsed();

    for &backend in TransformBackend::available() {
        let mut plan =
            SquaredDifference::with_backend(shape, shape, lags, Padding::Smooth, backend).unwrap();
        assert_eq!(plan.padded_shape(), [128, 1350]);
        assert!(!plan.wraps());

        let started = std::time::Instant::now();
        let got = plan.landscape(a.view(), b.view()).unwrap();
        let fast = started.elapsed();

        assert_eq!(got.overlap(), expected.overlap());
        let (absolute, relative) = agreement(&got, &expected);
        println!(
            "full window {:?} lags over {shape:?} on {}: {absolute:e} absolute, \
             {relative:e} relative; transform {:?}, direct {:?} ({:.0}x)",
            lags.extent(),
            backend.name(),
            fast,
            slow,
            slow.as_secs_f64() / fast.as_secs_f64()
        );
        assert!(relative < 1.0e-12, "{relative:e}");
        assert_eq!(got.argmin().unwrap().0, expected.argmin().unwrap().0);
    }
}

/// What the levers are worth, measured rather than asserted.
///
/// Four rows, each the same landscape computed a different way: the padding
/// rule, plan reuse, and threads over independent plane pairs. Printed rather
/// than asserted, because a timing assertion on a shared machine is a flaky test
/// dressed as a guarantee.
#[test]
#[ignore = "a measurement, not a check"]
fn the_speed_levers_are_measured() {
    let shape = [96usize, 1304];
    let lags = ShiftWindow::symmetric([30, 30]);
    let a = plane(shape[0], shape[1], 0xDEAD_BEEF_CAFE_0001);
    let b = plane(shape[0], shape[1], 0x0BAD_C0DE_F00D_0002);
    let rounds = 20;

    let best = |plan: &mut SquaredDifference| -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..rounds {
            let started = std::time::Instant::now();
            let landscape = plan.landscape(a.view(), b.view()).unwrap();
            let elapsed = started.elapsed().as_secs_f64();
            std::hint::black_box(&landscape);
            best = smaller(best, elapsed);
        }
        best
    };

    for (label, padding) in [
        ("Padding::Smooth", Padding::Smooth),
        ("Padding::Minimal", Padding::Minimal),
        // The geometry the consumer's own reference pads to: a bounding box
        // rather than a chosen length, and `157` is prime.
        ("Padding::Exact([157, 1335])", Padding::Exact([157, 1335])),
    ] {
        let mut plan = SquaredDifference::new(shape, shape, lags, padding).unwrap();
        println!(
            "{label:28} padded {:?}  {:8.3} ms per landscape",
            plan.padded_shape(),
            best(&mut plan) * 1e3
        );
    }

    // A held landscape buffer against a fresh one per call. This is the smallest
    // lever here and it is measured rather than assumed, because "it must be
    // faster" is how an API grows a signature that pays for nothing.
    let mut plan = SquaredDifference::new(shape, shape, lags, Padding::Smooth).unwrap();
    let allocating = best(&mut plan);
    let mut buffer = plan.empty_landscape();
    let mut into = f64::INFINITY;
    for _ in 0..rounds {
        let started = std::time::Instant::now();
        plan.landscape_into(a.view(), b.view(), &mut buffer)
            .unwrap();
        into = smaller(into, started.elapsed().as_secs_f64());
        std::hint::black_box(&buffer);
    }
    println!(
        "landscape {:8.3} ms, landscape_into {:8.3} ms ({:.3}x)",
        allocating * 1e3,
        into * 1e3,
        allocating / into
    );

    // Plan reuse against re-planning per landscape, which is the mistake this
    // API's shape exists to prevent.
    let reused = best(&mut plan);
    let mut fresh = f64::INFINITY;
    for _ in 0..rounds {
        let started = std::time::Instant::now();
        let mut throwaway = SquaredDifference::new(shape, shape, lags, Padding::Smooth).unwrap();
        let landscape = throwaway.landscape(a.view(), b.view()).unwrap();
        std::hint::black_box(&landscape);
        fresh = smaller(fresh, started.elapsed().as_secs_f64());
    }
    println!(
        "plan reused {:8.3} ms, planned per landscape {:8.3} ms ({:.2}x)",
        reused * 1e3,
        fresh * 1e3,
        fresh / reused
    );

    // Threads over independent plane pairs, one cloned plan each.
    for threads in [1usize, 2, 4, 8] {
        let started = std::time::Instant::now();
        let handles = (0..threads)
            .map(|_| {
                let mut mine = plan.clone();
                let a = a.clone();
                let b = b.clone();
                std::thread::spawn(move || {
                    let mut best = f64::INFINITY;
                    for _ in 0..rounds {
                        let started = std::time::Instant::now();
                        let landscape = mine.landscape(a.view(), b.view()).unwrap();
                        std::hint::black_box(&landscape);
                        best = smaller(best, started.elapsed().as_secs_f64());
                    }
                    best
                })
            })
            .collect::<Vec<_>>();
        let worst = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .fold(0.0f64, larger);
        println!(
            "{threads} threads: {:8.3} ms per landscape per thread, {:8.3} landscapes/s total, \
             wall {:?}",
            worst * 1e3,
            threads as f64 / worst,
            started.elapsed()
        );
    }
}

/// The backends against each other, measured rather than argued about.
///
/// Four numbers per backend and they are not all the same story, which is the
/// reason this prints a table rather than a verdict:
///
/// * the forward transform at `128 x 1350`, the shape `Padding::Smooth` chooses
///   for the consumer's geometry — the row FFTW was expected to win;
/// * the forward transform at `157 x 1335`, the shape the consumer's geometry
///   has *before* the padding rule is applied, where `157` is prime;
/// * a whole landscape, which is two forwards, one inverse and the rectangle
///   sums that no transform touches;
/// * and the same landscape from eight threads over cloned plans, because a
///   backend that cannot be shared across threads would lose the `6.4x` that is
///   worth more than either row above.
///
/// Best-of on a shared machine, so read the ratios and not the milliseconds.
#[test]
#[ignore = "a measurement, not a check"]
fn the_backends_are_measured_against_each_other() {
    let rounds = 50;
    for shape in [[128usize, 1350usize], [157, 1335]] {
        let source = plane(shape[0], shape[1], 0xDEAD_BEEF_CAFE_0001);
        for &backend in TransformBackend::available() {
            let mut transform = RealTransform2::with_backend(shape, backend).unwrap();
            let mut spectrum = transform.spectrum();
            let mut best = f64::INFINITY;
            for _ in 0..rounds {
                let started = std::time::Instant::now();
                transform.forward(source.view(), &mut spectrum).unwrap();
                std::hint::black_box(&spectrum);
                best = smaller(best, started.elapsed().as_secs_f64());
            }
            println!(
                "forward r2c {shape:?} on {:9}: {:8.3} ms",
                backend.name(),
                best * 1e3
            );
        }
    }

    let shape = [96usize, 1304];
    let lags = ShiftWindow::symmetric([30, 30]);
    let a = plane(shape[0], shape[1], 0xDEAD_BEEF_CAFE_0001);
    let b = plane(shape[0], shape[1], 0x0BAD_C0DE_F00D_0002);
    let rounds = 20;
    for &backend in TransformBackend::available() {
        let planning = std::time::Instant::now();
        let plan =
            SquaredDifference::with_backend(shape, shape, lags, Padding::Smooth, backend).unwrap();
        let planning = planning.elapsed();
        let mut mine = plan.clone();
        let mut best = f64::INFINITY;
        for _ in 0..rounds {
            let started = std::time::Instant::now();
            let landscape = mine.landscape(a.view(), b.view()).unwrap();
            std::hint::black_box(&landscape);
            best = smaller(best, started.elapsed().as_secs_f64());
        }

        let threads = 8usize;
        let handles = (0..threads)
            .map(|_| {
                let mut mine = plan.clone();
                let a = a.clone();
                let b = b.clone();
                std::thread::spawn(move || {
                    let mut best = f64::INFINITY;
                    for _ in 0..rounds {
                        let started = std::time::Instant::now();
                        let landscape = mine.landscape(a.view(), b.view()).unwrap();
                        std::hint::black_box(&landscape);
                        best = smaller(best, started.elapsed().as_secs_f64());
                    }
                    best
                })
            })
            .collect::<Vec<_>>();
        let worst = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .fold(0.0f64, larger);
        println!(
            "landscape {:?} on {:9}: {:8.3} ms one thread, {:8.3} landscapes/s on \
             {threads} ({:.1}x), planned in {:?}",
            plan.padded_shape(),
            backend.name(),
            best * 1e3,
            threads as f64 / worst,
            best * threads as f64 / worst,
            planning
        );
    }
}
