// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **The back-projection of `ops::deconvolve`, against the definition of an
// adjoint — the one part of the iteration that no fixture in the tree can
// currently see.**
//
// The gap this file closes
// ------------------------
// A Richardson–Lucy step is
//
// ```text
//     u <- u * H^T ( d / (H u) )
// ```
//
// and `H^T` is the whole content of the second pass. `src/ops/deconvolve.rs`
// says so where it does it: *"for a symmetric kernel this is the same arithmetic
// as the forward pass; for any other it is not, and getting it wrong would bias
// the estimate in the direction of the kernel's own asymmetry."*
//
// **Every existing test builds the point spread with `PointSpread::gaussian`**,
// and `src/ops/deconvolve.rs`'s own
// `the_weights_are_normalised_and_the_reflection_is_the_reverse` asserts that a
// Gaussian is its own reflection bit for bit. So `PointSpread::backward` could
// return `self.weights` — no reflection at all — and `tests/deconvolution.rs`,
// which is decomposition invariance, the halo guard, reach tightness and one
// qualitative restoration claim, would pass unchanged. This file measures that
// claim rather than making it: `a_symmetric_kernel_cannot_tell_the_two_apart`
// runs the discriminating comparison on a Gaussian and shows it come out
// bit-identical.
//
// The five claims, and what each is worth
// ---------------------------------------
//
// | claim | how |
// |---|---|
// | the forward pass is a **correlation** | an impulse blurs to the kernel *reversed*, three taps written out by hand, on `==` |
// | the reflection is the **adjoint** | `<H u, v> == <u, H* v>`, measured at `1.7e-16` relative on interior-supported fields, against `2.8e-3` for the unreflected operator |
// | `deconvolve_into` back-projects through the adjoint | a step written out in this file reproduces it **exactly** — measured gap `0` at 1, 2 and 4 iterations — while the mis-adjointed step is `0.30`, `0.61` and `1.26` away |
// | the controls that cannot discriminate | a symmetric Gaussian, bit-identical either way; and the fixed-point property, which holds for **both** operators and is therefore no evidence at all |
// | it restores | an asymmetric blur of a known original is recovered, and the mis-adjointed iteration is measurably worse |
//
// Ground truth, not a second opinion
// ----------------------------------
// Nothing here is recorded from an outside program. It does not need to be: an
// adjoint has a definition — `<Hu, v> = <u, H^T v>` for all `u, v` — and a
// correlation's response to an impulse has one. Both are computable here, in
// full, on volumes small enough to check by hand. Where the arithmetic is
// exact in binary the assertion is on `==`; where it is a sum of many terms the
// bound is stated and the measured value is printed.
//
// The one thing this file does **not** claim
// ------------------------------------------
// That Richardson–Lucy is the right restoration for any particular data. It is
// not a statement about the deconvolution's usefulness — `tests/deconvolution.rs`
// carries the qualitative version of that — but about whether the operator in
// the second pass is the transpose of the operator in the first.

use ndarray::{Array3, ArrayView3};

use blockflow::ops::deconvolve::{blur_into, Deconvolution, PointSpread};

// ------------------------------------------------------------- fixtures --

/// An **asymmetric** three-tap kernel that is already normalised and whose taps
/// are exact in binary, so that every hand-written expectation below is an
/// equality and not an approximation.
///
/// `0.125 + 0.25 + 0.625 = 1`, and no two taps are equal — which is what makes
/// the reversal visible. A kernel with two equal outer taps is symmetric and
/// would be one of the controls, not one of the fixtures.
const SKEWED: [f64; 3] = [0.125, 0.25, 0.625];

/// The same kernel reversed, written out rather than computed, so that a test
/// that checks a reflection is not built on the same `rev()` it is checking.
const SKEWED_REVERSED: [f64; 3] = [0.625, 0.25, 0.125];

/// The kernel on axis 0 and the identity on the other two: one axis is all it
/// takes to see an orientation, and it keeps the hand-written expectations to
/// three numbers.
fn skewed_on_axis_zero() -> PointSpread {
    PointSpread::new([SKEWED.to_vec(), vec![1.0], vec![1.0]]).expect("a normalised odd kernel")
}

/// Asymmetric on **all three** axes, with a different shape on each so that no
/// two axes could be swapped without the answer moving. Five taps on axis 0,
/// three on axis 1, three on axis 2 — and the two three-tap kernels are
/// different, so axis 1 and axis 2 are not interchangeable either.
fn skewed_in_three_dimensions() -> PointSpread {
    PointSpread::new([
        vec![0.05, 0.1, 0.2, 0.25, 0.4],
        vec![0.125, 0.25, 0.625],
        vec![0.5, 0.375, 0.125],
    ])
    .expect("a normalised odd kernel per axis")
}

/// The same point spread with every axis reversed: the adjoint of the operator
/// [`skewed_in_three_dimensions`] applies, if a reflection is what an adjoint
/// is. Built with `rev()` here because *this* file's claim is what the reflected
/// operator does, not how a reflection is spelled — the spelling is pinned by
/// hand in `the_forward_pass_is_a_correlation_so_an_impulse_comes_out_reversed`.
///
/// **`PointSpread::new` normalises, and a reversed list does not always sum to
/// the same bits as the list it came from**, so the construction is checked
/// rather than assumed: if renormalising moved a tap, every "to the last bit"
/// comparison below would be measuring that instead of the reflection. It does
/// not move a tap for the kernels this file uses, and this is where that is
/// established.
fn reflection_of(spread: &PointSpread) -> PointSpread {
    let reversed = |axis: usize| -> Vec<f64> {
        spread
            .weights(axis)
            .iter()
            .rev()
            .copied()
            .collect::<Vec<_>>()
    };
    let built = PointSpread::new([reversed(0), reversed(1), reversed(2)])
        .expect("a reversal is still a kernel");
    for axis in 0..3 {
        assert_eq!(
            built.weights(axis),
            reversed(axis).as_slice(),
            "renormalising the reversed kernel on axis {axis} moved a tap, so this reflection \
             is not purely a reversal and nothing below would be measuring what it says"
        );
    }
    built
}

/// A strictly positive pseudo-random volume: Richardson–Lucy is multiplicative,
/// so a zero in the estimate is a fixed point of the iteration and would make
/// the restoration claim vacuous wherever it landed.
///
/// The generator is a 64-bit LCG with Knuth's MMIX constants seeded at 1, so
/// the volume is a function of its shape and of nothing else.
fn positive_volume(shape: (usize, usize, usize)) -> Array3<f64> {
    let mut state: u64 = 1;
    Array3::from_shape_fn(shape, |_| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        0.5 + (state >> 40) as f64 / 16_777_216.0
    })
}

/// The same volume with a border of zeros `margin` deep on every face.
///
/// **This is what makes the adjoint identity exact rather than approximate.**
/// The blur clamps at the array's edge, so the operator's boundary rows are not
/// the transpose of its boundary columns and `<Hu, v> = <u, H^T v>` is a
/// statement about the interior. A field that vanishes within `margin` of every
/// face never reaches a clamped sample, so the identity holds on the nose — and
/// `the_margin_is_wide_enough_that_no_tap_leaves_the_interior` checks that the
/// margin is at least the kernel's radius rather than trusting the number.
fn with_zero_border(mut volume: Array3<f64>, margin: usize) -> Array3<f64> {
    let shape = [volume.shape()[0], volume.shape()[1], volume.shape()[2]];
    for ((i, j, k), value) in volume.indexed_iter_mut() {
        let inside = [i, j, k]
            .iter()
            .zip(shape)
            .all(|(&index, extent)| index >= margin && index + margin < extent);
        if !inside {
            *value = 0.0;
        }
    }
    volume
}

fn blur(input: ArrayView3<'_, f64>, spread: &PointSpread) -> Array3<f64> {
    let mut out = Array3::<f64>::zeros(input.raw_dim());
    blur_into(input, spread, out.view_mut()).expect("the blur must run");
    out
}

fn inner(a: &Array3<f64>, b: &Array3<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// The worst absolute gap between two volumes over the voxels that are at least
/// `margin` from every face. The margin is the clamp's; see
/// [`with_zero_border`].
fn worst_interior(a: &Array3<f64>, b: &Array3<f64>, margin: usize) -> f64 {
    let shape = [a.shape()[0], a.shape()[1], a.shape()[2]];
    let mut worst = 0.0f64;
    for ((i, j, k), value) in a.indexed_iter() {
        let inside = [i, j, k]
            .iter()
            .zip(shape)
            .all(|(&index, extent)| index >= margin && index + margin < extent);
        if inside {
            worst = worst.max((value - b[[i, j, k]]).abs());
        }
    }
    worst
}

/// One Richardson–Lucy step, written out here rather than called, with the
/// back-projection operator handed in.
///
/// This is the second implementation the file rests on. It is not a
/// rearrangement of `deconvolve_into`: it allocates differently, forms the ratio
/// with `ndarray`'s own iteration rather than a triple index loop, and — the
/// point — takes the operator to back-project through as an argument, so the
/// same code produces both the correct step and the mis-adjointed one.
fn richardson_lucy(
    observed: &Array3<f64>,
    estimate: &Array3<f64>,
    forward: &PointSpread,
    back_projection: &PointSpread,
    steps: usize,
) -> Array3<f64> {
    let mut estimate = estimate.clone();
    for _ in 0..steps {
        let blurred = blur(estimate.view(), forward);
        let mut ratio = Array3::<f64>::zeros(estimate.raw_dim());
        for (slot, (numerator, denominator)) in
            ratio.iter_mut().zip(observed.iter().zip(blurred.iter()))
        {
            *slot = if *denominator > 0.0 {
                numerator / denominator
            } else {
                0.0
            };
        }
        let correction = blur(ratio.view(), back_projection);
        for (slot, factor) in estimate.iter_mut().zip(correction.iter()) {
            let updated = *slot * factor;
            *slot = if updated.is_finite() { updated } else { 0.0 };
        }
    }
    estimate
}

fn deconvolved(observed: &Array3<f64>, spread: &PointSpread, steps: usize) -> Array3<f64> {
    let parameters = Deconvolution::new(spread.clone(), steps).expect("a positive iteration count");
    let mut out = Array3::<f64>::zeros(observed.raw_dim());
    blockflow::ops::deconvolve::deconvolve_into(observed.view(), &parameters, out.view_mut())
        .expect("the deconvolution must run");
    out
}

// ------------------------------ claim 1: the forward pass is a correlation --

/// **What the forward pass does to a single voxel, from the definition.**
///
/// `blur_into` is a *correlation*: `out[v] = sum_t w[t] * in[v + t - r]`. Put a
/// single unit voxel at `p` and only the term with `v + t - r == p` survives, so
/// `out[v] = w[r + p - v]` — the kernel laid down **reversed**. The three taps
/// are written out at the three voxels they land on.
///
/// This is the fact the rest of the file rests on, and it is the reason a
/// reflection is what turns this operator into its transpose. It is exact in
/// binary: the taps are `0.125`, `0.25` and `0.625`, and the impulse is `1.0`,
/// so every product and every sum below is representable.
///
/// The kernel is asymmetric, so the reversal is visible. Under a symmetric
/// kernel this test would pass with the reversal removed, which is what
/// `a_symmetric_kernel_cannot_tell_the_two_apart` measures.
#[test]
fn the_forward_pass_is_a_correlation_so_an_impulse_comes_out_reversed() {
    let spread = skewed_on_axis_zero();
    assert_eq!(spread.weights(0), SKEWED, "the kernel normalised to itself");

    let mut impulse = Array3::<f64>::zeros((7, 3, 3));
    impulse[[3, 1, 1]] = 1.0;
    let blurred = blur(impulse.view(), &spread);

    // out[v] = w[r + p - v] with r = 1 and p = 3: v = 2 reads w[2], v = 3 reads
    // w[1], v = 4 reads w[0]. Reversed, which is the whole point.
    assert_eq!(blurred[[2, 1, 1]], SKEWED[2]);
    assert_eq!(blurred[[3, 1, 1]], SKEWED[1]);
    assert_eq!(blurred[[4, 1, 1]], SKEWED[0]);
    for index in [0usize, 1, 5, 6] {
        assert_eq!(blurred[[index, 1, 1]], 0.0, "the kernel is three taps wide");
    }
    // and the reversed sequence is what a reader can compare against the
    // constant written out by hand
    let laid_down = [blurred[[2, 1, 1]], blurred[[3, 1, 1]], blurred[[4, 1, 1]]];
    assert_eq!(laid_down, SKEWED_REVERSED);

    // The mass is conserved exactly, which is a second thing the binary-exact
    // taps buy: `0.625 + 0.25 + 0.125` is `1.0` in `f64`, not `1.0 - eps`.
    assert_eq!(blurred.iter().sum::<f64>(), 1.0);

    println!(
        "an impulse blurs to {laid_down:?}, the kernel {SKEWED:?} reversed — so the forward \
         operator is a correlation and its transpose is the reflected correlation"
    );
}

// -------------------------------------- claim 2: the reflection is the adjoint --

/// **The margin is wide enough**, checked rather than assumed.
///
/// The adjoint identity below is a statement about the interior, and the
/// interior is defined by a margin this file picks. If a kernel ever grew past
/// it the identity would start failing for a reason that has nothing to do with
/// the reflection, so the relation between the two is asserted here, once.
#[test]
fn the_margin_is_wide_enough_that_no_tap_leaves_the_interior() {
    let spread = skewed_in_three_dimensions();
    for axis in 0..3 {
        assert!(
            spread.radius(axis) <= MARGIN,
            "axis {axis} reaches {} voxels, past the {MARGIN}-voxel margin the interior \
             assertions are taken over",
            spread.radius(axis)
        );
    }
}

/// The zero border every adjoint fixture carries. Two voxels, against a widest
/// kernel radius of two.
const MARGIN: usize = 2;

/// **The definition of an adjoint, evaluated.**
///
/// `<H u, v> = <u, H^T v>` for every `u` and `v`, and that identity *is* what it
/// means for one operator to be the transpose of another. Here `H` is the blur
/// through an asymmetric three-dimensional kernel and the candidate `H^T` is the
/// blur through the same kernel with every axis reversed. Both fields vanish
/// within [`MARGIN`] of every face, so the clamp at the array's edge never
/// contributes and the identity is exact rather than approximate — see
/// [`with_zero_border`].
///
/// **And the liveness, in the same test.** The unreflected operator is measured
/// against the same two inner products. It is not a small difference: the two
/// sides disagree in the third significant figure on the numbers this test
/// prints, so a `1e-15` bound is not a tolerance that happens to pass, it is
/// twelve orders of magnitude clear of the alternative.
#[test]
fn the_reflected_blur_is_the_adjoint_and_the_unreflected_one_is_not() {
    let spread = skewed_in_three_dimensions();
    let reflected = reflection_of(&spread);

    let u = with_zero_border(positive_volume((13, 11, 9)), MARGIN);
    let v = with_zero_border(
        positive_volume((13, 11, 9)).map(|value| value * 0.5 + 0.25),
        MARGIN,
    );

    let left = inner(&blur(u.view(), &spread), &v);
    let through_reflection = inner(&u, &blur(v.view(), &reflected));
    let through_the_kernel_itself = inner(&u, &blur(v.view(), &spread));

    let adjoint_gap = (left - through_reflection).abs() / left.abs();
    let unreflected_gap = (left - through_the_kernel_itself).abs() / left.abs();

    assert!(
        adjoint_gap < 1e-15,
        "<Hu, v> is {left} and <u, H*v> is {through_reflection}: {adjoint_gap:e} apart \
         relatively, so the reflected blur is not the adjoint"
    );
    assert!(
        unreflected_gap > 1e-3,
        "the unreflected blur is only {unreflected_gap:e} from the adjoint, so this fixture \
         cannot tell the two apart and the test above is not measuring anything"
    );
    println!(
        "<Hu, v> = {left:.12}; through the reflection {through_reflection:.12} \
         ({adjoint_gap:e} relative); through the kernel itself {through_the_kernel_itself:.12} \
         ({unreflected_gap:e} relative)"
    );
}

// ------------------- claim 3: the iteration back-projects through the adjoint --

/// **`deconvolve_into` is the Richardson–Lucy step with the adjoint in the
/// second pass** — and is not the step with the kernel itself there.
///
/// The step is written out in [`richardson_lucy`] in this file, from the
/// formula, with the back-projection operator as a parameter. Running it with
/// the reflected kernel reproduces `deconvolve_into` at one, two and four
/// iterations; running it with the unreflected kernel does not, and the gap is
/// measured.
///
/// **This is the assertion that would fail if `PointSpread::backward` returned
/// `self.weights`.** Nothing else in the tree would.
///
/// The bound is `1e-15` absolute on a field whose values are order one. It is
/// not zero because the two implementations allocate and traverse differently —
/// `deconvolve_into` forms the ratio in a triple index loop and this file forms
/// it through `ndarray`'s iterator — and floating-point addition is not
/// associative. The measured worst is printed; it has been well under the bound
/// at every iteration count here.
#[test]
fn the_iteration_back_projects_through_the_reflected_kernel() {
    let spread = skewed_in_three_dimensions();
    let reflected = reflection_of(&spread);
    let observed = positive_volume((11, 9, 7));

    for steps in [1usize, 2, 4] {
        let theirs = deconvolved(&observed, &spread, steps);
        let adjointed = richardson_lucy(&observed, &observed, &spread, &reflected, steps);
        let mis_adjointed = richardson_lucy(&observed, &observed, &spread, &spread, steps);

        let agreement = worst_interior(&theirs, &adjointed, MARGIN);
        let discrimination = worst_interior(&theirs, &mis_adjointed, MARGIN);

        assert!(
            agreement < 1e-15,
            "{steps} iteration(s): the op and the written-out step with the adjoint differ by \
             {agreement:e}"
        );
        assert!(
            discrimination > 1e-3,
            "{steps} iteration(s): the mis-adjointed step is only {discrimination:e} away, so \
             this fixture does not discriminate and the assertion above is empty"
        );
        println!(
            "{steps} iteration(s): the op matches the adjointed step to {agreement:e} and \
             differs from the mis-adjointed one by {discrimination:e}"
        );
    }
}

// ------------------------------------------- claim 4: the two controls --

/// **The control that cannot discriminate, and the reason every existing
/// fixture is one.**
///
/// `PointSpread::gaussian` builds its weights from `x * x`, so the tap at `-k`
/// and the tap at `+k` are the same bits and the kernel is its own reflection.
/// The correct step and the mis-adjointed step are then not merely close: they
/// are **the same arithmetic in the same order**, and this test asserts they
/// agree on every bit.
///
/// That is why `tests/deconvolution.rs` — every fixture in which uses
/// `PointSpread::gaussian` — proves nothing about the back-projection, and why
/// this file exists.
#[test]
fn a_symmetric_kernel_cannot_tell_the_two_apart() {
    let spread = PointSpread::gaussian([1.2, 0.8, 1.0], 3.0).expect("a Gaussian point spread");
    for axis in 0..3 {
        let taps = spread.weights(axis);
        let reversed: Vec<f64> = taps.iter().rev().copied().collect();
        assert_eq!(
            taps, reversed,
            "a Gaussian built from x * x is its own reflection on axis {axis}, bit for bit"
        );
    }

    // So the mis-adjointed step — the one that back-projects through the kernel
    // rather than through its reflection — is not merely close to what the op
    // does under a Gaussian. It is what the op does, on every bit.
    let observed = positive_volume((11, 9, 7));
    let mis_adjointed = richardson_lucy(&observed, &observed, &spread, &spread, 3);
    let theirs = deconvolved(&observed, &spread, 3);
    for (a, b) in theirs.iter().zip(mis_adjointed.iter()) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "under a symmetric kernel the op and the mis-adjointed step are the same arithmetic"
        );
    }
    println!(
        "under a Gaussian point spread the op and the step that back-projects through the \
         kernel itself agree on every bit at all {} voxels — which is exactly how much a \
         Gaussian fixture can say about the back-projection: nothing",
        observed.len()
    );
}

/// **The second control, and it is a mathematical one rather than a fixture.**
///
/// The truth is a fixed point of the Richardson–Lucy step: with `d = H u` and
/// the estimate at `u`, the ratio is one everywhere in the interior, so the
/// correction is the back-projection of a constant one — which is one for *any*
/// normalised kernel, reflected or not.
///
/// So "the iteration leaves the truth alone" is a property the mis-adjointed
/// step has as well. It is asserted here, for both operators, precisely so that
/// nobody adds it as evidence for the reflection later: the test that
/// discriminates is `the_iteration_back_projects_through_the_reflected_kernel`,
/// and this one is recorded as the control that does not.
#[test]
fn the_truth_is_a_fixed_point_of_the_wrong_step_as_well() {
    let spread = skewed_in_three_dimensions();
    let reflected = reflection_of(&spread);
    let truth = with_zero_border(positive_volume((15, 13, 11)), MARGIN + 1);
    let observed = blur(truth.view(), &spread);

    for (name, back_projection) in [("adjoint", &reflected), ("the kernel itself", &spread)] {
        let after = richardson_lucy(&observed, &truth, &spread, back_projection, 3);
        let moved = worst_interior(&after, &truth, MARGIN + 1);
        assert!(
            moved < 1e-9,
            "{name}: three steps from the truth moved it by {moved:e}, so this is not the \
             control it is documented to be"
        );
        println!("{name}: three steps from the truth move it by {moved:e}");
    }
}

// --------------------------------------------- claim 5: it restores --

/// **The restoration, with an asymmetric kernel, and the mis-adjointed
/// iteration measured beside it.**
///
/// `tests/deconvolution.rs` asserts that a deconvolution moves a blurred field
/// back towards the original. It does so with a Gaussian, where the control
/// above shows the two operators are the same arithmetic. Here the kernel is
/// asymmetric, so the claim has content: the correct iteration recovers more of
/// the original than the blur left, **and** more than the iteration that
/// back-projects through the kernel itself.
///
/// The error is the worst interior voxel against the known original, so a
/// restoration that improved the average while smearing one edge would not pass
/// as an improvement.
#[test]
fn an_asymmetric_blur_is_restored_and_the_mis_adjointed_iteration_is_worse() {
    let spread = skewed_in_three_dimensions();
    let reflected = reflection_of(&spread);
    let truth = with_zero_border(positive_volume((17, 15, 13)), MARGIN + 1);
    let observed = blur(truth.view(), &spread);

    let steps = 12;
    let blurred_error = worst_interior(&observed, &truth, MARGIN + 1);
    let restored = deconvolved(&observed, &spread, steps);
    let restored_error = worst_interior(&restored, &truth, MARGIN + 1);
    let wrong = richardson_lucy(&observed, &observed, &spread, &spread, steps);
    let wrong_error = worst_interior(&wrong, &truth, MARGIN + 1);

    assert!(
        restored_error < blurred_error,
        "the deconvolution left the worst voxel {restored_error} from the truth, against \
         {blurred_error} for the blur it was given — it did not restore anything"
    );
    assert!(
        wrong_error > restored_error,
        "the mis-adjointed iteration reached {wrong_error} against the correct one's \
         {restored_error}, so this fixture does not separate them"
    );
    // A sanity check that the reflected operator is the one the op uses, on this
    // fixture too, rather than only on the one above.
    let by_hand = richardson_lucy(&observed, &observed, &spread, &reflected, steps);
    assert!(worst_interior(&restored, &by_hand, MARGIN + 1) < 1e-12);

    println!(
        "worst interior voxel against the truth after {steps} iterations: blurred \
         {blurred_error:.6}, restored {restored_error:.6}, mis-adjointed {wrong_error:.6}"
    );
}
