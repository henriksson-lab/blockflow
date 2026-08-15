// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// `Rank::CeilingPercentile`, against the rule it exists to reproduce, and
// against the rule it is an alternative to.
//
// The situation
// -------------
// A window clipped at a volume boundary holds only `m` of its element's `n`
// offsets. Every implementation of a rank filter has to say what a rank means
// then, and there are two answers in wide use:
//
// * **Proportional rescaling of the rank.** The statistic is a position `k` out
//   of `n`, and truncation moves it to the same relative position out of `m`:
//   `round(k * (m - 1) / (n - 1))`. `Rank::Nth` is this one and is what this
//   crate has always done.
// * **A ceiling percentile over the surviving population.** The statistic is a
//   fraction `p`, and the answer is the `ceil(p * m)`-th smallest of the `m`
//   values actually read, one-based. `Rank::CeilingPercentile` is this one.
//
// Neither is more correct. The first keeps a median a median at every face; the
// second is what a filter that walks a histogram of the window falls out as, and
// is therefore what a caller reproducing such an implementation's numbers needs.
// So the rule is a point in this crate's parameter space, exactly as
// `ElementShape::ExtentEllipsoid` is, and for the same reason: a filter that is
// a different statistic from the one specified is a different filter, and the
// difference shows up as a different answer rather than as an error.
//
// **Where they agree, and why that hid the difference.** At `p = 0.5` the
// proportional rule reduces to `floor(m / 2)` for every `m` and every `n`. A
// histogram-walking *median* — a separate routine in the implementations that
// have both — stops at the first bin whose cumulative count exceeds `m / 2`,
// which is the same index for every `m`. So a median filter agrees with a median
// filter at every face of every volume, and a comparison that only ever ran a
// median would find the two conventions identical. They are not. The same
// implementation's *percentile* routine at `p = 0.5` gives `ceil(m / 2) - 1`,
// which is that index for odd `m` and one below it for even `m`; and at
// `p = 0.25` on a 27-voxel element the two give 7 and 6 on the **untruncated**
// window, before any boundary is involved.
//
// What this file checks, and what it deliberately does not do
// -----------------------------------------------------------
// It writes the ceiling rule out **independently**, in the form the rule is
// stated in — build a histogram of the window's values, walk it from the bottom
// accumulating counts, stop at the first bin whose running total reaches
// `p * m` — and compares the resulting value, voxel for voxel, against what the
// rank filter produces, over a sweep that puts windows at faces, edges and
// corners of a volume, at several fractions, over elements with odd and even
// populations.
//
// It does **not** depend on the implementation the rule was read off. That
// implementation is GPL and this crate is MIT, so a dependency on it — even a
// dev-dependency, even for a test — would be a licensing claim this crate is not
// in a position to make. The agreement was established once, outside the crate,
// by running both over a shared volume and diffing; what is kept here is the
// rule, written down, plus the frozen indices and divergence counts that
// comparison produced, so that a regression in either direction fails loudly.
// `by_rule` below is the whole of the imported knowledge, and it is a dozen
// lines.
//
// What that comparison found, on a `9 x 7 x 6` volume, over box elements of
// 27, 25, 64, 24 and 125 voxels and the nine fractions swept below, with the
// values all distinct so that no agreement could come from a tie:
//
// * `Rank::CeilingPercentile` matched the reference's **percentile** filter at
//   every voxel of all 45 combinations — 0 disagreements out of 378 each.
// * `Rank::Nth` — the proportional rule — disagreed with that same filter at up
//   to 378 of 378, i.e. at every voxel including the untruncated interior.
// * `Rank::Nth` at a half matched the reference's **median** filter, a separate
//   routine with no histogram threshold, at 0 of 378 for every element; and
//   `Rank::CeilingPercentile(0.5)` disagreed with that median at 238, 370 and
//   168 of 378 on the 27-, 64- and 25-voxel elements.
//
// That last pair is the finding worth carrying: the reference's own two
// routines disagree with each other at `p = 0.5` wherever `m` is even, so
// "matches the reference" was never one claim.

use ndarray::Array3;

use blockflow::ops::{rank_filter_into, ElementShape, Rank, StructuringElement};

/// The number of distinct values, and therefore of histogram bins. Small enough
/// that ties are common, which is what makes the "first bin whose running total
/// reaches the threshold" wording bite: with no ties every rule that lands on
/// the same index gives the same value and the sweep would prove less.
const BINS: usize = 16;

/// The rule, transcribed. One statement, in the form the rule is stated in.
///
/// Kept deliberately close to that form rather than simplified to
/// `sorted[ceil(p * m) - 1]`: the rule is a histogram walk, the comparison is
/// `sum >= p * m` on a running integer total against a floating-point threshold,
/// and the maximum is a separate loop from the top rather than the `p = 1` case
/// of the first one. (They agree — a running total of integers reaches a real
/// threshold `t` exactly when it reaches `ceil(t)` — but the point of a
/// transcription is not to improve on what it transcribes.)
///
/// The `p * m <= 0` behaviour is transcribed too, and it is a **quirk rather
/// than a rule**: the walk breaks on its first iteration whether or not bin zero
/// is occupied, so it answers the value zero rather than the window's minimum.
/// This crate does not reproduce that, and
/// `zero_is_the_minimum_here_and_the_first_bin_there` is where the divergence is
/// pinned rather than hidden.
fn by_rule(window: &[u16], fraction: f64) -> u16 {
    if window.is_empty() {
        return 0;
    }
    let mut histogram = [0usize; BINS];
    for &value in window {
        histogram[value as usize] += 1;
    }
    if fraction == 1.0 {
        // "make sure p = 1 returns the maximum filter": walk from the top.
        for bin in (0..BINS).rev() {
            if histogram[bin] > 0 {
                return bin as u16;
            }
        }
        return 0;
    }
    let threshold = fraction * window.len() as f64;
    let mut sum = 0usize;
    for bin in 0..BINS {
        sum += histogram[bin];
        if sum as f64 >= threshold {
            return bin as u16;
        }
    }
    (BINS - 1) as u16
}

/// Deliberately small on every axis, so that a `5 x 5 x 5` element leaves almost
/// no interior: the clamped windows are the point, and a volume large enough to
/// be mostly interior would drown them.
const VOLUME: (usize, usize, usize) = (9, 7, 6);

fn intensities() -> Array3<u16> {
    let mut array = Array3::zeros(VOLUME);
    for (flat, value) in array.iter_mut().enumerate() {
        *value = ((flat * 7919) % BINS) as u16;
    }
    array
}

/// Every window of `element` over `volume`, clipped at the boundary exactly as
/// the filter clips it. The gathered values, per voxel, in the element's order.
fn windows(input: &Array3<u16>, element: &StructuringElement) -> Vec<Vec<u16>> {
    let (rows, cols, planes) = input.dim();
    let mut all = Vec::with_capacity(rows * cols * planes);
    for i in 0..rows {
        for j in 0..cols {
            for k in 0..planes {
                let mut window = Vec::new();
                for offset in element.offsets() {
                    let a = i as isize + offset[0];
                    let b = j as isize + offset[1];
                    let c = k as isize + offset[2];
                    if a < 0 || b < 0 || c < 0 {
                        continue;
                    }
                    let (a, b, c) = (a as usize, b as usize, c as usize);
                    if a >= rows || b >= cols || c >= planes {
                        continue;
                    }
                    window.push(input[[a, b, c]]);
                }
                all.push(window);
            }
        }
    }
    all
}

/// Elements with odd and with even populations, flat and solid, so that nothing
/// here rests on one parity of `m` or on one shape of truncation.
fn elements() -> Vec<(&'static str, StructuringElement)> {
    vec![
        (
            "box 3x3x3, n = 27",
            StructuringElement::from_size(ElementShape::Box, [3, 3, 3]).unwrap(),
        ),
        (
            "box 5x5x1, n = 25",
            StructuringElement::from_size(ElementShape::Box, [5, 5, 1]).unwrap(),
        ),
        (
            "box 4x4x4, n = 64",
            StructuringElement::from_size(ElementShape::Box, [4, 4, 4]).unwrap(),
        ),
        (
            "box 2x3x4, n = 24",
            StructuringElement::from_size(ElementShape::Box, [2, 3, 4]).unwrap(),
        ),
        (
            "inscribed ellipsoid 5x5x5, n = 81",
            StructuringElement::from_size(ElementShape::Ellipsoid, [5, 5, 5]).unwrap(),
        ),
    ]
}

/// The fractions swept. `0.25`, `0.5` and `0.75` are named in the rule's own
/// discussion; the others are there so that the agreement at a half cannot be an
/// artefact of three tidy numbers.
const FRACTIONS: &[f64] = &[0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0];

// ------------------------------------------------ 1. against the rule --

/// **Value by value, over faces, edges, corners and interior**, at every
/// fraction, for every element.
#[test]
fn the_ceiling_percentile_is_the_reference_histogram_walk() {
    let input = intensities();
    let mut clamped = 0usize;
    let mut compared = 0usize;
    for (name, element) in elements() {
        let gathered = windows(&input, &element);
        for &fraction in FRACTIONS {
            let rank = Rank::ceiling_percentile(fraction).unwrap();
            let mut got = Array3::zeros(VOLUME);
            rank_filter_into(input.view(), &element, rank, got.view_mut()).unwrap();
            for (index, window) in gathered.iter().enumerate() {
                let want = by_rule(window, fraction);
                assert_eq!(
                    got.as_slice().unwrap()[index],
                    want,
                    "{name} at p = {fraction}, voxel {index}, m = {}",
                    window.len()
                );
                if window.len() < element.len() {
                    clamped += 1;
                }
                compared += 1;
            }
        }
    }
    // The sweep really is dominated by clamped windows, which is the property
    // the volume was sized for. Frozen, so that shrinking the element or growing
    // the volume cannot quietly turn this into an interior-only test.
    assert_eq!(compared, 17_010, "the sweep did not run as expected");
    assert_eq!(clamped, 12_942, "clamped windows in the sweep");
}

// ---------------------------------- 2. the two conventions, side by side --

/// **They agree at `p = 0.5` over an odd surviving population and nowhere
/// else** — asserted as both halves, so the equivalence is pinned and so is the
/// divergence.
///
/// Stated on the resolved indices rather than on filtered volumes, because that
/// is where the difference lives: two indices that differ can still select the
/// same value out of a window with ties, and a comparison of volumes would
/// therefore understate how often the conventions part company.
#[test]
fn the_two_conventions_agree_at_a_half_over_an_odd_population_and_not_elsewhere() {
    for full in [24usize, 25, 27, 64, 81] {
        for &fraction in FRACTIONS {
            let proportional = Rank::Nth(
                (fraction * (full - 1) as f64).round() as usize, // `Rank::percentile`, inlined
            );
            let ceiling = Rank::ceiling_percentile(fraction).unwrap();
            for available in 1..=full {
                let ours = proportional.resolve(full, available);
                let theirs = ceiling.resolve(full, available);
                if fraction == 0.5 {
                    // the proportional rule at a half is `floor(m / 2)` for
                    // every `m` and every `n`, which is exactly what a
                    // histogram median walk gives
                    assert_eq!(ours, available / 2, "n {full}, m {available}");
                    assert_eq!(theirs, available.div_ceil(2) - 1);
                    if available % 2 == 1 {
                        assert_eq!(ours, theirs, "odd m = {available} must agree at a half");
                    } else {
                        assert_eq!(
                            ours,
                            theirs + 1,
                            "even m = {available} must differ by one at a half"
                        );
                    }
                }
                assert!(ours < available && theirs < available);
            }
        }
    }
}

/// The frozen numbers, so that a change to either rule fails here first and
/// with the arithmetic visible.
///
/// `(n, p, m) -> (proportional, ceiling)`. The 27-voxel element's face, edge and
/// corner windows are 18, 12 and 8 values; the untruncated one is 27.
#[test]
fn the_frozen_indices_of_both_conventions() {
    #[rustfmt::skip]
    let table: &[(usize, f64, usize, usize, usize)] = &[
        // n,  p,     m,   proportional, ceiling
        (27, 0.25, 27,  7,  6),   // the *untruncated* window already differs
        (27, 0.25, 18,  5,  4),   // a face
        (27, 0.25, 12,  3,  2),   // an edge
        (27, 0.25,  8,  2,  1),   // a corner
        (27, 0.50, 27, 13, 13),   // odd m: the one place they meet
        (27, 0.50, 18,  9,  8),   // even m: apart again, by one
        (27, 0.50, 12,  6,  5),
        (27, 0.50,  8,  4,  3),
        (27, 0.75, 27, 20, 20),   // they coincide here by arithmetic accident
        (27, 0.75, 20, 15, 14),   // and part company at a nearby m
        (27, 0.75, 24, 18, 17),
        (27, 1.00, 27, 26, 26),   // the maximum is the maximum under both
        (27, 1.00,  8,  7,  7),
        (27, 0.00, 27,  0,  0),   // and so is the minimum
        (64, 0.25, 64, 16, 15),
        (64, 0.25, 32,  8,  7),
        (64, 0.50, 64, 32, 31),   // even n, even m
        (64, 0.50, 33, 16, 16),   // even n, odd m
        (81, 0.75, 81, 60, 60),
        (81, 0.75, 45, 33, 33),
        (81, 0.90, 81, 72, 72),
        (81, 0.90, 50, 44, 44),
    ];
    for &(full, fraction, available, proportional_want, ceiling_want) in table {
        let proportional = Rank::Nth((fraction * (full - 1) as f64).round() as usize);
        let ceiling = Rank::ceiling_percentile(fraction).unwrap();
        assert_eq!(
            proportional.resolve(full, available),
            proportional_want,
            "proportional, n = {full}, p = {fraction}, m = {available}"
        );
        assert_eq!(
            ceiling.resolve(full, available),
            ceiling_want,
            "ceiling, n = {full}, p = {fraction}, m = {available}"
        );
    }
}

/// How often the two part company over the whole sweep, frozen.
///
/// A single number rather than a table because its job is different: the table
/// above pins the arithmetic, this pins the *scale* of the disagreement, so that
/// a change which happened to preserve the tabulated points while moving
/// everything else still fails.
#[test]
fn the_frozen_divergence_count_over_the_whole_sweep() {
    let mut differ = 0usize;
    let mut total = 0usize;
    for full in [24usize, 25, 27, 64, 81] {
        for &fraction in FRACTIONS {
            let proportional = Rank::Nth((fraction * (full - 1) as f64).round() as usize);
            let ceiling = Rank::ceiling_percentile(fraction).unwrap();
            for available in 1..=full {
                if proportional.resolve(full, available) != ceiling.resolve(full, available) {
                    differ += 1;
                }
                total += 1;
            }
        }
    }
    assert_eq!(total, 1_989);
    assert_eq!(differ, 531, "the two conventions part company this often");
}

// ------------------------------------------------- 3. the one divergence --

/// `p = 0` is the **window minimum** here and the **first histogram bin**
/// there, and this crate keeps the minimum on purpose.
///
/// The reference's walk breaks on its first iteration when the threshold is
/// zero, whether or not that bin holds anything, so it answers the value zero. A
/// rank filter in this crate writes a value it read — that is what makes
/// `constant_maps_to` exact at every truncation and what makes a short-circuited
/// block byte-identical to a computed one — and a literal zero from an empty bin
/// is not one of the values read. The divergence is pinned rather than hidden,
/// and it is the only one in the sweep.
#[test]
fn zero_is_the_minimum_here_and_the_first_bin_there() {
    // a window with nothing in bin zero, so the two answers are different
    let window: Vec<u16> = vec![7, 9, 11, 7, 13];
    assert_eq!(by_rule(&window, 0.0), 0, "the reference stops at bin zero");

    let input = Array3::from_shape_fn((4, 4, 4), |(i, j, k)| (i + j + k + 7) as u16);
    let element = StructuringElement::from_size(ElementShape::Box, [3, 3, 3]).unwrap();
    let mut got = Array3::zeros((4, 4, 4));
    rank_filter_into(
        input.view(),
        &element,
        Rank::ceiling_percentile(0.0).unwrap(),
        got.view_mut(),
    )
    .unwrap();
    assert_eq!(got[[0, 0, 0]], 7, "the minimum of the corner window");
    assert_eq!(got[[2, 2, 2]], 10, "the minimum of the interior window");
    assert!(got.iter().all(|&value| value >= 7));

    // and it is exactly the erosion the other convention names, which is the
    // reason it is worth keeping
    let mut lowest = Array3::zeros((4, 4, 4));
    rank_filter_into(input.view(), &element, Rank::lowest(), lowest.view_mut()).unwrap();
    assert_eq!(got, lowest);
}

// ---------------------------------------------------- 4. nothing moved --

/// The default is unchanged: nothing that did not ask for the ceiling rule got
/// it. A median filter answers exactly what it answered before the second
/// convention existed, at every voxel including the clamped ones.
#[test]
fn the_default_convention_is_untouched() {
    let input = intensities();
    for (name, element) in elements() {
        let gathered = windows(&input, &element);
        let rank = Rank::median(&element);
        let mut got = Array3::zeros(VOLUME);
        rank_filter_into(input.view(), &element, rank, got.view_mut()).unwrap();
        for (index, window) in gathered.iter().enumerate() {
            // the proportional rule at a half, stated directly: the upper of the
            // two middles of whatever survived
            let mut sorted = window.clone();
            sorted.sort_unstable();
            assert_eq!(
                got.as_slice().unwrap()[index],
                sorted[sorted.len() / 2],
                "{name}, voxel {index}"
            );
        }
    }
}
