// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// `ElementShape::ExtentEllipsoid`, against the rule it exists to reproduce.
//
// Why this shape exists at all
// ----------------------------
// `ElementShape::Ellipsoid` is the ellipsoid inscribed in the element's bounding
// box, `sum (d_a / r_a)^2 <= 1`. That is the defensible rule and it stays this
// crate's default. It is **not** the only rule in use: a widely deployed array
// filtering implementation defines its ball by normalising each axis by
// `max(lower, upper)` — the half-extent rounded *up* — rather than by the
// radius, and by comparing strictly. For an odd extent those differ by a whole
// unit of semi-axis: an element of size 11 admits `d^2 < 36` under that rule and
// `d^2 <= 25` under this crate's, which is 895 voxels against 515.
//
// Neither is more correct. What matters is that a caller reproducing another
// implementation's numbers can *state* that implementation's element rather than
// being told the nearest available one is close enough — a filter that is a
// different set from the one specified is a different filter, and the difference
// shows up as a different answer, not as an error. So the rule is a point in
// this crate's parameter space.
//
// What this file checks, and what it deliberately does not do
// ----------------------------------------------------------
// It writes the rule out **independently**, in the arithmetic the rule is stated
// in — build the box, compute `1 - sum((mesh + add)/nrm)^2` at every position,
// keep the positions where that is positive — and compares the resulting set,
// voxel for voxel, against what `StructuringElement` generates, over a sweep of
// sizes that includes odd, even, mixed and degenerate extents.
//
// It does **not** depend on the implementation it was derived from. That
// implementation is GPL and this crate is MIT, so a dependency on it — even a
// dev-dependency, even for a test — would be a licensing claim this crate is not
// in a position to make. The agreement was established once, outside the crate,
// by running both and diffing the sets over 1521 sizes; what is kept here is the
// rule, written down, plus the voxel counts that comparison produced, so that a
// regression in either direction fails loudly. `by_rule` below is the whole of
// the imported knowledge, and it is nine lines.

use std::collections::BTreeSet;

use blockflow::ops::{ElementShape, StructuringElement};

/// The rule, transcribed. One statement, in the form the rule is stated in.
///
/// Kept deliberately close to that form rather than simplified: `lower` and
/// `upper` are the two halves of the extent, `add` is the half-voxel correction
/// that only an even extent needs, `nrm` is the larger half, and the membership
/// test is `1 - r^2 > 0` rather than `r^2 < 1` because that is the expression
/// being reproduced. (They agree exactly — `1 - x` is exact for `x` in
/// `[0.5, 2]` by Sterbenz, and for `x < 0.5` the difference is clearly positive
/// — but the point of a transcription is not to improve on what it transcribes.)
fn by_rule(size: [usize; 3]) -> BTreeSet<[isize; 3]> {
    let lower = [size[0] / 2, size[1] / 2, size[2] / 2];
    let upper = [size[0] - lower[0], size[1] - lower[1], size[2] - lower[2]];
    let add = [
        ((size[0] + 1) % 2) as f64 / 2.0,
        ((size[1] + 1) % 2) as f64 / 2.0,
        ((size[2] + 1) % 2) as f64 / 2.0,
    ];
    let nrm = [
        lower[0].max(upper[0]) as f64,
        lower[1].max(upper[1]) as f64,
        lower[2].max(upper[2]) as f64,
    ];

    let mut members = BTreeSet::new();
    for i in 0..size[0] {
        for j in 0..size[1] {
            for k in 0..size[2] {
                let index = [i, j, k];
                let mut squared = 0.0_f64;
                for axis in 0..3 {
                    let mesh = index[axis] as isize - lower[axis] as isize;
                    let normalised = (mesh as f64 + add[axis]) / nrm[axis];
                    squared += normalised * normalised;
                }
                if (1.0_f64 - squared).max(0.0) > 0.0 {
                    // The offsets are measured from the anchor, and the anchor
                    // is `lower` — the same index the rule meshes from, which is
                    // why `StructuringElement::from_size` anchors there too.
                    members.insert([
                        i as isize - lower[0] as isize,
                        j as isize - lower[1] as isize,
                        k as isize - lower[2] as isize,
                    ]);
                }
            }
        }
    }
    members
}

fn generated(shape: ElementShape, size: [usize; 3]) -> BTreeSet<[isize; 3]> {
    StructuringElement::from_size(shape, size)
        .unwrap()
        .offsets()
        .iter()
        .copied()
        .collect()
}

/// **The bar.** Set equality, voxel for voxel, over every combination of a range
/// of extents that mixes odd, even and flat on every axis.
#[test]
fn the_extent_ellipsoid_is_the_rule_it_transcribes_at_every_size() {
    let extents = [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
    let mut checked = 0;
    for &a in &extents {
        for &b in &extents {
            for &c in &[1usize, 2, 3, 4, 5, 10, 11] {
                let size = [a, b, c];
                let want = by_rule(size);
                let got = generated(ElementShape::ExtentEllipsoid, size);
                assert_eq!(
                    got,
                    want,
                    "size {size:?}: {} voxels generated, {} by the rule; only generated {:?}; \
                     only in the rule {:?}",
                    got.len(),
                    want.len(),
                    got.difference(&want).take(4).collect::<Vec<_>>(),
                    want.difference(&got).take(4).collect::<Vec<_>>(),
                );
                assert!(!got.is_empty(), "size {size:?} produced an empty element");
                checked += 1;
            }
        }
    }
    assert!(checked >= 1000, "the sweep ran only {checked} sizes");
}

/// The counts the cross-implementation comparison produced, frozen.
///
/// The sweep above proves the generated set is the transcribed rule; these say
/// what that rule *is*, in numbers a reader can check against the other
/// implementation by hand. They are the reason a change to either the rule or
/// the anchor cannot pass this file quietly by changing both consistently.
#[test]
fn the_frozen_voxel_counts_are_the_ones_the_other_implementation_produces() {
    for (size, count, inscribed) in [
        ([11usize, 11, 11], 895usize, 515usize),
        ([10, 10, 10], 552, 383),
        ([10, 10, 1], 80, 64),
        ([5, 5, 5], 93, 33),
        ([4, 6, 3], 44, 14),
        ([3, 3, 3], 27, 7),
        ([1, 1, 1], 1, 1),
    ] {
        let element = StructuringElement::from_size(ElementShape::ExtentEllipsoid, size).unwrap();
        assert_eq!(element.len(), count, "extent ellipsoid of {size:?}");
        let element = StructuringElement::from_size(ElementShape::Ellipsoid, size).unwrap();
        assert_eq!(element.len(), inscribed, "inscribed ellipsoid of {size:?}");
    }
}

/// The other two forms of the same reference, and the one that was already
/// shared.
///
/// A full box is a full box under any convention, so the agreement here is not
/// news — what is news is that it now holds for an **even** size, which this
/// crate could not express at all before. The `sphere` form of that reference is
/// the same boolean set as its `disk` form (one thresholds at `> 0`, the other
/// at `!= 0`, of a quantity that is `max(1 - r^2, 0)` and therefore never
/// negative), which is why one shape here covers both.
#[test]
fn a_box_of_an_even_size_is_the_whole_box() {
    for size in [[4usize, 4, 4], [10, 6, 1], [2, 3, 8], [1, 1, 2]] {
        let element = StructuringElement::from_size(ElementShape::Box, size).unwrap();
        assert_eq!(element.len(), size[0] * size[1] * size[2]);
        assert_eq!(element.size(), size);
        let lower = [size[0] / 2, size[1] / 2, size[2] / 2];
        for axis in 0..3 {
            assert_eq!(
                element.sides(axis),
                (lower[axis], size[axis] - 1 - lower[axis]),
                "size {size:?} axis {axis}"
            );
        }
    }
}

/// The two rules are ordered, which is the useful thing to know when choosing
/// between them: the extent ellipsoid contains the inscribed one, never the
/// other way round, and both stay inside the bounding box.
#[test]
fn the_extent_ellipsoid_contains_the_inscribed_one_and_both_stay_in_the_box() {
    for size in [
        [11usize, 11, 11],
        [10, 6, 4],
        [7, 7, 1],
        [2, 2, 2],
        [5, 4, 3],
    ] {
        let wide = generated(ElementShape::ExtentEllipsoid, size);
        let inscribed = generated(ElementShape::Ellipsoid, size);
        let boxed = generated(ElementShape::Box, size);
        assert!(
            inscribed.is_subset(&wide),
            "size {size:?}: the inscribed ellipsoid escapes the extent one"
        );
        assert!(wide.is_subset(&boxed), "size {size:?}: escapes its own box");
    }
}
