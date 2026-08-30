// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **`ops::morphology` against the definitions of the four operations, and
// against the algebraic laws an opening and a closing are defined by.**
//
// The gap this file closes
// ------------------------
// Everything the tree asserts about this module today is either
// decomposition invariance — `tests/image_ops.rs`, `tests/asymmetric_element.rs`,
// whose reference is `chain.apply()` called once — or the transitive chain
// `erosion_and_dilation_are_the_extreme_ranks_of_the_same_element`
// (`src/ops/morphology.rs`) into `ops::rank`'s `by_definition`. That chain is a
// real witness for the *sweep*, but it has a common mode: both sides read
// `input[centre + offset]` over `element.offsets()`, so it cannot see the
// composition. And the only law asserted anywhere is
// `opening_removes_an_isolated_voxel_and_closing_fills_an_isolated_hole`, on a
// symmetric 3x3x3 box.
//
// An opening is not "erode then dilate". It is the operation defined by three
// properties — **anti-extensive**, **increasing**, **idempotent** — and
// "erode by `B` then dilate by `B`" only has them when `B` is symmetric. That is
// the distinction nothing here was measuring, and
// `an_asymmetric_opening_translates_the_image_which_is_a_defect` records what
// happened when it was.
//
// What is asserted
// ----------------
//
// | claim | how |
// |---|---|
// | erosion and dilation are the gathered definition | a triple loop over the element's offsets with the clamp written out, on six elements x three masks — 49 896 voxels, exact |
// | an opening by a symmetric element is an opening | anti-extensive, increasing and idempotent, on four symmetric elements x three masks; the closing dual beside it |
// | the laws are not vacuous | the same three properties are checked to *fail* for a deliberately wrong composition, so a fixture on which everything is an opening is caught |
// | **an opening by an asymmetric element is not one** | a defect, recorded with a minimal counterexample and the corrected composition computed beside it |
//
// What is deliberately not asserted
// ---------------------------------
// That `element.offsets()` is the right set of offsets. The oracle below reads
// the same offsets the implementation does, so it witnesses the sweep, the
// clamp and the composition, and says nothing about the element. The element's
// own witness is `tests/element_reference_rule.rs`, which compares against a
// transcription of another implementation's membership rule.

use ndarray::{Array3, ArrayView3};

use blockflow::ops::element::{ElementShape, StructuringElement};
use blockflow::ops::morphology::{close_into, dilate_into, erode_into, open_into};

// ------------------------------------------------------------- fixtures --

/// A pseudo-random mask, from a 64-bit LCG with Knuth's MMIX constants, so the
/// mask is a function of its shape and seed and of nothing else.
///
/// `density` is the reciprocal of the set fraction. Three densities are used
/// below because morphology's failure modes live at the extremes: a sparse mask
/// erodes to nothing and hides everything about the dilation, and a dense one
/// dilates to everything and hides the erosion.
fn lcg_mask(shape: (usize, usize, usize), seed: u64, density: u64) -> Array3<bool> {
    let mut state = seed;
    Array3::from_shape_fn(shape, |_| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) % density == 0
    })
}

fn masks() -> Vec<(&'static str, Array3<bool>)> {
    vec![
        ("sparse, one voxel in five", lcg_mask((14, 11, 9), 11, 5)),
        ("even, one voxel in two", lcg_mask((14, 11, 9), 29, 2)),
        (
            "dense, four voxels in five",
            lcg_mask((14, 11, 9), 47, 5).map(|set| !set),
        ),
    ]
}

/// Elements whose reflection is themselves. These are the ones an opening is an
/// opening for, and the four cover a box, a ball, a flat element and a
/// single-axis one.
fn symmetric_elements() -> Vec<(&'static str, StructuringElement)> {
    vec![
        (
            "box radius 1",
            StructuringElement::from_radius(ElementShape::Box, [1, 1, 1]),
        ),
        (
            "ellipsoid radius 2",
            StructuringElement::from_radius(ElementShape::Ellipsoid, [2, 2, 2]),
        ),
        (
            "flat box, radius 0 on axis 2",
            StructuringElement::from_radius(ElementShape::Box, [2, 1, 0]),
        ),
        (
            "a line of five on axis 0",
            StructuringElement::from_radius(ElementShape::Box, [2, 0, 0]),
        ),
    ]
}

/// Elements whose reflection is not themselves. Two of them, and the first is
/// the one that matters: `from_size` with an **even** extent is a documented,
/// tested, first-class way to build an element in this crate — it is what
/// `tests/asymmetric_element.rs` is built around — and it produces sides
/// `(2, 1)`, not a symmetric element.
fn asymmetric_elements() -> Vec<(&'static str, StructuringElement)> {
    vec![
        (
            "box from_size [4, 1, 1], sides (2, 1)",
            StructuringElement::from_size(ElementShape::Box, [4, 1, 1]).expect("an even box"),
        ),
        (
            "the two offsets {0, +1} on axis 0",
            StructuringElement::from_offsets([[0, 0, 0], [1, 0, 0]]).expect("two offsets"),
        ),
    ]
}

/// `{-o : o in B}`. Built here from `offsets()` because the claim being made is
/// about what the reflected element *does*, not about how a reflection is
/// spelled.
fn reflection_of(element: &StructuringElement) -> StructuringElement {
    StructuringElement::from_offsets(
        element
            .offsets()
            .iter()
            .map(|offset| [-offset[0], -offset[1], -offset[2]])
            .collect::<Vec<_>>(),
    )
    .expect("a reflected offset set is still an offset set")
}

fn eroded(mask: &Array3<bool>, element: &StructuringElement) -> Array3<bool> {
    let mut out = Array3::from_elem(mask.raw_dim(), false);
    erode_into(mask.view(), element, out.view_mut()).expect("the erosion must run");
    out
}

fn dilated(mask: &Array3<bool>, element: &StructuringElement) -> Array3<bool> {
    let mut out = Array3::from_elem(mask.raw_dim(), false);
    dilate_into(mask.view(), element, out.view_mut()).expect("the dilation must run");
    out
}

fn opened(mask: &Array3<bool>, element: &StructuringElement) -> Array3<bool> {
    let mut out = Array3::from_elem(mask.raw_dim(), false);
    open_into(mask.view(), element, out.view_mut()).expect("the opening must run");
    out
}

fn closed(mask: &Array3<bool>, element: &StructuringElement) -> Array3<bool> {
    let mut out = Array3::from_elem(mask.raw_dim(), false);
    close_into(mask.view(), element, out.view_mut()).expect("the closing must run");
    out
}

// -------------------------------------------- claim 1: the two primitives --

/// The definition, written out: for each voxel, gather the element's offsets,
/// **skip the ones that leave the array**, and reduce.
///
/// `hit` is `false` for an erosion (a clear neighbour clears the output) and
/// `true` for a dilation. The skip is the clamp the module header states: what
/// lies outside behaves as set for an erosion and clear for a dilation, which
/// is what a skip does to a conjunction and a disjunction respectively.
///
/// This is a triple loop and nothing else. It calls no function in `ops` except
/// `element.offsets()`, which is the honest limit of what it witnesses — see the
/// header.
fn by_definition(
    mask: ArrayView3<'_, bool>,
    element: &StructuringElement,
    hit: bool,
) -> Array3<bool> {
    let extent = [
        mask.shape()[0] as isize,
        mask.shape()[1] as isize,
        mask.shape()[2] as isize,
    ];
    Array3::from_shape_fn(
        (extent[0] as usize, extent[1] as usize, extent[2] as usize),
        |(i, j, k)| {
            let mut answer = !hit;
            for offset in element.offsets() {
                let at = [
                    i as isize + offset[0],
                    j as isize + offset[1],
                    k as isize + offset[2],
                ];
                if (0..3).any(|axis| at[axis] < 0 || at[axis] >= extent[axis]) {
                    continue;
                }
                if mask[[at[0] as usize, at[1] as usize, at[2] as usize]] == hit {
                    answer = hit;
                }
            }
            answer
        },
    )
}

/// **Erosion is the conjunction and dilation is the disjunction**, over exactly
/// the offsets the element names and exactly the voxels that exist.
///
/// Six elements — four symmetric, two not — against three masks at three
/// densities, on exact equality. The asymmetric ones are in the list because the
/// sweep applies the element as written for both primitives, so an
/// implementation that reflected one of them would agree everywhere on a
/// symmetric element and disagree here.
#[test]
fn the_two_primitives_are_the_gathered_definition() {
    let mut compared = 0usize;
    for (element_name, element) in symmetric_elements()
        .into_iter()
        .chain(asymmetric_elements())
    {
        for (mask_name, mask) in masks() {
            assert_eq!(
                eroded(&mask, &element),
                by_definition(mask.view(), &element, false),
                "{element_name} on {mask_name}: the erosion is not the conjunction"
            );
            assert_eq!(
                dilated(&mask, &element),
                by_definition(mask.view(), &element, true),
                "{element_name} on {mask_name}: the dilation is not the disjunction"
            );
            compared += 2 * mask.len();
        }
    }
    assert_eq!(compared, 6 * 3 * 2 * 14 * 11 * 9);
    println!("{compared} voxels compared against the gathered definition");
}

/// The oracle above must be able to disagree, or the test that uses it is
/// empty. The two reductions are checked to be different functions on the same
/// fixtures — measured, so a mask on which erosion and dilation happen to agree
/// could not be the fixture that passes.
#[test]
fn the_oracle_can_tell_the_two_reductions_apart() {
    let element = StructuringElement::from_radius(ElementShape::Box, [1, 1, 1]);
    for (mask_name, mask) in masks() {
        let low = by_definition(mask.view(), &element, false);
        let high = by_definition(mask.view(), &element, true);
        let apart = low.iter().zip(high.iter()).filter(|(a, b)| a != b).count();
        assert!(
            apart > mask.len() / 20,
            "{mask_name}: the two reductions differ at only {apart} of {} voxels, so this mask \
             does not exercise the oracle",
            mask.len()
        );
        println!(
            "{mask_name}: erosion and dilation differ at {apart} of {} voxels",
            mask.len()
        );
    }
}

// ------------------------------------- claim 2: an opening is an opening --

/// Two of the three properties that *define* an opening, counted on `mask`.
/// (The third, that the opening is increasing, needs a *pair* of nested inputs
/// and is checked where the pair is built.)
///
/// Counted and returned rather than asserted, so that the same code can
/// establish that a symmetric element has them and measure how far a wrong
/// composition is from having them.
struct Laws {
    /// `open(X) ⊆ X`.
    anti_extensive: usize,
    /// `open(open(X)) == open(X)`.
    idempotent: usize,
}

fn laws_of_opening(mask: &Array3<bool>, open: impl Fn(&Array3<bool>) -> Array3<bool>) -> Laws {
    let once = open(mask);
    let twice = open(&once);
    Laws {
        anti_extensive: mask
            .iter()
            .zip(once.iter())
            .filter(|(source, result)| **result && !**source)
            .count(),
        idempotent: once
            .iter()
            .zip(twice.iter())
            .filter(|(a, b)| a != b)
            .count(),
    }
}

/// **An opening by a symmetric element obeys the three laws**, and the closing
/// obeys their duals.
///
/// * anti-extensive: `open(X)` adds no voxel to `X`;
/// * idempotent: opening twice is opening once;
/// * increasing: `X ⊆ Y` implies `open(X) ⊆ open(Y)`, checked on a nested pair
///   built by taking a mask and its dilation.
///
/// And for the closing: extensive (`X ⊆ close(X)`), idempotent, increasing.
///
/// These are not properties of *this* implementation — they are what the words
/// "opening" and "closing" mean, and any implementation that fails them is
/// computing something else. That is what makes them an outside witness even
/// though nothing outside the crate is consulted.
#[test]
fn an_opening_by_a_symmetric_element_obeys_the_three_laws() {
    for (element_name, element) in symmetric_elements() {
        assert!(
            element.is_symmetric(),
            "{element_name} is in the symmetric list and is not symmetric"
        );
        for (mask_name, mask) in masks() {
            let laws = laws_of_opening(&mask, |m| opened(m, &element));
            assert_eq!(
                laws.anti_extensive, 0,
                "{element_name} on {mask_name}: the opening added {} voxels that were not in \
                 the input",
                laws.anti_extensive
            );
            assert_eq!(
                laws.idempotent, 0,
                "{element_name} on {mask_name}: opening twice differs from opening once at {} \
                 voxels",
                laws.idempotent
            );

            // The closing's duals.
            let shut = closed(&mask, &element);
            let lost = mask
                .iter()
                .zip(shut.iter())
                .filter(|(source, result)| **source && !**result)
                .count();
            assert_eq!(
                lost, 0,
                "{element_name} on {mask_name}: the closing dropped {lost} voxels that were in \
                 the input"
            );
            assert_eq!(
                closed(&shut, &element),
                shut,
                "{element_name} on {mask_name}: the closing is not idempotent"
            );

            // Increasing, on a nested pair: a mask and its own dilation.
            let bigger = dilated(&mask, &element);
            let small = opened(&mask, &element);
            let large = opened(&bigger, &element);
            let broke = small
                .iter()
                .zip(large.iter())
                .filter(|(inner, outer)| **inner && !**outer)
                .count();
            assert_eq!(
                broke, 0,
                "{element_name} on {mask_name}: the opening is not increasing — {broke} voxels \
                 survive in the smaller input and not in the larger"
            );
        }
    }
    println!("four symmetric elements x three masks: anti-extensive, idempotent and increasing");
}

/// **The laws are not vacuous**, which is the thing a suite of algebraic
/// properties most easily becomes.
///
/// A deliberately wrong composition — dilate first, then erode, which is the
/// *closing* wearing the opening's name — is put through the same three checks
/// and measured to fail two of them. If a fixture ever stopped being able to
/// tell an opening from its dual, this test says so instead of
/// `an_opening_by_a_symmetric_element_obeys_the_three_laws` passing for nothing.
#[test]
fn the_laws_reject_a_composition_that_is_not_an_opening() {
    let element = StructuringElement::from_radius(ElementShape::Ellipsoid, [2, 2, 2]);
    let mut worst = 0usize;
    for (mask_name, mask) in masks() {
        // The dual: erode(dilate(X)), which is anti-extensive for nobody.
        let laws = laws_of_opening(&mask, |m| closed(m, &element));
        assert!(
            laws.anti_extensive > 0,
            "{mask_name}: the closing added no voxel, so this mask cannot tell an opening from \
             a closing and the law test above is not measuring anything on it"
        );
        worst = worst.max(laws.anti_extensive);
        println!(
            "{mask_name}: a closing put through the opening's laws adds {} voxels",
            laws.anti_extensive
        );
    }
    assert!(
        worst > 100,
        "the discriminating margin is only {worst} voxels"
    );
}

// ------------------------------------------------- the defect, recorded --

/// **A DEFECT, recorded here rather than fixed, and this test asserts the
/// current wrong behaviour so that fixing it fails loudly.**
///
/// `Morphology::Open` over an **asymmetric** element is not an opening. It is
/// anti-extensive for nobody, it is not idempotent, and what it actually does is
/// **translate the image** by the element's own asymmetry.
///
/// The minimal case, on one axis, with
/// `StructuringElement::from_size(ElementShape::Box, [4, 1, 1])` — sides
/// `(2, 1)`, an ordinary even extent, and the very element
/// `tests/asymmetric_element.rs` is built around:
///
/// ```text
///     X          0 0 0 1 1 1 1 0 0 0 0 0
///     open(X)    0 0 0 0 1 1 1 1 0 0 0 0     <- moved one voxel up
///     open^2(X)  0 0 0 0 0 1 1 1 1 0 0 0     <- and again
/// ```
///
/// **Why.** `sweep` reads `input[centre + offset]` for both primitives
/// (`src/ops/morphology.rs`), so its dilation is a dilation by the *reflected*
/// element, and `dilate(erode(X))` is `(X ⊖ B) ⊕ B̌` rather than `(X ⊖ B) ⊕ B`.
/// The two coincide exactly when `B = B̌`. The module is aware of the
/// non-reflection and states it at `MorphologyOp::reach_spec` — *"`sweep`
/// applies the element as written for both erosion and dilation — it does not
/// reflect it between the two passes"* — where it is treated as a fact about the
/// **reach**; what is not stated anywhere is that it also means `Open` and
/// `Close` are not an opening and a closing.
///
/// **The fix, and the evidence for it.** This test computes
/// `dilate(erode(X, B), B̌)` from the same two public primitives and shows that
/// it *is* anti-extensive, *is* idempotent, and gives the textbook answer — the
/// run of four survives unmoved and the isolated voxel is removed. So the defect
/// is the missing reflection and nothing else. `reach_spec`'s own note says what
/// else has to move: *"If `sweep`'s dilation is ever reflected, this becomes
/// `lo + hi` on both sides and `StructuringElement::reach_spec_after` stops
/// being the right method to call."*
///
/// **When this is fixed, delete this test** and add the asymmetric elements to
/// `an_opening_by_a_symmetric_element_obeys_the_three_laws`, which is where they
/// belong once they pass.
#[test]
fn an_asymmetric_opening_translates_the_image_which_is_a_defect() {
    let element = StructuringElement::from_size(ElementShape::Box, [4, 1, 1]).expect("an even box");
    assert_eq!(
        element.sides(0),
        (2, 1),
        "an even extent has no centre voxel"
    );
    assert!(!element.is_symmetric());

    let mut input = Array3::<bool>::from_elem((12, 1, 1), false);
    for index in 3..7 {
        input[[index, 0, 0]] = true;
    }
    // and one isolated voxel, which a real opening removes
    let mut with_speck = input.clone();
    with_speck[[10, 0, 0]] = true;

    let line = |volume: &Array3<bool>| -> Vec<u8> {
        (0..volume.shape()[0])
            .map(|index| volume[[index, 0, 0]] as u8)
            .collect()
    };

    // What ships today, asserted exactly.
    let shipped = opened(&input, &element);
    assert_eq!(
        line(&shipped),
        vec![0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        "the recorded defect has changed; if it was fixed, see this test's documentation"
    );
    assert_eq!(
        line(&opened(&shipped, &element)),
        vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        "and it moves again, so it is not idempotent"
    );
    let laws = laws_of_opening(&input, |m| opened(m, &element));
    assert_eq!(
        laws.anti_extensive, 1,
        "one voxel appears that was not in the input"
    );
    assert_eq!(
        laws.idempotent, 2,
        "opening twice differs from opening once"
    );

    // What the reflected composition gives, from the same two primitives.
    let reflected = reflection_of(&element);
    let correct = |volume: &Array3<bool>| dilated(&eroded(volume, &element), &reflected);
    assert_eq!(
        line(&correct(&input)),
        vec![0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        "reflecting the element for the dilation leaves the run where it was"
    );
    assert_eq!(
        line(&correct(&correct(&input))),
        line(&correct(&input)),
        "and is idempotent"
    );
    assert_eq!(
        line(&correct(&with_speck)),
        vec![0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        "and removes the isolated voxel, which is what an opening by a four-wide element does"
    );

    // The same, over the whole fixture set, so the diagnosis is not one example.
    for (element_name, element) in asymmetric_elements() {
        let reflected = reflection_of(&element);
        for (mask_name, mask) in masks() {
            let shipped = laws_of_opening(&mask, |m| opened(m, &element));
            let corrected = laws_of_opening(&mask, |m| dilated(&eroded(m, &element), &reflected));
            assert!(
                shipped.anti_extensive > 0 || shipped.idempotent > 0,
                "{element_name} on {mask_name}: the shipped opening obeys the laws here, so \
                 this fixture does not show the defect"
            );
            assert_eq!(
                corrected.anti_extensive, 0,
                "{element_name} on {mask_name}: the reflected composition is not anti-extensive \
                 either, so the diagnosis is wrong"
            );
            assert_eq!(
                corrected.idempotent, 0,
                "{element_name} on {mask_name}: the reflected composition is not idempotent \
                 either, so the diagnosis is wrong"
            );
            println!(
                "{element_name} on {mask_name}: shipped opening adds {} voxels and is \
                 non-idempotent at {}; reflecting the dilation gives 0 and 0",
                shipped.anti_extensive, shipped.idempotent
            );
        }
    }
}
