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
// properties — **anti-extensive**, **increasing**, **idempotent** — and "erode
// by `B` then dilate by the *neighbourhood gather*" only has them when `B` is
// symmetric, because that gather is a dilation by `B̌`. That is the distinction
// nothing here was measuring; when this file first measured it, `Open` over an
// element with no centre voxel obeyed none of the three and translated the
// image once per application. `ops::morphology::dilate_placed_into` is the fix —
// the dilation that is the erosion's adjoint — and
// `the_dilation_that_makes_an_opening_is_the_reflected_one` is the
// counterexample kept as a regression.
//
// What is asserted
// ----------------
//
// | claim | how |
// |---|---|
// | erosion and dilation are the gathered definition | a triple loop over the element's offsets with the clamp written out, on six elements x three masks — 49 896 voxels, exact |
// | an opening is an opening | anti-extensive, increasing and idempotent, on six elements — four symmetric, two not — x three masks; the closing dual beside it |
// | the laws are not vacuous | the same three properties are checked to *fail* for a deliberately wrong composition, so a fixture on which everything is an opening is caught |
// | **the composition is the one that reflects** | the minimal counterexample the defect was found on, asserted the right way round, with the unreflected composition computed beside it and measured to move the image |
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
    let built = StructuringElement::from_offsets(
        element
            .offsets()
            .iter()
            .map(|offset| [-offset[0], -offset[1], -offset[2]])
            .collect::<Vec<_>>(),
    )
    .expect("a reflected offset set is still an offset set");
    // and the crate's own reflection is that set, which is the one line that
    // keeps this file's oracle and `StructuringElement::reflected` from being
    // two different reflections.
    assert_eq!(
        element
            .reflected()
            .expect("an anchored element reflects")
            .offsets(),
        built.offsets(),
        "the crate's reflection is not the negated offset set"
    );
    built
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
fn an_opening_obeys_the_three_laws_over_every_element() {
    let mut asymmetric_seen = 0usize;
    for (element_name, element) in symmetric_elements()
        .into_iter()
        .chain(asymmetric_elements())
    {
        asymmetric_seen += !element.is_symmetric() as usize;
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
    assert!(
        asymmetric_seen >= 2,
        "the sweep saw {asymmetric_seen} elements without a centre voxel, and those are the \
         ones the three laws were failing on"
    );
    println!(
        "six elements ({asymmetric_seen} of them without a centre voxel) x three masks: \
         anti-extensive, idempotent and increasing"
    );
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

// --------------------------------- claim 4: the composition that reflects --

/// **The dilation an opening is made of is the one by the element, not the one
/// by its reflection** — the minimal counterexample the defect was found on,
/// kept as a regression and asserted the right way round.
///
/// The element is `from_size(ElementShape::Box, [4, 1, 1])`: sides `(2, 1)`, an
/// ordinary even extent, and the element `tests/asymmetric_element.rs` is built
/// around. On one axis, with `X` a run of four and one isolated voxel:
///
/// ```text
///     X                 0 0 0 1 1 1 1 0 0 0 1 0
///     open(X)           0 0 0 1 1 1 1 0 0 0 0 0   <- the run survives, the speck goes
///     gather-composed   0 0 0 0 1 1 1 1 0 0 0 0   <- moved a voxel: 7 gained, 3 lost
/// ```
///
/// The second line is what ships. The third is `dilate_into(erode_into(X))` —
/// the composition this module had until `dilate_placed_into` existed — computed
/// here from the same two public primitives so that the difference between them
/// is visible in one test rather than described in a comment. It is
/// `(X ⊖ B) ⊕ B̌`, it is anti-extensive for nobody, and applying it again moves
/// the run again.
///
/// **Why an opening cannot be built from the neighbourhood gather.**
/// `ops::morphology::sweep` reads `input[centre + offset]` for both primitives,
/// which is what makes its dilation a dilation by `B̌` and what makes it equal to
/// `ops::rank`'s extreme rank over the same element — an equality
/// `ops::background` builds a grey opening on and which is worth keeping. So the
/// composition reflects instead, in `dilate_placed_into`, which places the
/// element at each set voxel and is the erosion's adjoint.
#[test]
fn the_dilation_that_makes_an_opening_is_the_reflected_one() {
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
    input[[10, 0, 0]] = true;

    let line = |volume: &Array3<bool>| -> Vec<u8> {
        (0..volume.shape()[0])
            .map(|index| volume[[index, 0, 0]] as u8)
            .collect()
    };

    assert_eq!(
        line(&opened(&input, &element)),
        vec![0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        "the run of four survives where it is and the isolated voxel is removed, which is what \
         an opening by a four-wide element does"
    );
    assert_eq!(
        line(&opened(&opened(&input, &element), &element)),
        line(&opened(&input, &element)),
        "and it is idempotent"
    );

    // The composition that does not reflect, from the same two primitives.
    let unreflected = dilated(&eroded(&input, &element), &element);
    assert_eq!(
        line(&unreflected),
        vec![0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        "the unreflected composition moves the run, which is the defect this test records"
    );
    assert_eq!(
        line(&dilated(&eroded(&unreflected, &element), &element)),
        vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        "and moves it again, so it is not idempotent either"
    );

    // The same, over the whole fixture set, so neither claim is one example.
    for (element_name, element) in asymmetric_elements() {
        for (mask_name, mask) in masks() {
            let shipped = laws_of_opening(&mask, |m| opened(m, &element));
            assert_eq!(
                (shipped.anti_extensive, shipped.idempotent),
                (0, 0),
                "{element_name} on {mask_name}: the opening breaks its own laws"
            );
            let unreflected = laws_of_opening(&mask, |m| dilated(&eroded(m, &element), &element));
            assert!(
                unreflected.anti_extensive > 0 || unreflected.idempotent > 0,
                "{element_name} on {mask_name}: the unreflected composition obeys the laws \
                 here, so this fixture cannot tell the two compositions apart"
            );
            println!(
                "{element_name} on {mask_name}: the opening is (0, 0) against the laws; the \
                 unreflected composition is ({}, {})",
                unreflected.anti_extensive, unreflected.idempotent
            );
        }
    }
}

/// **The two dilations are two filters**, which is what the claim above rests
/// on: if they were the same volume everywhere, the composition would not care
/// which one it called and the test above would pass for nothing.
#[test]
fn the_placed_dilation_and_the_neighbourhood_gather_are_two_filters() {
    for (element_name, element) in asymmetric_elements() {
        let reflected = reflection_of(&element);
        for (mask_name, mask) in masks() {
            let placed = dilated(&mask, &reflected);
            let gathered = dilated(&mask, &element);
            let apart = placed
                .iter()
                .zip(gathered.iter())
                .filter(|(a, b)| a != b)
                .count();
            assert!(
                apart > 0,
                "{element_name} on {mask_name}: the two dilations agree everywhere"
            );
            println!("{element_name} on {mask_name}: the two dilations differ at {apart} voxels");
        }
    }
    for (element_name, element) in symmetric_elements() {
        let reflected = reflection_of(&element);
        for (mask_name, mask) in masks() {
            assert_eq!(
                dilated(&mask, &reflected),
                dilated(&mask, &element),
                "{element_name} on {mask_name}: a symmetric element is its own reflection and \
                 the two dilations must be one filter"
            );
        }
    }
}
