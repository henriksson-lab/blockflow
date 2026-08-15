// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **Step one of the output-side index map: the step that changes nothing.**
//
// `forme.md` specifies `Geometry` — one declaration of an op's output space and
// what each input must supply to fill it — replacing quantities that are
// currently stated twice and checked against each other. The migration's first
// step lands the type with a default that reproduces today's behaviour exactly,
// *before* anything consumes it, because a step that changes nothing is a step
// whose failure is unambiguous.
//
// So this file asserts that nothing changed:
//
// 1. Every shipped op's default geometry is its own `reach_spec`, on ops chosen
//    to cover the shapes that differ — symmetric, asymmetric from an even
//    element, per-block from a sample lattice, and zero.
// 2. The output volume is the input volume for all of them, which is what makes
//    the default correct rather than merely present.
// 3. `Placement::same` really is one anchor in both spaces, so an op ignoring
//    the output side behaves as it did.
//
// The file is deliberately small. Its value is that it fails loudly if a later
// step moves a declaration onto `Geometry` and gets it wrong for an op nobody
// was looking at.

use blockflow::op::{Anchor, BlockOp, Geometry, InputMap, Placement};
use blockflow::ops::{
    ElementShape, LocalStatistic, LocalStatisticOp, Morphology, MorphologyOp, RankFilterOp,
    Sampling, Statistic, StructuringElement, VoxelwiseMapOp,
};
use blockflow::reach::Reach;

const VOLUME: [usize; 3] = [40, 32, 12];

fn ops() -> Vec<(&'static str, Box<dyn BlockOp>)> {
    vec![
        // Reach zero: the case a wrong default would most easily hide in.
        (
            "identity",
            Box::new(VoxelwiseMapOp::new("identity", |value| value)) as Box<dyn BlockOp>,
        ),
        // Symmetric, from an odd element.
        (
            "dilate",
            Box::new(MorphologyOp::new(
                "dilate",
                Morphology::Dilate,
                StructuringElement::from_radius(ElementShape::Box, [2, 1, 1]),
            )),
        ),
        // **Asymmetric**, from an even element: a geometry derived from a
        // symmetric radius would be wrong on one side and this is where it shows.
        (
            "median-even",
            Box::new(RankFilterOp::median(
                "median-even",
                StructuringElement::from_size(ElementShape::Box, [4, 4, 2]).unwrap(),
            )),
        ),
        // Per-block, from a sample lattice: the reach is a table rather than one
        // integer per axis, and the default has to carry it whole.
        (
            "sampled-mean",
            Box::new(LocalStatisticOp::new(
                "sampled-mean",
                LocalStatistic::sampled(
                    StructuringElement::from_radius(ElementShape::Box, [1, 1, 1]),
                    Sampling::every([5, 4, 3]),
                    Statistic::Mean,
                )
                .unwrap(),
            )),
        ),
    ]
}

#[test]
fn the_default_geometry_is_the_ops_own_reach() {
    for (name, op) in ops() {
        let geometry = op.geometry(VOLUME);
        assert_eq!(
            geometry.primary_reach(),
            Some(&op.reach_spec(VOLUME)),
            "{name}: the default geometry must state the reach the op already declares"
        );
    }
}

#[test]
fn the_default_output_volume_is_the_input_volume() {
    for (name, op) in ops() {
        assert_eq!(op.geometry(VOLUME).output_volume(), VOLUME, "{name}");
    }
}

#[test]
fn the_default_declares_exactly_one_input() {
    for (name, op) in ops() {
        let geometry = op.geometry(VOLUME);
        assert_eq!(geometry.inputs().len(), 1, "{name}");
        assert!(
            matches!(geometry.inputs()[0], InputMap::Stencil(_)),
            "{name}: an op that has not moved to a map is a stencil"
        );
    }
}

/// The reaches under test are genuinely different from each other, so the three
/// assertions above are comparing something.
#[test]
fn the_ops_chosen_do_not_all_declare_the_same_reach() {
    let reaches: Vec<Reach> = ops().iter().map(|(_, op)| op.reach_spec(VOLUME)).collect();
    let first = &reaches[0];
    assert!(
        reaches.iter().any(|reach| reach != first),
        "every op sampled declares the same reach, so this file proves nothing"
    );
    // And at least one is not expressible as a symmetric triple, which is the
    // shape a lossy default would silently flatten it to.
    assert!(
        reaches.iter().any(|reach| reach.as_symmetric().is_none()),
        "no op sampled has a reach that a symmetric triple would lose"
    );
}

// ------------------------------------------------------------ placement --

#[test]
fn a_placement_of_one_anchor_is_that_anchor_in_both_spaces() {
    let at = Anchor::new([4, 8, 0], VOLUME);
    let placement = Placement::same(at.clone());
    assert_eq!(placement.input, at);
    assert_eq!(placement.output, at);
    assert!(placement.sources.is_empty());
    assert_eq!(placement.source(0), None);
}

#[test]
fn a_placement_can_differ_between_the_spaces() {
    let fine = Anchor::new([10, 0, 0], VOLUME);
    let coarse = Anchor::new([2, 0, 0], [8, 8, 4]);
    let placement = Placement::new(fine.clone(), coarse.clone())
        .with_sources(vec![(1, fine.clone()), (3, coarse.clone())]);
    assert_eq!(placement.input, fine);
    assert_eq!(placement.output, coarse);
    assert_eq!(placement.source(1), Some(&fine));
    assert_eq!(placement.source(3), Some(&coarse));
    assert_eq!(placement.source(2), None);
}

// ------------------------------------------------------------ the type --

#[test]
fn a_table_has_no_single_reach() {
    let geometry = Geometry::new(
        VOLUME,
        vec![InputMap::Table(vec![blockflow::region::Region::new(
            &[0, 0, 0],
            &[4, 4, 4],
        )])],
    );
    assert_eq!(geometry.primary_reach(), None);
    assert_eq!(geometry.output_volume(), VOLUME);
}

#[test]
fn the_identity_geometry_reaches_nothing() {
    let geometry = Geometry::same(VOLUME);
    assert_eq!(geometry.primary_reach(), Some(&Reach::none()));
}
