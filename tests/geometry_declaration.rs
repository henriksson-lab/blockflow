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

// ------------------------------------------- step two: geometry is consulted --
//
// `Chain::reach_spec` now folds through `BlockOp::geometry` rather than through
// `BlockOp::reach_spec`. For every shipped op that is the same number by the
// same route, which is what makes the step's failure unambiguous — so the tests
// below use ops built to make the two routes *differ*, because otherwise
// nothing here would distinguish a chain that consults the map from one that
// ignores it.

use blockflow::error::Result;
use blockflow::op::Chain;
use blockflow::reach::{AxisReach, Space};
use blockflow::voxels::Voxels;

/// States a symmetric bound of `3` and an **asymmetric** map of `1` below and
/// `3` above — the shape an even element really has.
///
/// If the fold ignored `geometry` it would get the default `reach_spec`, which
/// is symmetric `3`, so the two routes give visibly different answers.
struct AsymmetricByMap;

impl BlockOp for AsymmetricByMap {
    fn name(&self) -> &'static str {
        "asymmetric-by-map"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        3
    }

    fn geometry(&self, volume: [usize; 3]) -> Geometry {
        Geometry::stencil(volume, Reach::asymmetric([(1, 3), (1, 3), (1, 3)]))
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        out.assign(input)
    }
}

#[test]
fn the_chain_folds_the_reach_the_map_states() {
    let chain = Chain::op(AsymmetricByMap);
    let spec = chain.reach_spec(VOLUME).expect("a valid chain");
    for axis in 0..3 {
        assert_eq!(
            spec.axis(axis),
            &AxisReach::Bounded { lo: 1, hi: 3 },
            "axis {axis}: the fold must take the map's reach, not the symmetric default"
        );
    }
    // And the symmetric bound is still a bound on it, which is the invariant
    // `Chain::reach_spec` checks and which this op deliberately satisfies.
    assert_eq!(chain.reach(0, VOLUME[0]), 3);
}

/// A per-block map is not a reach and must not be flattened into one. It answers
/// nothing in `Space::source_index`, which is a *marked* zero: the space is
/// carried into the plan and cannot be converted into this phase's voxels.
struct TableMapped;

impl BlockOp for TableMapped {
    fn name(&self) -> &'static str {
        "table-mapped"
    }

    fn reach(&self, _axis: usize, volume_len: usize) -> usize {
        volume_len
    }

    fn geometry(&self, volume: [usize; 3]) -> Geometry {
        Geometry::new(
            volume,
            vec![InputMap::Table(vec![blockflow::region::Region::new(
                &[0, 0, 0],
                &[4, 4, 4],
            )])],
        )
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        out.assign(input)
    }
}

#[test]
fn a_table_map_reaches_nothing_in_the_source_index_space() {
    let chain = Chain::op(TableMapped);
    let spec = chain.reach_spec(VOLUME).expect("a valid chain");
    assert!(spec.is_none(), "a table states its dependency per block");
    assert_eq!(spec.space(), Space::source_index());
    assert!(
        !spec.space().converts_to_voxels(),
        "a source-index reach must not be convertible into this phase's voxels"
    );
}

/// An empty table is a map that gives some block nothing to read, and is refused
/// by name rather than folded to a zero that would look like a stencil.
struct EmptyTable;

impl BlockOp for EmptyTable {
    fn name(&self) -> &'static str {
        "empty-table"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn geometry(&self, volume: [usize; 3]) -> Geometry {
        Geometry::new(volume, vec![InputMap::Table(Vec::new())])
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        out.assign(input)
    }
}

#[test]
fn an_empty_table_map_is_refused_by_name() {
    let failed = Chain::op(EmptyTable).reach_spec(VOLUME).unwrap_err();
    let message = failed.to_string();
    assert!(
        message.contains("empty-table") && message.contains("one region per block"),
        "the refusal must name the op and what a table map is: {message}"
    );
}

// ----------------------------------- step three: the first declaration moves --
//
// `ResampleOp` is the first op to state an `InputMap` rather than to have one
// derived from its reach. Unlike steps one and two, a failure here means
// something real — so these compare the map against the two parallel
// declarations it is meant to replace, across factors that grow, shrink and do
// neither, and both interpolations.

use blockflow::ops::{Interpolation, Ratio, Resample, ResampleOp};

fn resamplings() -> Vec<(&'static str, Resample)> {
    vec![
        (
            "identity-nearest",
            Resample::uniform(Ratio::identity(), Interpolation::Nearest),
        ),
        (
            "halve-nearest",
            Resample::uniform(Ratio::smaller(2).unwrap(), Interpolation::Nearest),
        ),
        (
            "halve-linear",
            Resample::uniform(Ratio::smaller(2).unwrap(), Interpolation::Linear),
        ),
        (
            "triple-linear",
            Resample::uniform(Ratio::larger(3).unwrap(), Interpolation::Linear),
        ),
        (
            "anisotropic-linear",
            Resample::new(
                [
                    Ratio::new(2, 5).unwrap(),
                    Ratio::new(5, 2).unwrap(),
                    Ratio::identity(),
                ],
                Interpolation::Linear,
            ),
        ),
    ]
}

#[test]
fn the_resample_map_states_the_reach_the_op_already_declared() {
    for (name, resample) in resamplings() {
        let op = ResampleOp::new("resample", resample);
        assert_eq!(
            op.geometry(VOLUME).primary_reach(),
            Some(&resample.reach_spec()),
            "{name}: the map's window must be the interpolation's own reach"
        );
    }
}

#[test]
fn the_resample_map_states_the_volume_the_op_already_wrote() {
    for (name, resample) in resamplings() {
        let op = ResampleOp::new("resample", resample);
        assert_eq!(
            op.geometry(VOLUME).output_volume(),
            resample.output_volume(VOLUME).unwrap(),
            "{name}"
        );
        // And it is what `output_shape` says, which is the declaration the
        // executor allocates from today.
        assert_eq!(
            op.geometry(VOLUME).output_volume(),
            op.output_shape(VOLUME),
            "{name}"
        );
    }
}

#[test]
fn the_resample_map_carries_the_factor_it_was_built_from() {
    for (name, resample) in resamplings() {
        let op = ResampleOp::new("resample", resample);
        let geometry = op.geometry(VOLUME);
        match &geometry.inputs()[0] {
            InputMap::Affine { up, down, .. } => {
                for axis in 0..3 {
                    assert_eq!(up[axis], resample.ratio(axis).up(), "{name} axis {axis} up");
                    assert_eq!(
                        down[axis],
                        resample.ratio(axis).down(),
                        "{name} axis {axis} down"
                    );
                }
            }
            other => panic!("{name}: a resampling states an affine map, got {other:?}"),
        }
    }
}

/// The factor and the reach are genuinely different quantities in this sample —
/// a large factor with a small window — so the separation the map makes is
/// being tested rather than assumed.
#[test]
fn a_large_factor_does_not_imply_a_large_reach() {
    let resample = Resample::uniform(Ratio::smaller(8).unwrap(), Interpolation::Linear);
    let op = ResampleOp::new("decimate", resample);
    let geometry = op.geometry(VOLUME);
    assert_eq!(geometry.output_volume()[0], VOLUME[0] / 8);
    let reach = geometry
        .primary_reach()
        .expect("an affine map has a window");
    let widest = (0..3)
        .map(|axis| reach.axis(axis).widest(VOLUME[axis]))
        .max()
        .unwrap();
    assert!(
        widest < 8,
        "an eightfold decimation declared a reach of {widest}, which would make the halo the \
         factor rather than the window"
    );
}

/// The fold takes the map's window, which is what makes step three a change and
/// not a restatement.
#[test]
fn the_chain_folds_the_resample_map() {
    for (name, resample) in resamplings() {
        let chain = Chain::op(ResampleOp::new("resample", resample));
        assert_eq!(
            chain.reach_spec(VOLUME).expect("a valid chain"),
            resample.reach_spec(),
            "{name}"
        );
    }
}
