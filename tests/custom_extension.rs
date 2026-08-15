// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **Two extension points, exercised from outside the crate.**
//
// A user of this library must be able to add operations and the functions those
// operations are parameterised by. Two places prevented that until now, and they
// needed different fixes because the obstacles were different:
//
// * `ops::local::Statistic` was a closed enum, so a windowed statistic the crate
//   did not ship could not be written at all. `Isodata` had to be added *inside*
//   the crate for exactly that reason. It is open now, through `Reducer`.
// * `StructuringElement` had no constructor from an explicit offset list, so a
//   neighbourhood shape the crate did not name could not be built — not because
//   the shape rule was closed to extension in any costly way, but because there
//   was simply no way in. `from_offsets` is that way in.
//
// This file is deliberately an *integration* test rather than a unit test: it
// can only use what the crate exports, which is the property being asserted. A
// unit test inside the module could reach private items and would prove nothing
// about what a third party can write.
//
// The bar is the crate's own: a custom reducer over a custom element must be
// **decomposition invariant**, byte-identical to a whole-volume run of the same
// kernels over several block sizes and split patterns. Being able to *write* an
// extension is worth nothing if the extension is not held to the same standard
// as the shipped ops.

use std::sync::Arc;

use ndarray::Array3;

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, BlockOp, Chain};
use blockflow::ops::local::Reducer;
use blockflow::ops::{LocalStatistic, LocalStatisticOp, Statistic, StructuringElement};
use blockflow::strategy::{execute, Hints, Workflow};
use blockflow::voxels::Voxels;
use blockflow::Dtype;

const VOLUME: [usize; 3] = [24, 18, 14];

// ------------------------------------------------ a third-party reducer --

/// The **midrange**: the mean of the window's extremes.
///
/// Chosen because the crate ships nothing that computes it and nothing that
/// approximates it — a mean, a deviation, a rank and an isodata threshold all
/// answer different questions — so a test that passes here could not have been
/// satisfied by reaching for an existing variant.
///
/// It is also exactly declarable: a uniform window has `max == min == value`, so
/// the midrange is that value, and the arithmetic returns one of its own
/// operands rather than computing a new one. That makes `constant_maps_to` true
/// in the strong sense `ops/mod.rs` requires rather than true to within a
/// rounding.
struct Midrange;

impl Reducer for Midrange {
    fn reduce(&self, window: &mut [f64], _full: usize) -> f64 {
        let mut low = f64::INFINITY;
        let mut high = f64::NEG_INFINITY;
        for &value in window.iter() {
            low = low.min(value);
            high = high.max(value);
        }
        if window.is_empty() {
            return 0.0;
        }
        // Written as a mean of two rather than `(low + high) / 2.0` in one step
        // for no reason but symmetry with the shipped statistics; both are exact
        // here because a uniform window makes both operands equal.
        (low + high) / 2.0
    }

    /// Exactly the constant. A uniform window's extremes are both that value.
    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        value.is_finite().then_some(value)
    }

    fn cost_per_sample(&self, _window: usize) -> f64 {
        0.0
    }

    fn key(&self) -> (&'static str, u64) {
        ("midrange", 0)
    }
}

// ------------------------------------------------------------ fixtures --

fn intensities() -> Array3<f64> {
    Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(i, j, k)| {
        ((i * 7919 + j * 104729 + k * 1013) % 251) as f64 / 251.0
    })
}

/// A neighbourhood no `ElementShape` produces: the six face neighbours and the
/// centre, asymmetric on the first axis so that the derived reach has two
/// different sides and a symmetric reading of it would be wrong.
fn custom_element() -> StructuringElement {
    StructuringElement::from_offsets(vec![
        [0, 0, 0],
        [-1, 0, 0],
        [2, 0, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, 0, -1],
        [0, 0, 1],
    ])
    .expect("seven distinct offsets are an element")
}

fn plan(workflow: &Workflow, block: usize, split_axes: &[usize]) -> Decomposition {
    let slots = workflow.chain.slots();
    let names: Vec<String> = slots.iter().map(|slot| slot.display_name()).collect();
    let reach = workflow.chain.reach3(&VOLUME);
    let grid = BlockGrid::along(VOLUME, split_axes, block).unwrap();
    let phase = PhaseDecomposition::derive((0..slots.len()).collect(), names, reach, reach, grid);
    Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases: vec![phase],
        chain_reach: reach,
    }
}

fn chain() -> Chain {
    let statistic = LocalStatistic::new(
        custom_element(),
        [1, 1, 1],
        Statistic::Custom(Arc::new(Midrange)),
    )
    .expect("a custom reducer over a custom element");
    Chain::op(LocalStatisticOp::new("midrange", statistic))
}

// ---------------------------------------------------------- the checks --

/// The bar every shipped op meets, applied to an op the crate does not ship.
#[test]
fn a_custom_reducer_over_a_custom_element_is_decomposition_invariant() {
    let input = intensities();
    let workflow = Workflow::new(chain(), VOLUME, Dtype::F64);

    let source: Voxels = input.clone().into();
    let mut whole = Voxels::zeros(Dtype::F64, VOLUME).unwrap();
    workflow
        .chain
        .apply(&source, &mut whole, &Anchor::whole(VOLUME))
        .expect("the whole-volume reference must run");
    let want = whole.view::<f64>().unwrap().to_owned();

    // not vacuous: the statistic must actually change the volume
    assert!(
        want.iter().zip(input.iter()).any(|(a, b)| a != b),
        "the midrange of this scene is its own input, so the test asserts nothing"
    );

    let mut checked = 0;
    for block in [4, 6, 8, 24] {
        for split_axes in [vec![0], vec![2], vec![0, 1], vec![0, 1, 2]] {
            let decomposition = plan(&workflow, block, &split_axes);
            let env =
                ArrayEnvironment::new(input.clone().into(), decomposition.n_phases(), [4, 4, 4])
                    .unwrap();
            execute("custom", &workflow, &decomposition, &Hints::default(), &env).unwrap();
            let got = env.output().view::<f64>().unwrap().to_owned();
            assert_eq!(got, want, "block {block}, axes {split_axes:?}");
            checked += 1;
        }
    }
    assert_eq!(checked, 16);
}

/// The reach of a hand-built element is **derived from its offsets**, and the
/// op's reach follows it. A symmetric reading of the first axis would report
/// two below the anchor where only one is read, and would fetch a plane nothing
/// depends on.
#[test]
fn the_reach_of_a_hand_built_element_follows_its_offsets() {
    let element = custom_element();
    assert_eq!(element.sides(0), (1, 2));
    assert_eq!(element.sides(1), (1, 1));
    assert_eq!(element.sides(2), (1, 1));

    let statistic =
        LocalStatistic::new(element, [1, 1, 1], Statistic::Custom(Arc::new(Midrange))).unwrap();
    // A spacing of one is the no-lattice case, so the op's reach is the
    // element's own widest side and nothing else.
    let op = LocalStatisticOp::new("midrange", statistic);
    assert_eq!(op.reach(0, VOLUME[0]), 2);
    assert_eq!(op.reach(1, VOLUME[1]), 1);
}

/// A third-party reducer states its own constant algebra, and the declaration
/// must agree with computing the block — the crate's rule for every shipped op.
#[test]
fn a_custom_reducer_declares_a_constant_it_actually_produces() {
    let statistic = Statistic::Custom(Arc::new(Midrange));
    assert_eq!(statistic.constant_maps_to(0.25), Some(0.25));
    assert_eq!(statistic.constant_maps_to(f64::NAN), None);

    let local = LocalStatistic::new(
        custom_element(),
        [1, 1, 1],
        Statistic::Custom(Arc::new(Midrange)),
    )
    .unwrap();
    let op = LocalStatisticOp::new("midrange", local);
    assert_eq!(op.constant_maps_to(0.25), Some(0.25));

    // and computing it agrees with declaring it
    let constant: Voxels = Array3::from_elem((8, 8, 8), 0.25).into();
    let mut out = Voxels::zeros(Dtype::F64, [8, 8, 8]).unwrap();
    op.apply(&constant, &mut out, &Anchor::whole([8, 8, 8]))
        .unwrap();
    assert!(out.view::<f64>().unwrap().iter().all(|&v| v == 0.25));
}

/// `Statistic` is compared and hashed because plans are, and a trait object has
/// no derivable identity — so a custom reducer is identified by the key it
/// states. Two of the same reducer are the same statistic; a custom one is never
/// equal to a shipped one.
#[test]
fn a_custom_statistic_is_identified_by_the_key_its_reducer_states() {
    let one = Statistic::Custom(Arc::new(Midrange));
    let two = Statistic::Custom(Arc::new(Midrange));
    assert_eq!(one, two);
    assert_ne!(one, Statistic::Mean);

    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let digest = |statistic: &Statistic| {
        let mut hasher = DefaultHasher::new();
        statistic.hash(&mut hasher);
        hasher.finish()
    };
    assert_eq!(digest(&one), digest(&two));
    assert_ne!(digest(&one), digest(&Statistic::Mean));
}
