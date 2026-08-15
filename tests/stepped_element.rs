// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// A **decimated** structuring element, taken through the same bar
// `asymmetric_element.rs` holds an off-centre one to.
//
// The situation
// -------------
// The cost of a rank filter is its element, and the elements that dominate a
// chain are the large flat ones: a `200 x 200 x 1` window is 40 000 values
// gathered and selected from at every voxel. Decimating it by two on each wide
// axis leaves 10 000 — a quarter of the cost — and for a statistic describing a
// smooth low-frequency field the two answers are close.
//
// **This is not the sampling lattice.** `ops::local`'s `SampleLattice` evaluates
// a statistic on a coarse grid of *positions* and interpolates back; a step
// makes the *window* sparse and leaves the answer at full resolution. The two
// compose and are not interchangeable, and the difference matters because only
// one of them changes which voxels a block must be able to read.
//
// Three things are asserted here, and only the first is routine:
//
// 1. **Decomposition invariance.** Every block size and split, byte-identical to
//    a whole-volume run. A sparse window is not exempt from the property the
//    crate exists for, and it is the property that would break first if the
//    decimation's phase were anchored to anything but the element — an
//    implementation that strided the *clipped* range would re-phase itself at
//    every seam.
// 2. **The reach is the widest surviving offset.** A step can strand the far
//    pole: an 8-wide axis anchored at `(4, 3)` and stepped by two keeps
//    `-4, -2, 0, 2`, so it reads four below the anchor and **two** above, not
//    three. The declared reach must be the two, derived from the offsets, with
//    nothing able to say otherwise.
// 3. **It is really a different filter.** Otherwise (1) would be a test of the
//    unstepped element wearing a step, and (2) would be pinning a number nobody
//    depends on.
//
// One difference from an implementation that strides a slice, stated
// -------------------------------------------------------------------
// The rule was read off an implementation that expresses the same idea as
// `a[max(0, c - lo) : min(c + hi, n) : step]`, and that expression is not quite
// a fixed offset set: the stride is counted from the **clipped** start, so at a
// low boundary the decimation re-phases by `(lo - c) mod step` and a centre one
// voxel along reads a different parity of the volume. Compared outside this
// crate on a `24 x 18 x 1` volume with a `9 x 7 x 1` box stepped by `(2, 2, 1)`,
// evaluated per voxel: the fixed-offset element agreed with that implementation
// at every one of the 300 voxels outside the low margin and differed at 80 of
// the 132 inside it, and a transcription of the re-phasing rule accounted for
// all 432. The same comparison against the *undecimated* element differed at
// 281–308 of 432, which is what settles that a step decimates the element rather
// than sampling the positions.
//
// This crate keeps the fixed offsets. A window whose membership depends on how
// close the anchor is to a boundary is not one element, and property (1) above
// is exactly what it would cost: a block seam is not a volume boundary, so an
// implementation that re-phased at clipped starts would have to know which of
// the two it was looking at.

use ndarray::Array3;

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, BlockOp, Chain};
use blockflow::ops::{ElementShape, Rank, RankFilterOp, StructuringElement};
use blockflow::reach::Reach;
use blockflow::strategy::{execute, Hints, Workflow};
use blockflow::synthetic::{Scene, SceneSpec};
use blockflow::voxels::Voxels;
use blockflow::Dtype;

/// Long on axis 0, where the elements below are widest and where a reach of five
/// has to leave room for several blocks; short on the other two, because the
/// sweep's cost is the product of the three and the property is per axis.
const VOLUME: [usize; 3] = [30, 16, 12];

fn intensities() -> Array3<f64> {
    let scene = Scene::new(
        SceneSpec::new(VOLUME, 20260806)
            .with_objects(35)
            .with_radius(1.5, 4.0)
            .with_noise(0.02),
    )
    .unwrap();
    let rendered = scene.render();
    let mut array = Array3::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    for i in 0..VOLUME[0] {
        for j in 0..VOLUME[1] {
            for k in 0..VOLUME[2] {
                array[[i, j, k]] = rendered.intensity[[i, j, k]];
            }
        }
    }
    array
}

/// The elements under test. Odd and even extents, a step that divides the
/// half-extent and one that does not, a flat axis, and both a box and a ball —
/// so that nothing here rests on one arithmetic coincidence.
fn elements() -> Vec<(&'static str, StructuringElement)> {
    vec![
        (
            // the shape the consumer that prompted this asked for, scaled to a
            // volume a test can run: a wide flat window decimated on both wide
            // axes and left alone on the thin one
            "box 9x7x1 step 2,2,1",
            StructuringElement::from_size_stepped(ElementShape::Box, [9, 7, 1], [2, 2, 1]).unwrap(),
        ),
        (
            // an even extent, where the anchor is off centre *and* the step
            // strands the far pole
            "box 8x8x1 step 2,2,1",
            StructuringElement::from_size_stepped(ElementShape::Box, [8, 8, 1], [2, 2, 1]).unwrap(),
        ),
        (
            // a step that does not divide the half-extent on any axis
            "box 10x7x4 step 3,2,1",
            StructuringElement::from_size_stepped(ElementShape::Box, [10, 7, 4], [3, 2, 1])
                .unwrap(),
        ),
        (
            "inscribed ellipsoid 9x9x5 step 2,2,2",
            StructuringElement::from_size_stepped(ElementShape::Ellipsoid, [9, 9, 5], [2, 2, 2])
                .unwrap(),
        ),
    ]
}

/// One case per element per convention, because the two truncation conventions
/// resolve a rank differently at exactly the voxels a short halo would also get
/// wrong, and a sweep that only ran one of them would leave the other untested
/// under decomposition.
fn cases() -> Vec<(String, Chain)> {
    let mut cases = Vec::new();
    for (name, element) in elements() {
        cases.push((
            format!("median, {name}"),
            Chain::op(RankFilterOp::median("median", element.clone())),
        ));
        cases.push((
            format!("ceiling percentile 0.25, {name}"),
            Chain::op(RankFilterOp::new(
                "percentile",
                element.clone(),
                Rank::ceiling_percentile(0.25).unwrap(),
            )),
        ));
        cases.push((
            format!("proportional quarter rank, {name}"),
            Chain::op(RankFilterOp::new(
                "rank",
                element.clone(),
                Rank::Nth(element.len() / 4),
            )),
        ));
    }
    cases
}

fn workflow(chain: Chain) -> Workflow {
    Workflow::new(chain, VOLUME, Dtype::F64)
}

/// A plan built from the chain's **own** per-side reach — nothing here states a
/// reach, so nothing here can hide one that is wrong.
fn plan(workflow: &Workflow, block: usize, split_axes: &[usize]) -> Decomposition {
    let spec = workflow.chain.reach_spec(VOLUME).expect("a foldable reach");
    plan_with_halo(workflow, block, split_axes, spec.clone(), spec)
}

fn plan_with_halo(
    workflow: &Workflow,
    block: usize,
    split_axes: &[usize],
    reach: Reach,
    halo: Reach,
) -> Decomposition {
    let slots = workflow.chain.slots();
    let names: Vec<String> = slots.iter().map(|slot| slot.display_name()).collect();
    let grid = BlockGrid::along(VOLUME, split_axes, block).unwrap();
    let phase = PhaseDecomposition::derive((0..slots.len()).collect(), names, reach, halo, grid);
    Decomposition {
        volume: VOLUME,
        dtype: workflow.dtype,
        phases: vec![phase],
        chain_reach: workflow.chain.reach3(&VOLUME),
    }
}

fn reference(chain: &Chain, input: &Array3<f64>) -> Array3<f64> {
    let source: Voxels = input.clone().into();
    let mut out = Voxels::zeros(Dtype::F64, VOLUME).unwrap();
    chain
        .apply(&source, &mut out, &Anchor::whole(VOLUME))
        .expect("the whole-volume reference must run");
    out.view::<f64>().unwrap().to_owned()
}

fn run(workflow: &Workflow, decomposition: &Decomposition, input: &Array3<f64>) -> Array3<f64> {
    let env =
        ArrayEnvironment::new(input.clone().into(), decomposition.n_phases(), [4, 4, 4]).unwrap();
    execute("stepped", workflow, decomposition, &Hints::default(), &env).unwrap();
    env.output().view::<f64>().unwrap().to_owned()
}

// ------------------------------------------------------ what is derived --

/// **The reach is the widest surviving offset**, on each side of each axis, and
/// the op reports exactly that.
///
/// The 8-wide axis is the case that separates a derived reach from a declared
/// one: the box asks for `(4, 3)`, the step keeps `-4, -2, 0, 2`, and an element
/// that reported three above the anchor would state a dependency it does not
/// have — a plane fetched into every block that no voxel of the answer reads.
#[test]
fn a_stepped_elements_reach_is_its_widest_surviving_offset() {
    let element =
        StructuringElement::from_size_stepped(ElementShape::Box, [8, 8, 1], [2, 2, 1]).unwrap();
    assert_eq!(element.step(), [2, 2, 1]);
    assert_eq!(
        element.len(),
        4 * 4,
        "a quarter of the 8x8 box, less the odd rows"
    );
    assert_eq!(
        element.sides(0),
        (4, 2),
        "the far pole at +3 did not survive"
    );
    assert_eq!(element.sides(1), (4, 2));
    assert_eq!(element.sides(2), (0, 0));
    assert_eq!(element.reach(0), 4, "the wider side");
    assert_eq!(element.size(), [7, 7, 1], "the span of what survived");

    let op = RankFilterOp::median("median", element.clone());
    assert_eq!(op.reach(0, VOLUME[0]), 4);
    assert_eq!(op.reach(2, VOLUME[2]), 0);
    let spec = op.reach_spec(VOLUME);
    assert_eq!(spec.at(0, 0, VOLUME[0]), (4, 2));
    assert_eq!(spec.at(1, 0, VOLUME[1]), (4, 2));
    assert_eq!(
        spec.as_symmetric(),
        None,
        "a stranded pole makes the reach asymmetric even for a symmetric box"
    );

    // brute force, over every element in the sweep: the reported sides are the
    // extremes of the offsets and are not a restatement of the constructor's
    // arguments
    for (name, element) in elements() {
        for axis in 0..3 {
            let below = element
                .offsets()
                .iter()
                .map(|offset| (-offset[axis]).max(0) as usize)
                .max()
                .unwrap();
            let above = element
                .offsets()
                .iter()
                .map(|offset| offset[axis].max(0) as usize)
                .max()
                .unwrap();
            assert_eq!(element.sides(axis), (below, above), "{name} axis {axis}");
        }
    }
}

/// The count is the surviving count, and every cost and every rank is built from
/// it. A median that still named `n / 2` of the *undecimated* element would be a
/// maximum of the sparse one.
#[test]
fn the_length_is_the_surviving_count_and_the_cost_follows_it() {
    let whole = StructuringElement::from_size(ElementShape::Box, [9, 7, 1]).unwrap();
    let stepped =
        StructuringElement::from_size_stepped(ElementShape::Box, [9, 7, 1], [2, 2, 1]).unwrap();
    assert_eq!(whole.len(), 63);
    assert_eq!(stepped.len(), 5 * 4);

    assert_eq!(Rank::median(&stepped), Rank::Nth(10));
    assert_eq!(Rank::highest(&stepped), Rank::Nth(19));

    let dense = RankFilterOp::median("dense", whole);
    let sparse = RankFilterOp::median("sparse", stepped);
    assert!(
        sparse.cost_per_voxel() < dense.cost_per_voxel(),
        "a sparser window must cost less: {} against {}",
        sparse.cost_per_voxel(),
        dense.cost_per_voxel()
    );
    // the ratio is the ratio of the populations, because that is what the cost
    // is a function of
    let ratio = sparse.cost_per_voxel() / dense.cost_per_voxel();
    assert!((ratio - 20.0 / 63.0).abs() < 1e-12, "ratio {ratio}");
}

// ---------------------------------------------- 1. it is a real filter --

/// A stepped element gives a **different answer** from the element it decimates,
/// and from a smaller dense element that happens to hold the same number of
/// voxels. Without this the invariance sweep below could pass while testing
/// nothing new.
#[test]
fn a_stepped_element_is_not_the_element_it_decimates_nor_a_smaller_dense_one() {
    let input = intensities();
    let filtered = |element: StructuringElement| -> Array3<f64> {
        reference(&Chain::op(RankFilterOp::median("median", element)), &input)
    };

    let dense = filtered(StructuringElement::from_size(ElementShape::Box, [9, 7, 1]).unwrap());
    let sparse = filtered(
        StructuringElement::from_size_stepped(ElementShape::Box, [9, 7, 1], [2, 2, 1]).unwrap(),
    );
    // 20 voxels, like the sparse window, but gathered from a 5x4 neighbourhood
    // instead of a 9x7 one
    let small = filtered(StructuringElement::from_size(ElementShape::Box, [5, 4, 1]).unwrap());

    let differing = |left: &Array3<f64>, right: &Array3<f64>| -> usize {
        left.iter()
            .zip(right.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count()
    };
    let total = VOLUME[0] * VOLUME[1] * VOLUME[2];
    assert!(
        differing(&dense, &sparse) * 4 > total,
        "the decimated window must be a materially different filter, differed at {} of {total}",
        differing(&dense, &sparse)
    );
    assert!(
        differing(&small, &sparse) * 4 > total,
        "and not merely a smaller dense window, differed at {} of {total}",
        differing(&small, &sparse)
    );
}

// -------------------------------------------------- 2. the ordinary bar --

/// Byte-identical to the whole-volume reference, under every decomposition.
#[test]
fn a_stepped_element_is_decomposition_invariant() {
    let input = intensities();
    for (name, chain) in cases() {
        let want = reference(&chain, &input);
        let workflow = workflow(chain);
        let mut ran = 0;
        for block in [4usize, 7, 13, 64] {
            for split_axes in [vec![0], vec![2], vec![0, 1, 2]] {
                let decomposition = plan(&workflow, block, &split_axes);
                decomposition
                    .check()
                    .unwrap_or_else(|err| panic!("{name}: an honest plan must tile: {err}"));
                assert_eq!(
                    run(&workflow, &decomposition, &input),
                    want,
                    "{name}: block {block}, axes {split_axes:?} disagreed with the \
                     whole-volume reference"
                );
                ran += 1;
            }
        }
        assert!(ran >= 12, "{name}: the sweep did not run");
    }
}

// ------------------------------------- 3. the guard, on the derived side --

/// A halo short of the **derived** reach is refused, and the derived reach
/// itself is accepted.
///
/// The pair is the point. Refusing a short halo alone is also what a crate that
/// demanded the undecimated box's reach would do; accepting `(4, 2)` — two above
/// the anchor, where the box spans three — is what says the demand is the
/// element's actual dependency rather than its bounding box.
#[test]
fn a_halo_short_of_the_derived_reach_is_refused_and_the_derived_one_is_not() {
    let input = intensities();
    let element =
        StructuringElement::from_size_stepped(ElementShape::Box, [8, 8, 1], [2, 2, 1]).unwrap();
    let workflow = workflow(Chain::op(RankFilterOp::median("median", element)));
    let reach = workflow.chain.reach_spec(VOLUME).unwrap();
    assert_eq!(reach.at(0, 0, VOLUME[0]), (4, 2));

    // Split on two axes, because a halo can only be short where there is a
    // seam.
    let exact = plan_with_halo(&workflow, 8, &[0, 1], reach.clone(), reach.clone());
    exact.check().expect("the exact per-side halo must tile");
    let want = reference(&workflow.chain, &input);
    assert_eq!(run(&workflow, &exact, &input), want);

    for short in [
        // one short on the stranded side, which is the side a reach taken from
        // the bounding box would have over-granted and therefore cannot notice
        Reach::asymmetric([(4, 1), (4, 2), (0, 0)]),
        // and one short on the wide side, so the guard is not merely sensitive
        // to the entry it looks at first
        Reach::asymmetric([(3, 2), (4, 2), (0, 0)]),
        Reach::asymmetric([(4, 2), (4, 1), (0, 0)]),
    ] {
        let plan = plan_with_halo(&workflow, 8, &[0, 1], reach.clone(), short.clone());
        assert!(
            plan.check().is_err(),
            "a halo of {short:?} is short of {reach:?} and must be refused"
        );
    }

    // and the shortfall really is observable: made self-consistent so the tiling
    // check is satisfied, it changes the answer. A reach nobody depends on could
    // be understated with no symptom, and then the refusals above would be
    // guarding nothing.
    let understated = Reach::asymmetric([(4, 1), (4, 2), (0, 0)]);
    let plan = plan_with_halo(
        &workflow,
        8,
        &[0, 1],
        understated.clone(),
        understated.clone(),
    );
    plan.check().expect("a self-consistent plan tiles");
    assert_ne!(
        run(&workflow, &plan, &input),
        want,
        "the offset at +2 is really read, so understating it must change the answer"
    );
}
