// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// An element with no centre voxel, taken through the same bar `image_ops.rs`
// holds every op to — and through the one case a symmetric assumption gets
// wrong.
//
// The situation
// -------------
// An even extent has no centre voxel, so the anchor sits off centre and the
// element reads one voxel further below it than above:
// `from_size(shape, [10, ..])` gives sides `(5, 4)`. Every op that derives
// geometry from an element therefore has an **asymmetric** dependency, and the
// crate has carried per-side reaches since `AxisReach::Bounded { lo, hi }`
// existed — what it did not have was any way to *build* an element that needed
// them.
//
// Three things are asserted here, and only the first is routine:
//
// 1. **Decomposition invariance.** Every block size and split, byte-identical to
//    a whole-volume run of the same kernels. This is the property the crate
//    exists for and an off-centre window is not exempt from it.
// 2. **The guard fires on the narrow side specifically.** A halo of `(5, 3)`
//    against a reach of `(5, 4)` is short by one on the side an implementation
//    that took `max(lo, hi)` and applied it symmetrically would have granted
//    five voxels of — so it would look satisfied, and the seam would be wrong.
//    The plan must be refused. A halo of exactly `(5, 4)` must be accepted, or
//    the test would pass by over-granting rather than by being exact.
// 3. **Understating the narrow side is silently wrong.** The same shortfall,
//    made self-consistent so that the tiling check is happy, must change the
//    answer. That is the evidence that the narrow side is really read — a reach
//    nobody depends on could be understated with no symptom, and then the guard
//    in (2) would be guarding nothing.

use ndarray::Array3;

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, Chain};
use blockflow::ops::StructuringElement;
use blockflow::ops::{ElementShape, Morphology, MorphologyOp, Rank, RankFilterOp};
use blockflow::reach::Reach;
use blockflow::strategy::{execute, Hints, Workflow};
use blockflow::synthetic::{Scene, SceneSpec};
use blockflow::voxels::Voxels;
use blockflow::Dtype;

/// Long on axis 0, where the elements below are widest and where the biggest
/// reach — an opening's `(9, 9)`, which is `lo + hi` on both sides — has to
/// leave room for several blocks;
/// deliberately short on the other two, because the sweep's cost is the product
/// of the three and the property being tested is per axis.
const VOLUME: [usize; 3] = [30, 16, 12];

fn intensities() -> Array3<f64> {
    let scene = Scene::new(
        SceneSpec::new(VOLUME, 20260213)
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

fn mask(input: &Array3<f64>, level: f64) -> Array3<f64> {
    input.mapv(|value| if value > level { 1.0 } else { 0.0 })
}

/// The elements under test. Even on some axes and odd on others, in both ball
/// rules and in the box, so that nothing here rests on one shape's arithmetic.
fn elements() -> Vec<(&'static str, StructuringElement)> {
    vec![
        (
            "box 10x5x4",
            StructuringElement::from_size(ElementShape::Box, [10, 5, 4]).unwrap(),
        ),
        (
            "extent ellipsoid 10x6x3",
            StructuringElement::from_size(ElementShape::ExtentEllipsoid, [10, 6, 3]).unwrap(),
        ),
        (
            "inscribed ellipsoid 6x6x4",
            StructuringElement::from_size(ElementShape::Ellipsoid, [6, 6, 4]).unwrap(),
        ),
        (
            // Not derived from an extent at all: an anchor placed where the
            // caller wants it. The reach is asymmetric on two axes and in
            // opposite directions, which no `from_size` can produce.
            "hand-placed anchor",
            StructuringElement::from_sides(ElementShape::Box, [3, 0, 1], [1, 2, 1]),
        ),
    ]
}

fn cases(input: &Array3<f64>) -> Vec<(String, Chain, Array3<f64>)> {
    let masked = mask(input, 0.35);
    let mut cases = Vec::new();
    for (name, element) in elements() {
        cases.push((
            format!("median, {name}"),
            Chain::op(RankFilterOp::median("median", element.clone())),
            input.clone(),
        ));
        cases.push((
            format!("erode, {name}"),
            Chain::op(MorphologyOp::new(
                "erode",
                Morphology::Erode,
                element.clone(),
            )),
            masked.clone(),
        ));
        cases.push((
            // The composition, which is the reach most likely to be wrong per
            // side: an opening reflects between its two passes — see
            // `ops::morphology::dilate_placed_into` — so an element reading
            // `(5, 4)` makes an opening reading `(9, 9)` and not `(10, 8)`.
            // Both numbers are wrong for a crate that assumed symmetry, and
            // only one of them belongs to an operation that is idempotent.
            format!("open, {name}"),
            Chain::op(MorphologyOp::new("open", Morphology::Open, element.clone())),
            masked.clone(),
        ));
        cases.push((
            format!("rank 1/4 then dilate, {name}"),
            Chain::sequence(vec![
                Chain::op(RankFilterOp::new(
                    "rank",
                    element.clone(),
                    Rank::Nth(element.len() / 4),
                )),
                Chain::op(MorphologyOp::new(
                    "dilate",
                    Morphology::Dilate,
                    element.clone(),
                )),
            ]),
            input.clone(),
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
    execute("asym", workflow, decomposition, &Hints::default(), &env).unwrap();
    env.output().view::<f64>().unwrap().to_owned()
}

// ------------------------------------------------------ what is derived --

/// The reaches really are asymmetric, and the symmetric integer really is only
/// the wider side. Without this the two properties below could pass while
/// testing a symmetric element by accident.
#[test]
fn an_even_element_derives_an_asymmetric_reach_and_a_symmetric_bound() {
    let element = StructuringElement::from_size(ElementShape::Box, [10, 5, 4]).unwrap();
    assert_eq!(element.sides(0), (5, 4));
    assert_eq!(element.sides(1), (2, 2));
    assert_eq!(element.sides(2), (2, 1));

    let erode = Chain::op(MorphologyOp::new(
        "erode",
        Morphology::Erode,
        element.clone(),
    ));
    let spec = erode.reach_spec(VOLUME).unwrap();
    assert_eq!(spec.as_symmetric(), None, "an even element is not a triple");
    assert_eq!(spec.at(0, 0, VOLUME[0]), (5, 4));
    assert_eq!(spec.at(2, 0, VOLUME[2]), (2, 1));
    assert_eq!(
        erode.reach3(&VOLUME),
        [5, 2, 2],
        "the bound is the wider side"
    );

    // and the composition **reflects**, so it reads `lo + hi` on both sides:
    // symmetric again, and narrower than the `(10, 8)` an unreflected repetition
    // would read. The two rules agree for every centred element, which is why
    // this is the file the difference shows up in.
    let open = Chain::op(MorphologyOp::new("open", Morphology::Open, element));
    let spec = open.reach_spec(VOLUME).unwrap();
    assert_eq!(spec.at(0, 0, VOLUME[0]), (9, 9));
    assert_eq!(spec.at(2, 0, VOLUME[2]), (3, 3));
    assert_eq!(open.reach3(&VOLUME), [9, 4, 3]);
}

// -------------------------------------------------- 1. the ordinary bar --

/// Byte-identical to the whole-volume reference, under every decomposition.
#[test]
fn an_asymmetric_element_is_decomposition_invariant() {
    let input = intensities();
    for (name, chain, source) in cases(&input) {
        let want = reference(&chain, &source);
        let workflow = workflow(chain);
        let mut ran = 0;
        for block in [4usize, 9, 64] {
            for split_axes in [vec![0], vec![2], vec![0, 1, 2]] {
                let decomposition = plan(&workflow, block, &split_axes);
                decomposition
                    .check()
                    .unwrap_or_else(|err| panic!("{name}: an honest plan must tile: {err}"));
                assert_eq!(
                    run(&workflow, &decomposition, &source),
                    want,
                    "{name}: block {block}, axes {split_axes:?} disagreed with the \
                     whole-volume reference"
                );
                ran += 1;
            }
        }
        assert!(ran >= 9, "{name}: the sweep did not run");
    }
}

// ------------------------------------- 2. the guard, on the narrow side --

/// A halo short on the **narrow** side is refused, and the exact per-side halo
/// is accepted.
///
/// The pair is the whole point. Refusing `(5, 3)` alone would also be what a
/// crate that demanded five on both sides did; accepting `(5, 4)` is what says
/// the demand is the element's actual dependency and not its bounding box.
#[test]
fn a_halo_short_on_the_narrow_side_is_refused_and_the_exact_one_is_not() {
    let input = intensities();
    let source = mask(&input, 0.35);
    let element = StructuringElement::from_size(ElementShape::Box, [10, 5, 4]).unwrap();
    let workflow = workflow(Chain::op(MorphologyOp::new(
        "erode",
        Morphology::Erode,
        element,
    )));
    let reach = workflow.chain.reach_spec(VOLUME).unwrap();
    assert_eq!(reach.at(0, 0, VOLUME[0]), (5, 4));

    // Split on two axes, because a halo can only be short where there is a
    // seam: an axis nobody cut has one block spanning it and no shortfall on it
    // is observable. That is a fact about decompositions rather than about
    // reaches, and getting it wrong here would produce a test that passed for
    // the wrong reason on the axes it did cut.
    let exact = plan_with_halo(&workflow, 8, &[0, 2], reach.clone(), reach.clone());
    exact.check().expect("the exact per-side halo must tile");
    let want = reference(&workflow.chain, &source);
    assert_eq!(run(&workflow, &exact, &source), want);

    // One short on the narrow side — the side a symmetric `max(lo, hi)` would
    // have handed five voxels of, and therefore the one it cannot notice.
    for short in [
        Reach::asymmetric([(5, 3), (2, 2), (2, 1)]),
        // and one short on the wide side, so the guard is not merely sensitive
        // to the first entry it looks at
        Reach::asymmetric([(4, 4), (2, 2), (2, 1)]),
        // and one short on the narrow side of a different axis
        Reach::asymmetric([(5, 4), (2, 2), (2, 0)]),
    ] {
        let forced = exact.with_forced_halo(short.clone());
        let err = forced
            .check()
            .expect_err(&format!("{short} must not check out"))
            .to_string();
        assert!(
            err.contains("do not tile the volume exactly"),
            "{short}: expected the tiling guard, got: {err}"
        );

        let env = ArrayEnvironment::new(source.clone().into(), 1, [4, 4, 4]).unwrap();
        let err = execute("short", &workflow, &forced, &Hints::default(), &env)
            .expect_err(&format!("{short}: the executor must refuse it"))
            .to_string();
        assert!(
            err.contains("do not tile the volume exactly"),
            "{short}: {err}"
        );
    }
}

// ------------------------------- 3. the same shortfall, made consistent --

/// Understating the narrow side by one tiles perfectly and gives wrong values.
///
/// This is what makes the guard above worth having: the dependency on the narrow
/// side is real, so a plan that quietly drops it produces a complete,
/// well-formed, wrong volume rather than an error. Searched over block edges
/// because a reach is a bound over every block placement and a particular
/// placement need not attain it — the same reason one decomposition proves
/// nothing about a halo.
///
/// **A small element on continuous data, deliberately.** A minimum over a
/// 200-voxel window of a *mask* changes only when the one voxel dropped was the
/// only clear one in the window, which is rare enough that a real shortfall can
/// hide behind it — the first draft of this test used exactly that and reported
/// "no difference" for a plan that was genuinely reading the wrong data. A
/// four-voxel window over an intensity field puts the dropped voxel in the
/// answer about a quarter of the time, so the shortfall is visible rather than
/// merely present.
#[test]
fn understating_the_narrow_side_tiles_and_changes_the_answer() {
    let input = intensities();
    let source = input.clone();
    let element = StructuringElement::from_size(ElementShape::Box, [4, 1, 1]).unwrap();
    assert_eq!(element.sides(0), (2, 1));
    let workflow = workflow(Chain::op(RankFilterOp::new(
        "lowest",
        element,
        Rank::lowest(),
    )));
    assert_eq!(
        workflow
            .chain
            .reach_spec(VOLUME)
            .unwrap()
            .at(0, 0, VOLUME[0]),
        (2, 1)
    );
    let want = reference(&workflow.chain, &source);

    let understated = Reach::asymmetric([(2, 0), (0, 0), (0, 0)]);
    let mut differed = 0;
    let mut tiled = 0;
    for block in [4usize, 6, 7, 9, 11] {
        let plan = plan_with_halo(
            &workflow,
            block,
            &[0],
            understated.clone(),
            understated.clone(),
        );
        // Self-consistent, so the structural guard has nothing to say.
        if plan.check().is_err() {
            continue;
        }
        tiled += 1;
        if run(&workflow, &plan, &source) != want {
            differed += 1;
        }
    }
    assert!(tiled >= 4, "only {tiled} understated plans tiled at all");
    assert!(
        differed > 0,
        "an understated narrow side never changed the answer, so nothing reads it and \
         the reach on that side is bigger than the dependency it declares"
    );
}
