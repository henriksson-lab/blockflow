// SPDX-License-Identifier: MIT
//
// The acceptance criterion for `assemble::PlanBuilder`, and it is one sentence:
// **a builder-assembled plan is the hand-assembled plan.** Not equivalent to it,
// not close enough to run — equal, field for field, and with the same
// fingerprint, which is the number a parity figure is attached to.
//
// That is the only bar worth setting. A `Decomposition` is the binding half of a
// plan: change a slot range, a halo, a per-phase element type or which level an
// arm reads and the output changes. A builder that produced a *different* plan
// that happened to run would be a second planner, and this crate would then have
// two of them disagreeing silently. So the tests below build the same plan twice
// — once through the builder and once the way it had to be written before the
// builder existed — and compare the plans rather than the outputs. Comparing
// outputs would pass for a plan that differed in any way the fixture is not
// sensitive to, which is most of them.
//
// The two shapes are the two the hand-assembly hurt on:
//
// * **the mixed-kind chain**: `Pixels`, `Pixels`, `Iterate`, `Fragments`,
//   `Fragments`, `Pixels` reading a level back through a source leaf,
//   `Fragments`, `Fragments`. Eight phases, three kinds, two fragment streams,
//   one source leaf, and two phases that change the element type without owning
//   a chain slot. Every one of the six responsibilities appears in it.
// * **the two-phase global op appended to a chain**, which is the shape
//   `ops::fill::fill_phases` builds as a whole plan and therefore cannot
//   contribute to a larger one.
//
// Beside them: that the plans *run* to the same answer, that a level freed
// after its last reader says so by name, and that naming the wrong phase in a
// fragment address is refused with both phases in the message.

use std::collections::BTreeSet;

use ndarray::Array3;

use blockflow::assemble::PlanBuilder;
use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::dtype::Dtype;
use blockflow::env::{ArrayEnvironment, Environment};
use blockflow::fragment::{fragment_phase, FragmentInput, FragmentOp, PhaseWork};
use blockflow::geometry::BlockGrid;
use blockflow::iterate::{iterative_phase, SubstageLimit};
use blockflow::op::Chain;
use blockflow::ops::detect::{LabelRegionsOp, RegionPointsOp};
use blockflow::ops::element::{ElementShape, StructuringElement};
use blockflow::ops::fill::{FillHolesOp, LabelBackgroundOp};
use blockflow::ops::reconstruct::HExtremaOp;
use blockflow::ops::regional::{LabelPlateauxOp, RegionalMaximaOp};
use blockflow::ops::voxelwise::{Logic, LogicCombine, VoxelwiseMapOp};
use blockflow::points::PointStore;
use blockflow::region::Region;
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints, Workflow};
use blockflow::voxels::Voxels;

const VOLUME: [usize; 3] = [16, 16, 8];
const BLOCK: [usize; 3] = [8, 8, 8];
const PLATEAUX: &str = "test.plateaux";
const MOMENTS: &str = "test.moments";
const POINTS: &str = "test.points";

/// The level the second phase writes and the sixth reads back. A constant in the
/// hand-built half because that is what it had to be: `Chain::source` took a
/// number, so "the level phase 1 wrote" was a statement made in two places.
const SECOND_LEVEL: usize = 2;

fn grid() -> BlockGrid {
    BlockGrid::new(VOLUME, BLOCK).expect("a lattice")
}

fn element() -> StructuringElement {
    StructuringElement::from_radius(ElementShape::Ellipsoid, [1, 1, 1])
}

/// Generous, because this fixture is not testing the guard: the h-maxima flood
/// walks one element radius per substage across a flat background, so the
/// geometric bound is the volume's L1 diameter and a tight number here would be
/// a test that fails for a reason that is not the builder's.
fn limit() -> SubstageLimit {
    SubstageLimit::of(256).expect("a positive limit")
}

/// A chain fragment with more than one slot and a non-zero reach, so that the
/// slot cursor and the per-phase reach are both exercised rather than being
/// trivially right.
fn first_fragment() -> Chain {
    Chain::sequence(vec![
        Chain::op(VoxelwiseMapOp::new("scale", |value| value * 2.0)),
        Chain::op(VoxelwiseMapOp::new("floor", |value| value.max(0.0))),
    ])
}

fn second_fragment() -> Chain {
    Chain::op(VoxelwiseMapOp::new("identity", |value| value))
}

/// A fan-in of the mask against the level the second phase wrote, which is the
/// slot that makes a source leaf part of the plan.
fn gate_fragment(level: impl Into<blockflow::assemble::Level>) -> Chain {
    Chain::parallel(
        vec![
            Chain::op(VoxelwiseMapOp::new("mask", |value| value)),
            Chain::sequence(vec![
                Chain::source(level, Dtype::F64),
                Chain::op(VoxelwiseMapOp::new("above", |value| {
                    if value >= 0.5 {
                        1.0
                    } else {
                        0.0
                    }
                })),
            ]),
        ],
        Box::new(LogicCombine::new("kept", Logic::And)),
    )
    .expect("a fan-in")
}

// --------------------------------------------------- the mixed-kind plan --

/// Eight phases, three kinds, built the way it had to be built before the
/// builder existed.
///
/// Kept verbatim rather than tidied: the cursors, the three name slices, the
/// `dtype` lines and the trailing `declare_source_levels` are what the builder
/// is measured against, and a tidier version of this would be measuring
/// something else.
struct HandBuilt {
    workflow: Workflow,
    decomposition: Decomposition,
    hmax: HExtremaOp,
    plateaux: LabelPlateauxOp,
    maxima: RegionalMaximaOp,
    regions: LabelRegionsOp,
    points: RegionPointsOp,
}

impl HandBuilt {
    fn assemble() -> Self {
        let grid = grid();

        let first = first_fragment();
        let second = second_fragment();
        let gate = gate_fragment(SECOND_LEVEL);
        let (n_first, first_reach) = (first.slots().len(), first.reach3(&VOLUME));
        let (n_second, second_reach) = (second.slots().len(), second.reach3(&VOLUME));
        let (n_gate, gate_reach) = (gate.slots().len(), gate.reach3(&VOLUME));

        let chain = Chain::sequence(vec![first, second, gate]);
        let chain_reach = chain.reach3(&VOLUME);
        let workflow = Workflow::new(chain, VOLUME, Dtype::F64);
        let names: Vec<String> = workflow
            .chain
            .slots()
            .iter()
            .map(|slot| slot.display_name())
            .collect();

        let hmax = HExtremaOp::maxima("hmax", element(), 0.1, limit()).expect("a transform");

        let mut phases = vec![
            PhaseDecomposition::derive(
                (0..n_first).collect(),
                names[0..n_first].to_vec(),
                first_reach,
                first_reach,
                grid.clone(),
            ),
            PhaseDecomposition::derive(
                (n_first..n_first + n_second).collect(),
                names[n_first..n_first + n_second].to_vec(),
                second_reach,
                second_reach,
                grid.clone(),
            ),
            iterative_phase(&hmax, grid.clone()).expect("an iterative phase"),
        ];

        let plateaux_phase = phases.len();
        let plateaux = LabelPlateauxOp::new("plateaux", PLATEAUX, Lifecycle::DeleteOnExit);
        let mut labelling = fragment_phase(&plateaux, grid.clone()).expect("a fragment phase");
        labelling.dtype = Some(Dtype::U32);
        phases.push(labelling);

        let maxima = RegionalMaximaOp::new("maxima", PLATEAUX, plateaux_phase, Dtype::F64, &grid);
        let mut merging = fragment_phase(&maxima, grid.clone()).expect("a fragment phase");
        merging.dtype = Some(Dtype::F64);
        phases.push(merging);

        let gate_slots = n_first + n_second;
        phases.push(PhaseDecomposition::derive(
            (gate_slots..gate_slots + n_gate).collect(),
            names[gate_slots..gate_slots + n_gate].to_vec(),
            gate_reach,
            gate_reach,
            grid.clone(),
        ));

        let regions = LabelRegionsOp::new("regions", MOMENTS, Lifecycle::DeleteOnExit);
        let moments_phase = phases.len();
        phases.push(fragment_phase(&regions, grid.clone()).expect("a fragment phase"));

        let points = RegionPointsOp::new(
            "points",
            MOMENTS,
            moments_phase,
            POINTS,
            Lifecycle::Persistent,
            &grid,
        );
        phases.push(fragment_phase(&points, grid).expect("a fragment phase"));

        let mut decomposition = Decomposition {
            volume: VOLUME,
            dtype: Dtype::F64,
            phases,
            chain_reach,
        };
        decomposition
            .declare_source_levels(&workflow.chain)
            .expect("source levels");
        decomposition.check().expect("a plan");

        Self {
            workflow,
            decomposition,
            hmax,
            plateaux,
            maxima,
            regions,
            points,
        }
    }

    fn work(&self) -> Vec<PhaseWork<'_>> {
        vec![
            PhaseWork::Pixels,
            PhaseWork::Pixels,
            PhaseWork::Iterate(&self.hmax),
            PhaseWork::Fragments(&self.plateaux),
            PhaseWork::Fragments(&self.maxima),
            PhaseWork::Pixels,
            PhaseWork::Fragments(&self.regions),
            PhaseWork::Fragments(&self.points),
        ]
    }
}

/// The same plan, assembled.
fn assembled() -> blockflow::assemble::Assembly {
    let mut plan = PlanBuilder::new(VOLUME, Dtype::F64, grid());
    plan.pixels(first_fragment()).expect("a pixel phase");
    let second = plan.pixels(second_fragment()).expect("a pixel phase");
    plan.iterate(HExtremaOp::maxima("hmax", element(), 0.1, limit()).expect("a transform"))
        .expect("an iterative phase");
    blockflow::ops::regional::append_to(&mut plan, PLATEAUX, Lifecycle::DeleteOnExit, Dtype::F64)
        .expect("the two maxima phases");
    plan.pixels(gate_fragment(second.level().expect("a level")))
        .expect("a pixel phase");
    blockflow::ops::detect::append_to(
        &mut plan,
        MOMENTS,
        Lifecycle::DeleteOnExit,
        POINTS,
        Lifecycle::Persistent,
    )
    .expect("the two point phases");
    plan.finish().expect("a plan")
}

/// **The criterion.** Field for field, and by fingerprint.
#[test]
fn the_assembled_mixed_kind_plan_is_the_hand_built_one() {
    let hand = HandBuilt::assemble();
    let built = assembled();

    assert_eq!(built.decomposition.n_phases(), 8);
    assert_eq!(
        built.decomposition, hand.decomposition,
        "the assembled plan differs from the hand-built one"
    );
    assert_eq!(
        built.decomposition.fingerprint(),
        hand.decomposition.fingerprint()
    );
    // The chain is the other half of what `execute_phases` is handed, and a
    // plan that partitions a different chain is a different program.
    assert_eq!(
        built.workflow.chain.slots().len(),
        hand.workflow.chain.slots().len()
    );
    assert_eq!(
        built.decomposition.op_names_in_order(),
        hand.decomposition.op_names_in_order()
    );

    // Item four, as a shape rather than as a length assertion: the kinds come
    // out of the phase list, so there is no second list to be out of order.
    let work = built.work();
    let expected = hand.work();
    assert_eq!(work.len(), expected.len());
    for (index, (got, want)) in work.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            got.is_fragments(),
            want.is_fragments(),
            "phase {index} kind"
        );
        assert_eq!(
            got.is_iterative(),
            want.is_iterative(),
            "phase {index} kind"
        );
    }
}

/// The plans are equal, so the runs must be. Asserted rather than assumed,
/// because equality of a `Decomposition` is a statement about the binding half
/// and the ops are the other half.
#[test]
fn the_two_plans_produce_the_same_points() {
    let hand = HandBuilt::assemble();
    let built = assembled();
    let input = input_volume();

    let from_hand = run_points(
        &hand.workflow,
        &hand.decomposition,
        &hand.work(),
        input.clone(),
    );
    let from_built = run_points(
        &built.workflow,
        &built.decomposition,
        &built.work(),
        input.clone(),
    );
    assert_eq!(from_hand, from_built);
    // and it is an answer, not two empty lists agreeing
    assert!(!from_hand.is_empty(), "the fixture found nothing at all");
}

fn input_volume() -> Voxels {
    let mut values = Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), 0.0f64);
    for (index, centre) in [[3usize, 3, 3], [11, 4, 4], [5, 12, 2]].iter().enumerate() {
        let height = 0.6 + 0.1 * index as f64;
        for di in 0..2 {
            for dj in 0..2 {
                for dk in 0..2 {
                    values[[centre[0] + di, centre[1] + dj, centre[2] + dk]] = height;
                }
            }
        }
    }
    values.into()
}

fn run_points(
    workflow: &Workflow,
    plan: &Decomposition,
    work: &[PhaseWork<'_>],
    input: Voxels,
) -> Vec<[usize; 3]> {
    let env = ArrayEnvironment::for_decomposition(input, plan, [4, 4, 4]).expect("environment");
    execute_phases(
        "builder",
        workflow,
        plan,
        &Hints::default(),
        &env,
        &[],
        work,
    )
    .expect("a run");
    let last = plan.n_phases() - 1;
    let mut store = PointStore::new(plan.volume).expect("a store");
    for core in plan.phases[last].grid.cores() {
        let bytes = env
            .read_sidecar(POINTS, last, core.index)
            .expect("a read")
            .expect("every block writes a blob");
        store.write(core.index, &bytes).expect("a write");
    }
    store.seal().expect("sealed");
    store
        .query(&Region::whole(&plan.volume))
        .expect("a query")
        .into_iter()
        .map(|point| point.at)
        .collect()
}

// ------------------------------------- the two-phase global op, appended --

/// The second real case: a chain fragment, then `ops::fill`'s two phases behind
/// it. `fill_phases` builds a whole `Decomposition`, so before `append_to` this
/// had to be its body re-derived by hand — including the two `dtype` lines.
#[test]
fn the_assembled_appended_global_op_is_the_hand_built_one() {
    let grid = grid();
    let threshold = Chain::op(VoxelwiseMapOp::new("above", |value| {
        if value >= 0.5 {
            1.0
        } else {
            0.0
        }
    }));

    let (n, reach) = (threshold.slots().len(), threshold.reach3(&VOLUME));
    let chain = Chain::sequence(vec![threshold]);
    let names: Vec<String> = chain
        .slots()
        .iter()
        .map(|slot| slot.display_name())
        .collect();
    let background = LabelBackgroundOp::new("background", PLATEAUX, Lifecycle::DeleteOnExit);
    let mut labelling = fragment_phase(&background, grid.clone()).expect("a fragment phase");
    labelling.dtype = Some(Dtype::U32);
    let fill = FillHolesOp::new("fill", PLATEAUX, 1, Dtype::Bool, &grid);
    let mut filling = fragment_phase(&fill, grid.clone()).expect("a fragment phase");
    filling.dtype = Some(Dtype::Bool);
    let mut hand = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases: vec![
            PhaseDecomposition::derive(
                (0..n).collect(),
                names[0..n].to_vec(),
                reach,
                reach,
                grid.clone(),
            ),
            labelling,
            filling,
        ],
        chain_reach: chain.reach3(&VOLUME),
    };
    hand.declare_source_levels(&chain).expect("source levels");
    hand.check().expect("a plan");

    let mut plan = PlanBuilder::new(VOLUME, Dtype::F64, grid);
    plan.pixels(Chain::op(VoxelwiseMapOp::new("above", |value| {
        if value >= 0.5 {
            1.0
        } else {
            0.0
        }
    })))
    .expect("a pixel phase");
    blockflow::ops::fill::append_to(&mut plan, PLATEAUX, Lifecycle::DeleteOnExit, Dtype::Bool)
        .expect("the two fill phases");
    let built = plan.finish().expect("a plan");

    assert_eq!(built.decomposition, hand);
    assert_eq!(built.decomposition.fingerprint(), hand.fingerprint());
}

// ------------------------------------------------ the two new refusals --

/// A level freed after its last reader must name **itself, the phase that freed
/// it, and the way to keep it**. Before this it handed back a `1 x 1 x 1`
/// placeholder, which failed somewhere else as a shape mismatch against the
/// volume — an error naming a symptom and no part of the cause.
#[test]
fn reading_a_freed_level_names_the_level_the_phase_and_the_hint() {
    let built = assembled();
    let input = input_volume();
    let env =
        ArrayEnvironment::for_decomposition(input, &built.decomposition, [4, 4, 4]).expect("env");
    execute_phases(
        "builder",
        &built.workflow,
        &built.decomposition,
        &Hints::default(),
        &env,
        &[],
        &built.work(),
    )
    .expect("a run");

    // Level 1 is written by phase 0 and read by phase 1, so phase 1's
    // completion is what frees it.
    assert!(env.is_discarded(1));
    assert_eq!(env.freed_after(1), Some(1));
    let message = env.try_level(1).expect_err("a freed level").to_string();
    assert!(message.contains("level 1"), "{message}");
    assert!(message.contains("after phase 1"), "{message}");
    assert!(message.contains("keep_levels"), "{message}");

    // And the hint is the answer: pin it and the same read succeeds.
    let env = ArrayEnvironment::for_decomposition(input_volume(), &built.decomposition, [4, 4, 4])
        .expect("env");
    execute_phases(
        "builder",
        &built.workflow,
        &built.decomposition,
        &Hints {
            keep_levels: BTreeSet::from([1usize]),
            ..Hints::default()
        },
        &env,
        &[],
        &built.work(),
    )
    .expect("a run");
    assert!(!env.is_discarded(1));
    assert_eq!(env.try_level(1).expect("a level").shape(), VOLUME);
}

/// A fragment op reads a stream **from a phase**, and naming the wrong one reads
/// the wrong generation rather than failing. Refused at plan time, with both
/// phases in the message.
#[test]
fn naming_the_wrong_producing_phase_is_refused_and_names_both() {
    let hand = HandBuilt::assemble();
    // The plateau stream is written by phase 3 and by nothing else. Address it
    // at phase 2, which is in range, is not a forward reference, is on the same
    // lattice, and is a phase that exists — so every check that was here before
    // passes it, and the run reads a stream with no fragments in it.
    let wrong = RegionalMaximaOp::new("maxima", PLATEAUX, 2, Dtype::F64, &grid());
    let work = vec![
        PhaseWork::Pixels,
        PhaseWork::Pixels,
        PhaseWork::Iterate(&hand.hmax),
        PhaseWork::Fragments(&hand.plateaux),
        PhaseWork::Fragments(&wrong),
        PhaseWork::Pixels,
        PhaseWork::Fragments(&hand.regions),
        PhaseWork::Fragments(&hand.points),
    ];
    // The plan itself is the hand-built one and is unchanged: only the address
    // inside the op moved, which is exactly the mistake that used to be
    // undetectable.
    blockflow::fragment::check_phase_work(&hand.decomposition, &hand.work()).expect("the plan");
    let message = blockflow::fragment::check_phase_work(&hand.decomposition, &work)
        .expect_err("a wrong address")
        .to_string();
    assert!(message.contains("phase 4"), "{message}");
    assert!(message.contains("phase 2"), "{message}");
    assert!(message.contains("[3]"), "{message}");
}

/// The builder path has no number to get wrong: a [`Phase`] comes from the call
/// that made the phase, and there is no constructor from a `usize`. The nearest
/// thing to a negative test that compiles is that the handle really does carry
/// the phase the op needs.
#[test]
fn a_handle_addresses_the_phase_that_produced_it() {
    let mut plan = PlanBuilder::new(VOLUME, Dtype::F64, grid());
    plan.pixels(first_fragment()).expect("a pixel phase");
    let plateaux = plan
        .fragments(LabelPlateauxOp::new(
            "plateaux",
            PLATEAUX,
            Lifecycle::DeleteOnExit,
        ))
        .expect("a fragment phase");
    assert_eq!(plateaux.index(), 1);
    let maxima = RegionalMaximaOp::reading("maxima", PLATEAUX, plateaux, Dtype::Bool, plan.grid());
    assert_eq!(
        maxima.inputs(),
        vec![FragmentInput::own(PLATEAUX, 1).with_reach(maxima.lattice())]
    );
    plan.fragments(maxima).expect("a fragment phase");
    plan.finish().expect("a plan");
}
