// SPDX-License-Identifier: MIT
//
// **Can a phase collapse an axis, and can it say so?** Both, now — and the
// second is the part that took a change.
//
// The smallest honest projection: maximum along axis 0 of an `[N, Y, X]` volume
// into `[1, Y, X]`. That output is a legal 3-D volume — a degenerate axis is how
// this crate models lower rank — so the question was never about the type
// system. It was only ever: **can the geometry declare a collapsed axis?**
//
// Four plans for one op, and the difference between them is the *declaration*:
//
// | plan | phase reach | fetch | what it is |
// |---|---|---|---|
// | truthful | `All` on axis 0 in `Space::source_voxels()` | stated per block | saying what the op means |
// | phase frame | `All` on axis 0 in `Space::phase_voxels()` | stated per block | `All` against an axis of extent 1 — vacuous |
// | escape | `none()` in `Space::source_index()` | stated per block | the `ops::lattice` escape |
// | short read | `none()` | *not* stated — the block reads its own extent | the negative control |
//
// The negative control is the point of the fixture. A projection that quietly
// reads one plane instead of the axis produces a complete, well-formed volume of
// exactly the right shape, and on a fixture whose answer happens to be uniform
// along the collapsed axis it is also *correct*. `texture` is built so that the
// maximum along axis 0 is attained on a different plane at every position and
// never on plane 0, so the wrong answer is wrong at every voxel.
//
// Why the truthful declaration could not be made, and can now
// -----------------------------------------------------------
// A reach stated in `Frame::Source` is denied the clamp exception in
// `BlockGeometry::derive_with`, on this argument: a **cropping** phase's edge is
// an interior position of the array it reads, so a neighbour exists there and a
// halo could have reached it; trusting the clamp would trust a voxel whose
// context was never fetched. Correct, and it is why the frame exists.
//
// **That reasoning does not hold for an axis the op consumes entirely.** There
// is no beyond, and so no neighbour a halo could have reached into. `All` is not
// a distance that ran off the end; it is the statement that the end is where the
// op stops. So the exception is granted per axis, on the two conditions that
// make it mean something: the reach on that axis is `AxisReach::All`, and this
// block's read spans the whole of the axis.
//
// The second condition is what keeps it honest, and it pays for itself. Where
// the axis *is* cut with a finite halo, no block spans it, every block stays
// degenerate and the tiling check fires exactly as it did before — so the
// declaration is now the **whole-axis mandate it always implied**: a plan may
// leave the axis whole, or grant a whole-axis halo, and there is no third
// option. That is a free partial answer to **G9** in `docs/ops-survey/README.md`'s
// register (`BlockConstraint::FullExtent(axis)`), obtained without a constraint
// type — declared by the op rather than configured by the planner, and enforced
// by the guard that already ran.
//
// What that does *not* do is check that the block fetched the axis it declared.
// A declaration nothing compares against the fetch is decoration: a projection
// that reads only its own block, and one that fetches half the axis, are both
// well-formed and wrong at every position. So `Decomposition::check` compares
// them, and that is what makes the truthful declaration worth more than the
// `Space::source_index()` escape rather than merely as much — the escape records
// *that* a dependency exists, this records **what would satisfy it**.
//
// One thing the change does not alter, recorded here because it is the reason
// the phase-frame plan is not an answer: `All` stated in the phase's own frame
// plans, and is **vacuous** on a collapsed axis. `AxisReach::is_whole` requires
// `extent > 1`, so against an axis of extent 1 the words are accepted without
// being a statement of anything.

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::{BlockCore, BlockGeometry, BlockGrid};
use blockflow::op::{Anchor, BlockOp, Chain};
use blockflow::reach::{AxisReach, Frame, Reach, Space};
use blockflow::region::Region;
use blockflow::strategy::{execute, Hints, Workflow};
use blockflow::voxels::Voxels;
use blockflow::Dtype;
use ndarray::Array3;
use std::cmp::Ordering;

/// Deliberately not a round number on any axis, so a block edge divides none of
/// them unless it was chosen to.
const VOLUME: [usize; 3] = [11, 13, 17];

/// What the projection writes: one plane, the free axes untouched.
const COLLAPSED: [usize; 3] = [1, VOLUME[1], VOLUME[2]];

/// Ordering by `total_cmp`, never `f64::max`: a maximum that disagrees with the
/// reference on one representation of one value would make every parity
/// assertion below a statement about `f64::max` instead of about the plan.
fn larger(a: f64, b: f64) -> f64 {
    match a.total_cmp(&b) {
        Ordering::Less => b,
        _ => a,
    }
}

// --------------------------------------------------------------- the op --

/// Maximum along axis 0. Reads the whole of that axis, writes `[1, Y, X]`.
///
/// The reach is stated in whichever [`Space`] the case under test wants, because
/// the space *is* the experiment: the arithmetic below is identical in all of
/// them and only the declaration differs.
struct MaxAlongFirstAxis {
    space: Space,
}

impl MaxAlongFirstAxis {
    fn new(space: Space) -> Self {
        Self { space }
    }
}

impl BlockOp for MaxAlongFirstAxis {
    fn name(&self) -> &'static str {
        "max along axis 0"
    }

    /// The symmetric bound `reach_spec` has to stay inside. On the collapsed
    /// axis that is the whole extent, which is the number `AxisReach::All` means
    /// and the reason the two agree.
    fn reach(&self, axis: usize, volume_len: usize) -> usize {
        if axis == 0 {
            volume_len
        } else {
            0
        }
    }

    fn reach_spec(&self, _volume: [usize; 3]) -> Reach {
        Reach::per_axis([AxisReach::All, AxisReach::none(), AxisReach::none()]).in_space(self.space)
    }

    fn output_shape(&self, input: [usize; 3]) -> [usize; 3] {
        [1, input[1], input[2]]
    }

    fn apply(
        &self,
        input: &Voxels,
        out: &mut Voxels,
        _at: &Anchor,
    ) -> Result<(), blockflow::Error> {
        let source = input.view::<f64>()?;
        let mut target = out.view_mut::<f64>()?;
        let shape = input.shape();
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let mut best = f64::NEG_INFINITY;
                for i in 0..shape[0] {
                    best = larger(best, source[[i, j, k]]);
                }
                target[[0, j, k]] = best;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------- the fixture --

/// A texture whose maximum along axis 0 is attained on a different plane at
/// every `(y, x)` and **never** on plane 0.
///
/// Plane 0 is pushed to a floor below every other plane, so a projection that
/// reads only its own extent on the collapsed axis — the default fetch, which is
/// the thing the negative control exists to catch — is wrong at every one of the
/// `13 x 17` positions rather than at a few.
fn texture(shape: [usize; 3]) -> Array3<f64> {
    Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
        if i == 0 {
            -1000.0
        } else {
            ((i * 7919 + j * 104729 + k * 1299709) % 1013) as f64 + 1.0
        }
    })
}

/// The whole-volume reference, written out rather than asked of the op: for each
/// `(y, x)`, the maximum over the whole of axis 0 of the *whole* array.
fn reference(input: &Array3<f64>) -> Array3<f64> {
    let shape = input.shape();
    Array3::from_shape_fn((1, shape[1], shape[2]), |(_, j, k)| {
        (0..shape[0]).fold(f64::NEG_INFINITY, |best, i| larger(best, input[[i, j, k]]))
    })
}

/// Where the maximum sits, per position — used only to assert the fixture is the
/// trap it claims to be.
fn argmax_planes(input: &Array3<f64>) -> Vec<usize> {
    let shape = input.shape();
    let mut planes = Vec::new();
    for j in 0..shape[1] {
        for k in 0..shape[2] {
            let mut best = (0usize, f64::NEG_INFINITY);
            for i in 0..shape[0] {
                if input[[i, j, k]].total_cmp(&best.1) == Ordering::Greater {
                    best = (i, input[[i, j, k]]);
                }
            }
            planes.push(best.0);
        }
    }
    planes
}

// ------------------------------------------------------------ the plans --

/// The grid a collapsing phase is cut from: **the volume it writes**, which has
/// extent 1 on the collapsed axis. Cutting the axis it consumes is not on offer;
/// cutting the two free ones is the whole question of decomposition invariance.
fn collapsed_grid(block: [usize; 3]) -> BlockGrid {
    BlockGrid::new(COLLAPSED, block).unwrap()
}

/// The fetch every one of these plans states, and the only honest one: the whole
/// of the source's axis 0, at this block's own range on the free axes.
fn whole_axis_fetch(block: &BlockGeometry) -> Region {
    Region::new(
        &[0, block.read.start[1], block.read.start[2]],
        &[VOLUME[0], block.read.shape[1], block.read.shape[2]],
    )
}

/// The same fetch, stopping halfway up the axis it claims to consume.
fn half_axis_fetch(block: &BlockGeometry) -> Region {
    Region::new(
        &[0, block.read.start[1], block.read.start[2]],
        &[VOLUME[0] / 2, block.read.shape[1], block.read.shape[2]],
    )
}

fn plan_with(
    reach: Reach,
    block: [usize; 3],
    sources: Option<fn(&BlockGeometry) -> Region>,
) -> Decomposition {
    let phase = PhaseDecomposition::derive(
        vec![0],
        vec!["max along axis 0".to_string()],
        reach,
        Reach::none(),
        collapsed_grid(block),
    );
    let phase = match sources {
        Some(map) => phase.with_sources(map),
        None => phase,
    };
    Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases: vec![phase],
        chain_reach: [0, 0, 0],
    }
}

/// The truthful declaration: whole-axis reach on the collapsed axis, in the
/// frame the numbers are actually measured against — the image below.
fn truthful_reach() -> Reach {
    Reach::per_axis([AxisReach::All, AxisReach::none(), AxisReach::none()])
        .in_space(Space::source_voxels())
}

/// The same words in the phase's own frame, where the axis has extent 1.
fn phase_frame_reach() -> Reach {
    Reach::per_axis([AxisReach::All, AxisReach::none(), AxisReach::none()])
}

/// What `ops::lattice` declares: nothing, in the space that says the dependency
/// lives in the fetch region instead. The *marked* zero.
fn escape_reach() -> Reach {
    Reach::none().in_space(Space::source_index())
}

/// The three declarations that plan, with the space each op must state to match.
fn declarations() -> [(&'static str, Reach, Space); 3] {
    [
        ("truthful", truthful_reach(), Space::source_voxels()),
        ("escape", escape_reach(), Space::source_index()),
        ("phase frame", phase_frame_reach(), Space::phase_voxels()),
    ]
}

fn run(chain: Chain, input: &Voxels, plan: &Decomposition) -> Voxels {
    let workflow = Workflow::new(chain, input.shape(), input.dtype());
    let env = ArrayEnvironment::for_decomposition(input.clone(), plan, [1, 1, 1]).unwrap();
    execute("collapse", &workflow, plan, &Hints::default(), &env).unwrap();
    env.output()
}

fn chain(space: Space) -> Chain {
    Chain::op(MaxAlongFirstAxis::new(space))
}

// ----------------------------------- 1. does the truthful declaration plan? --

/// **Yes.** `AxisReach::All` in `Space::source_voxels()` is granted the clamp
/// exception on the axis it consumes, so the one block keeps its whole core and
/// the plan checks.
///
/// The inverse of what this file recorded when it was an experiment, and the
/// assertion that pins the change: this used to leave every block with an empty
/// valid region and refuse with the tiling message.
#[test]
fn the_truthful_declaration_plans() {
    let plan = plan_with(truthful_reach(), COLLAPSED, Some(whole_axis_fetch));
    let phase = &plan.phases[0];
    assert_eq!(
        phase.blocks[0].valid, phase.blocks[0].core,
        "the one block's valid region is its whole core"
    );
    assert!(
        phase.blocks_missing_valid_core().is_empty(),
        "no block loses part of its core"
    );
    plan.check().unwrap();
}

/// And **no halo is needed to rescue it**, which is what makes the grant a
/// property of the declaration rather than of the configuration.
///
/// The read already spans the axis — it is one plane long in this phase's own
/// volume — so the exception applies at every halo, including none.
#[test]
fn every_halo_satisfies_a_whole_axis_reach_on_a_consumed_axis() {
    for halo in [
        Reach::none(),
        Reach::symmetric([4, 0, 0]),
        Reach::symmetric([64, 64, 64]),
        Reach::all(),
    ] {
        let grid = collapsed_grid(COLLAPSED);
        let phase = PhaseDecomposition::derive(
            vec![0],
            vec!["max along axis 0".to_string()],
            truthful_reach(),
            halo.clone(),
            grid,
        );
        assert_eq!(
            phase.blocks[0].valid, phase.blocks[0].core,
            "halo {halo} should not have mattered"
        );
    }
}

/// The same statement one level down and away from the projection, on an axis of
/// any extent: **a whole-axis reach in the source frame is a whole-axis
/// mandate.** A block whose read spans the axis keeps its core; a block that
/// only reads part of it keeps nothing, at every extent and every edge.
///
/// This is the shape of the partial answer to G9. Nobody configured a
/// constraint: the op declared what it consumes, and a lattice that cuts the
/// axis without a halo to match is refused by the guard that was already there.
#[test]
fn a_whole_axis_reach_in_the_source_frame_is_a_whole_axis_mandate() {
    for extent in [1usize, 2, 5, 32] {
        for edge in [1usize, 2, 3, extent] {
            let edge = edge.min(extent);
            let volume = [extent, 4, 4];
            let grid = BlockGrid::new(volume, [edge, 4, 4]).unwrap();
            let reach = truthful_reach();
            for core in grid.cores() {
                // Left whole by the halo: every block reads the whole axis.
                let spanning = BlockGeometry::derive_with(&core, volume, &Reach::all(), &reach);
                assert_eq!(
                    spanning.read.start[0], 0,
                    "extent {extent} edge {edge} block {:?}",
                    core.index
                );
                assert_eq!(spanning.read.shape[0], extent);
                assert_eq!(
                    spanning.valid, spanning.core,
                    "extent {extent} edge {edge} block {:?}",
                    core.index
                );

                // Cut, with nothing granted: the block reads its core only, and
                // spans the axis exactly when the lattice never cut it.
                let cut = BlockGeometry::derive_with(&core, volume, &Reach::none(), &reach);
                if edge == extent {
                    assert_eq!(
                        cut.valid, cut.core,
                        "extent {extent} uncut block {:?}",
                        core.index
                    );
                } else {
                    assert_eq!(
                        cut.valid.shape,
                        vec![0, 0, 0],
                        "extent {extent} edge {edge} block {:?}",
                        core.index
                    );
                }
            }
        }
    }
}

/// The op's own declaration is *not* what ever refused it. `Chain::reach_spec`
/// accepts `AxisReach::All` on one axis against its symmetric bound, in every
/// space. What decided the plan was the geometry, and only in the source frame.
#[test]
fn the_ops_declaration_is_accepted_in_every_space() {
    for space in [
        Space::phase_voxels(),
        Space::source_voxels(),
        Space::source_index(),
    ] {
        let spec = chain(space).reach_spec(VOLUME).unwrap();
        assert_eq!(spec.axis(0), &AxisReach::All, "{space:?}");
        assert!(spec.is_whole_axis(0, VOLUME[0]), "{space:?}");
        assert!(!spec.is_whole_axis(1, VOLUME[1]), "{space:?}");
    }
}

// -------------------------------------- 2. does anything plan and run? --

/// **All three of them.** The truthful declaration, the `ops::lattice` escape,
/// and the same words in the phase's own frame. All plan, all run, all give the
/// whole-volume answer.
#[test]
fn all_three_declarations_plan_and_are_right() {
    let fine = texture(VOLUME);
    let input: Voxels = fine.clone().into();
    let expected: Voxels = reference(&fine).into();
    assert_eq!(expected.shape(), COLLAPSED);

    for (label, reach, space) in declarations() {
        let plan = plan_with(reach, COLLAPSED, Some(whole_axis_fetch));
        plan.check().unwrap_or_else(|err| panic!("{label}: {err}"));
        let got = run(chain(space), &input, &plan);
        assert_eq!(got.shape(), COLLAPSED, "{label}");
        assert_eq!(got, expected, "{label}");
    }
}

// ---------------------------- 3. is it decomposition-invariant? --

/// Byte-identical to the whole-volume reference at every cut of the two free
/// axes, including edges that divide neither extent and a one-voxel block.
///
/// `13 x 17` with edges 1, 2, 3, 4, 5, 6, 8 and 13: 1 divides both, 13 divides
/// one exactly and leaves a short last block on the other, and 6 and 8 divide
/// neither. Twenty-five cuts, three declarations.
#[test]
fn the_collapse_is_decomposition_invariant_across_the_free_axes() {
    let fine = texture(VOLUME);
    let input: Voxels = fine.clone().into();
    let expected: Voxels = reference(&fine).into();

    for edge_y in [1usize, 2, 5, 6, 13] {
        for edge_x in [1usize, 3, 4, 8, 17] {
            for (label, reach, space) in declarations() {
                let plan = plan_with(reach, [1, edge_y, edge_x], Some(whole_axis_fetch));
                plan.check()
                    .unwrap_or_else(|err| panic!("{label} {edge_y}x{edge_x}: {err}"));
                let got = run(chain(space), &input, &plan);
                assert_eq!(got, expected, "{label}, block [1, {edge_y}, {edge_x}]");
            }
        }
    }
}

/// The cuts above really are cuts: the grid holds more than one block and the
/// last block on each axis is short where the edge does not divide the extent.
#[test]
fn the_free_axis_cuts_are_real_cuts() {
    for (label, reach, _) in declarations() {
        let plan = plan_with(reach, [1, 6, 8], Some(whole_axis_fetch));
        let phase = &plan.phases[0];
        assert_eq!(phase.grid.blocks_per_axis(), [1, 3, 3], "{label}");
        assert_eq!(phase.blocks.len(), 9, "{label}");
        let last = phase.blocks.last().unwrap();
        assert_eq!(
            last.core.shape,
            vec![1, 1, 1],
            "{label}: 13 = 6+6+1, 17 = 8+8+1"
        );
        // And every block fetches the whole of the collapsed axis, not part of it.
        for block in &phase.blocks {
            assert_eq!(block.source.start[0], 0, "{label}");
            assert_eq!(block.source.shape[0], VOLUME[0], "{label}");
        }
    }
}

// --------------------------------------------- 4. the negative control --

/// **A projection that reads only its own extent on the collapsed axis**, with
/// the dependency declared. Refused by name.
///
/// This is the plan every other guard accepts: the phase's volume has extent 1
/// on axis 0, so the block's read is one plane, the default fetch is that plane,
/// the op's declared output shape matches, and the valid regions tile. Only the
/// declaration says otherwise — and only because something now compares it
/// against the fetch.
#[test]
fn a_projection_that_reads_only_its_own_extent_is_refused_by_name() {
    let plan = plan_with(truthful_reach(), COLLAPSED, None);
    // Everything except the claim is in order: the block keeps its whole core.
    assert!(plan.phases[0].blocks_missing_valid_core().is_empty());

    let err = plan.check().unwrap_err().to_string();
    assert!(err.contains("the whole of axis 0 of image 0"), "{err}");
    assert!(err.contains("fetches 0..1 of that axis"), "{err}");
    assert!(err.contains("the whole of it is 0..11"), "{err}");
    assert!(err.contains("PhaseDecomposition::with_sources"), "{err}");
}

/// A fetch covering **half** the collapsed axis is the same fault with the fetch
/// region stated, and it is refused in the same words — with the half it did
/// fetch in them.
///
/// The contrast is the argument for the whole change. The escape and the
/// phase-frame declaration accept this plan, because neither says anything a
/// fetch could fail to meet: the escape records *that* a dependency exists, and
/// `All` against an axis of extent 1 records nothing at all. Both then run and
/// return the wrong numbers.
#[test]
fn a_fetch_covering_half_the_collapsed_axis_is_refused_by_name() {
    let plan = plan_with(truthful_reach(), COLLAPSED, Some(half_axis_fetch));
    let err = plan.check().unwrap_err().to_string();
    assert!(err.contains("the whole of axis 0 of image 0"), "{err}");
    assert!(err.contains("fetches 0..5 of that axis"), "{err}");
    assert!(err.contains("the whole of it is 0..11"), "{err}");

    let fine = texture(VOLUME);
    let input: Voxels = fine.clone().into();
    let expected: Voxels = reference(&fine).into();
    for (label, reach, space) in [
        ("escape", escape_reach(), Space::source_index()),
        ("phase frame", phase_frame_reach(), Space::phase_voxels()),
    ] {
        let plan = plan_with(reach, COLLAPSED, Some(half_axis_fetch));
        plan.check()
            .unwrap_or_else(|err| panic!("{label} has nothing to check against: {err}"));
        let got = run(chain(space), &input, &plan);
        assert_ne!(got, expected, "{label}: half the axis is not the axis");
    }
}

/// The fault the declaration exists to catch, with nothing declared: the plan
/// still checks, still runs, and is wrong at **every** position.
///
/// Undeclared is the one case no guard can reach — a phase that says it reaches
/// nothing and reaches nothing is self-consistent, and the fetch it is compared
/// against is the one it stated. What the change buys is that an op which means
/// the whole axis has somewhere to say so and is held to it; it does not make
/// silence into a claim.
#[test]
fn an_undeclared_short_read_runs_and_is_wrong_at_every_position() {
    let fine = texture(VOLUME);
    let input: Voxels = fine.clone().into();
    let expected: Voxels = reference(&fine).into();

    // The fixture is the trap: the maximum is never on plane 0 and moves.
    let planes = argmax_planes(&fine);
    assert!(planes.iter().all(|&plane| plane != 0), "never plane 0");
    assert!(
        planes
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            > 1,
        "and not one plane everywhere either"
    );

    let plan = plan_with(Reach::none(), COLLAPSED, None);
    plan.check()
        .expect("nothing in the plan says this fetch is short");
    let got = run(chain(Space::phase_voxels()), &input, &plan);
    assert_eq!(got.shape(), COLLAPSED, "the right shape");
    assert_ne!(got, expected, "and the wrong answer");

    // Wrong at *every* position, so a test on this fixture cannot pass by luck.
    let wrong = got.view::<f64>().unwrap();
    let right = expected.view::<f64>().unwrap();
    for j in 0..COLLAPSED[1] {
        for k in 0..COLLAPSED[2] {
            assert_ne!(wrong[[0, j, k]], right[[0, j, k]], "at ({j}, {k})");
        }
    }
}

/// The same trap on a fixture whose answer is uniform along the collapsed axis:
/// the short read is **indistinguishable from correct**. This is why the fixture
/// above is built the way it is, and it is the reason a projection needs a check
/// on the fetch rather than a test on convenient data.
#[test]
fn on_a_uniform_fixture_the_short_read_looks_right() {
    let flat = Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(_, j, k)| {
        (j * 31 + k * 7) as f64
    });
    let input: Voxels = flat.clone().into();
    let expected: Voxels = reference(&flat).into();

    let plan = plan_with(Reach::none(), COLLAPSED, None);
    let got = run(chain(Space::phase_voxels()), &input, &plan);
    assert_eq!(got, expected, "the wrong plan agrees, on the wrong fixture");
}

// ------------------------------------------- 5. what the frame decides --

/// The clamp exception in `BlockGeometry::derive_with`, exercised directly with
/// everything else held equal. On the axis the reach consumes, the frame is no
/// longer the difference — and on every other axis it still is.
#[test]
fn the_consumed_axis_is_the_exception_and_nothing_else_is() {
    let grid = collapsed_grid(COLLAPSED);
    let core: BlockCore = grid.cores().into_iter().next().unwrap();

    assert!(Space::phase_voxels().clamp_is_an_edge());
    assert!(!Space::source_voxels().clamp_is_an_edge());
    assert_eq!(Space::source_voxels().frame, Frame::Source);

    // Consumed: `All` on axis 0, and the block's read spans axis 0.
    let whole = Reach::per_axis([AxisReach::All, AxisReach::none(), AxisReach::none()]);
    let phase_frame = BlockGeometry::derive_with(&core, COLLAPSED, &Reach::none(), &whole);
    let source_frame = BlockGeometry::derive_with(
        &core,
        COLLAPSED,
        &Reach::none(),
        &whole.clone().in_space(Space::source_voxels()),
    );
    assert_eq!(phase_frame.read, source_frame.read, "same read extent");
    assert_eq!(phase_frame.valid, phase_frame.core, "granted the exception");
    assert_eq!(source_frame.valid, source_frame.core, "granted it too");

    // Not consumed: a bounded reach on a free axis, in the same two frames. The
    // phase frame trusts its clamped edges; the source frame does not, because
    // there a neighbour a halo could have reached really does exist, and the
    // block loses the plane at each end that had context it never fetched.
    let bounded = Reach::from([0, 1, 0]);
    let trusted = BlockGeometry::derive_with(&core, COLLAPSED, &Reach::none(), &bounded);
    let untrusted = BlockGeometry::derive_with(
        &core,
        COLLAPSED,
        &Reach::none(),
        &bounded.in_space(Space::source_voxels()),
    );
    assert_eq!(trusted.read, untrusted.read, "same read extent");
    assert_eq!(trusted.valid, trusted.core, "granted the exception");
    assert_ne!(untrusted.valid, untrusted.core, "denied it");
    assert_eq!(untrusted.valid.start[1], 1, "the bottom plane is dropped");
    assert_eq!(
        untrusted.valid.shape[1],
        COLLAPSED[1] - 2,
        "and the top one"
    );
}

/// And the phase-frame declaration is vacuous where the source-frame one is now
/// a claim: against an axis of extent 1, `AxisReach::All` is not even a planning
/// barrier, so it is accepted without being a statement of the dependency.
#[test]
fn a_whole_axis_reach_against_an_extent_of_one_is_not_a_barrier() {
    let reach = phase_frame_reach();
    assert!(!reach.is_whole_axis(0, COLLAPSED[0]), "extent 1");
    assert!(reach.is_whole_axis(0, VOLUME[0]), "extent 11");
    assert!(!reach.is_barrier(COLLAPSED));
}

// ------------------------------- 6. the mandate, on an axis that is not --
//                                    collapsed but is consumed

/// An axis a phase keeps and consumes: the same declaration, on a grid that
/// leaves the axis whole. It plans without a halo, because the block already
/// reads everything there is.
#[test]
fn an_uncut_consumed_axis_plans_without_a_halo() {
    let volume = [11, 4, 4];
    let phase = PhaseDecomposition::derive(
        vec![0],
        vec!["sweep".to_string()],
        truthful_reach(),
        Reach::none(),
        BlockGrid::new(volume, [11, 2, 2]).unwrap(),
    );
    assert!(phase.blocks.len() > 1, "the free axes are still cut");
    for block in &phase.blocks {
        assert_eq!(block.valid, block.core, "block {:?}", block.index);
    }
    Decomposition {
        volume,
        dtype: Dtype::F64,
        phases: vec![phase],
        chain_reach: [0, 0, 0],
    }
    .check()
    .unwrap();
}

/// And cutting it is refused — which is the mandate, and the partial answer to
/// G9. A finite halo leaves every block short of the axis, every block
/// degenerate, and the tiling check fires exactly as it does for a short halo,
/// because that is what a short halo is here.
///
/// The one way a cut axis stays honest is a whole-axis halo: every block then
/// reads the whole of the axis, spans it, and keeps its core. Redundant, and
/// correct — the cost model's problem rather than the guard's.
#[test]
fn cutting_a_consumed_axis_is_refused_unless_the_halo_spans_it() {
    let volume = [11, 4, 4];
    let cut = |halo: Reach| {
        let phase = PhaseDecomposition::derive(
            vec![0],
            vec!["sweep".to_string()],
            truthful_reach(),
            halo,
            BlockGrid::new(volume, [4, 4, 4]).unwrap(),
        );
        Decomposition {
            volume,
            dtype: Dtype::F64,
            phases: vec![phase],
            chain_reach: [0, 0, 0],
        }
    };

    for halo in [Reach::none(), Reach::symmetric([3, 0, 0])] {
        let plan = cut(halo.clone());
        for block in &plan.phases[0].blocks {
            assert_eq!(
                block.valid.shape,
                vec![0, 0, 0],
                "halo {halo}, block {:?}",
                block.index
            );
        }
        let err = plan.check().unwrap_err().to_string();
        assert!(
            err.contains("valid regions do not tile the volume exactly"),
            "halo {halo}: {err}"
        );
    }

    let spanning = cut(Reach::all());
    for block in &spanning.phases[0].blocks {
        assert_eq!(block.valid, block.core, "block {:?}", block.index);
        assert_eq!(block.read.start[0], 0);
        assert_eq!(block.read.shape[0], volume[0]);
    }
    spanning.check().unwrap();
}
