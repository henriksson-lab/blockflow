// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The acceptance suite for a reach that is more than one integer per axis.
//
// The bar is the one this project already uses, and "the tests pass" is not it:
//
// 1. **Decomposition invariance.** An operation whose dependency is genuinely
//    one-sided is run at several block sizes and split axes and asserted
//    byte-identical to a whole-volume reference computed by the same kernel. A
//    run that agrees at one decomposition is exactly how a bad halo passes — and
//    an asymmetric halo applied to the wrong side is a bad halo that a symmetric
//    test could never see, because a symmetric halo is right on both sides by
//    accident.
// 2. **The guard seen to fire, on every new form.** Asymmetric, per-block and
//    whole-axis halos are each forced below the phase's reach and the tiling
//    check is watched failing. A guard nobody has watched fail is not known to
//    work.
// 3. **The silent version of the same failure**, so that what the structural
//    guard is protecting is visible: a phase that declares the dependency on the
//    wrong side tiles perfectly and produces wrong values.
//
// The operation is a one-sided running sum: `out[v] = in[v-2] + in[v-1] + in[v]`
// along axis 0, clamped at the array's start. Nothing about it is symmetric, so
// a halo granted on the wrong side is wrong *values*, not merely wasted reads.

use ndarray::Array3;

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::{BlockCore, BlockGeometry, BlockGrid};
use blockflow::op::{Anchor, BlockOp, Chain};
use blockflow::reach::{AxisReach, Frame, Reach, Space, Units};
use blockflow::strategy::{execute, Hints, Strategy, Workflow};
use blockflow::voxels::Voxels;
use blockflow::{Dtype, Region};

const VOLUME: [usize; 3] = [32, 6, 5];

/// `out[v] = in[v - back] + ... + in[v]` along axis 0.
///
/// The dependency is `back` below and **nothing** above, and the op is written
/// against the buffer it is handed with the anchor saying where that buffer
/// sits — so a block whose read starts inside the volume knows it, and a block
/// at the start of the volume clamps because there is nothing before it.
struct TrailingSumOp {
    back: usize,
}

impl BlockOp for TrailingSumOp {
    fn name(&self) -> &'static str {
        "trailing-sum"
    }

    /// The symmetric bound: the widest side. Kept honest — it is what a caller
    /// holding one integer per axis gets, and it must remain a bound on the
    /// statement below.
    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        if axis == 0 {
            self.back
        } else {
            0
        }
    }

    /// The dependency as it is: `back` below the written voxel and none above.
    fn reach_spec(&self, _volume: [usize; 3]) -> Reach {
        Reach::asymmetric([(self.back, 0), (0, 0), (0, 0)])
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, at: &Anchor) -> blockflow::Result<()> {
        let source = input.view::<f64>()?;
        let mut target = out.view_mut::<f64>()?;
        let offset = at.offset[0];
        let shape = input.shape();
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    // How far back this voxel may look: `back`, or the distance
                    // to the start of the volume, whichever is smaller.
                    let global = offset + i;
                    let reach = self.back.min(global).min(i);
                    let mut total = 0.0;
                    for step in 0..=reach {
                        total += source[[i - step, j, k]];
                    }
                    target[[i, j, k]] = total;
                }
            }
        }
        Ok(())
    }
}

fn intensities() -> Array3<f64> {
    let mut array = Array3::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    for i in 0..VOLUME[0] {
        for j in 0..VOLUME[1] {
            for k in 0..VOLUME[2] {
                array[[i, j, k]] = (i * 7 + j * 13 + k * 31) as f64 % 11.0;
            }
        }
    }
    array
}

fn workflow(back: usize) -> Workflow {
    Workflow::new(Chain::op(TrailingSumOp { back }), VOLUME, Dtype::F64)
}

fn plan(workflow: &Workflow, block: usize, split_axes: &[usize], reach: Reach) -> Decomposition {
    let slots = workflow.chain.slots();
    let names: Vec<String> = slots.iter().map(|slot| slot.display_name()).collect();
    let grid = BlockGrid::along(VOLUME, split_axes, block).unwrap();
    let chain_reach = reach.bound(VOLUME);
    let phase = PhaseDecomposition::derive(
        (0..slots.len()).collect(),
        names,
        reach.clone(),
        reach,
        grid,
    );
    Decomposition {
        volume: VOLUME,
        dtype: workflow.dtype,
        phases: vec![phase],
        chain_reach,
    }
}

fn reference(workflow: &Workflow, input: &Array3<f64>) -> Array3<f64> {
    let source: Voxels = input.clone().into();
    let mut out = Voxels::zeros(Dtype::F64, VOLUME).unwrap();
    workflow
        .chain
        .apply(&source, &mut out, &Anchor::whole(VOLUME))
        .expect("the whole-volume reference must run");
    out.view::<f64>().unwrap().to_owned()
}

fn run(workflow: &Workflow, decomposition: &Decomposition, input: &Array3<f64>) -> Array3<f64> {
    let env =
        ArrayEnvironment::new(input.clone().into(), decomposition.n_phases(), [4, 4, 4]).unwrap();
    execute("reach", workflow, decomposition, &Hints::default(), &env).unwrap();
    env.output().view::<f64>().unwrap().to_owned()
}

// ------------------------------------------------- decomposition invariance --

/// The bar. A one-sided dependency, stated one-sided, at eight decompositions,
/// each byte-identical to the whole-volume reference.
///
/// What this catches that a symmetric test cannot: an asymmetric halo applied to
/// the wrong side. A symmetric halo covers both sides, so a `lo`/`hi` mix-up in
/// `BlockGeometry::derive` is invisible under one; here the seam is one-sided
/// and the values diverge the moment it is.
#[test]
fn a_one_sided_dependency_gives_the_same_volume_at_every_decomposition() {
    let input = intensities();
    let workflow = workflow(2);
    let want = reference(&workflow, &input);
    let spec = workflow.chain.reach_spec(VOLUME).unwrap();
    assert_eq!(spec.as_symmetric(), None, "the statement is not a triple");

    let mut fetched = Vec::new();
    for block in [4, 8, 16, 32] {
        for axes in [&[0usize][..], &[0, 1][..]] {
            let decomposition = plan(&workflow, block, axes, spec.clone());
            decomposition.check().unwrap();
            let got = run(&workflow, &decomposition, &input);
            assert_eq!(got, want, "block {block} axes {axes:?}");
            fetched.push(
                decomposition.phases[0]
                    .blocks
                    .iter()
                    .map(|geometry| geometry.read.voxels())
                    .sum::<usize>(),
            );
        }
    }

    // And the same run with the dependency declared symmetrically is correct
    // too — and reads more. That is the whole measured motivation, at the size
    // of a test rather than of a lattice.
    let symmetric = plan(&workflow, 4, &[0], Reach::from([2, 0, 0]));
    symmetric.check().unwrap();
    assert_eq!(run(&workflow, &symmetric, &input), want);
    let symmetric_reads: usize = symmetric.phases[0]
        .blocks
        .iter()
        .map(|geometry| geometry.read.voxels())
        .sum();
    assert!(
        symmetric_reads > fetched[0],
        "one-sided read {} voxels, symmetric read {symmetric_reads}",
        fetched[0]
    );
}

/// The silent failure the guard exists for: declared on the wrong side, the plan
/// tiles perfectly and the values are wrong.
///
/// `(0, 2)` says "I read two voxels *after* the one I write", which is the exact
/// opposite of the truth. Every valid region still covers its core, `check`
/// passes, the executor is happy — and the answer is wrong at every seam. This
/// is why the structural guard is worth having and why the reach must never be
/// derived from the halo.
#[test]
fn a_dependency_declared_on_the_wrong_side_tiles_and_lies() {
    let input = intensities();
    let workflow = workflow(2);
    let want = reference(&workflow, &input);
    let backwards = plan(
        &workflow,
        4,
        &[0],
        Reach::asymmetric([(0, 2), (0, 0), (0, 0)]),
    );
    backwards.check().expect("it tiles: that is the point");
    for block in &backwards.phases[0].blocks {
        assert!(block.valid_covers_core());
    }
    assert_ne!(run(&workflow, &backwards, &input), want);
}

// ------------------------------------------------------ the guard firing --

/// The tiling check fires on a short halo of **every** new form, not only on the
/// symmetric one it was written against.
///
/// Each case grants less than the phase's reach in a different shape: one side
/// of an asymmetric halo, one block of a per-block halo, and a whole-axis reach
/// against a bounded halo. `with_forced_halo` is the provocation, which is what
/// it exists for.
#[test]
fn the_halo_guard_fires_on_every_new_form_of_short_halo() {
    let workflow = workflow(2);
    let spec = workflow.chain.reach_spec(VOLUME).unwrap();
    let good = plan(&workflow, 4, &[0], spec);
    good.check().unwrap();

    let cases: Vec<(&str, Reach)> = vec![
        // the low side, which is the side this op actually depends on
        ("asymmetric", Reach::asymmetric([(1, 0), (0, 0), (0, 0)])),
        // generous everywhere except block 3, which is where it fires
        (
            "per-block",
            Reach::per_axis([
                AxisReach::PerBlock(vec![
                    (2, 2),
                    (2, 2),
                    (2, 2),
                    (0, 2),
                    (2, 2),
                    (2, 2),
                    (2, 2),
                    (2, 2),
                ]),
                AxisReach::none(),
                AxisReach::none(),
            ]),
        ),
        ("nothing at all", Reach::from([0, 0, 0])),
    ];
    for (what, halo) in cases {
        let short = good.with_forced_halo(halo);
        let message = short
            .check()
            .expect_err(&format!("{what}: a short halo must not check"))
            .to_string();
        assert!(
            message.contains("do not tile the volume exactly"),
            "{what}: {message}"
        );
    }

    // A whole-axis reach granted a whole-axis halo is the redundant but correct
    // configuration: every block reads everything, so every block's core is
    // trustworthy and the regions tile. That is what the design records — the
    // cost model, not the guard, is what drives such a phase to a single block.
    let whole = plan(&workflow, 4, &[0], Reach::all());
    whole.check().unwrap();
    for block in &whole.phases[0].blocks {
        assert_eq!(block.read.shape[0], VOLUME[0]);
    }
    // Grant it anything less and no interior block has a trustworthy voxel.
    let message = whole
        .with_forced_halo([8, 8, 8])
        .check()
        .unwrap_err()
        .to_string();
    assert!(
        message.contains("do not tile the volume exactly"),
        "{message}"
    );
}

/// A whole-axis reach is a barrier because of its **type**, not because somebody
/// compared it against the volume.
#[test]
fn all_is_a_barrier_without_anybody_measuring_it() {
    struct Reduction;
    impl BlockOp for Reduction {
        fn name(&self) -> &'static str {
            "reduce"
        }
        fn reach(&self, _axis: usize, volume_len: usize) -> usize {
            volume_len
        }
        fn reach_spec(&self, _volume: [usize; 3]) -> Reach {
            Reach::per_axis([AxisReach::All, AxisReach::none(), AxisReach::none()])
        }
        fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> blockflow::Result<()> {
            out.assign(input)
        }
    }

    let chain = Chain::op(Reduction);
    assert!(blockflow::is_planning_barrier(&chain, VOLUME));
    let spec = chain.reach_spec(VOLUME).unwrap();
    assert!(spec.is_whole_axis(0, VOLUME[0]));
    assert!(!spec.is_whole_axis(1, VOLUME[1]));
    // and the plan it forces is the single block that can run it
    let workflow = Workflow::new(chain, VOLUME, Dtype::F64);
    let plan = blockflow::Enumerating::default()
        .decompose(&workflow, &blockflow::Constraints::default())
        .unwrap();
    plan.check().unwrap();
    assert_eq!(plan.phases[0].grid.split_axes(), Vec::<usize>::new());
}

// -------------------------------------------------- the coordinate space --

/// A phase whose edges are not the array's edges: the clamp exception is the
/// difference, and stating the frame is what turns a silent pass into a
/// refusal.
///
/// `derive` trusts a read clamped at the phase's own volume edge, because at a
/// real array boundary the operation saw everything that exists. A cropping
/// phase's boundary is an interior position of the level below, so the same
/// clamp trusts a voxel whose context was never fetched. Declared in the source
/// frame, that trust is withdrawn and the guard reports the hole.
#[test]
fn a_phase_whose_edges_are_not_the_arrays_edges_is_not_granted_the_clamp() {
    let core = BlockCore {
        index: [0, 0, 0],
        flat: 0,
        core: Region::new(&[0, 0, 0], &[8, 4, 4]),
    };
    let own = Reach::from([2, 0, 0]);
    let below = Reach::from([2, 0, 0]).in_space(Space::source_voxels());
    let halo = Reach::from([2, 0, 0]);

    // In the phase's own frame the volume edge is an edge: the whole core is
    // trustworthy even though the read ran off the end.
    let trusted = BlockGeometry::derive_with(&core, [32, 4, 4], &halo, &own);
    assert_eq!(trusted.valid.start, vec![0, 0, 0]);
    assert!(trusted.valid_covers_core());

    // Stated against the level below, it is not: the first two planes had
    // context that was never fetched, and they are excluded.
    let untrusted = BlockGeometry::derive_with(&core, [32, 4, 4], &halo, &below);
    assert_eq!(untrusted.valid.start, vec![2, 0, 0]);
    assert!(!untrusted.valid_covers_core());

    // Which the tiling check turns into a refusal, so a cropping phase that
    // reaches is told rather than trusted. Before the frame existed the only
    // way to plan one was to declare reach 0 at its own boundaries, which is
    // the silent version of exactly this.
    let workflow = workflow(2);
    let refused = plan(&workflow, 8, &[0], below);
    let message = refused.check().unwrap_err().to_string();
    assert!(
        message.contains("do not tile the volume exactly") && message.contains("source/voxels"),
        "{message}"
    );
}

/// Whole blocks become voxels where — and only where — a grid is known.
#[test]
fn a_reach_in_blocks_is_converted_by_the_grid_it_is_cut_on() {
    let workflow = workflow(0);
    let in_blocks = Reach::from([1, 0, 0]).in_space(Space::blocks());
    for edge in [4usize, 8] {
        let decomposition = plan(&workflow, edge, &[0], in_blocks.clone());
        // The plan records the statement, not the conversion: two plans that
        // differ only in block size record the same reach and derive different
        // read extents from it.
        assert_eq!(decomposition.phases[0].reach.space().units, Units::Blocks);
        // one block below and one above, each a whole block edge wide
        let interior = &decomposition.phases[0].blocks[1];
        assert_eq!(interior.read.shape[0], edge * 3);
    }
}

/// A dependency in the level below's own lattice is carried, and a phase that
/// declares one without saying where its blocks read is refused.
///
/// This is the form that cannot be converted: there is no factor turning a step
/// of somebody else's lattice into a voxel of this one. What the plan does with
/// it is record it — into the fingerprint and onto the wire — and refuse the
/// combination that would be a lie, which is declaring the dependency and then
/// fetching one's own read extent.
#[test]
fn a_dependency_in_the_level_belows_lattice_needs_a_fetch_region_to_meet_it() {
    let workflow = workflow(0);
    let stated = Reach::from([2, 0, 0]).in_space(Space::source_index());
    assert!(!stated.space().converts_to_voxels());
    assert_eq!(stated.space().frame, Frame::Source);

    let refused = plan(&workflow, 8, &[0], stated.clone());
    let message = refused.check().unwrap_err().to_string();
    assert!(
        message.contains("steps of the level below's own lattice")
            && message.contains("where each block reads"),
        "{message}"
    );

    // With the fetch regions stated, it checks — and the dependency is in the
    // fingerprint rather than being a zero somebody wrote to get past a guard.
    let mut carried = refused.clone();
    carried.phases[0] = carried.phases[0]
        .clone()
        .with_sources(|block| Region::new(&block.read.start.clone(), &block.read.shape.clone()));
    // `with_sources` that reproduces the read extent is still "its own read
    // extent", so the refusal stands; the mapping has to be a real one.
    assert!(carried.check().is_err());

    // A real mapping: every block reads the same window of the level below,
    // which is nothing like its own read extent and is inside the level.
    let mut mapped = refused.clone();
    mapped.phases[0] = mapped.phases[0].clone().with_sources(|block| {
        let mut start = block.read.start.clone();
        start[0] = 0;
        Region::new(&start, &block.read.shape)
    });
    mapped.check().unwrap();
    assert_ne!(
        mapped.fingerprint(),
        plan(&workflow, 8, &[0], Reach::from([0, 0, 0])).fingerprint()
    );
}

/// The degenerate form is the plan it always was, fingerprint included.
///
/// The property that makes every historical figure still attach to its plan: a
/// workflow that says one integer per axis produces a decomposition that
/// fingerprints exactly as it did before a reach could say anything else.
#[test]
fn a_plan_that_says_nothing_new_fingerprints_as_it_always_did() {
    let workflow = workflow(2);
    let symmetric = plan(&workflow, 4, &[0], Reach::from([2, 0, 0]));
    assert_eq!(symmetric.phases[0].reach, [2, 0, 0]);
    assert_eq!(symmetric.phases[0].halo, [2, 0, 0]);

    // The asymmetric statement of the same dependency is a *different* plan, and
    // says so — it reads less, so it had better not be mistaken for the old one.
    let one_sided = plan(
        &workflow,
        4,
        &[0],
        Reach::asymmetric([(2, 0), (0, 0), (0, 0)]),
    );
    assert_ne!(one_sided.fingerprint(), symmetric.fingerprint());
    assert_ne!(one_sided.phases[0].reach, [2, 0, 0]);
}
