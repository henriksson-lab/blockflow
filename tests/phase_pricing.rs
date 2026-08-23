// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **What a phase's price is a price of.** `decomposition::price_phase` turns a
// candidate grid into a number the partition search minimises, and the whole
// value of that number is that it ranks candidates the way a run would. This
// file pins the two places the ranking was found to be a fact about the model
// rather than about the work:
//
// 1. **The read charge is the phase's halo.** A block fetches `core (+) halo`;
//    the *reach* is the narrower thing the halo is required to cover, and the
//    two part exactly where a halo is granted wider than the ops asked for — a
//    mandated per-block window, or a fragment phase whose halo is raised by a
//    fragment input's reach in blocks. Charging the reach there under-charged
//    by up to 91.3%, and — the part that makes it a defect rather than a
//    calibration — by an amount that grows as the block edge falls, so the
//    price preferred the smallest grid on offer for a reason unconnected to the
//    work.
//
// 2. **The negative control is the same program with one thing changed**: the
//    identical arithmetic fed the reach instead of the halo. It is asserted to
//    be wrong, and wrong in the unsafe direction, so that a revert cannot pass
//    this file quietly.
//
// The reference both are measured against is `Decomposition::exact_read_voxels`,
// which is not another model: it adds up the read extents the plan's own
// `BlockGeometry`s hold, clamped at the volume boundary, which is what the
// executor will fetch.

use blockflow::decomposition::{
    price_phase, CostModel, Decomposition, PhaseDecomposition, PhaseTraffic,
};
use blockflow::geometry::BlockGrid;
use blockflow::reach::Reach;
use blockflow::Dtype;

const VOLUME: [usize; 3] = [64, 64, 64];
/// The op's own reach, held fixed while the *granted* halo widens. Non-zero, so
/// that "halo == reach" is a row of the table rather than the degenerate case.
const REACH: [usize; 3] = [1, 1, 1];

/// The predicted whole-phase read, and the read the plan's geometry says will
/// happen, for one grid and one granted halo.
fn predicted_and_exact(edge: usize, halo: [usize; 3], charged: [usize; 3]) -> (f64, usize) {
    let grid = BlockGrid::along(VOLUME, &[0, 1, 2], edge).expect("a grid at every edge here");
    let plan = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases: vec![PhaseDecomposition::derive(
            Vec::new(),
            vec!["priced".to_string()],
            REACH,
            halo,
            grid.clone(),
        )],
        chain_reach: REACH,
    };
    let cost = price_phase(
        &grid,
        &Reach::symmetric(charged),
        1.0,
        1,
        false,
        8.0,
        &CostModel::default(),
        1.0,
        // One image in, one image out: the case this file is about is the halo,
        // and holding the traffic at the classic assumption keeps it that way.
        PhaseTraffic::one_in_one_out(),
    );
    (
        cost.read_voxels_per_block * grid.n_blocks() as f64,
        plan.exact_read_voxels()[0],
    )
}

/// Relative error of a predicted read against the exact one, as a fraction.
fn error(edge: usize, halo: [usize; 3], charged: [usize; 3]) -> f64 {
    let (predicted, exact) = predicted_and_exact(edge, halo, charged);
    predicted / exact as f64 - 1.0
}

/// The grids and granted halos the table below is taken over. `halo == REACH` is
/// the first row deliberately: it is the case the two charges agree on, and a
/// change that broke it would be a change to something other than this.
const EDGES: [usize; 4] = [64, 32, 16, 8];
const HALOS: [[usize; 3]; 4] = [[1, 1, 1], [2, 2, 2], [4, 4, 4], [8, 8, 8]];

/// **The read charge is the geometry's own read, to the voxel** — not a bound on
/// it, not a conservative approximation of it, the same integer.
///
/// This started life as a band. Charging the halo over-stated by 7.9% to 72.8%
/// across this sweep, because the charge was made on the infinite grid with
/// every block assumed interior; that was defended as conservative, and the
/// defence did not survive a phase's array count entering the price, since a
/// phase reading three arrays pays the boundary fraction three times and the
/// size of the model's error then depends on the partition being ranked.
/// `price_phase` now charges `BlockGrid::mean_read_voxels` and the band became
/// an equality.
///
/// An equality is a much better test than a band: there is no width left for a
/// future defect to hide in, and no constant anyone can re-baseline. If it
/// fails, the price and the geometry have stopped being the same statement.
#[test]
fn charging_the_halo_is_exactly_the_read_the_geometry_will_perform() {
    for edge in EDGES {
        for halo in HALOS {
            let (predicted, exact) = predicted_and_exact(edge, halo, halo);
            assert_eq!(
                predicted, exact as f64,
                "edge {edge}, halo {halo:?}: the price says {predicted} voxels and the plan's \
                 own geometry says {exact}"
            );
        }
    }
}

/// **The negative control.** The same arithmetic on the reach — what
/// `price_phase` was given before — under-states every grid the moment the
/// granted halo exceeds the reach, and this test fails if that ever stops being
/// true, which is what a revert would do.
#[test]
fn charging_the_reach_instead_under_states_the_read_wherever_the_halo_is_wider() {
    for edge in EDGES {
        for halo in HALOS {
            if halo == REACH || edge == VOLUME[0] {
                // Nothing to distinguish: the charges are the same number, and
                // at one block every read is the whole volume either way.
                continue;
            }
            let err = error(edge, halo, REACH);
            assert!(
                err <= 0.0,
                "edge {edge}, halo {halo:?}: charging the reach was expected to under-state the \
                 read and it over-stated by {:.1}%. Either the geometry changed or the two \
                 charges are no longer distinguishable here, and this control has stopped being \
                 one",
                100.0 * err
            );
        }
    }
    // The worst case of the sweep, quoted so that a shrinking error is visible
    // as a change rather than as a pass: the smallest block with the widest
    // granted halo.
    let worst = error(8, [8, 8, 8], REACH);
    assert!(
        worst < -0.90,
        "the reach charge under-stated the smallest grid by only {:.1}%, and the measurement in \
         `price_phase`'s doc says 91.3%",
        -100.0 * worst
    );
}

/// **The part that makes it a defect rather than a calibration constant.** The
/// under-charge is not a fixed factor a model could absorb: it is zero at one
/// block and deepens monotonically as the edge falls, so it is a property of the
/// candidate being priced. That is precisely the test `BlockGrid::mean_core_voxels`
/// was introduced under, applied to the other half of the same expression.
#[test]
fn the_reach_charges_error_is_a_property_of_the_candidate_not_a_constant() {
    let halo = [4, 4, 4];
    let errors: Vec<f64> = EDGES.iter().map(|&edge| error(edge, halo, REACH)).collect();
    assert!(
        errors[0].abs() < 1e-12,
        "one block should read the whole volume however it is charged: {:?}",
        errors
    );
    for pair in errors.windows(2) {
        assert!(
            pair[1] < pair[0],
            "the under-charge did not deepen as the block shrank: {errors:?}. If it has become a \
             constant the ranking argument no longer applies and this file needs rewriting"
        );
    }
    // Charging the halo has no such trend to exploit, and no residual at all:
    // the error is zero at every one of the same four grids. That is the
    // contrast this file exists to record — not "smaller", but "not a function
    // of the candidate, because it is not there".
    let charged: Vec<f64> = EDGES.iter().map(|&edge| error(edge, halo, halo)).collect();
    assert!(
        charged.iter().all(|err| *err == 0.0),
        "the halo charge has acquired an error that varies with the grid: {charged:?}"
    );
}

// ------------------------------------------------ phases with no chain slot --
//
// A fragment or iterative phase owns no slot of the chain, so every fold the
// price is built from — reach, compute, element type — visits nothing for it.
// Reach and element type were already given their own answers (`fragment_phase`
// derives one, `check_dtypes` asks the op). Compute was not: it folded to `0.0`,
// and `IterativeOp::cost_per_voxel` sat unread while the phase most likely to be
// one is a thinning, whose entire cost is compute.
//
// The read side was worse than unpriced, it was fabricated: a
// `fragments -> fragments` phase reads no pixel and writes no image, and was
// charged a full traversal of the volume plus a full write of it.

use blockflow::fragment::{fragment_only, PhaseWork};
use blockflow::iterate::{iterative_phase, IterativeOp, Substage, SubstageLimit, SubstageOperand};
use blockflow::op::Chain;
use blockflow::probes::NullFragmentOp;
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::predicted_makespan;
use blockflow::voxels::Voxels;
use blockflow::Result;

/// A one-substage spread along axis 0 whose per-voxel cost the test sets.
///
/// The cost is the only thing that varies between instances, which is what makes
/// it a control: two plans differing in nothing else must be priced apart by
/// exactly the amount the declaration says.
struct Thinning {
    reach: usize,
    cost: f64,
}

impl IterativeOp for Thinning {
    fn name(&self) -> &'static str {
        "thinning"
    }

    fn operands(&self) -> Vec<SubstageOperand> {
        vec![
            SubstageOperand::running([self.reach, self.reach, self.reach]),
            SubstageOperand::fixed([0, 0, 0]),
        ]
    }

    fn limit(&self) -> SubstageLimit {
        SubstageLimit::of(64).expect("a positive limit")
    }

    fn substage(&self, at: &Substage<'_>, out: &mut Voxels) -> Result<()> {
        out.assign(at.operand(0)?)
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

fn iterative_plan(op: &dyn IterativeOp, edge: usize) -> Decomposition {
    let grid = BlockGrid::along(VOLUME, &[0, 1, 2], edge).expect("a grid");
    Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases: vec![iterative_phase(op, grid).expect("a well-formed iterative op")],
        chain_reach: [0, 0, 0],
    }
}

/// **The headline.** An iterative phase's declared compute now reaches the
/// price, and a phase that declares more costs more. Before this, both plans
/// priced identically because the fold over `phase.slots` had nothing to fold.
#[test]
fn an_iterative_phases_declared_compute_reaches_its_price() {
    let chain = Chain::sequence(Vec::new());
    let model = CostModel::default();
    let cheap = Thinning {
        reach: 1,
        cost: 1.0,
    };
    let dear = Thinning {
        reach: 1,
        cost: 40.0,
    };
    let cheap_price = predicted_makespan(
        &chain,
        &iterative_plan(&cheap, 16),
        &[PhaseWork::Iterate(&cheap)],
        &model,
        4,
    )
    .expect("an iterative phase prices");
    let dear_price = predicted_makespan(
        &chain,
        &iterative_plan(&dear, 16),
        &[PhaseWork::Iterate(&dear)],
        &model,
        4,
    )
    .expect("an iterative phase prices");
    assert!(
        dear_price > cheap_price,
        "a 40x per-voxel declaration priced at {dear_price} against {cheap_price} for 1x; the \
         op's own cost is not reaching the price"
    );
}

/// **A claim this file used to make, and the measurement that killed it.**
///
/// It asserted that scaling an iterative op's `cost_per_voxel` cannot move the
/// block edge the makespan prefers, on the argument that compute is charged over
/// the same extent as the read, so scaling it scales a term the choice was
/// already made on. That is **false above one worker**, and the reason the
/// original sweep missed it is worth keeping: it ran `cost_per_voxel` over
/// `1 .. 1000`, and on this fixture the transition sits between `0.1` and `1`.
/// The whole sweep was inside the compute-bound plateau, so it measured a
/// plateau and reported an invariance.
///
/// The mechanism is [`phase_makespan`]'s roofline, `max(pool, channel)`:
///
/// * **compute is in the pool and not in the channel.** The channel bound is
///   bytes — read plus write — and carries no compute term at all.
/// * at **`workers == 1`** the pool is always the larger, the objective collapses
///   to serial work, and serial work is monotone in the block edge. The largest
///   candidate wins for every op, every reach and every compute figure, so
///   nothing about compute can move it.
/// * **above one worker** the pool is divided by [`rounds`] and the channel is
///   not, so raising the compute walks the phase from bandwidth-bound to
///   compute-bound and walks the argmin with it.
///
/// Measured here over `1e-3 .. 1e3` at three reaches and six worker counts: the
/// argmin is flat at `workers == 1` in every row and moves in every row above it.
/// The boundary is at the worker count, not at the reach and not at the compute.
///
/// **What this costs:** the safety argument for adding a compute figure to an
/// iterative phase — that it cannot destabilise an existing choice — holds only
/// at one worker, and real runs use forty. The figure is still right to charge;
/// it is not free.
///
/// [`phase_makespan`]: blockflow::strategy::phase_makespan
/// [`rounds`]: blockflow::strategy::rounds
#[test]
fn the_iterative_edge_is_flat_in_the_compute_at_one_worker_and_moves_above_it() {
    let chain = Chain::sequence(Vec::new());
    let model = CostModel::default();
    // Wide enough to cross the roofline's knee. A range that starts at 1.0 is
    // already past it on this fixture, which is how the false claim was made.
    const COSTS: [f64; 6] = [1e-3, 1e-1, 1.0, 10.0, 100.0, 1000.0];
    let argmin = |reach: usize, cost: f64, workers: usize| -> usize {
        let op = Thinning { reach, cost };
        let work = [PhaseWork::Iterate(&op)];
        let mut best: Option<(f64, usize)> = None;
        for edge in [8usize, 16, 32, 64] {
            let price =
                predicted_makespan(&chain, &iterative_plan(&op, edge), &work, &model, workers)
                    .expect("an iterative phase prices");
            let better = match best {
                None => true,
                Some((seen, _)) => price.total_cmp(&seen).is_lt(),
            };
            if better {
                best = Some((price, edge));
            }
        }
        best.expect("at least one edge").1
    };
    for reach in [1usize, 3, 8] {
        let serial: Vec<usize> = COSTS.iter().map(|&c| argmin(reach, c, 1)).collect();
        assert!(
            serial.windows(2).all(|pair| pair[0] == pair[1]),
            "reach {reach}: the edge moved with the compute at one worker ({serial:?}). At one \
             worker the objective is serial work, which is monotone in the edge, so no compute \
             figure can move it — if this fires, `phase_makespan` is no longer a roofline"
        );
        for workers in [2usize, 3, 4, 8, 40] {
            let parallel: Vec<usize> = COSTS.iter().map(|&c| argmin(reach, c, workers)).collect();
            assert!(
                parallel.windows(2).any(|pair| pair[0] != pair[1]),
                "reach {reach}, {workers} workers: the edge did not move across a 10^6 span of \
                 declared compute ({parallel:?}). Compute is in the pool and not in the channel, \
                 so above one worker it must be able to walk the phase across the roofline's knee"
            );
        }
    }
}

/// A `fragments -> fragments` phase reads no pixel and writes no image, and the
/// price now says so. The reference is `exact_read_voxels`, which reports zero.
#[test]
fn a_fragments_only_phase_is_not_charged_for_pixels_it_never_touches() {
    let merge = NullFragmentOp::new("merge", "merged", Lifecycle::DeleteOnExit);
    let plan = fragment_only(VOLUME, [16, 16, 16], Dtype::F64, &[&merge])
        .expect("a fragments-only decomposition");
    assert_eq!(
        plan.exact_read_voxels(),
        vec![0],
        "the geometry says this phase fetches nothing"
    );
    let price = blockflow::decomposition::predicted_cost(
        &Chain::sequence(Vec::new()),
        &plan,
        &[PhaseWork::Fragments(&merge)],
        &CostModel::default(),
    )
    .expect("a fragment phase prices");
    assert_eq!(
        price, 0.0,
        "a phase that reads no pixel and writes no image was charged {price}"
    );
}

/// **The negative control for the whole arrangement.** A phase that owns no
/// chain slot and is not named in `work` is refused rather than priced at the
/// zero the fold would have produced — which is the failure this parameter
/// exists to remove, and it is invisible in the answer.
#[test]
fn a_slotless_phase_the_caller_did_not_describe_is_refused_by_name() {
    let merge = NullFragmentOp::new("merge", "merged", Lifecycle::DeleteOnExit);
    let plan = fragment_only(VOLUME, [16, 16, 16], Dtype::F64, &[&merge])
        .expect("a fragments-only decomposition");
    let message = blockflow::decomposition::predicted_cost(
        &Chain::sequence(Vec::new()),
        &plan,
        &[],
        &CostModel::default(),
    )
    .expect_err("a slotless phase with no work entry must not price")
    .to_string();
    assert!(
        message.contains("owns no chain slot"),
        "the refusal should say what is missing: {message}"
    );
    assert!(
        message.contains("PhaseWork"),
        "the refusal should name the thing to pass: {message}"
    );
}

/// An iterative phase reads and writes an image like any pixel phase, so the
/// traffic gate must not swallow it. Without this the previous two tests would
/// pass against an implementation that priced every slotless phase at nothing.
#[test]
fn an_iterative_phase_is_still_charged_for_the_image_it_reads_and_writes() {
    let op = Thinning {
        reach: 1,
        cost: 1.0,
    };
    let plan = iterative_plan(&op, 16);
    let price = blockflow::decomposition::predicted_cost(
        &Chain::sequence(Vec::new()),
        &plan,
        &[PhaseWork::Iterate(&op)],
        &CostModel::default(),
    )
    .expect("an iterative phase prices");
    assert!(
        price > 0.0,
        "an iterative phase reads and writes a real image and was charged {price}"
    );
    assert!(
        plan.exact_read_voxels()[0] > 0,
        "and the geometry agrees that it fetches something"
    );
}

/// A phase that traverses a second array is charged for traversing it.
///
/// `Chain::Source` is cheaper than materialising the array it replaces because
/// it adds nothing to the halo — not because reading it is free. The price
/// charged one traversal for every phase however many arrays it named, which is
/// an under-charge, and under-charging is the direction this model is not
/// permitted to be wrong in.
#[test]
fn a_phase_that_reads_a_second_array_is_charged_for_both() {
    use blockflow::probes::IdentityOp;

    let grid = BlockGrid::along(VOLUME, &[0, 1, 2], 16).expect("a grid");
    let chain = Chain::op(IdentityOp::new("identity", [0, 0, 0]));
    let plan = |images: Vec<usize>| Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases: vec![PhaseDecomposition::derive(
            vec![0],
            vec!["identity".to_string()],
            [0usize, 0, 0],
            [0usize, 0, 0],
            grid.clone(),
        )
        .with_source_images(images)],
        chain_reach: [0, 0, 0],
    };
    let model = CostModel::default();
    let one = blockflow::decomposition::predicted_cost(&chain, &plan(Vec::new()), &[], &model)
        .expect("a price");
    let two = blockflow::decomposition::predicted_cost(&chain, &plan(vec![0]), &[], &model)
        .expect("a price");
    assert!(
        two > one,
        "a phase naming a second array priced at {two} against {one} for one array"
    );
    // And the exact figure agrees on the factor: the read doubles, the write
    // does not, so the two prices differ by exactly one traversal's read cost.
    let plans = (plan(Vec::new()), plan(vec![0]));
    assert_eq!(
        plans.1.exact_read_voxels()[0],
        2 * plans.0.exact_read_voxels()[0],
        "the geometry says two traversals, so the price should be built on two"
    );
}
