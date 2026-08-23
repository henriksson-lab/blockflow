// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The dynamic program and the enumeration must choose the **same partition**.
//
// Not the same cost — the same cuts, the same block edges, the same halos, the
// same derived geometry. `Decomposition` is the binding half of the contract and
// its phase boundaries are asserted by name all over this crate and recorded in
// `statistics::PlanIdentity`, so a different-but-equally-priced partition would
// be a silent parity change dressed up as an optimisation. Equal cost is the
// property that is easy to get and not the one that is wanted.
//
// The DP can promise that because the enumeration's tie-break is *reproducible*
// rather than incidental: it keeps the candidate minimising
// `(total, n_phases, cut mask)` lexicographically, and all three of those are
// additive over the groups of a partition — see `PartitionSearch` for why the
// mask is, which is the only one that is not obvious. So the DP minimises the
// same triple over prefixes and lands on the same partition.
//
// What this file sweeps, because a single hand-built chain would prove nothing:
// reaches that are zero, symmetric, one-sided and whole-axis; costs and
// traversal preferences that make cutting pay and that make it not pay;
// mandated block extents that conflict with each other; barrier cuts present
// and absent; several volumes, candidate lists and split-axis sets; and budgets
// tight enough that some runs of slots cannot be a phase at all while the chain
// as a whole still plans.

use blockflow::decomposition::{predicted_cost, Constraints, CostModel, Decomposition};
use blockflow::dtype::Dtype;
use blockflow::error::Result;
use blockflow::op::{Anchor, BlockOp, Chain};
use blockflow::probes::MandatedExtentOp;
use blockflow::reach::{AxisReach, Reach};
use blockflow::strategy::{Enumerating, PartitionSearch, Strategy, Workflow};
use blockflow::voxels::Voxels;

// ------------------------------------------------------------- the probe --

/// An op that states whatever reach the sweep wants it to state.
///
/// The point of it is the reaches the shipped probes cannot express: a one-sided
/// dependency, and `AxisReach::All` — which is a planning barrier *declared*
/// rather than inferred from a number that happens to equal the extent.
struct SweepOp {
    name: &'static str,
    reach: Reach,
    cost: f64,
    order: Option<[usize; 3]>,
}

impl BlockOp for SweepOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// The symmetric bound `Chain::reach_spec` checks the full statement
    /// against. It has to be a bound, so it is derived from the full statement
    /// rather than declared beside it.
    fn reach(&self, axis: usize, volume_len: usize) -> usize {
        match self.reach.axis(axis) {
            AxisReach::All => volume_len,
            AxisReach::Bounded { lo, hi } => *lo.max(hi),
            AxisReach::PerBlock(_) => volume_len,
            // The unaligned answer, which is what a bound has to be: a bound
            // that assumed the discount would be narrower than the reach it
            // bounds on every lattice the stride does not divide.
            AxisReach::Aligned { unaligned, .. } => unaligned.0.max(unaligned.1),
        }
    }

    fn reach_spec(&self, _volume: [usize; 3]) -> Reach {
        self.reach.clone()
    }

    fn accepts(&self, _dtype: Dtype) -> bool {
        true
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        out.assign(input)
    }

    fn preferred_iteration(&self) -> Option<[usize; 3]> {
        self.order
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

// ------------------------------------------------------- the generator ----

/// splitmix64. A generator rather than a fixed table because the property is
/// meant to hold for chains nobody thought of, and a seed printed in a failure
/// message reproduces the one that broke.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1))
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn pick<T: Copy>(&mut self, from: &[T]) -> T {
        from[self.below(from.len())]
    }
}

/// Enough names for the longest chain either search will plan, plus one.
const NAMES: [&str; 34] = [
    "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14",
    "s15", "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23", "s24", "s25", "s26", "s27",
    "s28", "s29", "s30", "s31", "s32", "s33",
];

const VOLUMES: [[usize; 3]; 5] = [
    [64, 16, 16],
    [48, 24, 12],
    [96, 8, 8],
    [32, 32, 32],
    [40, 20, 10],
];

const ORDERS: [[usize; 3]; 3] = [[0, 1, 2], [2, 1, 0], [1, 2, 0]];

/// Whether the generated cost model is **exactly representable in binary**.
///
/// This is the axis the sweep is split along, and the reason is the one thing
/// that stops the two searches agreeing everywhere.
///
/// Fusing two ops and pricing them apart can come to the *same* total: with a
/// block that spans the volume on every axis the reach touches, redundancy is
/// `1.0` whatever the cuts are, and with `materialise_cost_per_voxel` equal to
/// `write_cost_per_voxel` a cut costs exactly what it saves. The partitions are
/// then genuinely tied, and the enumeration's `(total, n_phases, mask)` is
/// *designed* to decide such a tie — on the phase count, then on the cuts.
///
/// It can only do that if the tie is visible. `sum(0.1 + 0.1 + 0.1)` grouped two
/// ways gives two different `f64`s, so a mathematical tie becomes a
/// last-bit-apart comparison and whichever way it falls is rounding noise rather
/// than the tie-break. The enumeration compares whole totals and the DP compares
/// prefixes, so the noise reaches them differently and they may keep different
/// members of one tied set.
///
/// * [`Arithmetic::Dyadic`] — every cost, coefficient and block edge is a
///   dyadic rational, so mathematically equal totals are **bit** equal, the
///   tie-break is what decides, and the two searches must agree exactly.
/// * [`Arithmetic::Arbitrary`] — decimals that are not, so ties are decided by
///   rounding. Agreement is then asserted up to a tie: equal plans, or plans of
///   equal cost.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arithmetic {
    Dyadic,
    Arbitrary,
}

/// One slot: an op with a randomly chosen reach, cost and traversal preference,
/// or — one time in twelve — an op that mandates a block extent.
fn slot(rng: &mut Rng, index: usize, volume: [usize; 3], arithmetic: Arithmetic) -> Chain {
    let name = NAMES[index];
    if rng.below(12) == 0 {
        // A mandate. Two different ones in one phase cannot be fused, which is
        // an infeasible *run* rather than an infeasible chain — the case the
        // search is meant to route around rather than refuse.
        let axis = rng.below(3);
        let mut extent = volume;
        extent[axis] = rng.pick(&[1usize, 2, 4]).min(volume[axis]);
        return Chain::op(MandatedExtentOp::new(name, extent));
    }
    let mut axes = [AxisReach::none(), AxisReach::none(), AxisReach::none()];
    for (axis, entry) in axes.iter_mut().enumerate() {
        *entry = match rng.below(10) {
            0..=3 => AxisReach::none(),
            4..=6 => AxisReach::symmetric(rng.pick(&[1usize, 2, 4, 8])),
            7..=8 => {
                let lo = rng.pick(&[0usize, 1, 3, 6]);
                let hi = rng.pick(&[0usize, 1, 3, 6]);
                AxisReach::Bounded { lo, hi }
            }
            // Whole-axis: a declared planning barrier, so `barrier_cuts` forces
            // boundaries on both sides of this slot.
            _ if volume[axis] > 1 => AxisReach::All,
            _ => AxisReach::none(),
        };
    }
    Chain::op(SweepOp {
        name,
        reach: Reach::per_axis(axes),
        cost: match arithmetic {
            Arithmetic::Dyadic => rng.pick(&[0.25f64, 0.5, 1.0, 2.0, 8.0]),
            Arithmetic::Arbitrary => rng.pick(&[0.1f64, 0.7, 1.3, 3.0, 12.5]),
        },
        order: match rng.below(3) {
            0 => None,
            other => Some(ORDERS[other]),
        },
    })
}

fn constraints(rng: &mut Rng, arithmetic: Arithmetic) -> Constraints {
    let candidates: Vec<usize> = match rng.below(4) {
        0 => vec![8],
        1 => vec![4, 8, 16],
        2 => vec![8, 16, 32],
        _ => vec![4, 16, 64],
    };
    let mut model = CostModel::default();
    match arithmetic {
        Arithmetic::Dyadic => {
            model.order_conflict_penalty = rng.pick(&[0.0f64, 1.0, 128.0]);
            model.read_cost_per_voxel = rng.pick(&[0.5f64, 1.0, 4.0]);
            model.write_cost_per_voxel = rng.pick(&[0.0f64, 1.0, 2.0]);
            model.compute_scale = rng.pick(&[0.25f64, 1.0, 4.0]);
            // Equal to `write_cost_per_voxel` a third of the time, which is what
            // makes a cut exactly free and the tie a real one.
            model.materialise_cost_per_voxel = rng.pick(&[0.0f64, 1.0, 32.0]);
        }
        Arithmetic::Arbitrary => {
            model.order_conflict_penalty = rng.pick(&[0.0f64, 1.1, 100.0]);
            model.read_cost_per_voxel = rng.pick(&[0.6f64, 1.0, 4.0]);
            model.write_cost_per_voxel = rng.pick(&[0.0f64, 1.0, 2.2]);
            model.compute_scale = rng.pick(&[0.3f64, 1.0, 3.0]);
            model.materialise_cost_per_voxel = rng.pick(&[0.0f64, 2.2, 30.0]);
        }
    }
    Constraints {
        // A budget that binds is the point of a third of these: it makes a run
        // of slots infeasible without making the chain unplannable, which is
        // exactly the edge the DP must decline to take.
        budget_bytes: match rng.below(3) {
            0 => None,
            1 => Some(rng.pick(&[1u64 << 14, 1 << 16, 1 << 18])),
            _ => Some(rng.pick(&[1u64 << 20, 1 << 24])),
        },
        expected_concurrency: rng.pick(&[1usize, 2, 8]),
        model,
        block_candidates: candidates,
        split_axes: match rng.below(4) {
            0 => vec![2],
            1 => vec![0, 2],
            2 => vec![0, 1, 2],
            _ => vec![1],
        },
        ..Default::default()
    }
}

/// One generated planning problem.
///
/// Rebuilt from the seed rather than cloned, because `Workflow` owns its chain
/// and each search needs its own. The generator draws from the seed in a fixed
/// order, so two calls with one seed are the same problem.
struct Case {
    chain: Chain,
    volume: [usize; 3],
    constraints: Constraints,
    dtype: Dtype,
    n: usize,
}

fn case(seed: u64, min_slots: usize, span: usize, arithmetic: Arithmetic) -> Case {
    let mut rng = Rng::new(seed);
    let volume = rng.pick(&VOLUMES);
    let n = min_slots + rng.below(span);
    let chain = Chain::sequence(
        (0..n)
            .map(|index| slot(&mut rng, index, volume, arithmetic))
            .collect(),
    );
    let constraints = constraints(&mut rng, arithmetic);
    let dtype = rng.pick(&[Dtype::F64, Dtype::U8, Dtype::F32]);
    Case {
        chain,
        volume,
        constraints,
        dtype,
        n,
    }
}

/// Counts, so the sweep can assert that it actually swept.
#[derive(Default, Debug)]
struct Tally {
    cases: usize,
    planned: usize,
    refused: usize,
    multi_phase: usize,
    fused: usize,
    with_barrier: usize,
    with_mandate: usize,
    budget_bound: usize,
    phases: usize,
    slots: usize,
    /// Plans that differ while their costs do not. Zero under
    /// [`Arithmetic::Dyadic`]; see it for what these are.
    tied: usize,
}

fn plan(case: Case, search: PartitionSearch) -> Result<Decomposition> {
    Enumerating {
        search,
        ..Enumerating::default()
    }
    .decompose(
        &Workflow::new(case.chain, case.volume, case.dtype),
        &case.constraints,
    )
}

fn sweep(
    seeds: std::ops::Range<u64>,
    min_slots: usize,
    span: usize,
    arithmetic: Arithmetic,
) -> Tally {
    let mut tally = Tally::default();
    for seed in seeds {
        let subject = case(seed, min_slots, span, arithmetic);
        let (volume, n) = (subject.volume, subject.n);
        tally.cases += 1;
        tally.slots += n;
        if subject.constraints.budget_bytes.is_some() {
            tally.budget_bound += 1;
        }
        {
            let slots = subject.chain.slots();
            if slots
                .iter()
                .any(|slot| slot.block_constraint(volume).ok().flatten().is_some())
            {
                tally.with_mandate += 1;
            }
            if slots.iter().any(|slot| {
                slot.reach_spec(volume)
                    .map(|reach| (0..3).any(|axis| reach.is_whole_axis(axis, volume[axis])))
                    .unwrap_or(false)
            }) {
                tally.with_barrier += 1;
            }
        }

        let model = subject.constraints.model.clone();
        let dp = plan(subject, PartitionSearch::Dp);
        let exhaustive = plan(
            case(seed, min_slots, span, arithmetic),
            PartitionSearch::Exhaustive,
        );

        match (dp, exhaustive) {
            (Ok(dp), Ok(exhaustive)) => {
                // The whole plan, not a summary of it: phase slots, block grids,
                // reaches, halos and every derived block geometry.
                if dp != exhaustive {
                    // Under exact arithmetic this is a failure. Under inexact
                    // arithmetic it is allowed only where the two plans cost the
                    // same, which is to say only where the enumeration's own
                    // choice was made by rounding rather than by its tie-break.
                    let cost = |plan: &Decomposition| {
                        predicted_cost(
                            &case(seed, min_slots, span, arithmetic).chain,
                            plan,
                            &[],
                            &model,
                        )
                        .expect("a plan the search returned prices")
                    };
                    let (mine, theirs) = (cost(&dp), cost(&exhaustive));
                    assert!(
                        arithmetic == Arithmetic::Arbitrary
                            && (mine - theirs).abs() <= 1e-9 * theirs.abs().max(1.0),
                        "seed {seed}: the DP and the enumeration planned {n} slots on \
                         {volume:?} differently, and not as a tie — {mine} against {theirs}.\n\
                         DP  {:?}\nENUM {:?}",
                        dp.phases
                            .iter()
                            .map(|p| p.slots.clone())
                            .collect::<Vec<_>>(),
                        exhaustive
                            .phases
                            .iter()
                            .map(|p| p.slots.clone())
                            .collect::<Vec<_>>(),
                    );
                    tally.tied += 1;
                }
                tally.planned += 1;
                tally.phases += dp.n_phases();
                if dp.n_phases() > 1 {
                    tally.multi_phase += 1;
                }
                if dp.phases.iter().any(|phase| phase.slots.len() > 1) {
                    tally.fused += 1;
                }
            }
            (Err(dp), Err(exhaustive)) => {
                tally.refused += 1;
                // Neither may refuse a chain the other plans, which this arm
                // asserts by being the arm that was taken. The wording is the
                // one thing allowed to differ, and only for the search's own
                // refusal: one counts partitions and the other counts runs of
                // slots. A refusal raised by the plan's own checks is the same
                // plan failing the same check, so it is the same string.
                assert!(
                    dp.to_string() == exhaustive.to_string()
                        || (dp.to_string().contains("enumerating:")
                            && exhaustive.to_string().contains("enumerating:")),
                    "seed {seed}: {dp} / {exhaustive}"
                );
            }
            (dp, exhaustive) => panic!(
                "seed {seed}: the searches disagreed about feasibility — DP {:?}, exhaustive {:?}",
                dp.map(|plan| plan.n_phases())
                    .map_err(|err| err.to_string()),
                exhaustive
                    .map(|plan| plan.n_phases())
                    .map_err(|err| err.to_string()),
            ),
        }
    }
    tally
}

// -------------------------------------------------------------- the tests --

/// The load-bearing one: over three thousand generated chains, the two searches
/// return **equal** `Decomposition`s or both refuse.
///
/// Exact arithmetic, so there is nothing to excuse. Every partition either wins
/// on cost or is decided by the tie-break the DP reproduces, and the plans must
/// match phase for phase and block for block.
#[test]
fn the_dp_and_the_enumeration_choose_the_same_partition() {
    let tally = sweep(0..3000, 1, 8, Arithmetic::Dyadic);
    eprintln!("short chains, dyadic: {tally:?}");
    assert_eq!(tally.tied, 0, "{tally:?}");
    // The sweep has to have swept. These floors are not the property under
    // test — they are the guard against the property being asserted over three
    // thousand single-phase plans that never had a cut to disagree about.
    assert!(tally.planned > 2000, "{tally:?}");
    assert!(tally.refused > 20, "{tally:?}");
    assert!(tally.multi_phase > 500, "{tally:?}");
    assert!(tally.fused > 500, "{tally:?}");
    assert!(tally.with_barrier > 300, "{tally:?}");
    assert!(tally.with_mandate > 300, "{tally:?}");
    assert!(tally.budget_bound > 1500, "{tally:?}");
}

/// And again where the enumeration is genuinely expensive: 12 to 16 slots is
/// `2048` to `32768` partitions each, which is the range the DP was written to
/// replace and the range where a tie-break that was merely *nearly* right would
/// have room to show.
#[test]
fn the_two_searches_agree_on_long_chains_too() {
    let tally = sweep(10_000..10_400, 12, 5, Arithmetic::Dyadic);
    eprintln!("long chains, dyadic: {tally:?}");
    assert_eq!(tally.tied, 0, "{tally:?}");
    assert!(tally.planned > 250, "{tally:?}");
    assert!(tally.multi_phase > 100, "{tally:?}");
}

/// **The one thing that stops the agreement being unconditional**, measured
/// rather than asserted away.
///
/// With costs that binary cannot hold exactly, two partitions of mathematically
/// equal price differ in their last bits, and *which* of them each search keeps
/// is then decided by rounding: the enumeration compares whole totals, the DP
/// compares prefixes, and the noise reaches the two comparisons differently. The
/// plans may therefore differ — but only where the cost does not, which is what
/// is asserted here, at every seed, to within `1e-9` relative.
///
/// Nothing about this is peculiar to the DP. The enumeration's answer on a tied
/// pair was already an artefact of the summation order rather than a decision;
/// what the DP changes is which artefact. The fix, if the tie ever matters, is a
/// cost model whose ties are visible — see [`Arithmetic`] — not a different
/// search.
#[test]
fn where_the_arithmetic_cannot_see_a_tie_the_two_searches_may_part_but_never_on_cost() {
    let tally = sweep(0..3000, 1, 8, Arithmetic::Arbitrary);
    eprintln!("short chains, inexact: {tally:?}");
    assert!(tally.planned > 2000, "{tally:?}");
    // A rate, so a regression that made the DP part from the enumeration
    // *structurally* rather than by a last bit shows up as a number instead of
    // being absorbed. It has been 4 in 3000.
    assert!(
        tally.tied * 100 < tally.planned,
        "the searches parted on {} of {} plans, which is too many to be rounding",
        tally.tied,
        tally.planned
    );
}

/// Past the enumeration's limit only the DP answers, and it says so.
///
/// This is the case the pipeline on this crate is at: a 20-slot chain today and
/// two arms to add, after which `Exhaustive` refuses outright and the DP does
/// not.
#[test]
fn a_chain_too_long_to_enumerate_is_planned_by_the_dp_and_refused_by_the_enumeration() {
    let volume = [64, 16, 16];
    let build = |n: usize| {
        Chain::sequence(
            (0..n)
                .map(|index| {
                    Chain::op(SweepOp {
                        name: NAMES[index],
                        reach: Reach::symmetric([1 + index % 3, 0, 0]),
                        cost: 1.0 + index as f64,
                        order: None,
                    })
                })
                .collect(),
        )
    };
    let constraints = Constraints::default();

    let plan = Enumerating::default()
        .decompose(&Workflow::new(build(24), volume, Dtype::F64), &constraints)
        .expect("the DP plans a chain the enumeration will not");
    plan.check().unwrap();
    assert_eq!(
        plan.phases
            .iter()
            .map(|phase| phase.slots.len())
            .sum::<usize>(),
        24,
        "every slot lands in exactly one phase"
    );

    let refusal = Enumerating {
        search: PartitionSearch::Exhaustive,
        ..Enumerating::default()
    }
    .decompose(&Workflow::new(build(24), volume, Dtype::F64), &constraints)
    .unwrap_err()
    .to_string();
    assert!(refusal.contains("exhaustive search's limit"), "{refusal}");

    // And the DP has a ceiling of its own, which is the cut mask rather than the
    // search: `u32`.
    let too_long = Enumerating::default()
        .decompose(&Workflow::new(build(33), volume, Dtype::F64), &constraints)
        .unwrap_err()
        .to_string();
    assert!(too_long.contains("`u32`"), "{too_long}");
}
