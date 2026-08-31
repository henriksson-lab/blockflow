// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **Which terms of a phase repeat, and which happen once.**
//
// An iterative phase runs `S` substages that read and compute, ping-ponging two
// private buffers, and `strategy::run_iterative_phase` writes the image **once**,
// after the loop. So the shape is `S x (read + compute) + write`.
//
// `phase_compute_per_voxel`'s own documentation is where this file comes from.
// It records that the planner priced such a phase at `S == 1`, that the claim
// "a missing count cannot move the block edge, because a common factor cannot
// move an argmin" was **measured false**, and why:
//
// > the count is not a common factor of the whole price. A substage reads and
// > computes; the image is written **once**, after the loop. So `S` substages
// > are `S x (read + compute) + write`, and pricing at `S == 1` over-weights the
// > write against the rest by a residual that varies with the block edge.
//
// It then says what is wanted, and this file is the acceptance for it:
//
// > what is wanted is a statement of **which terms of a phase repeat and which
// > happen once**, which is a fact about `strategy::run_iterative_phase` rather
// > than about any builder.
//
// That statement is `PhaseTraffic::repeats`. What is asserted here is that the
// split is real arithmetic and not a scale factor: the price is exactly
// `S x (read + compute) + write`, and the **ratio between two candidate block
// edges moves with `S`** — which is the whole of why an argmin can change its
// mind, and is what a common factor could never do.

use blockflow::decomposition::{price_phase, CostModel, PhaseTraffic};
use blockflow::geometry::BlockGrid;
use blockflow::reach::Reach;

const VOLUME: [usize; 3] = [64, 64, 64];

/// One phase's price at `edge`, run for `substages`.
fn price(edge: usize, substages: usize) -> f64 {
    let grid = BlockGrid::new(VOLUME, [edge, edge, edge]).expect("a grid");
    // A reach of one on every axis, so the read extent genuinely exceeds the
    // core and the read term has something to say about the edge. With a reach
    // of zero the read is the core at every candidate and the residual this test
    // is about would be flat by construction.
    let halo = Reach::symmetric([1, 1, 1]);
    price_phase(
        &grid,
        &halo,
        1.0,
        1,
        false,
        8.0,
        &CostModel::default(),
        CostModel::default().materialise_cost_per_voxel,
        PhaseTraffic::one_in_one_out().repeating(substages),
    )
    .cost_per_block
}

/// **The price is `S x (read + compute) + write`, exactly.**
///
/// Asserted as an identity rather than an inequality: two substages must cost
/// exactly one substage's read-and-compute more than one substage does, and the
/// write must be paid once in both. Anything else means the count reached a term
/// it should not have.
#[test]
fn a_substage_repeats_the_read_and_the_compute_and_not_the_write() {
    let once = price(16, 1);
    let twice = price(16, 2);
    let thrice = price(16, 3);

    // The increments are equal, which is what "linear in `S` with a constant
    // term" means and is the strongest form available without naming the terms.
    let first = twice - once;
    let second = thrice - twice;
    assert!(
        (first - second).abs() < 1e-9,
        "each further substage must cost the same read-and-compute: {first} then {second}"
    );
    assert!(first > 0.0, "a substage has to cost something");

    // And the constant term is the write, so extrapolating back to zero
    // substages leaves exactly it.
    let write_only = once - first;
    assert!(
        write_only > 0.0,
        "the once-only term must be positive — it is the image write, and a phase that writes \
         nothing would make this test vacuous"
    );

    // Zero is read as one, because `Stats::substages` reports zero for a phase
    // that is not an iteration and charging that phase nothing would be reading
    // "not an iteration" as "no work".
    assert_eq!(price(16, 0), once);
}

/// **The ratio between two candidate edges moves with `S`.**
///
/// This is the finding the doc records and the reason the split had to exist. A
/// common factor on the whole price cannot move an argmin — that was the
/// argument for leaving the count out, and it fails because the write is outside
/// the factor. Here it is, measured: the two candidates' prices stand in a
/// different ratio at one substage than at eight, so a search between them is
/// answering a different question.
#[test]
fn the_ratio_between_two_block_edges_moves_with_the_substage_count() {
    let ratio_at = |substages: usize| price(32, substages) / price(8, substages);

    let one = ratio_at(1);
    let many = ratio_at(8);
    assert!(
        (one - many).abs() > 1e-6,
        "the price ratio between edge 32 and edge 8 was {one} at one substage and {many} at \
         eight. If a count really were a common factor of the whole price these would be equal \
         — which is exactly the argument that put `S == 1` in the planner, and it is false \
         because the write sits outside the factor."
    );

    // And the direction: more substages weigh the read-and-compute more heavily
    // against the write, so the ratio moves toward the read term's own ratio.
    let reads = |edge: usize| {
        let grid = BlockGrid::new(VOLUME, [edge, edge, edge]).expect("a grid");
        price_phase(
            &grid,
            &Reach::symmetric([1, 1, 1]),
            1.0,
            1,
            false,
            8.0,
            &CostModel::default(),
            CostModel::default().materialise_cost_per_voxel,
            PhaseTraffic {
                images_read: 1,
                // No write at all, so what is left is the repeating part.
                writes_an_image: false,
                repeats: 1,
                // A single op holds nothing of its own; see
                // `Chain::resident_block_buffers`.
                chain_buffers: 0,
            },
        )
        .cost_per_block
    };
    let read_ratio = reads(32) / reads(8);
    assert!(
        (many - read_ratio).abs() < (one - read_ratio).abs(),
        "at eight substages the ratio {many} should sit nearer the read-and-compute ratio \
         {read_ratio} than the one-substage ratio {one} does — that is the residual moving, \
         and it is what lets the chosen edge depart"
    );
}

/// **`PhaseCost` reports the count it charged**, so a plan that priced an
/// iteration is distinguishable from one that assumed a single pass.
#[test]
fn the_price_records_how_many_times_it_charged() {
    let grid = BlockGrid::new(VOLUME, [16, 16, 16]).expect("a grid");
    let of = |substages: usize| {
        price_phase(
            &grid,
            &Reach::symmetric([1, 1, 1]),
            1.0,
            1,
            false,
            8.0,
            &CostModel::default(),
            CostModel::default().materialise_cost_per_voxel,
            PhaseTraffic::one_in_one_out().repeating(substages),
        )
        .repeats
    };
    assert_eq!(of(1), 1);
    assert_eq!(of(7), 7);
    assert_eq!(of(0), 1, "zero is not an iteration, and is charged once");
}

/// **A plan the planner builds is charged once**, because the count is measured
/// and no plan holds it — `IterativeOp::limit` is a runaway bound and
/// deliberately not an estimate.
#[test]
fn a_phase_the_plan_describes_repeats_once() {
    use blockflow::assemble::PlanBuilder;
    use blockflow::op::Chain;
    use blockflow::probes::IdentityOp;
    use blockflow::Dtype;

    let grid = BlockGrid::new(VOLUME, [16, 16, 16]).expect("a grid");
    let mut builder = PlanBuilder::new(VOLUME, Dtype::F64, grid);
    builder
        .pixels(Chain::op(IdentityOp::new("one", [1, 1, 1])))
        .expect("a pixel phase");
    let assembly = builder.finish().expect("an assembly");
    let work = assembly.work();
    let traffic = blockflow::decomposition::phase_traffic_of(0, &assembly.decomposition, &work)
        .expect("a phase's traffic");
    assert_eq!(
        traffic.repeats, 1,
        "the plan cannot know a count that is a fixed point over data, so it says one and a \
         caller with a measurement says otherwise"
    );
    assert_eq!(traffic.repeating(5).repeats, 5);
}
