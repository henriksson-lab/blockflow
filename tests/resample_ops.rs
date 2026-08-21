// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The acceptance suite for a phase that **changes the shape of the image it
// writes**. Everything else in this crate's test tree resamples nothing, so
// every property here is being asserted for the first time on a plan whose two
// images are different sizes.
//
// Eight things, and each catches something the others cannot
// ----------------------------------------------------------
// 1. **Byte identity with the whole-volume reference** — the same kernel called
//    once — at several factors, several block edges and several split-axis
//    choices, including a factor that does not divide the volume evenly and a
//    block edge that is not a multiple of the factor's period.
// 2. **The round trip**: upsample then downsample by the same factor with
//    nearest is the identity, and what linear does instead is stated and shown.
// 3. **The halo guard fires**, by name, on a resampling phase — provoked rather
//    than assumed.
// 4. **The two-coordinate-space clamp** does not bite here, and the check is the
//    volume's own edges rather than an argument: the reference and the run agree
//    at the faces, which is where a false clamp exception would show first.
// 5. **Why the halo has to be per block and per side** — one number for all of
//    them is refused rather than merely wasteful — and **what the lattice
//    alignment costs**, measured off the plan's own `exact_read_voxels` against
//    the tight dependency rather than argued: 1.00x for every downsampling and
//    every integer growth, 1.52x to 2.01x for a rational growth at a small block
//    edge.
// 6. **Through `ZarrEnvironment`**: a resize changes the image's shape, and
//    `prepare` creates images from `volume_at`/`dtype_at`, so this is the first
//    workload that exercises that path with two different shapes.
// 7. **Label data survives** a nearest resampling through the executor, which is
//    the case a general library is asked for and the one linear cannot serve.
// 8. **An output extent the caller states** rather than derives from the factor:
//    that the factor-exact default is byte-unchanged, that a stated extent is
//    byte identical to the whole-volume answer across block sizes including a
//    one-voxel block, that it cuts where the factor's period is the whole output
//    axis and by how much the fetch drops, and that the check the extent waiver
//    owes is seen to fire on a short fetch.

use blockflow::decomposition::Decomposition;
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, Chain};
use blockflow::ops::{
    resample_nearest_into, resample_phase, Interpolation, Ratio, Resample, ResampleOp,
};
use blockflow::reach::AxisReach;
use blockflow::strategy::{execute, Hints, Workflow};
use blockflow::voxels::Voxels;
use blockflow::Dtype;
use ndarray::Array3;

/// Not a round number on any axis, so a factor divides none of them evenly
/// unless it was chosen to.
const VOLUME: [usize; 3] = [24, 18, 14];

/// Structure at several scales, so an interpolation has something to do and a
/// seam has something to get wrong. Deliberately not smooth: a ramp would agree
/// with itself under a wrong sample map.
fn texture(shape: [usize; 3]) -> Array3<f64> {
    Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
        ((i * 7919 + j * 104729 + k * 1299709) % 1013) as f64 + 1.0
    })
}

fn ratios(specs: [(usize, usize); 3]) -> [Ratio; 3] {
    [
        Ratio::new(specs[0].0, specs[0].1).unwrap(),
        Ratio::new(specs[1].0, specs[1].1).unwrap(),
        Ratio::new(specs[2].0, specs[2].1).unwrap(),
    ]
}

/// The whole-volume reference: the same op, applied once, to everything.
fn reference(input: &Voxels, resample: &Resample) -> Voxels {
    let op = ResampleOp::new("resample", *resample);
    let shape = input.shape();
    let chain = Chain::op(op);
    let mut out = Voxels::zeros(
        chain.produces(input.dtype()).unwrap(),
        chain.output_shape(shape).unwrap(),
    )
    .unwrap();
    chain.apply(input, &mut out, &Anchor::whole(shape)).unwrap();
    out
}

/// A one-phase plan for a resampling op, at a given block edge and split axes.
///
/// Everything geometric comes from the op: the grid is over what the op says it
/// produces, and the halo and the fetch regions come from `resample_phase`.
/// Nothing here supplies a reach, so nothing here can hide one that is wrong.
fn plan(
    resample: &Resample,
    input: [usize; 3],
    edge: usize,
    split_axes: &[usize],
) -> Decomposition {
    let output = resample.output_volume(input).unwrap();
    let grid = BlockGrid::along(output, split_axes, edge).unwrap();
    let phase =
        resample_phase(vec![0], vec!["resample".to_string()], resample, input, grid).unwrap();
    let reach = [
        resample.reach(0).0.max(resample.reach(0).1),
        resample.reach(1).0.max(resample.reach(1).1),
        resample.reach(2).0.max(resample.reach(2).1),
    ];
    Decomposition {
        volume: input,
        dtype: Dtype::F64,
        phases: vec![phase],
        chain_reach: reach,
    }
}

fn run(resample: &Resample, input: &Voxels, decomposition: &Decomposition) -> Voxels {
    let workflow = Workflow::new(
        Chain::op(ResampleOp::new("resample", *resample)),
        input.shape(),
        input.dtype(),
    );
    let env = ArrayEnvironment::for_decomposition(input.clone(), decomposition, [4, 4, 4]).unwrap();
    execute(
        "resample",
        &workflow,
        decomposition,
        &Hints::default(),
        &env,
    )
    .unwrap();
    env.output()
}

// ------------------------------------------------- 1. the identity bar --

/// Every decomposition of every factor reproduces the whole-volume answer, bit
/// for bit.
///
/// The list of factors is chosen for what each one breaks:
///
/// * `1/2`, `1/3` — integer decimation, where the fetch is trivially aligned;
/// * `1/5` on an axis of 24 — **does not divide evenly**: 24 becomes 4 and the
///   top 4 input voxels are sampled by nothing, which is where an off-by-one
///   lives;
/// * `2/1`, `3/1` — integer growth, where the fetch must be padded and the reach
///   is not zero;
/// * `3/2`, `5/3` — neither, on all three axes at once, so no axis's arithmetic
///   is the identity by accident;
/// * a mix of growing and shrinking on different axes, which is the case a
///   single "factor" parameter would have hidden.
///
/// The block edges include 5 and 7, which are multiples of no factor's period,
/// so the alignment the fetch needs is bought by the halo at every block.
#[test]
fn every_decomposition_reproduces_the_whole_volume_answer() {
    let input: Voxels = texture(VOLUME).into();
    let factors = [
        [(1usize, 2usize), (1, 2), (1, 2)],
        [(1, 3), (1, 1), (1, 2)],
        [(1, 5), (1, 5), (1, 5)],
        [(2, 1), (2, 1), (1, 1)],
        [(3, 1), (1, 1), (1, 1)],
        [(3, 2), (3, 2), (3, 2)],
        [(5, 3), (2, 3), (1, 1)],
        [(1, 4), (3, 1), (2, 3)],
    ];
    for interpolation in [Interpolation::Nearest, Interpolation::Linear] {
        for specs in factors {
            let resample = Resample::new(ratios(specs), interpolation);
            let want = reference(&input, &resample);
            assert!(
                want.uniform().is_none(),
                "the reference is constant, so it discriminates nothing"
            );
            for edge in [2usize, 3, 5, 7, 64] {
                for split_axes in [vec![0], vec![1, 2], vec![0, 1, 2]] {
                    let decomposition = plan(&resample, VOLUME, edge, &split_axes);
                    decomposition.check().unwrap();
                    assert_eq!(
                        decomposition.output_volume(),
                        want.shape(),
                        "the plan's image is not the shape the op produces"
                    );
                    assert_eq!(decomposition.uniform_volume(), None);
                    let got = run(&resample, &input, &decomposition);
                    assert_eq!(
                        got, want,
                        "{specs:?} {interpolation:?}, edge {edge}, axes {split_axes:?}"
                    );
                }
            }
        }
    }
}

/// The factor that does not divide the volume, in the open: the output extent,
/// the input voxels nothing samples, and the answer, all of them checked at the
/// edge where the arithmetic changes.
#[test]
fn a_factor_that_does_not_divide_the_volume_drops_a_named_tail() {
    let resample = Resample::uniform(Ratio::smaller(5).unwrap(), Interpolation::Nearest);
    let input: Voxels = texture(VOLUME).into();
    assert_eq!(resample.output_volume(VOLUME).unwrap(), [4, 3, 2]);
    assert_eq!(resample.unsampled_tail(0, VOLUME[0]), 4);
    assert_eq!(resample.unsampled_tail(1, VOLUME[1]), 3);
    assert_eq!(resample.unsampled_tail(2, VOLUME[2]), 4);

    let want = reference(&input, &resample);
    let source = input.view::<f64>().unwrap();
    let got = want.view::<f64>().unwrap();
    // nearest at 1/5 takes source 5o + 2, so the last output voxel is at 17 and
    // 20..24 is the tail nothing reads
    assert_eq!(got[[3, 2, 1]], source[[17, 12, 7]]);
    for edge in [1usize, 2, 3] {
        let decomposition = plan(&resample, VOLUME, edge, &[0, 1, 2]);
        decomposition.check().unwrap();
        assert_eq!(run(&resample, &input, &decomposition), want, "edge {edge}");
    }
}

/// The short circuit over a resizing phase: a uniform input lets every block be
/// skipped, and a skipped block must produce the block the work would have
/// produced — **at the output's shape, not the input's**.
///
/// That last clause is the reason this is here rather than left to the kernel
/// test of `constant_maps_to`: the executor allocates the replacement buffer
/// from the read extent and the image's own element type, and no other workload
/// in this tree has ever asked it to do that for an image of a different shape.
#[test]
fn a_uniform_volume_short_circuits_a_resizing_phase_at_the_output_shape() {
    for interpolation in [Interpolation::Nearest, Interpolation::Linear] {
        let resample = Resample::new(ratios([(3, 2), (1, 2), (2, 1)]), interpolation);
        let input = Voxels::filled(Dtype::F64, VOLUME, 7.0).unwrap();
        let decomposition = plan(&resample, VOLUME, 4, &[0, 1, 2]);
        decomposition.check().unwrap();
        let got = run(&resample, &input, &decomposition);
        assert_eq!(got.shape(), resample.output_volume(VOLUME).unwrap());
        assert_eq!(got, reference(&input, &resample), "{interpolation:?}");
        assert_eq!(got.uniform(), Some(7.0));
    }
}

/// A resampling phase **in a chain**: resample, then filter the result on its
/// own grid over the smaller volume.
///
/// This is the arrangement a caller actually wants — decimate, then do the
/// expensive thing on less data — and it is the one that exercises the seam:
/// image 1 is a different shape from image 0, phase 1's blocks read image 1
/// through their own halo, and `TaskGraph` has to join the two across the
/// change of extent. `dependencies_cover_reads` is asked directly, because a
/// missing edge there is a race rather than a wrong number and would not show
/// as a difference on a single-threaded run.
#[test]
fn a_resampling_phase_chains_into_a_filter_on_the_smaller_volume() {
    use blockflow::graph::TaskGraph;
    use blockflow::ops::{ElementShape, RankFilterOp, StructuringElement};

    let resample = Resample::uniform(Ratio::smaller(2).unwrap(), Interpolation::Linear);
    let element = StructuringElement::from_radius(ElementShape::Box, [1, 1, 1]);
    let chain = || {
        Chain::sequence(vec![
            Chain::op(ResampleOp::new("resample", resample)),
            Chain::op(RankFilterOp::median("median", element.clone())),
        ])
    };
    let input: Voxels = texture(VOLUME).into();
    let smaller = resample.output_volume(VOLUME).unwrap();

    let mut want = Voxels::zeros(Dtype::F64, smaller).unwrap();
    chain()
        .apply(&input, &mut want, &Anchor::whole(VOLUME))
        .unwrap();

    for edge in [3usize, 5] {
        let first = resample_phase(
            vec![0],
            vec!["resample".to_string()],
            &resample,
            VOLUME,
            BlockGrid::along(smaller, &[0, 1, 2], edge).unwrap(),
        )
        .unwrap();
        // The second phase is an ordinary one — over the *smaller* volume, which
        // is what `volume_at(1)` now returns.
        let second = blockflow::decomposition::PhaseDecomposition::derive(
            vec![1],
            vec!["median".to_string()],
            [1, 1, 1],
            [1, 1, 1],
            BlockGrid::along(smaller, &[0, 1, 2], edge).unwrap(),
        );
        let decomposition = Decomposition {
            volume: VOLUME,
            dtype: Dtype::F64,
            phases: vec![first, second],
            chain_reach: [1, 1, 1],
        };
        decomposition.check().unwrap();
        assert_eq!(decomposition.volume_at(0), VOLUME);
        assert_eq!(decomposition.volume_at(1), smaller);
        assert_eq!(decomposition.output_volume(), smaller);
        TaskGraph::build(&decomposition)
            .dependencies_cover_reads(&decomposition)
            .unwrap();

        let workflow = Workflow::new(chain(), VOLUME, Dtype::F64);
        let env =
            ArrayEnvironment::for_decomposition(input.clone(), &decomposition, [4, 4, 4]).unwrap();
        execute(
            "resample-then-filter",
            &workflow,
            &decomposition,
            &Hints::default(),
            &env,
        )
        .unwrap();
        assert_eq!(env.output(), want, "edge {edge}");
    }
}

// ---------------------------------------------------- 2. the round trip --

/// Upsample then downsample by the same integer factor with nearest is the
/// identity — and it is the identity **through the executor**, at a block edge
/// that divides neither image.
///
/// Why it holds: growing by `n` writes each source voxel into `n` consecutive
/// output voxels, and shrinking by `n` takes the one at `n*o + (n-1)/2`, which
/// is inside that run. Both halves are the same centred map, so the composition
/// is exact rather than approximate — no value is combined with any other at any
/// point.
///
/// **Linear does not do this and must not be expected to.** Growing by `n`
/// interpolates, so the intermediate volume holds values that were never in the
/// input; shrinking again averages neighbours of the original and returns a
/// smoothed copy. The test below measures how far off it is rather than
/// asserting it is not the identity, because "not equal" would pass for a kernel
/// that returned nothing.
#[test]
fn nearest_upsampling_then_downsampling_is_the_identity() {
    let input: Voxels = texture(VOLUME).into();
    for factor in [2usize, 3, 4] {
        let grow = Resample::uniform(Ratio::larger(factor).unwrap(), Interpolation::Nearest);
        let shrink = Resample::uniform(Ratio::smaller(factor).unwrap(), Interpolation::Nearest);

        // through the executor, both ways, at a block edge that divides neither
        // the input nor the intermediate volume
        let up_plan = plan(&grow, VOLUME, 5, &[0, 1, 2]);
        up_plan.check().unwrap();
        let bigger = run(&grow, &input, &up_plan);
        assert_eq!(
            bigger.shape(),
            [VOLUME[0] * factor, VOLUME[1] * factor, VOLUME[2] * factor]
        );

        let down_plan = plan(&shrink, bigger.shape(), 5, &[0, 1, 2]);
        down_plan.check().unwrap();
        let back = run(&shrink, &bigger, &down_plan);
        assert_eq!(back, input, "factor {factor}");
    }
}

/// What linear does instead, stated as a measurement: the round trip is a
/// smoothing, close to the input but not it, and the error is bounded by the
/// local variation rather than growing without limit.
#[test]
fn linear_upsampling_then_downsampling_smooths_rather_than_restoring() {
    let input: Voxels = texture(VOLUME).into();
    let grow = Resample::uniform(Ratio::larger(2).unwrap(), Interpolation::Linear);
    let shrink = Resample::uniform(Ratio::smaller(2).unwrap(), Interpolation::Linear);
    let bigger = reference(&input, &grow);
    let back = reference(&bigger, &shrink);
    assert_eq!(back.shape(), VOLUME);

    let source = input.view::<f64>().unwrap();
    let got = back.view::<f64>().unwrap();
    let mut differing = 0usize;
    let mut worst = 0.0f64;
    for (a, b) in source.iter().zip(got.iter()) {
        if a != b {
            differing += 1;
        }
        worst = worst.max((a - b).abs());
    }
    assert!(
        differing > source.len() / 2,
        "a linear round trip that reproduced most voxels would mean the \
         interpolation is not interpolating"
    );
    // the data spans 1..=1013, and no voxel moves further than the range it was
    // averaged over
    assert!(worst < 1013.0, "worst deviation {worst}");
}

// ----------------------------------------------------- 3. the guard fires --

/// A halo below the phase's reach makes the valid regions stop tiling, and the
/// plan says so by name — on a resampling phase, which reaches through a fetch
/// mapping rather than through its own read extent, so this is the guard *seen*
/// to fire on the new shape of plan rather than assumed to still be there.
///
/// The provocation is `with_forced_halo`, which keeps each block's fetch region
/// — so what changes is exactly one thing, the granted halo.
#[test]
fn a_halo_below_the_reach_is_caught_on_a_resampling_phase() {
    let resample = Resample::uniform(Ratio::larger(3).unwrap(), Interpolation::Linear);
    let input: Voxels = texture(VOLUME).into();
    let honest = plan(&resample, VOLUME, 5, &[0, 1, 2]);
    honest.check().unwrap();
    assert_eq!(
        resample.reach(0),
        (1, 1),
        "the reach must be non-zero or the guard has nothing to catch"
    );

    let forced = honest.with_forced_halo([0, 0, 0]);
    let message = forced.check().unwrap_err().to_string();
    assert!(
        message.contains("do not tile the volume exactly"),
        "{message}"
    );

    let workflow = Workflow::new(
        Chain::op(ResampleOp::new("resample", resample)),
        VOLUME,
        Dtype::F64,
    );
    let env = ArrayEnvironment::for_decomposition(input, &forced, [4, 4, 4]).unwrap();
    let message = execute("short", &workflow, &forced, &Hints::default(), &env)
        .unwrap_err()
        .to_string();
    assert!(
        message.contains("do not tile the volume exactly"),
        "{message}"
    );
}

/// And the other guard this op can trip: a plan whose grid and whose op disagree
/// about the shape. It is the same refusal `tests/element_type.rs` provokes with
/// a decimating op, reached here from the other direction — the plan is built
/// for one factor and executed with another.
#[test]
fn a_plan_built_for_another_factor_is_refused_and_names_both_extents() {
    let planned = Resample::uniform(Ratio::smaller(2).unwrap(), Interpolation::Nearest);
    let executed = Resample::uniform(Ratio::smaller(3).unwrap(), Interpolation::Nearest);
    let decomposition = plan(&planned, VOLUME, 4, &[0]);
    decomposition.check().unwrap();

    let workflow = Workflow::new(
        Chain::op(ResampleOp::new("resample", executed)),
        VOLUME,
        Dtype::F64,
    );
    let input: Voxels = texture(VOLUME).into();
    let env = ArrayEnvironment::for_decomposition(input, &decomposition, [4, 4, 4]).unwrap();
    let message = execute(
        "mismatch",
        &workflow,
        &decomposition,
        &Hints::default(),
        &env,
    )
    .unwrap_err()
    .to_string();
    assert!(
        message.contains("has nowhere to land") && message.contains("its ops turn that into"),
        "{message}"
    );
}

// -------------------------------------------- 4. the faces of the volume --

/// The clamp exception spans two coordinate spaces for a phase that resizes, so
/// the frame the reach is stated in decides whether a volume-edge block is
/// trusted. This op states `Frame::Phase` and the claim that licenses it is that
/// its output volume's edges *are* the image of the input volume's edges.
///
/// Asserted where a false premise would show first: every voxel of all six
/// faces, at a decomposition that puts a block boundary next to each of them,
/// against the whole-volume reference — for the growing factor, whose reach is
/// non-zero and which therefore actually asks for the exception.
#[test]
fn the_faces_of_the_volume_agree_with_the_reference_where_the_clamp_is_trusted() {
    let input: Voxels = texture(VOLUME).into();
    for interpolation in [Interpolation::Nearest, Interpolation::Linear] {
        let resample = Resample::new(ratios([(3, 1), (2, 1), (1, 3)]), interpolation);
        let want = reference(&input, &resample);
        let want = want.view::<f64>().unwrap();
        let decomposition = plan(&resample, VOLUME, 3, &[0, 1, 2]);
        decomposition.check().unwrap();
        let got = run(&resample, &input, &decomposition);
        let got = got.view::<f64>().unwrap();
        let shape = [want.shape()[0], want.shape()[1], want.shape()[2]];
        for axis in 0..3 {
            for face in [0, shape[axis] - 1] {
                let mut checked = 0usize;
                for i in 0..shape[0] {
                    for j in 0..shape[1] {
                        for k in 0..shape[2] {
                            if [i, j, k][axis] != face {
                                continue;
                            }
                            assert_eq!(
                                got[[i, j, k]],
                                want[[i, j, k]],
                                "{interpolation:?} face {axis}={face} at {i},{j},{k}"
                            );
                            checked += 1;
                        }
                    }
                }
                assert!(checked > 0);
            }
        }
    }
}

// -------------------------------------- 5. what the per-block halo buys --

/// **The per-block, per-side halo is not an optimisation here — it is the only
/// form that plans at all**, and this is where that is established rather than
/// claimed. Two refusals and one measurement:
///
/// * one number granted to every block cannot put every read boundary on the
///   fetch lattice, because the blocks' cores start at different offsets within
///   a cell. The refusal names the lattice.
/// * neither can one number per block granted to both of its sides: a core at
///   `a` needs `a mod up` below it and `-(b) mod up` above, and those are equal
///   only when `a + b` happens to be a multiple of `up`.
/// * what the alignment costs is then measured against the **tight** dependency
///   — [`Resample::source_span`], the source voxels a block's core actually
///   reads — off the plan's own `exact_read_voxels`, which counts the fetch
///   regions the run will move.
#[test]
fn the_halo_must_be_per_block_and_per_side_and_the_alignment_is_priced() {
    let resample = Resample::uniform(Ratio::larger(3).unwrap(), Interpolation::Linear);
    let output = resample.output_volume(VOLUME).unwrap();
    let grid = BlockGrid::along(output, &[0, 1, 2], 5).unwrap();

    let halo = resample.halo(&grid);
    assert!(
        matches!(halo.axis(0), AxisReach::PerBlock(_)),
        "the halo does not differ per block here, so this measures nothing"
    );
    let widest = halo.bound(output);

    // (1) one number for every block: the reads it produces are not on the
    // lattice, and saying so is the refusal rather than a fetch that is merely
    // larger.
    let planned = plan(&resample, VOLUME, 5, &[0, 1, 2]);
    planned.check().unwrap();
    let uniform = planned.with_forced_halo(widest);
    let refusal = uniform.phases[0]
        .blocks
        .iter()
        .find_map(|block| resample.source_region(&block.read, VOLUME).err())
        .expect("a uniform halo cannot be aligned at this edge, which is the point")
        .to_string();
    assert!(refusal.contains("multiples of 3"), "{refusal}");

    // (2) one number per block, granted to both of its sides: the same refusal,
    // for the other reason. The two sides of a block are different distances
    // from the cell boundary, so a symmetric grant lands on it at most one way.
    let symmetric: Vec<(usize, usize)> = match halo.axis(0) {
        AxisReach::PerBlock(table) => table
            .iter()
            .map(|&(lo, hi)| (lo.max(hi), lo.max(hi)))
            .collect(),
        other => panic!("{other:?}"),
    };
    let asymmetric = match halo.axis(0) {
        AxisReach::PerBlock(table) => table.clone(),
        other => panic!("{other:?}"),
    };
    assert!(
        symmetric != asymmetric,
        "the two sides agree on every block, so the asymmetric form is buying nothing here"
    );

    // (3) what the alignment costs, against the tight dependency of each core.
    for (specs, edge) in [
        ([(3usize, 1usize); 3], 5usize),
        ([(3, 2); 3], 5),
        ([(3, 2); 3], 12),
        ([(5, 2); 3], 4),
        ([(1, 2); 3], 5),
    ] {
        let resample = Resample::new(ratios(specs), Interpolation::Linear);
        let decomposition = plan(&resample, VOLUME, edge, &[0, 1, 2]);
        decomposition.check().unwrap();
        let fetched = decomposition.exact_read_voxels()[0];
        let mut needed = 0usize;
        for block in &decomposition.phases[0].blocks {
            let mut span = 1usize;
            for axis in 0..3 {
                let (low, high) = resample.source_span(
                    axis,
                    block.core.start[axis],
                    block.core.shape[axis],
                    VOLUME[axis],
                );
                span *= high - low;
            }
            needed += span;
        }
        println!(
            "{:?} at edge {edge}: {fetched} voxels fetched, {needed} depended on — the lattice \
             alignment costs {:.2}x",
            specs[0],
            fetched as f64 / needed as f64
        );
        assert!(fetched >= needed);
        // The measured spread is 1.00x to 2.01x and what drives it is the block
        // edge against the factor's period: a growing factor at a small edge
        // rounds both ends of every block out to a cell, and a larger edge
        // amortises the same two roundings over more voxels. What it never does
        // is approach a re-read of the block per block.
        assert!(
            fetched < needed * 3,
            "the alignment is costing more than twice the dependency: {fetched} against {needed}"
        );
    }
}

// -------------------------------------------------------- 6. through Zarr --

/// The same run through Zarr arrays on a disk, against the in-memory answer.
///
/// A resize changes the image's shape, and `ZarrEnvironment::prepare` creates
/// images from `volume_at`/`dtype_at` — so this is the first workload where
/// those two calls return different answers per image, and the first that
/// exercises writing an image that is not the shape of the one below it.
#[cfg(feature = "zarr")]
#[test]
fn a_resizing_phase_runs_through_zarr_and_agrees_with_memory() {
    use blockflow::zarr_env::ZarrEnvironment;

    let root = std::env::temp_dir().join(format!(
        "blockflow-resample-zarr-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    struct Scratch(std::path::PathBuf);
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let scratch = Scratch(root.clone());

    let input: Voxels = texture(VOLUME).into();
    for (specs, interpolation) in [
        ([(1usize, 2usize), (1, 2), (1, 2)], Interpolation::Nearest),
        ([(3, 2), (1, 1), (2, 3)], Interpolation::Linear),
    ] {
        let resample = Resample::new(ratios(specs), interpolation);
        let decomposition = plan(&resample, VOLUME, 4, &[0, 1, 2]);
        decomposition.check().unwrap();
        let want = run(&resample, &input, &decomposition);

        let path = scratch.0.join(format!("{specs:?}-{interpolation:?}"));
        let workflow = Workflow::new(
            Chain::op(ResampleOp::new("resample", resample)),
            VOLUME,
            Dtype::F64,
        );
        let env = ZarrEnvironment::create(&path, &input, [4, 4, 4]).unwrap();
        execute(
            "resample-zarr",
            &workflow,
            &decomposition,
            &Hints::default(),
            &env,
        )
        .unwrap();
        let got = env.output().unwrap();
        assert_eq!(got.shape(), want.shape());
        assert_eq!(got, want, "{specs:?} {interpolation:?} through storage");
    }
}

// ------------------------------------------------------- 7. label data --

/// A label volume resampled by nearest, through the executor, in its own element
/// type: every value in the output is a value that was in the input, and the
/// answer is the whole-volume answer.
///
/// This is the case linear cannot serve — the shell refuses `bool` and would
/// blend an integer id with its neighbour into an id that names nothing — and it
/// is why the nearest kernel's bound is `T: Copy` and not a numeric one.
#[test]
fn a_label_volume_survives_a_nearest_resampling_through_the_executor() {
    let labels: Voxels = Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(i, j, k)| {
        ((i / 3 + j / 2 + k) % 7) as u16 * 1000
    })
    .into();
    let resample = Resample::uniform(Ratio::smaller(2).unwrap(), Interpolation::Nearest);

    let mut want = Voxels::zeros(Dtype::U16, resample.output_volume(VOLUME).unwrap()).unwrap();
    let all = [resample.ratio(0), resample.ratio(1), resample.ratio(2)];
    resample_nearest_into(
        labels.view::<u16>().unwrap(),
        &Anchor::whole(VOLUME),
        &all,
        want.view_mut::<u16>().unwrap(),
    )
    .unwrap();

    let workflow = Workflow::new(
        Chain::op(ResampleOp::new("resample", resample)),
        VOLUME,
        Dtype::U16,
    );
    for edge in [3usize, 5] {
        let output = resample.output_volume(VOLUME).unwrap();
        let grid = BlockGrid::along(output, &[0, 1, 2], edge).unwrap();
        let phase = resample_phase(
            vec![0],
            vec!["resample".to_string()],
            &resample,
            VOLUME,
            grid,
        )
        .unwrap();
        let decomposition = Decomposition {
            volume: VOLUME,
            dtype: Dtype::U16,
            phases: vec![phase],
            chain_reach: [0, 0, 0],
        };
        decomposition.check().unwrap();
        let env =
            ArrayEnvironment::for_decomposition(labels.clone(), &decomposition, [4, 4, 4]).unwrap();
        execute("labels", &workflow, &decomposition, &Hints::default(), &env).unwrap();
        let got = env.output();
        assert_eq!(got, want, "edge {edge}");
        assert_eq!(got.dtype(), Dtype::U16);
        let source = labels.view::<u16>().unwrap();
        assert!(got
            .view::<u16>()
            .unwrap()
            .iter()
            .all(|value| source.iter().any(|had| had == value)));
    }
}

// ------------------------------------- 8. an output extent the caller states --
//
// `Resample::to_extent` takes the two extents and derives the factor, instead of
// taking the factor and deriving the output extent. Three things have to hold
// and each is asserted below rather than argued:
//
// * the **default is byte-unchanged** — every quantity the factor path declares,
//   and the voxels themselves, are what they were;
// * a stated extent **plans at any cut**, and the blocked answer is byte
//   identical to the whole-volume one, at block edges that divide the extent and
//   at edges that do not, down to one voxel;
// * the waiver is paid for: the check `placed_output_shape` traded away is in
//   `apply_placed` and is **seen to fire**.

/// The extents an axis of 24, 18 and 14 gets under the two rules at one factor.
///
/// `ceil` and `floor` of the same product, so the fixture discriminates the two
/// conventions on every axis rather than on one: `24 * 13/80 = 3.9`, `18 * 13/80
/// = 2.925`, `14 * 13/80 = 2.275`.
const STATED: [usize; 3] = [4, 3, 3];
const FACTORED: [usize; 3] = [3, 2, 2];

fn stated_ratios() -> [Ratio; 3] {
    ratios([(13, 80), (13, 80), (13, 80)])
}

/// The two rules disagree about the extent on **every axis of this fixture**,
/// which is what makes everything below a discriminating test rather than a
/// tautology. A fixture where they agreed would pass whichever rule was wired.
#[test]
fn the_stated_and_the_factored_extent_differ_on_every_axis_here() {
    let factored = Resample::new(stated_ratios(), Interpolation::Linear);
    assert_eq!(factored.output_volume(VOLUME).unwrap(), FACTORED);
    let stated = Resample::to_extent(VOLUME, STATED, Interpolation::Linear).unwrap();
    assert_eq!(stated.output_volume(VOLUME).unwrap(), STATED);
    for axis in 0..3 {
        assert_ne!(
            STATED[axis], FACTORED[axis],
            "axis {axis} does not discriminate the two rules"
        );
    }
    // And the sample map really is the stated one: the scale is `n/out` and not
    // `down/up`, so the two spellings put the last output voxel in different
    // places.
    assert_eq!(stated.ratio(0), Ratio::new(4, 24).unwrap());
    assert_eq!(factored.ratio(0), Ratio::new(13, 80).unwrap());
    // A stated extent is bound to the volume it was stated against, and being
    // asked about another is refused rather than answered by the factor.
    let message = stated.output_volume([25, 18, 14]).unwrap_err().to_string();
    assert!(message.contains("bound to the volume"), "{message}");
}

/// **The factor-exact default is byte-unchanged**, asserted rather than assumed.
///
/// Every quantity the factor arm declares, at a factor whose extent divides
/// nothing evenly, plus the voxels through the executor. A change to the shared
/// arithmetic that moved the default fails here, and the whole of the rest of
/// this file fails with it.
#[test]
fn the_factor_exact_default_is_unmoved_by_the_stated_extent() {
    let input: Voxels = texture(VOLUME).into();
    for interpolation in [Interpolation::Nearest, Interpolation::Linear] {
        for specs in [
            [(1usize, 5usize), (1, 5), (1, 5)],
            [(3, 2), (3, 2), (3, 2)],
            [(13, 80), (13, 80), (13, 80)],
        ] {
            let resample = Resample::new(ratios(specs), interpolation);
            assert_eq!(
                resample.extent(),
                blockflow::ops::OutputExtent::Factor,
                "the default arm moved"
            );
            for axis in 0..3 {
                // the extent rule
                assert_eq!(
                    resample.output_extent(axis, VOLUME[axis]),
                    VOLUME[axis] * specs[axis].0 / specs[axis].1
                );
                // the alignment is the factor's period, not 1
                assert_eq!(resample.alignment()[axis], resample.ratio(axis).up());
                // and the reach is the interpolation's, not zero
                assert_eq!(
                    resample.reach(axis),
                    (0..1)
                        .map(|_| Resample::uniform(resample.ratio(axis), interpolation).reach(0))
                        .next()
                        .unwrap()
                );
            }
            let op = ResampleOp::new("resample", resample);
            assert!(
                !blockflow::op::BlockOp::takes_extent_from_placement(&op),
                "the default must not take its extent from the plan"
            );
            let want = reference(&input, &resample);
            let decomposition = plan(&resample, VOLUME, 3, &[0, 1, 2]);
            decomposition.check().unwrap();
            assert_eq!(run(&resample, &input, &decomposition), want);
        }
    }
}

/// **The payoff**: the blocked path is byte identical to the whole-volume
/// answer, at a stated extent, across block sizes.
///
/// The edges include ones that divide the extent, ones that do not, and **one
/// voxel** — which is the cut the factor path cannot make at all here, because
/// `Ratio::new(4, 24)` reduces to `1/6` and `Ratio::new(3, 14)` does not reduce
/// at all, so the factor path's period is the whole axis on two of the three.
#[test]
fn every_stated_extent_decomposition_reproduces_the_whole_volume_answer() {
    let input: Voxels = texture(VOLUME).into();
    for interpolation in [Interpolation::Nearest, Interpolation::Linear] {
        for output in [STATED, [7, 5, 29], [23, 19, 13], [1, 3, 2], [37, 41, 43]] {
            let resample = Resample::to_extent(VOLUME, output, interpolation).unwrap();
            let want = reference(&input, &resample);
            assert_eq!(want.shape(), output);
            assert!(
                want.uniform().is_none(),
                "the reference is constant, so it discriminates nothing"
            );
            // The one-voxel cut is on one axis rather than three, and only
            // where the extent is small: `[37, 41, 43]` cut to single voxels on
            // every axis is 65 231 blocks of one voxel each, which measures the
            // executor's per-block cost and not this op's arithmetic. What the
            // one-voxel cut is here to show — a block whose read extent is one
            // output voxel and whose fetch is the two source voxels it brackets
            // — is shown by cutting one axis.
            let cases: Vec<(usize, Vec<usize>)> = vec![
                (1, vec![0]),
                (2, vec![0]),
                (3, vec![1, 2]),
                (5, vec![0, 1, 2]),
                (7, vec![0, 1, 2]),
                (64, vec![0, 1, 2]),
            ];
            for (edge, split_axes) in cases {
                let decomposition = plan(&resample, VOLUME, edge, &split_axes);
                decomposition.check().unwrap();
                assert_eq!(decomposition.output_volume(), output);
                let got = run(&resample, &input, &decomposition);
                assert_eq!(
                    got, want,
                    "{output:?} {interpolation:?}, edge {edge}, axes {split_axes:?}"
                );
            }
        }
    }
}

/// What the stated extent buys, as a number: the same map and the same answer,
/// with the fetch dropping from the **whole axis** to each block's own span.
///
/// `Ratio::new(7, 24)` is in lowest terms, so the factor path's alignment period
/// is 7 — the entire output axis — and every block's halo snaps its read out to
/// cover all of it. The plan is legal and it is not a decomposition: each of the
/// three blocks fetches all 24 input voxels. Stated, the same three blocks —
/// output 0..3, 3..6 and 6..7 — fetch 9, 9 and 2.
#[test]
fn a_stated_extent_cuts_where_the_factor_period_is_the_whole_axis() {
    let input: Voxels = texture(VOLUME).into();
    let factored = Resample::new(
        [
            Ratio::new(7, 24).unwrap(),
            Ratio::identity(),
            Ratio::identity(),
        ],
        Interpolation::Linear,
    );
    assert_eq!(factored.output_volume(VOLUME).unwrap(), [7, 18, 14]);
    assert_eq!(factored.alignment()[0], 7, "the period is the whole axis");
    let stated = Resample::to_extent(VOLUME, [7, 18, 14], Interpolation::Linear).unwrap();
    assert_eq!(stated.alignment(), [1, 1, 1]);

    let factored_plan = plan(&factored, VOLUME, 3, &[0]);
    let stated_plan = plan(&stated, VOLUME, 3, &[0]);
    let fetched = |decomposition: &Decomposition| -> Vec<usize> {
        decomposition.phases[0]
            .blocks
            .iter()
            .map(|block| block.source.shape[0])
            .collect()
    };
    assert_eq!(fetched(&factored_plan), vec![24, 24, 24]);
    assert_eq!(fetched(&stated_plan), vec![9, 9, 2]);

    // The same answer both ways, which is what makes the fetch a saving rather
    // than a different computation: at this factor the two extent rules agree
    // (`floor(24 * 7/24) == 7`), so the voxels have to be identical.
    let want = reference(&input, &factored);
    assert_eq!(reference(&input, &stated), want);
    assert_eq!(run(&factored, &input, &factored_plan), want);
    assert_eq!(run(&stated, &input, &stated_plan), want);
}

/// The op declares the waiver, and the framework's own check is satisfied rather
/// than bypassed.
#[test]
fn a_stated_extent_declares_that_it_takes_its_extent_from_the_plan() {
    use blockflow::op::BlockOp;

    let stated = Resample::to_extent(VOLUME, STATED, Interpolation::Linear).unwrap();
    let op = ResampleOp::new("resample", stated);
    assert!(op.takes_extent_from_placement());
    // `output_shape` has no answer for anything but the whole volume, and says
    // so by returning a shape the executor rejects rather than an arithmetic
    // that is right for the interior blocks.
    assert_eq!(op.output_shape(VOLUME), STATED);
    assert_eq!(op.output_shape([12, 9, 7]), [12, 9, 7]);
    // The plan the framework checks, checked.
    let decomposition = plan(&stated, VOLUME, 2, &[0, 1, 2]);
    decomposition.check().unwrap();
    let chain = Chain::op(ResampleOp::new("resample", stated));
    blockflow::decomposition::check_output_shapes(&chain, &decomposition, &[]).unwrap();
}

/// **The check the waiver owes, seen to fire.** A fetch one voxel short of what
/// the block's output voxels bracket is refused by name, rather than clamped to
/// the buffer's edge and returned as a plausible volume.
///
/// This is the replacement for the executor's own comparison of declared shape
/// against derived read extent, which an op answering out of the plan makes
/// vacuous. It is the stronger of the two because it is against the buffer the
/// block was handed rather than against a declaration.
#[test]
fn a_short_fetch_under_a_stated_extent_is_refused_rather_than_clamped() {
    use blockflow::op::{Anchor, BlockOp, Placement, SourceInputs};
    use blockflow::region::Region;

    let stated = Resample::to_extent(VOLUME, [7, 18, 14], Interpolation::Linear).unwrap();
    let op = ResampleOp::new("resample", stated);
    let input: Voxels = texture(VOLUME).into();
    // The middle block of the cut above: output 3..6, whose voxels bracket
    // source 11..20 and nothing else.
    let read = Region::new(&[3, 0, 0], &[3, 18, 14]);
    let fetch = stated.source_region(&read, VOLUME).unwrap();
    assert_eq!((fetch.start[0], fetch.shape[0]), (11, 9));

    let slice = |start: usize, len: usize| -> Voxels {
        let full = input.view::<f64>().unwrap();
        full.slice(ndarray::s![start..start + len, .., ..])
            .to_owned()
            .into()
    };
    let placed = |start: usize| {
        Placement::new(
            Anchor::new([start, 0, 0], VOLUME),
            Anchor::new([3, 0, 0], [7, 18, 14]),
        )
        .writing([3, 18, 14])
    };

    // The honest fetch computes, and its voxels are the whole-volume answer.
    let mut out = Voxels::zeros(Dtype::F64, [3, 18, 14]).unwrap();
    op.apply_placed(
        &slice(fetch.start[0], fetch.shape[0]),
        SourceInputs::new(&[]),
        &mut out,
        &placed(fetch.start[0]),
    )
    .unwrap();
    let whole = reference(&input, &stated);
    let want = whole
        .view::<f64>()
        .unwrap()
        .slice(ndarray::s![3..6, .., ..])
        .to_owned();
    assert_eq!(out.view::<f64>().unwrap(), want.view());

    // One voxel short on the far side, which is where a clamp would be silent.
    let mut out = Voxels::zeros(Dtype::F64, [3, 18, 14]).unwrap();
    let message = op
        .apply_placed(
            &slice(fetch.start[0], fetch.shape[0] - 1),
            SourceInputs::new(&[]),
            &mut out,
            &placed(fetch.start[0]),
        )
        .unwrap_err()
        .to_string();
    assert!(
        message.contains("reads source") && message.contains("was handed source"),
        "{message}"
    );
    // And one voxel short on the near side, which is the other end of the same
    // interval and would clamp just as quietly.
    let mut out = Voxels::zeros(Dtype::F64, [3, 18, 14]).unwrap();
    let message = op
        .apply_placed(
            &slice(fetch.start[0] + 1, fetch.shape[0] - 1),
            SourceInputs::new(&[]),
            &mut out,
            &placed(fetch.start[0] + 1),
        )
        .unwrap_err()
        .to_string();
    assert!(
        message.contains("bracket between source voxels"),
        "{message}"
    );
}
