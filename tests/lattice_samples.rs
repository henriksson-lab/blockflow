// SPDX-License-Identifier: MIT
// Original work for this crate.

//! `ops::lattice_samples` against the fused `LocalStatisticOp`, on the bits.
//!
//! The two compute the same samples from the same whole windows and blend them
//! by the same expression, so a plan built either way writes the same image —
//! under every decomposition, because each is decomposition-invariant on its
//! own. Checked with a population mask, a stepped element, two statistics
//! sharing a gather, an isodata threshold and a narrowing.

use std::collections::BTreeSet;
use std::sync::Arc;

use ndarray::Array3;

use blockflow::assemble::{ImageId, PlanBuilder};
use blockflow::dtype::Dtype;
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::op::Chain;
use blockflow::ops::element::{ElementShape, Rank, StructuringElement};
use blockflow::ops::lattice_samples::{append_lattice_sample_phases, LatticeSamplesOp};
use blockflow::ops::local::{
    Isodata, LatticeNarrowing, LocalStatistic, LocalStatisticOp, Sampling, Statistic,
};
use blockflow::ops::voxelwise::{FnCombine, VoxelwiseMaskOp};
use blockflow::strategy::{execute_phases, Hints};
use blockflow::voxels::Voxels;

const VOLUME: [usize; 3] = [19, 37, 31];

fn texture() -> Voxels {
    Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(z, y, x)| {
        let v = (z * 7 + y * 13 + x * 3) % 29 + (z * y + x * x) % 11;
        (v as f64 * 37.0).floor()
    })
    .into()
}

fn run(plan: PlanBuilder) -> Voxels {
    let built = plan.finish().expect("a plan");
    let last = built.decomposition.phases.len();
    let env = ArrayEnvironment::for_decomposition(texture(), &built.decomposition, [4, 8, 8])
        .expect("an environment");
    execute_phases(
        "lattice samples",
        &built.workflow,
        &built.decomposition,
        &Hints {
            keep_images: BTreeSet::from([ImageId::from(last)]),
            ..Hints::default()
        },
        &env,
        &[],
        &built.work(),
    )
    .expect("a run");
    env.try_image(last).expect("the output")
}

fn bits(voxels: &Voxels) -> Vec<u64> {
    voxels.widened().iter().map(|v| v.to_bits()).collect()
}

fn rank(element: &StructuringElement, spacing: [usize; 3], p: f64) -> LocalStatistic {
    LocalStatistic::sampled(
        element.clone(),
        Sampling::every(spacing),
        Statistic::Rank(Rank::ceiling_percentile(p).unwrap()),
    )
    .unwrap()
}

/// Two percentiles over a masked, stepped window, combined with the voxel.
fn masked_pair(grid: BlockGrid, split: bool) -> Voxels {
    let element =
        StructuringElement::from_size_stepped(ElementShape::Box, [3, 12, 10], [1, 2, 2]).unwrap();
    let spacing = [3, 7, 6];
    let low = rank(&element, spacing, 0.25);
    let high =
        rank(&element, spacing, 0.9).narrowed(LatticeNarrowing::through(Dtype::U16).unwrap());
    let combine = |x: &[f64]| x[0] * 0.5 + x[1] - 0.25 * x[2];
    let mut plan = PlanBuilder::new(VOLUME, Dtype::F64, grid);
    let mask = plan
        .pixels(Chain::op(VoxelwiseMaskOp::threshold("population", 300.0)))
        .unwrap()
        .image()
        .unwrap();
    let image = ImageId::from(0usize);
    if split {
        let samples = LatticeSamplesOp::new("samples", "pair", image, vec![low, high])
            .unwrap()
            .masked_by(mask);
        append_lattice_sample_phases(
            &mut plan,
            samples,
            "combine",
            vec![image],
            Arc::new(combine),
            Dtype::F64,
        )
        .unwrap();
    } else {
        let source = || Chain::source(image, Dtype::F64);
        plan.pixels(
            Chain::parallel(
                vec![
                    source(),
                    Chain::sequence(vec![
                        source(),
                        Chain::op(LocalStatisticOp::new("low", low).masked_by(mask)),
                    ]),
                    Chain::sequence(vec![
                        source(),
                        Chain::op(LocalStatisticOp::new("high", high).masked_by(mask)),
                    ]),
                ],
                Box::new(FnCombine::new("combine", 3, combine).unwrap()),
            )
            .unwrap(),
        )
        .unwrap();
    }
    run(plan)
}

/// An unmasked isodata threshold, compared into a `u16` written image.
fn isodata(grid: BlockGrid, split: bool) -> Voxels {
    let element = StructuringElement::from_size(ElementShape::Box, [3, 16, 16]).unwrap();
    let statistic = LocalStatistic::sampled(
        element,
        Sampling::every([3, 8, 8]),
        Statistic::Isodata(Isodata::new(64, 0.0).unwrap()),
    )
    .unwrap();
    let combine = |x: &[f64]| if x[0] > x[1] { 7.0 } else { 2.0 };
    let image = ImageId::from(0usize);
    let mut plan = PlanBuilder::new(VOLUME, Dtype::F64, grid);
    if split {
        let samples = LatticeSamplesOp::new("samples", "isodata", image, vec![statistic]).unwrap();
        append_lattice_sample_phases(
            &mut plan,
            samples,
            "combine",
            vec![image],
            Arc::new(combine),
            Dtype::U16,
        )
        .unwrap();
    } else {
        plan.pixels(
            Chain::parallel(
                vec![
                    Chain::op(blockflow::ops::voxelwise::CarryOp::new("carry")),
                    Chain::op(LocalStatisticOp::new("isodata", statistic)),
                ],
                Box::new(
                    FnCombine::new("combine", 2, combine)
                        .unwrap()
                        .producing(Dtype::U16)
                        .unwrap(),
                ),
            )
            .unwrap(),
        )
        .unwrap();
    }
    run(plan)
}

fn grids() -> Vec<BlockGrid> {
    [
        [19, 37, 31],
        [4, 37, 31],
        [5, 9, 31],
        [7, 16, 11],
        [3, 8, 8],
    ]
    .into_iter()
    .map(|block| BlockGrid::new(VOLUME, block).unwrap())
    .collect()
}

#[test]
fn a_masked_pair_of_percentiles_is_the_fused_answer_on_every_grid() {
    let reference = bits(&masked_pair(BlockGrid::new(VOLUME, VOLUME).unwrap(), false));
    for grid in grids() {
        let block = grid.block();
        assert_eq!(
            bits(&masked_pair(grid.clone(), false)),
            reference,
            "fused, block {block:?}"
        );
        assert_eq!(
            bits(&masked_pair(grid, true)),
            reference,
            "split, block {block:?}"
        );
    }
}

#[test]
fn an_isodata_threshold_is_the_fused_answer_on_every_grid() {
    let reference = isodata(BlockGrid::new(VOLUME, VOLUME).unwrap(), false);
    assert_eq!(reference.dtype(), Dtype::U16);
    for grid in grids() {
        let block = grid.block();
        let split = isodata(grid, true);
        assert_eq!(split.dtype(), Dtype::U16);
        assert_eq!(bits(&split), bits(&reference), "split, block {block:?}");
    }
}

#[test]
fn statistics_over_different_windows_are_refused() {
    let a = rank(
        &StructuringElement::from_size(ElementShape::Box, [3, 3, 3]).unwrap(),
        [2, 2, 2],
        0.5,
    );
    let b = rank(
        &StructuringElement::from_size(ElementShape::Box, [3, 5, 3]).unwrap(),
        [2, 2, 2],
        0.5,
    );
    let refused = LatticeSamplesOp::new("samples", "s", ImageId::from(0usize), vec![a, b]);
    assert!(refused.is_err());
}

/// Timing, not correctness: the adaptive-threshold arm of the vessel example
/// at its own parameters, on a z-slab grid, fused against split, one worker.
/// Ignored because it takes tens of seconds; run with `--ignored --nocapture`.
#[test]
#[ignore]
fn timing_an_isodata_lattice_on_a_slab_grid() {
    let volume = [48usize, 512, 404];
    let input: Voxels = Array3::from_shape_fn((volume[0], volume[1], volume[2]), |(z, y, x)| {
        ((z * 7 + y * 13 + x * 3) % 997) as f64
    })
    .into();
    let statistic = || {
        LocalStatistic::sampled(
            StructuringElement::from_size(ElementShape::Box, [3, 250, 250]).unwrap(),
            Sampling::every([3, 50, 50]),
            Statistic::Isodata(Isodata::new(256, 0.0).unwrap()),
        )
        .unwrap()
    };
    let combine = |x: &[f64]| if x[0] > x[1] { 1.0 } else { 0.0 };
    for split in [false, true, false, true] {
        let grid = BlockGrid::new(volume, [8, 512, 404]).unwrap();
        let mut plan = PlanBuilder::new(volume, Dtype::F64, grid);
        let image = ImageId::from(0usize);
        if split {
            let samples =
                LatticeSamplesOp::new("samples", "isodata", image, vec![statistic()]).unwrap();
            append_lattice_sample_phases(
                &mut plan,
                samples,
                "combine",
                vec![image],
                Arc::new(combine),
                Dtype::F64,
            )
            .unwrap();
        } else {
            plan.pixels(
                Chain::parallel(
                    vec![
                        Chain::op(blockflow::ops::voxelwise::CarryOp::new("carry")),
                        Chain::op(LocalStatisticOp::new("isodata", statistic())),
                    ],
                    Box::new(FnCombine::new("combine", 2, combine).unwrap()),
                )
                .unwrap(),
            )
            .unwrap();
        }
        let built = plan.finish().unwrap();
        let env =
            ArrayEnvironment::for_decomposition(input.clone(), &built.decomposition, [8, 64, 64])
                .unwrap();
        let started = std::time::Instant::now();
        execute_phases(
            "timing",
            &built.workflow,
            &built.decomposition,
            &Hints {
                concurrency: 1,
                ..Hints::default()
            },
            &env,
            &[],
            &built.work(),
        )
        .unwrap();
        eprintln!(
            "{}: {:.2} s",
            if split { "split" } else { "fused" },
            started.elapsed().as_secs_f64()
        );
    }
}

/// The redundancy the fused form pays: one evaluation per sample where a
/// block holds an axis whole, more where halos overlap.
#[test]
fn the_fused_form_evaluates_overlapping_halos_more_than_once() {
    let volume = [128usize, 512, 404];
    let equalize = rank(
        &StructuringElement::from_size(ElementShape::Box, [5, 200, 200]).unwrap(),
        [5, 50, 50],
        0.4,
    );
    let lightsheet = rank(
        &StructuringElement::from_size_stepped(ElementShape::Box, [1, 200, 200], [1, 2, 2])
            .unwrap(),
        [1, 25, 25],
        0.25,
    );
    let whole = BlockGrid::new(volume, volume).unwrap();
    let slabs = BlockGrid::new(volume, [8, 512, 404]).unwrap();
    let bricks = BlockGrid::new(volume, [64, 256, 256]).unwrap();
    assert_eq!(equalize.evaluations_per_sample(&whole).unwrap(), 1.0);
    assert_eq!(lightsheet.evaluations_per_sample(&slabs).unwrap(), 1.0);
    let slab = equalize.evaluations_per_sample(&slabs).unwrap();
    let brick = lightsheet.evaluations_per_sample(&bricks).unwrap();
    eprintln!("equalize on 8-plane slabs: {slab:.3}; lightsheet on 64x256x256: {brick:.3}");
    assert!(slab > 2.0, "{slab}");
    assert!(brick > 1.5, "{brick}");
}

/// Timing, not correctness: the equalize arm's two masked percentiles at the
/// vessel example's parameters, fused against split, one worker.
#[test]
#[ignore]
fn timing_a_masked_pair_of_percentiles() {
    for (volume, block) in [
        ([64usize, 512, 256], [32usize, 512, 256]),
        ([40, 512, 404], [8, 512, 404]),
    ] {
        let input: Voxels =
            Array3::from_shape_fn((volume[0], volume[1], volume[2]), |(z, y, x)| {
                ((z * 7 + y * 13 + x * 3) % 3001) as f64
            })
            .into();
        let element = StructuringElement::from_size(ElementShape::Box, [5, 200, 200]).unwrap();
        let low = rank(&element, [5, 50, 50], 0.4);
        let high = rank(&element, [5, 50, 50], 0.975);
        let combine = |x: &[f64]| x[0] * x[1] / (x[2] + 1.0);
        for split in [false, true, false, true] {
            let grid = BlockGrid::new(volume, block).unwrap();
            let mut plan = PlanBuilder::new(volume, Dtype::F64, grid);
            let mask = plan
                .pixels(Chain::op(VoxelwiseMaskOp::threshold("population", 100.0)))
                .unwrap()
                .image()
                .unwrap();
            let image = ImageId::from(0usize);
            if split {
                let samples = LatticeSamplesOp::new(
                    "samples",
                    "pair",
                    image,
                    vec![low.clone(), high.clone()],
                )
                .unwrap()
                .masked_by(mask);
                append_lattice_sample_phases(
                    &mut plan,
                    samples,
                    "combine",
                    vec![image],
                    Arc::new(combine),
                    Dtype::F64,
                )
                .unwrap();
            } else {
                let source = || Chain::source(image, Dtype::F64);
                plan.pixels(
                    Chain::parallel(
                        vec![
                            source(),
                            Chain::sequence(vec![
                                source(),
                                Chain::op(
                                    LocalStatisticOp::new("low", low.clone()).masked_by(mask),
                                ),
                            ]),
                            Chain::sequence(vec![
                                source(),
                                Chain::op(
                                    LocalStatisticOp::new("high", high.clone()).masked_by(mask),
                                ),
                            ]),
                        ],
                        Box::new(FnCombine::new("combine", 3, combine).unwrap()),
                    )
                    .unwrap(),
                )
                .unwrap();
            }
            let built = plan.finish().unwrap();
            let env = ArrayEnvironment::for_decomposition(
                input.clone(),
                &built.decomposition,
                [8, 64, 64],
            )
            .unwrap();
            let started = std::time::Instant::now();
            execute_phases(
                "timing",
                &built.workflow,
                &built.decomposition,
                &Hints {
                    concurrency: 1,
                    ..Hints::default()
                },
                &env,
                &[],
                &built.work(),
            )
            .unwrap();
            eprintln!(
                "{volume:?} on {block:?}, {}: {:.2} s",
                if split { "split" } else { "fused" },
                started.elapsed().as_secs_f64()
            );
        }
    }
}
