// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// Where the `cost_per_voxel` figures in this module come from.
//
// `BlockOp::cost_per_voxel` says the number must be measured, and this is the
// measurement. It is ordinary code rather than a benchmark harness for two
// reasons: it runs through `BlockOp::apply`, which is the entry point the
// executor uses and therefore the thing whose cost the planner is pricing; and
// it can be re-run on another machine by calling one function, which is the only
// way a number like this stays true.
//
// Robustness, and why the minimum
// -------------------------------
// `docs/design/BLOCK_OPS.md` §"Estimating from noisy samples" observes that the
// contamination on a shared machine is **one-sided** — contention can only make
// a sample slower — so the low-order statistic is the right one and a mean over
// such samples is worthless. The minimum of several repetitions is taken here
// for exactly that reason.
//
// What is reported, and what is stored
// ------------------------------------
// The report gives nanoseconds per voxel and the ratio to the voxelwise map,
// which is the unit the module's constants are expressed in. For a
// neighbourhood op the useful figure is the ratio **per element voxel**, because
// the op's cost scales with its element and a single number would be right for
// one filter size only; the report prints that column too, and those are the
// constants the ops multiply by.

use std::time::Instant;

use ndarray::Array3;

use crate::op::{Anchor, BlockOp};

use super::element::{ElementShape, Rank, StructuringElement};
use super::local::{AdaptiveThresholdOp, Isodata, LocalStatistic, LocalStatisticOp, Statistic};
use super::morphology::{Morphology, MorphologyOp};
use super::rank::RankFilterOp;
use super::smooth::{Gaussian, SmoothOp};
use super::structure_tensor::{Eigenvalue, StructureTensor, StructureTensorOp};
use super::voxelwise::{CombineOp, Logic, VoxelwiseMapOp};

/// One measured op.
#[derive(Debug, Clone)]
pub struct Sample {
    pub name: String,
    /// The best of the repetitions, per voxel of the volume.
    pub nanos_per_voxel: f64,
    /// Relative to the voxelwise map, which is the module's unit of work.
    pub relative: f64,
    /// How many voxels the op's neighbourhood holds, where it has one. The
    /// per-element-voxel ratio is `relative / divisor`, and it is that figure the
    /// ops store, because it is the one that does not depend on the filter size
    /// this measurement happened to use.
    pub divisor: f64,
}

/// Time every op in this module over `shape`, taking the best of `repetitions`.
pub fn measure(shape: [usize; 3], repetitions: usize) -> Vec<Sample> {
    let voxels = (shape[0] * shape[1] * shape[2]) as f64;
    let input = ramp(shape);
    let anchor = Anchor::whole(shape);
    let operand = std::sync::Arc::new(ramp(shape));

    let box27 = StructuringElement::from_size(ElementShape::Box, [3, 3, 3]).unwrap();
    let window = StructuringElement::from_size(ElementShape::Box, [9, 9, 9]).unwrap();
    let median = Rank::median(&box27);
    let narrow = Gaussian::isotropic(1.0, 3.0).unwrap();
    let wide = Gaussian::isotropic(2.0, 3.0).unwrap();

    let cases: Vec<(String, Box<dyn BlockOp>, f64)> = vec![
        (
            "voxelwise map".to_string(),
            Box::new(VoxelwiseMapOp::threshold("map", 500.0, 1.0, 0.0)),
            1.0,
        ),
        (
            "combine (AND against a second volume)".to_string(),
            Box::new(CombineOp::new("combine", Logic::And, operand)),
            1.0,
        ),
        (
            format!("rank filter, {}-voxel element", box27.len()),
            Box::new(RankFilterOp::new("rank", box27.clone(), median)),
            box27.len() as f64,
        ),
        (
            format!("erode, {}-voxel element", box27.len()),
            Box::new(MorphologyOp::new("erode", Morphology::Erode, box27.clone())),
            box27.len() as f64,
        ),
        (
            format!("open, {}-voxel element (two passes)", box27.len()),
            Box::new(MorphologyOp::new("open", Morphology::Open, box27.clone())),
            2.0 * box27.len() as f64,
        ),
        (
            format!("local mean, {}-voxel window, spacing 16", box27.len()),
            Box::new(LocalStatisticOp::new(
                "mean",
                LocalStatistic::new(box27.clone(), [16, 16, 16], Statistic::Mean).unwrap(),
            )),
            box27.len() as f64 / 4096.0,
        ),
        (
            format!("local mean, {}-voxel window, spacing 8", window.len()),
            Box::new(LocalStatisticOp::new(
                "mean",
                LocalStatistic::new(window.clone(), [8, 8, 8], Statistic::Mean).unwrap(),
            )),
            window.len() as f64 / 512.0,
        ),
        (
            format!("local mean, {}-voxel window, spacing 4", window.len()),
            Box::new(LocalStatisticOp::new(
                "mean",
                LocalStatistic::new(window.clone(), [4, 4, 4], Statistic::Mean).unwrap(),
            )),
            window.len() as f64 / 64.0,
        ),
        (
            format!(
                "adaptive threshold, {}-voxel window, spacing 8",
                window.len()
            ),
            Box::new(AdaptiveThresholdOp::new(
                "adaptive",
                LocalStatistic::new(window.clone(), [8, 8, 8], Statistic::Mean).unwrap(),
                1.0,
                0.0,
            )),
            window.len() as f64 / 512.0,
        ),
        // Isodata at two bin counts and two window sizes, and a mean beside
        // each, for the same reason as the two smoothings below: the question
        // about this reducer's cost is whether it scales with the window or
        // with the bin count, and the answer is **both** — it walks the window
        // three times to bin it and the histogram twice more. One sample could
        // not have told them apart. Each isodata row is read against the mean
        // row with the same window and lattice, because the difference between
        // the two is exactly the reducer, the gather being identical.
        (
            format!("local mean, {}-voxel window, spacing 8", box27.len()),
            Box::new(LocalStatisticOp::new(
                "mean",
                LocalStatistic::new(box27.clone(), [8, 8, 8], Statistic::Mean).unwrap(),
            )),
            box27.len() as f64 / 512.0,
        ),
        (
            format!(
                "local isodata, {}-voxel window, 256 bins, spacing 8",
                box27.len()
            ),
            Box::new(LocalStatisticOp::new(
                "isodata",
                LocalStatistic::new(
                    box27.clone(),
                    [8, 8, 8],
                    Statistic::Isodata(Isodata::new(256, 1.0).unwrap()),
                )
                .unwrap(),
            )),
            256.0 / 512.0,
        ),
        (
            format!(
                "local isodata, {}-voxel window, 64 bins, spacing 8",
                window.len()
            ),
            Box::new(LocalStatisticOp::new(
                "isodata",
                LocalStatistic::new(
                    window.clone(),
                    [8, 8, 8],
                    Statistic::Isodata(Isodata::new(64, 1.0).unwrap()),
                )
                .unwrap(),
            )),
            64.0 / 512.0,
        ),
        (
            format!(
                "local isodata, {}-voxel window, 256 bins, spacing 8",
                window.len()
            ),
            Box::new(LocalStatisticOp::new(
                "isodata",
                LocalStatistic::new(
                    window.clone(),
                    [8, 8, 8],
                    Statistic::Isodata(Isodata::new(256, 1.0).unwrap()),
                )
                .unwrap(),
            )),
            256.0 / 512.0,
        ),
        // Two smoothings rather than one, because the whole question about a
        // separable convolution's cost is whether it scales with the sum of the
        // kernel lengths or with their product, and one sample cannot answer it.
        // The divisor is the tap count, so the `per element` column is the
        // per-tap figure `smooth.rs` stores — and if the two rows disagree
        // badly, the model is the wrong shape rather than the constant being
        // slightly off.
        (
            format!(
                "gaussian smooth, sigma 1 truncate 3 ({} taps)",
                narrow.taps()
            ),
            Box::new(SmoothOp::new("smooth", narrow.clone())),
            narrow.taps() as f64,
        ),
        (
            format!("gaussian smooth, sigma 2 truncate 3 ({} taps)", wide.taps()),
            Box::new(SmoothOp::new("smooth", wide.clone())),
            wide.taps() as f64,
        ),
        // **Three rows, because the structure tensor's cost has two terms and a
        // shape.** One separable smoothing of the intensity, then *six* of the
        // tensor's components, then a fixed per-voxel slab — the gradient
        // stencil, the six products and a cubic root-finder. Two unknowns and an
        // asymmetry: a single row would fit any of them.
        //
        // The rows are chosen so the three questions are separable. The first
        // and second differ only in the derivative scale, the first and third
        // only in the integration scale, and the second and third have the same
        // total kernel width — so the gap between them is the six-fold factor
        // and nothing else. The divisor is the tap count `cost_for` charges,
        // `taps(sigma) + 6 taps(rho)`, so the `per element` column is directly
        // comparable with the smoothing rows above it: if it does not land near
        // their per-tap figure, the model is the wrong shape rather than the
        // constant being off.
        // **The anchor.** `ops::ridge` decomposes the same six numbers with the
        // same closed form, so the ratio between these two rows is what decides
        // whether the structure tensor's per-voxel slab is priced consistently
        // with its only sibling. Without it the constant below would be in the
        // units of whatever day it was taken, and `mod.rs` is explicit that the
        // *relative* weighting between op families is the part a re-measurement
        // cannot repair.
        (
            "ridge, sigmas [1.0], truncate 3 (21 taps)".to_string(),
            Box::new(super::ridge::RidgeFilterOp::new(
                "ridge",
                super::ridge::ScaleSpace::isotropic(&[1.0], 3.0, 1.0).unwrap(),
                super::ridge::RidgeResponse::new(0.5, 0.5, 0.05, super::ridge::Polarity::Ridge)
                    .unwrap(),
            )),
            21.0,
        ),
        structure_tensor_case("sigma 1, rho 1", [1.0; 3], [1.0; 3]),
        structure_tensor_case("sigma 3, rho 1", [3.0; 3], [1.0; 3]),
        structure_tensor_case("sigma 1, rho 3", [1.0; 3], [3.0; 3]),
    ];

    let mut raw = Vec::new();
    for (name, op, divisor) in cases {
        // One buffer for the whole case, and one untimed pass over it before
        // any measurement: a freshly allocated output pays a page fault per
        // page on first touch, and at a few nanoseconds per voxel that fault
        // *is* the measurement for the cheapest ops. What is wanted is the
        // steady-state compute, which is what a run spends its time in.
        let mut out =
            crate::voxels::Voxels::zeros(op.produces(input.dtype()), op.output_shape(shape))
                .unwrap();
        op.apply(&input, &mut out, &anchor).unwrap();
        let mut best = f64::INFINITY;
        for _ in 0..repetitions.max(1) {
            let started = Instant::now();
            op.apply(&input, &mut out, &anchor).unwrap();
            let elapsed = started.elapsed().as_secs_f64() * 1e9;
            // consume the result so nothing is elided
            std::hint::black_box(out.view::<f64>().unwrap().iter().take(1).sum::<f64>());
            best = best.min(elapsed / voxels);
        }
        raw.push((name, best, divisor));
    }

    let unit = raw.first().map(|(_, nanos, _)| *nanos).unwrap_or(1.0);
    raw.into_iter()
        .map(|(name, nanos_per_voxel, divisor)| Sample {
            name,
            nanos_per_voxel,
            relative: nanos_per_voxel / unit,
            divisor,
        })
        .collect()
}

/// The measurement as a table, for pasting into a report or a header.
pub fn report(shape: [usize; 3], repetitions: usize) -> String {
    let samples = measure(shape, repetitions);
    let mut out = format!(
        "op cost, {}x{}x{}, best of {repetitions}\n\
         {:<52} {:>10} {:>10} {:>14}\n",
        shape[0], shape[1], shape[2], "op", "ns/voxel", "relative", "per element"
    );
    for sample in &samples {
        out.push_str(&format!(
            "{:<52} {:>10.3} {:>10.2} {:>14.3}\n",
            sample.name,
            sample.nanos_per_voxel,
            sample.relative,
            sample.relative / sample.divisor.max(1e-12)
        ));
    }
    out
}

/// One structure tensor row: the op, and the tap count `cost_for` charges for
/// it. `Largest` because which eigenvalue is emitted cannot change the cost —
/// all three come out of the same decomposition and one is copied out.
fn structure_tensor_case(
    label: &str,
    sigma: [f64; 3],
    rho: [f64; 3],
) -> (String, Box<dyn BlockOp>, f64) {
    let truncate = 3.0;
    let taps = |scale: [f64; 3]| -> usize {
        (0..3)
            .map(|axis| 2 * super::ridge::gaussian_radius(scale[axis], truncate) + 1)
            .sum::<usize>()
    };
    let charged = taps(sigma) + 6 * taps(rho);
    (
        format!("structure tensor, {label} truncate 3 ({charged} taps charged)"),
        Box::new(StructureTensorOp::new(
            "structure tensor",
            StructureTensor::new(sigma, rho, truncate).unwrap(),
            Eigenvalue::Largest,
        )),
        charged as f64,
    )
}

fn ramp(shape: [usize; 3]) -> crate::voxels::Voxels {
    let mut array = Array3::<f64>::zeros((shape[0], shape[1], shape[2]));
    for (flat, value) in array.iter_mut().enumerate() {
        *value = ((flat * 7919) % 1013) as f64;
    }
    array.into()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Retaking the measurement. Ignored because timing in a test suite is a
    /// measurement of the machine's mood, not of the code — but it is here, it
    /// runs, and it is one command:
    ///
    /// ```text
    /// cargo test --release -- --ignored --nocapture ops::cost
    /// ```
    #[test]
    #[ignore = "a measurement, not an assertion"]
    fn print_the_cost_table() {
        println!("{}", report([96, 64, 64], 5));
    }

    /// What *can* be asserted about a measured cost without measuring: that the
    /// ordering the constants encode is the ordering the ops actually have. A
    /// neighbourhood op must cost more per voxel than a voxelwise one, and an
    /// opening must cost more than the erosion it contains. If a future change
    /// inverts either, the stored constants are describing a different program.
    #[test]
    fn the_stored_costs_keep_the_order_the_measurement_found() {
        let element = StructuringElement::from_size(ElementShape::Box, [3, 3, 3]).unwrap();
        let map = VoxelwiseMapOp::new("map", |value| value);
        let rank = RankFilterOp::median("median", element.clone());
        let erode = MorphologyOp::new("erode", Morphology::Erode, element.clone());
        let open = MorphologyOp::new("open", Morphology::Open, element.clone());
        assert!(rank.cost_per_voxel() > map.cost_per_voxel());
        assert!(erode.cost_per_voxel() > map.cost_per_voxel());
        assert!(open.cost_per_voxel() > erode.cost_per_voxel());

        // and a bigger element costs more, which is why the constant is per
        // element voxel rather than per op
        let bigger = StructuringElement::from_size(ElementShape::Box, [7, 7, 7]).unwrap();
        assert!(
            RankFilterOp::median("median", bigger.clone()).cost_per_voxel() > rank.cost_per_voxel()
        );

        // Separability, as an ordering the measurement found and the constants
        // must keep: a Gaussian reaching three voxels touches 21 taps where a
        // 7x7x7 rank filter touches 343, and it must be priced far below it. A
        // model that had charged the kernel's volume would invert this.
        let smooth = SmoothOp::new("smooth", Gaussian::isotropic(1.0, 3.0).unwrap());
        let dense = RankFilterOp::median("median", bigger);
        assert_eq!(smooth.reach(0, 1000), 3);
        assert!(smooth.cost_per_voxel() < dense.cost_per_voxel() / 8.0);
    }

    /// A lattice divides the window's cost: sampling every eighth voxel on each
    /// axis evaluates the window 512 times less often.
    #[test]
    fn a_coarser_lattice_costs_less_per_voxel() {
        let window = StructuringElement::from_size(ElementShape::Box, [9, 9, 9]).unwrap();
        let dense = LocalStatisticOp::new(
            "dense",
            LocalStatistic::new(window.clone(), [1, 1, 1], Statistic::Mean).unwrap(),
        );
        let sparse = LocalStatisticOp::new(
            "sparse",
            LocalStatistic::new(window, [8, 8, 8], Statistic::Mean).unwrap(),
        );
        assert!(dense.cost_per_voxel() > sparse.cost_per_voxel());
    }
}
