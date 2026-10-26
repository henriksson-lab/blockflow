// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Deep vessel filling against the reference's, on masks of a real
//! vasculature crop that `reference-clearmap/run_deep_fill.sh` prepared and
//! ran the reference on. Needs the `deep-fill` feature, the converted weights
//! (`$VESSEL_FILL_WEIGHTS`) and the reference's outputs under
//! `$VESSEL_FILL_REFERENCE` (default
//! `/husky/henriksson/lightsheet-reference/vessel-filling`).

#![cfg(feature = "deep-fill")]

use std::path::{Path, PathBuf};
use std::time::Instant;

use blockflow::npy::{read_array_file, OrderPolicy};
use blockflow::voxels::Voxels;
use blockflow_vessel_graph::segment::{
    deep_fill, DeepFillDevice, DeepFillOutput, DeepFillParams, DeepFillSetup,
};
use ndarray::{Array3, Ix3};

fn reference() -> PathBuf {
    PathBuf::from(
        std::env::var("VESSEL_FILL_REFERENCE").unwrap_or_else(|_| {
            "/husky/henriksson/lightsheet-reference/vessel-filling".to_string()
        }),
    )
}

/// Where the comparisons run the network: `$VESSEL_FILL_DEVICE` (cpu, gpu or
/// auto), default cpu.
fn device() -> DeepFillDevice {
    std::env::var("VESSEL_FILL_DEVICE")
        .map(|text| DeepFillDevice::parse(&text).expect("cpu, gpu or auto"))
        .unwrap_or(DeepFillDevice::Cpu)
}

fn setup() -> DeepFillSetup {
    DeepFillSetup {
        device: device(),
        ..DeepFillSetup::default()
    }
}

fn load<T: Clone + blockflow::npy::NpyElement>(path: &Path) -> Array3<T> {
    let array: ndarray::ArrayD<T> = read_array_file(path, OrderPolicy::Either)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    array.into_dimensionality::<Ix3>().expect("a volume")
}

/// Max absolute difference and Pearson correlation over the voxels where
/// `want` is a number.
fn agreement(got: &Array3<f32>, want: &Array3<f32>) -> (f64, f64, usize) {
    let (mut n, mut sa, mut sb, mut saa, mut sbb, mut sab, mut max) =
        (0usize, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0f64);
    for (&a, &b) in got.iter().zip(want) {
        if b.is_nan() {
            continue;
        }
        let (a, b) = (a as f64, b as f64);
        n += 1;
        sa += a;
        sb += b;
        saa += a * a;
        sbb += b * b;
        sab += a * b;
        max = max.max((a - b).abs());
    }
    let n_f = n as f64;
    let cov = sab / n_f - sa / n_f * sb / n_f;
    let var = (saa / n_f - (sa / n_f).powi(2)) * (sbb / n_f - (sb / n_f).powi(2));
    (max, cov / var.sqrt(), n)
}

#[test]
#[ignore = "needs the converted weights ($VESSEL_FILL_WEIGHTS) and dev/small_{mask,prob}.npy \
            under $VESSEL_FILL_REFERENCE, written by the reference's network in evaluation mode"]
fn small_crop_probability_matches_the_reference_network() {
    let dir = reference().join("dev");
    let mask: Array3<bool> = load(&dir.join("small_mask.npy"));
    let want: Array3<f32> = load(&dir.join("small_prob.npy"));
    let shape = [mask.shape()[0], mask.shape()[1], mask.shape()[2]];
    let started = Instant::now();
    let got = deep_fill(
        Voxels::Bool(mask),
        &DeepFillParams::default(),
        &DeepFillSetup {
            block: shape,
            ..setup()
        },
        DeepFillOutput::Probability,
    )
    .unwrap();
    let seconds = started.elapsed().as_secs_f64();
    let got = got.view::<f32>().unwrap().to_owned();
    let (max, r, n) = agreement(&got, &want);
    let flips = got
        .iter()
        .zip(&want)
        .filter(|(a, b)| (**a >= 0.5) != (**b >= 0.5))
        .count();
    println!("{shape:?}: max |diff| {max:.2e}, r {r:.7}, {n} voxels, {flips} threshold flips, {seconds:.2}s");
    assert!(max < 1e-3, "max difference {max}");
}

fn count(mask: &Array3<bool>) -> usize {
    mask.iter().filter(|set| **set).count()
}

fn dice(a: &Array3<bool>, b: &Array3<bool>) -> f64 {
    let both = a.iter().zip(b).filter(|(x, y)| **x && **y).count();
    2.0 * both as f64 / (count(a) + count(b)).max(1) as f64
}

/// `mask` without its last `margin` planes on each axis's far side: where the
/// reference's sink is left at zero (its network output is shorter than its
/// input).
fn inner(mask: &Array3<bool>, margin: usize) -> Array3<bool> {
    let s = mask.shape();
    mask.slice(ndarray::s![
        ..s[0] - margin,
        ..s[1] - margin,
        ..s[2] - margin
    ])
    .to_owned()
}

/// The two channels' filled masks united, smoothed and filled again,
/// skeletonised and made a graph, as `segment` and `graph` do: vertices,
/// edges and total edge length in µm.
fn graph_of(vessel: &Array3<bool>, artery: &Array3<bool>) -> (usize, usize, f64, usize) {
    use blockflow_vessel_graph::graph::{vessel_graph, GraphParams, Inputs};
    use blockflow_vessel_graph::segment::{mask_steps, MaskSteps, PostProcess, SkeletonParams};

    let union = ndarray::Zip::from(vessel)
        .and(artery)
        .map_collect(|a, b| *a || *b);
    let shape = [union.shape()[0], union.shape()[1], union.shape()[2]];
    let (_, mask, skeleton) = mask_steps(
        Voxels::Bool(union),
        &MaskSteps {
            post: Some(PostProcess::default()),
            skeleton: Some(SkeletonParams {
                delete_border: true,
                ..SkeletonParams::default()
            }),
        },
        [64, 256, 256].map(|b: usize| b.min(shape[0].max(shape[1]).max(shape[2]))),
    )
    .unwrap();
    let skeleton = skeleton.unwrap();
    let artery = Voxels::Bool(artery.clone());
    let spacing = [1.6, 1.625, 1.625];
    let inputs = Inputs {
        vessels: Some(&mask),
        artery_mask: Some(&artery),
        ..Inputs::new(&skeleton, spacing)
    };
    let graph = vessel_graph(&inputs, &GraphParams::default()).unwrap();
    let (_, um) = graph.edge_lengths(spacing).unwrap();
    let mask_voxels = mask.view::<bool>().unwrap().iter().filter(|s| **s).count();
    (
        graph.n_vertices(),
        graph.n_edges(),
        um.iter().sum(),
        mask_voxels,
    )
}

#[test]
#[ignore = "needs the converted weights ($VESSEL_FILL_WEIGHTS) and the crop's masks and the \
            reference's filled masks under $VESSEL_FILL_REFERENCE/crop, made by \
            reference-clearmap/run_deep_fill.sh"]
fn crop_against_the_reference() {
    let dir = reference().join("crop");
    let roles = [
        ("vessel", "vessel-vessel-mask", 1usize),
        ("artery", "vessel-artery-mask", 2),
    ];
    let variants = ["cpu-eval", "cpu", "cpu-2", "gpu"];
    let mut ours_filled = Vec::new();
    let mut theirs_filled: Vec<Vec<Option<Array3<bool>>>> = Vec::new();
    for (role, label, resample) in roles {
        let input: Array3<bool> = load(&dir.join("input").join(format!("{label}.npy")));
        let shape = [input.shape()[0], input.shape()[1], input.shape()[2]];
        let params = DeepFillParams::resampled(resample);
        let setup = setup();
        let started = Instant::now();
        let filled = deep_fill(
            Voxels::Bool(input.clone()),
            &params,
            &setup,
            DeepFillOutput::Filled,
        )
        .unwrap()
        .view::<bool>()
        .unwrap()
        .to_owned();
        let filled_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let probability = deep_fill(
            Voxels::Bool(input.clone()),
            &params,
            &setup,
            DeepFillOutput::Probability,
        )
        .unwrap()
        .view::<f32>()
        .unwrap()
        .to_owned();
        let probability_seconds = started.elapsed().as_secs_f64();
        let n_in = count(&input);
        println!(
            "{role} {shape:?}, resample {resample}: input {n_in} voxels; ours fills to {} (+{}) in \
             {filled_seconds:.2}s (probability {probability_seconds:.2}s)",
            count(&filled),
            count(&filled) - n_in
        );
        let reference_probability = dir
            .join("clearmap")
            .join(format!("{role}-cpu-eval/probability.npy"));
        if reference_probability.exists() {
            let want: Array3<f32> = load(&reference_probability);
            let (max, r, n) = agreement(&probability, &want);
            let flips = probability
                .iter()
                .zip(&want)
                .filter(|(a, b)| !b.is_nan() && ((**a >= 0.5) != (**b >= 0.5)))
                .count();
            println!(
                "  probability against the reference's network (whole crop, evaluation mode, \
                 {n} voxels): max |diff| {max:.2e}, r {r:.7}, {flips} voxels on the other side of 0.5"
            );
            assert!(max < 1e-3, "{role}: probability max difference {max}");
            // The reference's network over the crop as one block, thresholded
            // and OR-ed with the mask as its `fill_vessels` does, where its
            // output reaches.
            let reached = want.mapv(|p| !p.is_nan());
            let one_block = ndarray::Zip::from(&input)
                .and(&want)
                .map_collect(|m, p| *m || *p >= 0.5);
            let ours_reached = ndarray::Zip::from(&filled)
                .and(&reached)
                .map_collect(|f, r| *f && *r);
            let theirs_reached = ndarray::Zip::from(&one_block)
                .and(&reached)
                .map_collect(|f, r| *f && *r);
            println!(
                "  against that network as one block, OR the mask, where it reaches: Dice {:.5}",
                dice(&ours_reached, &theirs_reached)
            );
        }
        let mut row = Vec::new();
        for variant in variants {
            let path = dir
                .join("clearmap")
                .join(format!("{role}-{variant}/filled.npy"));
            if !path.exists() {
                row.push(None);
                continue;
            }
            let theirs: Array3<bool> = load(&path);
            let margin = 8 * resample;
            println!(
                "  reference {variant:8}: {} (+{}); Dice {:.4}, without the far {margin} planes {:.4}; \
                 input voxels it drops {}",
                count(&theirs),
                count(&theirs) as i64 - n_in as i64,
                dice(&filled, &theirs),
                dice(&inner(&filled, margin), &inner(&theirs, margin)),
                input.iter().zip(&theirs).filter(|(a, b)| **a && !**b).count()
            );
            row.push(Some(theirs));
        }
        ours_filled.push(filled);
        theirs_filled.push(row);
    }
    let (v, e, length, mask) = graph_of(&ours_filled[0], &ours_filled[1]);
    println!(
        "graph, ours:             mask {mask}, {v} vertices, {e} edges, length {length:.0} µm"
    );
    for (index, variant) in variants.iter().enumerate() {
        if let (Some(a), Some(b)) = (&theirs_filled[0][index], &theirs_filled[1][index]) {
            let (v, e, length, mask) = graph_of(a, b);
            println!(
                "graph, reference {variant:8}: mask {mask}, {v} vertices, {e} edges, length {length:.0} µm"
            );
        }
    }
    let input_v: Array3<bool> = load(&dir.join("input/vessel-vessel-mask.npy"));
    let input_a: Array3<bool> = load(&dir.join("input/vessel-artery-mask.npy"));
    let (v, e, length, mask) = graph_of(&input_v, &input_a);
    println!(
        "graph, no deep fill:     mask {mask}, {v} vertices, {e} edges, length {length:.0} µm"
    );
}

#[test]
#[ignore = "timing; needs the converted weights ($VESSEL_FILL_WEIGHTS) and the crop's masks under \
            $VESSEL_FILL_REFERENCE/crop/input"]
fn crop_timing() {
    let dir = reference().join("crop").join("input");
    for (label, resample) in [("vessel-vessel-mask", 1usize), ("vessel-artery-mask", 2)] {
        let input: Array3<bool> = load(&dir.join(format!("{label}.npy")));
        for round in 0..2 {
            let started = Instant::now();
            let filled = deep_fill(
                Voxels::Bool(input.clone()),
                &DeepFillParams::resampled(resample),
                &setup(),
                DeepFillOutput::Filled,
            )
            .unwrap();
            let set = filled
                .view::<bool>()
                .unwrap()
                .iter()
                .filter(|s| **s)
                .count();
            println!(
                "{label}, resample {resample}, round {round}: {set} voxels in {:.2}s",
                started.elapsed().as_secs_f64()
            );
        }
    }
}

#[cfg(feature = "deep-fill-gpu")]
#[test]
#[ignore = "needs a GPU, the converted weights ($VESSEL_FILL_WEIGHTS) and the crop's masks under \
            $VESSEL_FILL_REFERENCE/crop/input"]
fn crop_gpu_against_cpu() {
    let dir = reference().join("crop").join("input");
    for (label, resample) in [("vessel-vessel-mask", 1usize), ("vessel-artery-mask", 2)] {
        let input: Array3<bool> = load(&dir.join(format!("{label}.npy")));
        let params = DeepFillParams::resampled(resample);
        let mut answers = Vec::new();
        for device in [DeepFillDevice::Cpu, DeepFillDevice::Gpu] {
            let setup = DeepFillSetup {
                device,
                ..DeepFillSetup::default()
            };
            let mut seconds = Vec::new();
            let mut last = None;
            for _ in 0..3 {
                let started = Instant::now();
                let p = deep_fill(
                    Voxels::Bool(input.clone()),
                    &params,
                    &setup,
                    DeepFillOutput::Probability,
                )
                .unwrap();
                seconds.push(started.elapsed().as_secs_f64());
                last = Some(p);
            }
            let probability = last.unwrap().view::<f32>().unwrap().to_owned();
            let filled = deep_fill(
                Voxels::Bool(input.clone()),
                &params,
                &setup,
                DeepFillOutput::Filled,
            )
            .unwrap()
            .view::<bool>()
            .unwrap()
            .to_owned();
            println!(
                "{label}, resample {resample}, {device:?}: probability in {:?} s, fills to {}",
                seconds
                    .iter()
                    .map(|s| format!("{s:.2}"))
                    .collect::<Vec<_>>(),
                count(&filled)
            );
            answers.push((probability, filled));
        }
        let ((p_cpu, f_cpu), (p_gpu, f_gpu)) = (&answers[0], &answers[1]);
        let (max, r, n) = agreement(p_gpu, p_cpu);
        let mean = p_gpu
            .iter()
            .zip(p_cpu)
            .map(|(a, b)| (a - b).abs() as f64)
            .sum::<f64>()
            / n as f64;
        let differ = f_gpu.iter().zip(f_cpu).filter(|(a, b)| a != b).count();
        println!(
            "  gpu against cpu: probability max |diff| {max:.2e}, mean {mean:.2e}, r {r:.9}; \
             filled masks differ in {differ} of {n} voxels, Dice {:.7}",
            dice(f_gpu, f_cpu)
        );
        assert!(max < 1e-3, "{label}: {max}");
    }
}
