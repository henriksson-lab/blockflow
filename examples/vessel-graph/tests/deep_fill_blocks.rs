// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Deep filling is the same answer whatever the block grid: blocks start the
//! network at multiples of its stride and read a halo wider than its reach, so
//! every block computes what one pass over the whole volume would. Run with
//! seeded random weights (the trained ones are not part of the crate), which
//! exercise the same arithmetic.

#![cfg(feature = "deep-fill")]

use std::io::Write;
use std::path::{Path, PathBuf};

use blockflow::voxels::Voxels;
use blockflow_vessel_graph::segment::{
    deep_fill, DeepFillDevice, DeepFillOutput, DeepFillParams, DeepFillSetup,
};
use ndarray::Array3;

/// The network's tensors, in its state-dict names.
const TENSORS: [(&str, &[usize]); 14] = [
    ("conv1.weight", &[16, 1, 7, 7, 7]),
    ("conv1.bias", &[16]),
    ("conv2.weight", &[32, 16, 5, 5, 5]),
    ("conv2.bias", &[32]),
    ("desepconv4.depthwise.weight", &[32, 1, 3, 3, 3]),
    ("desepconv4.depthwise.bias", &[32]),
    ("desepconv4.pointwise.weight", &[32, 32, 1, 1, 1]),
    ("desepconv4.pointwise.bias", &[32]),
    ("convbin1.weight", &[32, 32, 3, 3, 3]),
    ("convbin1.bias", &[32]),
    ("conv3.weight", &[32, 16, 7, 7, 7]),
    ("conv3.bias", &[16]),
    ("conv4.weight", &[2, 17, 3, 3, 3]),
    ("conv4.bias", &[2]),
];

struct Lcg(u64);

impl Lcg {
    /// Uniform in [-1, 1).
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// A safetensors file of seeded weights scaled by fan-in, so activations
/// neither vanish nor saturate.
fn random_weights() -> PathBuf {
    let mut rng = Lcg(7);
    let mut header = serde_json::Map::new();
    let mut body = Vec::new();
    for (name, shape) in TENSORS {
        let count: usize = shape.iter().product();
        let fan_in: usize = if shape.len() > 1 {
            if name == "conv3.weight" {
                // Transposed: [in, out, k^3]; each output sums in * k^3 / 8.
                shape[0] * 343 / 8
            } else {
                shape[1..].iter().product()
            }
        } else {
            1
        };
        let scale = if shape.len() > 1 {
            (3.0 / fan_in as f32).sqrt()
        } else {
            0.1
        };
        let start = body.len();
        for _ in 0..count {
            body.extend_from_slice(&(rng.next() * scale).to_le_bytes());
        }
        header.insert(
            name.to_string(),
            serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, body.len()]}),
        );
    }
    let mut text = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    while !text.len().is_multiple_of(8) {
        text.push(b' ');
    }
    let path = std::env::temp_dir().join(format!(
        "vessel-graph-deep-fill-test-{}.safetensors",
        std::process::id()
    ));
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(&(text.len() as u64).to_le_bytes()).unwrap();
    file.write_all(&text).unwrap();
    file.write_all(&body).unwrap();
    path
}

/// Hollow tubes along each axis and a few blobs, in a volume whose lengths are
/// multiples of nothing in particular.
fn hollow_tubes(shape: [usize; 3]) -> Array3<bool> {
    Array3::from_shape_fn(shape, |(z, y, x)| {
        let ring = |a: usize, b: usize, ca: usize, cb: usize, r: f64| {
            let d = ((a as f64 - ca as f64).powi(2) + (b as f64 - cb as f64).powi(2)).sqrt();
            d <= r && d >= r - 2.0
        };
        ring(y, x, 37, 51, 7.0)
            || ring(z, x, 21, 60, 9.0)
            || ring(z, y, 30, 80, 5.0)
            || (z / 9 + y / 13 + x / 11) % 17 == 0
    })
}

fn run_on(
    device: DeepFillDevice,
    mask: &Array3<bool>,
    weights: &Path,
    resample: usize,
    block: [usize; 3],
    output: DeepFillOutput,
) -> Voxels {
    deep_fill(
        Voxels::Bool(mask.clone()),
        &DeepFillParams::resampled(resample),
        &DeepFillSetup {
            weights: Some(weights.to_path_buf()),
            block,
            device,
        },
        output,
    )
    .unwrap()
}

fn run(
    mask: &Array3<bool>,
    weights: &Path,
    resample: usize,
    block: [usize; 3],
    output: DeepFillOutput,
) -> Voxels {
    run_on(DeepFillDevice::Cpu, mask, weights, resample, block, output)
}

/// Blocks of every grid here start inside the volume on some axis (x, past
/// the 128-voxel halo of resample 2), so the aligned start is exercised.
#[test]
fn the_answer_does_not_depend_on_the_block_grid() {
    let weights = random_weights();
    let shape = [46, 100, 190];
    let mask = hollow_tubes(shape);
    for resample in [1, 2] {
        let whole = run(
            &mask,
            &weights,
            resample,
            shape,
            DeepFillOutput::Probability,
        );
        let whole = whole.view::<f32>().unwrap().to_owned();
        let filled_whole = run(&mask, &weights, resample, shape, DeepFillOutput::Filled);
        let filled_whole = filled_whole.view::<bool>().unwrap().to_owned();
        let set = filled_whole.iter().filter(|s| **s).count();
        let input = mask.iter().filter(|s| **s).count();
        let mean = whole.iter().map(|&p| p as f64).sum::<f64>() / whole.len() as f64;
        println!("resample {resample}: input {input}, filled {set}, mean probability {mean:.3}");
        assert!(set > input, "the fixture should fill something");
        assert!(filled_whole.iter().zip(&mask).all(|(f, m)| *f || !*m));
        for block in [[24, 48, 40], [32, 100, 56]] {
            let blocked = run(
                &mask,
                &weights,
                resample,
                block,
                DeepFillOutput::Probability,
            );
            let blocked = blocked.view::<f32>().unwrap();
            let max = whole
                .iter()
                .zip(blocked.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let filled = run(&mask, &weights, resample, block, DeepFillOutput::Filled);
            let flips = filled
                .view::<bool>()
                .unwrap()
                .iter()
                .zip(&filled_whole)
                .filter(|(a, b)| a != b)
                .count();
            println!("  block {block:?}: max |diff| {max:.2e}, {flips} voxels differ");
            assert!(max < 1e-5, "resample {resample}, block {block:?}: {max}");
            assert_eq!(flips, 0, "resample {resample}, block {block:?}");
        }
    }
    let _ = std::fs::remove_file(weights);
}

#[test]
fn weights_of_another_network_are_refused() {
    let path = std::env::temp_dir().join(format!(
        "vessel-graph-deep-fill-bad-{}.safetensors",
        std::process::id()
    ));
    let text = br#"{"conv1.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}  "#;
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(&(text.len() as u64).to_le_bytes()).unwrap();
    file.write_all(text).unwrap();
    file.write_all(&[0; 4]).unwrap();
    drop(file);
    let error = deep_fill(
        Voxels::Bool(Array3::from_elem([8, 8, 8], false)),
        &DeepFillParams::default(),
        &DeepFillSetup {
            weights: Some(path.clone()),
            block: [8, 8, 8],
            device: DeepFillDevice::Cpu,
        },
        DeepFillOutput::Filled,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("conv1.weight"), "{error}");
    let _ = std::fs::remove_file(path);
}

/// The GPU network against the CPU one, on the seeded weights, whole and in
/// blocks: the probabilities within 1e-4, and the masks differing only where
/// the CPU's probability is within that of the threshold.
#[cfg(feature = "deep-fill-gpu")]
#[test]
#[ignore = "needs a GPU (a `deep-fill-cuda` or `deep-fill-wgpu` build and a reachable device)"]
fn the_gpu_agrees_with_the_cpu() {
    let weights = random_weights();
    let shape = [46, 100, 190];
    let mask = hollow_tubes(shape);
    for resample in [1, 2] {
        let cpu = run(
            &mask,
            &weights,
            resample,
            shape,
            DeepFillOutput::Probability,
        );
        let cpu = cpu.view::<f32>().unwrap().to_owned();
        for block in [shape, [24, 48, 40]] {
            let gpu = run_on(
                DeepFillDevice::Gpu,
                &mask,
                &weights,
                resample,
                block,
                DeepFillOutput::Probability,
            );
            let gpu = gpu.view::<f32>().unwrap();
            let max = cpu
                .iter()
                .zip(gpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let filled = run_on(
                DeepFillDevice::Gpu,
                &mask,
                &weights,
                resample,
                block,
                DeepFillOutput::Filled,
            );
            let unexplained = filled
                .view::<bool>()
                .unwrap()
                .iter()
                .zip(&mask)
                .zip(&cpu)
                .filter(|((f, m), p)| **f != (**m || **p >= 0.5) && (**p - 0.5).abs() > 1e-4)
                .count();
            println!("resample {resample}, block {block:?}: max |gpu - cpu| {max:.2e}");
            assert!(max < 1e-4, "resample {resample}, block {block:?}: {max}");
            assert_eq!(unexplained, 0, "resample {resample}, block {block:?}");
        }
    }
    let _ = std::fs::remove_file(weights);
}
