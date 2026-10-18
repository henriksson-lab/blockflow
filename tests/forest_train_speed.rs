//! **Where a pixel-classifier fit spends its time**: the feature gather or the
//! forest. A release benchmark, ignored by default:
//!
//! ```text
//! cargo test --release --test forest_train_speed -- --ignored --nocapture
//! RAYON_NUM_THREADS=1 cargo test --release --test forest_train_speed -- --ignored --nocapture
//! ```
//!
//! A 64x256x256 random-smooth `f64` volume, ~1% of its voxels labelled with two
//! classes, Labkit's stack at `[0.7, 1.0, 1.6, 3.5]` (71 channels), and the
//! default `TrainingSpec`. `FOREST_BENCH_TREES` overrides the tree count and
//! `FOREST_BENCH_Z` the depth of the volume. `per_channel_timing` prints what
//! each channel costs over the whole volume on one thread.
//!
//! Measured on the 40-core development machine (shared, so loaded), 100 trees,
//! 42,236 rows:
//!
//! ```text
//!                         gather_samples   Forest::train
//!   before, 40 threads        149.1 s         27.1 s    (both single-threaded)
//!   after,  40 threads         10.3 s          0.54 s
//!   after,   1 thread         169.4 s          9.0 s
//! ```
//!
//! The rows are identical before and after, and the forest is identical at 1
//! and 40 threads.

use std::time::Instant;

use ndarray::Array3;

use blockflow::forest::{Forest, TrainingSpec};
use blockflow::op::{Anchor, Chain};
use blockflow::ops::classify::gather_samples;
use blockflow::ops::features::FeatureStack;
use blockflow::ops::smooth::{Gaussian, SmoothOp};
use blockflow::voxels::Voxels;

fn hash(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

fn env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[test]
#[ignore = "release benchmark; run with --ignored --nocapture"]
fn gather_and_train_timing() {
    let shape = [env("FOREST_BENCH_Z", 64), 256, 256];
    let noise = Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
        (hash(((i * shape[1] + j) * shape[2] + k) as u64) >> 11) as f64 / (1u64 << 53) as f64
    });
    let noise = Voxels::F64(noise);
    let smooth = Chain::op(SmoothOp::new(
        "bench.smooth",
        Gaussian::new([2.0; 3], 3.0).unwrap(),
    ));
    let mut input = Voxels::zeros(blockflow::dtype::Dtype::F64, shape).unwrap();
    smooth
        .apply(&noise, &mut input, &Anchor::whole(shape))
        .unwrap();

    let values = input.view::<f64>().unwrap();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let labels = Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
        let at = ((i * shape[1] + j) * shape[2] + k) as u64;
        if hash(at ^ 0xabcdef) % 100 != 0 {
            0u32
        } else if values[[i, j, k]] > mean {
            2
        } else {
            1
        }
    });
    let labels = Voxels::U32(labels);

    let stack = FeatureStack::labkit(&[0.7, 1.0, 1.6, 3.5]).unwrap();
    let threads = rayon::current_num_threads();

    let started = Instant::now();
    let (samples, _) = gather_samples(&stack, &input, &labels, 0).unwrap();
    let gather = started.elapsed();

    let spec = TrainingSpec {
        trees: env("FOREST_BENCH_TREES", TrainingSpec::default().trees),
        ..TrainingSpec::default()
    };
    let started = Instant::now();
    let forest = Forest::train(&samples, &spec).unwrap();
    let train = started.elapsed();

    println!(
        "threads {threads}  volume {shape:?}  channels {}  rows {}  trees {}  nodes {}",
        samples.columns(),
        samples.rows(),
        forest.trees(),
        forest.nodes().len()
    );
    println!("gather_samples {:>9.3} s", gather.as_secs_f64());
    println!("Forest::train  {:>9.3} s", train.as_secs_f64());
    // A fingerprint, so runs at different thread counts can be compared by eye.
    let bytes = forest.to_bytes();
    let fingerprint = bytes.iter().fold(0u64, |acc, &b| hash(acc ^ b as u64));
    let sample_print = (0..samples.rows())
        .flat_map(|row| samples.row(row).iter().copied())
        .fold(0u64, |acc, v| hash(acc ^ v.to_bits()));
    println!("forest fingerprint {fingerprint:016x}  samples fingerprint {sample_print:016x}");
}

#[test]
#[ignore = "release benchmark; run with --ignored --nocapture"]
fn per_channel_timing() {
    let shape = [env("FOREST_BENCH_Z", 64), 256, 256];
    let input = Voxels::F64(Array3::from_shape_fn(
        (shape[0], shape[1], shape[2]),
        |(i, j, k)| {
            (hash(((i * shape[1] + j) * shape[2] + k) as u64) >> 11) as f64 / (1u64 << 53) as f64
        },
    ));
    let stack = FeatureStack::labkit(&[0.7, 1.0, 1.6, 3.5]).unwrap();
    for channel in stack.channels().unwrap() {
        let mut out = Voxels::zeros(blockflow::dtype::Dtype::F64, shape).unwrap();
        let started = Instant::now();
        channel
            .chain
            .apply(&input, &mut out, &Anchor::whole(shape))
            .unwrap();
        println!(
            "{:>32} {:>8.3} s  reach {:?}",
            channel.name,
            started.elapsed().as_secs_f64(),
            channel.chain.reach3(&shape)
        );
    }
}
