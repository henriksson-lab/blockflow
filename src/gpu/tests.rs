// SPDX-License-Identifier: MIT
//
// CPU/GPU parity, admission, and the executor running GPU blocks. Every test
// that needs a device returns early (and says so) when none is reachable, so
// the suite passes on a machine without one; the parity it asserts is then
// simply not exercised.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use ndarray::Array3;

use super::stencil::{self, vote_sweep_into, Colouring, VoteSweep, VoteSweepOp};
use super::{admit, available, budgets, smooth, DevicePolicy};
use crate::decomposition::{Constraints, CostModel};
use crate::dtype::Dtype;
use crate::env::ArrayEnvironment;
use crate::op::{Anchor, BlockOp, Chain, Combine};
use crate::ops::ridge::Boundary;
use crate::ops::smooth::{Gaussian, SmoothOp};
use crate::strategy::{Greedy, Strategy, Workflow};
use crate::voxels::Voxels;

fn device_or_skip(test: &str) -> bool {
    if !available() {
        eprintln!("{test}: no GPU reachable, skipped");
    }
    available()
}

fn ramp(shape: [usize; 3]) -> Array3<f32> {
    Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
        ((i * 7919 + j * 104_729 + k * 1013) % 251) as f32
    })
}

fn cpu_smooth(gaussian: &Gaussian, input: &Voxels) -> Array3<f64> {
    let op = SmoothOp::new("cpu", gaussian.clone());
    let mut out = Voxels::zeros(Dtype::F64, input.shape()).unwrap();
    op.apply(input, &mut out, &Anchor::whole(input.shape()))
        .unwrap();
    out.view::<f64>().unwrap().to_owned()
}

/// The stated tolerance: `1e-5` of the input's largest magnitude.
fn assert_close(gpu: &Array3<f64>, cpu: &Array3<f64>, scale: f64, what: &str) {
    let worst = gpu
        .iter()
        .zip(cpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst <= 1e-5 * scale,
        "{what}: worst |gpu - cpu| = {worst:e} against a bound of {:e}",
        1e-5 * scale
    );
}

#[test]
fn gpu_gaussian_matches_the_cpu_reference() {
    if !device_or_skip("gpu_gaussian_matches_the_cpu_reference") {
        return;
    }
    let shape = [13, 37, 29];
    let input_f32 = Voxels::F32(ramp(shape));
    let input_u16 = Voxels::U16(ramp(shape).mapv(|v| v as u16 * 200));
    for sigma in [[1.0, 2.0, 2.0], [0.0, 1.5, 3.0], [2.5, 0.7, 0.0]] {
        for boundary in [Boundary::Clamp, Boundary::Reflect] {
            for (input, scale) in [(&input_f32, 250.0), (&input_u16, 50_000.0)] {
                let gaussian = Gaussian::new(sigma, 4.0).unwrap().with_boundary(boundary);
                let op = smooth::accelerated(
                    SmoothOp::new("smooth", gaussian.clone()),
                    DevicePolicy::Gpu,
                );
                assert!(op.runs_on_gpu());
                let mut out = Voxels::zeros(Dtype::F64, shape).unwrap();
                op.apply(input, &mut out, &Anchor::whole(shape)).unwrap();
                let gpu = out.view::<f64>().unwrap().to_owned();
                assert_close(
                    &gpu,
                    &cpu_smooth(&gaussian, input),
                    scale,
                    &format!(
                        "sigma {sigma:?}, {boundary:?}, {}",
                        input.dtype().numpy_name()
                    ),
                );
            }
        }
    }
}

/// A kernel wider than the axis: every tap past the edge folds, repeatedly
/// under `Reflect`.
#[test]
fn gpu_gaussian_matches_on_an_axis_thinner_than_the_kernel() {
    if !device_or_skip("gpu_gaussian_matches_on_an_axis_thinner_than_the_kernel") {
        return;
    }
    let shape = [3, 1, 40];
    let input = Voxels::F32(ramp(shape));
    for boundary in [Boundary::Clamp, Boundary::Reflect] {
        let gaussian = Gaussian::new([4.0, 4.0, 1.0], 4.0)
            .unwrap()
            .with_boundary(boundary);
        let op = smooth::accelerated(SmoothOp::new("smooth", gaussian.clone()), DevicePolicy::Gpu);
        let mut out = Voxels::zeros(Dtype::F64, shape).unwrap();
        op.apply(&input, &mut out, &Anchor::whole(shape)).unwrap();
        assert_close(
            &out.view::<f64>().unwrap().to_owned(),
            &cpu_smooth(&gaussian, &input),
            250.0,
            "thin",
        );
    }
}

#[test]
fn the_cpu_policy_is_the_reference_to_the_bit_and_declares_no_device() {
    let shape = [6, 7, 8];
    let input = Voxels::F32(ramp(shape));
    let gaussian = Gaussian::new([1.0, 1.0, 1.0], 3.0).unwrap();
    let op = smooth::accelerated(SmoothOp::new("smooth", gaussian.clone()), DevicePolicy::Cpu);
    assert!(!op.runs_on_gpu());
    assert_eq!(op.device_bytes(shape), None);
    assert_eq!(op.cost_per_voxel(), op.reference().cost_per_voxel());
    let mut out = Voxels::zeros(Dtype::F64, shape).unwrap();
    op.apply(&input, &mut out, &Anchor::whole(shape)).unwrap();
    assert_eq!(out.view::<f64>().unwrap(), cpu_smooth(&gaussian, &input));
    // The plan-shaping answers are the CPU op's whatever the policy.
    let auto = smooth::accelerated(SmoothOp::new("smooth", gaussian), DevicePolicy::Auto);
    assert_eq!(auto.reach(1, 100), op.reach(1, 100));
    assert_eq!(auto.produces(Dtype::U16), Dtype::F64);
    assert_eq!(auto.device_bytes(shape).is_some(), available());
}

#[test]
fn gpu_vote_sweep_matches_the_cpu_reference_exactly() {
    if !device_or_skip("gpu_vote_sweep_matches_the_cpu_reference_exactly") {
        return;
    }
    let shape = [11, 19, 23];
    let labels = Array3::from_shape_fn((11, 19, 23), |(z, y, x)| {
        stencil::mix((z * 131 + y * 17 + x) as u32) % 5
    });
    let field = Array3::from_shape_fn((11, 19, 23), |(z, y, x)| {
        // Coarse values so ties happen and the hash is exercised.
        (stencil::mix((z * 7 + y * 3 + x * 11) as u32) % 4) as f32 * 0.25
    });
    let inputs = [Voxels::U32(labels.clone()), Voxels::F32(field.clone())];
    let at = Anchor::new([5, 2, 9], [100, 100, 100]);
    for colouring in [Colouring::Checkerboard, Colouring::Octant] {
        for class in 0..colouring.classes() {
            let sweep = VoteSweep::new(colouring, class, 1234, 7).unwrap();
            let op = stencil::accelerated(VoteSweepOp::new("vote", sweep), DevicePolicy::Gpu);
            let mut out = Voxels::zeros(Dtype::U32, shape).unwrap();
            op.apply(&[&inputs[0], &inputs[1]], &mut out, &at).unwrap();
            let mut want = Array3::<u32>::zeros(labels.dim());
            vote_sweep_into(
                labels.view(),
                field.view(),
                &sweep,
                at.offset,
                want.view_mut(),
            )
            .unwrap();
            assert_eq!(
                out.view::<u32>().unwrap(),
                want,
                "{colouring:?} class {class}"
            );
        }
    }
}

#[test]
fn admission_is_reentrant_and_bounded_by_the_device_budget() {
    if !device_or_skip("admission_is_reentrant_and_bounded_by_the_device_budget") {
        return;
    }
    let (slots, memory) = budgets().unwrap();
    {
        let _outer = admit(1 << 20);
        // The same block again from the same thread must not wait on itself.
        let _inner = admit(1 << 20);
        let _wider = admit(2 << 20);
    }
    // Threads each asking for 40% of the device: at most two fit at once
    // (fewer if another test holds some), and never more than the task slots.
    // Other tests share the process-wide budget, so this counts its own.
    let share = memory.total() / 5 * 2;
    let holding = AtomicUsize::new(0);
    let most = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..6 {
            scope.spawn(|| {
                let _held = admit(share);
                let now = holding.fetch_add(1, Ordering::SeqCst) + 1;
                most.fetch_max(now, Ordering::SeqCst);
                assert!(memory.in_use() <= memory.total());
                assert!(slots.in_use() <= slots.total());
                std::thread::sleep(std::time::Duration::from_millis(20));
                holding.fetch_sub(1, Ordering::SeqCst);
            });
        }
    });
    let most = most.load(Ordering::SeqCst);
    assert!(
        (1..=2).contains(&most),
        "{most} blocks of 40% were admitted at once"
    );
}

/// The executor runs GPU blocks concurrently from its worker threads, admits
/// each through the chain walk, and the decomposed result is the whole-volume
/// CPU answer to the stated tolerance.
#[test]
fn a_decomposed_run_on_the_gpu_matches_the_cpu_volume() {
    if !device_or_skip("a_decomposed_run_on_the_gpu_matches_the_cpu_volume") {
        return;
    }
    let shape = [20, 40, 36];
    let input = ramp(shape).mapv(f64::from);
    let gaussian = Gaussian::new([1.0, 2.0, 2.0], 3.0).unwrap();
    let op = smooth::accelerated(SmoothOp::new("smooth", gaussian.clone()), DevicePolicy::Gpu);
    let workflow = Workflow::new(Chain::op(op), shape, Dtype::F64);
    let constraints = Constraints {
        budget_bytes: None,
        expected_concurrency: 4,
        model: CostModel::default(),
        block_candidates: vec![8, 16],
        split_axes: vec![0, 1, 2],
        ..Default::default()
    };
    let strategy = Greedy { concurrency: 4 };
    let decomposition = strategy.decompose(&workflow, &constraints).unwrap();
    assert!(decomposition.n_tasks() > 1, "the test is about many blocks");
    let (slots, _) = budgets().unwrap();
    let grants = slots.grants();
    let env =
        ArrayEnvironment::new(input.clone().into(), decomposition.n_phases(), [4, 4, 4]).unwrap();
    strategy.run(&workflow, &decomposition, &env).unwrap();
    assert!(slots.grants() >= grants + decomposition.n_tasks() as u64);
    assert!(slots.high_water() <= slots.total());
    let got = env.output().view::<f64>().unwrap().to_owned();
    assert_close(
        &got,
        &cpu_smooth(&gaussian, &Voxels::F64(input)),
        250.0,
        "decomposed",
    );
}

fn seconds(f: impl FnOnce()) -> f64 {
    let start = Instant::now();
    f();
    start.elapsed().as_secs_f64()
}

/// Print-only. `cargo test --release --features gpu-cuda --lib
/// benchmark_gaussian -- --ignored --nocapture`.
///
/// Recorded on a Quadro RTX 5000 (PCIe 3 x16), 64x512x512 `f32` -> `f64`,
/// sigma [1, 2, 2] truncate 4 (43 taps), on a shared, loaded host:
///
/// ```text
/// cpu op (one thread)          0.60-0.87 s   35-52 ns/voxel
/// gpu op, first call (JIT)     0.9-1.5 s
/// gpu op incl. transfers       0.045 s       2.7 ns/voxel   13-16x
///   upload 0.013 s, kernels ~0.007 s, device read 0.013 s, host f32->f64 ~0.012 s
/// ```
#[test]
#[ignore]
fn benchmark_gaussian_cpu_against_gpu() {
    if !device_or_skip("benchmark_gaussian_cpu_against_gpu") {
        return;
    }
    let shape = [64, 512, 512];
    let voxels = shape.iter().product::<usize>() as f64;
    let input = Voxels::F32(ramp(shape));
    let gaussian = Gaussian::new([1.0, 2.0, 2.0], 4.0).unwrap();
    let cpu_op = SmoothOp::new("smooth", gaussian.clone());
    let gpu_op = smooth::accelerated(SmoothOp::new("smooth", gaussian.clone()), DevicePolicy::Gpu);
    let at = Anchor::whole(shape);
    let mut cpu_out = Voxels::zeros(Dtype::F64, shape).unwrap();
    let mut gpu_out = Voxels::zeros(Dtype::F64, shape).unwrap();

    let cpu = seconds(|| cpu_op.apply(&input, &mut cpu_out, &at).unwrap());
    let first = seconds(|| gpu_op.apply(&input, &mut gpu_out, &at).unwrap());
    let gpu = (0..3)
        .map(|_| seconds(|| gpu_op.apply(&input, &mut gpu_out, &at).unwrap()))
        .fold(f64::INFINITY, f64::min);
    assert_close(
        &gpu_out.view::<f64>().unwrap().to_owned(),
        &cpu_out.view::<f64>().unwrap().to_owned(),
        250.0,
        "benchmark",
    );
    let gpu_dev = super::device().unwrap();
    let transfer = (0..3)
        .map(|_| {
            seconds(|| {
                let block = gpu_dev.upload_f32(&input).unwrap();
                gpu_dev.download_into(block, &mut gpu_out).unwrap();
            })
        })
        .fold(f64::INFINITY, f64::min);
    let upload = (0..3)
        .map(|_| {
            seconds(|| {
                let block = gpu_dev.upload_f32(&input).unwrap();
                cubecl::future::block_on(gpu_dev.client().sync()).unwrap();
                drop(block);
            })
        })
        .fold(f64::INFINITY, f64::min);
    let block = gpu_dev.upload_f32(&input).unwrap();
    let read = seconds(|| {
        let _ = gpu_dev.client().read_one(block.handle.clone()).unwrap();
    });
    let unit_ns = cpu * 1e9 / voxels / cpu_op.cost_per_voxel();
    println!(
        "gaussian sigma [1,2,2] truncate 4 on {shape:?} f32 -> f64, taps {}",
        gaussian.taps()
    );
    println!(
        "  cpu op (one thread)        {:8.3} s  {:6.2} ns/voxel",
        cpu,
        cpu * 1e9 / voxels
    );
    println!("  gpu op, first (with JIT)   {:8.3} s", first);
    println!(
        "  gpu op incl. transfers     {:8.3} s  {:6.2} ns/voxel  speed-up {:.1}x",
        gpu,
        gpu * 1e9 / voxels,
        cpu / gpu
    );
    println!("  of which upload+download   {:8.3} s  (upload {upload:.3} s, device read {read:.3} s, rest is host f32->f64)", transfer);
    println!(
        "  cost units: cpu declares {:.1}; gpu measured {:.2} (transfer {:.2}), declares {:.2}",
        cpu_op.cost_per_voxel(),
        gpu * 1e9 / voxels / unit_ns,
        transfer * 1e9 / voxels / unit_ns,
        gpu_op.cost_per_voxel()
    );
}

/// Print-only, as above: one class update of the vote sweep.
#[test]
#[ignore]
fn benchmark_vote_sweep() {
    if !device_or_skip("benchmark_vote_sweep") {
        return;
    }
    let shape = [64, 256, 256];
    let voxels = shape.iter().product::<usize>() as f64;
    let labels = Voxels::U32(Array3::from_shape_fn((64, 256, 256), |(z, y, x)| {
        stencil::mix((z / 4 * 131 + y / 8 * 17 + x / 8) as u32) % 50
    }));
    let field = Voxels::F32(Array3::from_elem((64, 256, 256), 1.0f32));
    let sweep = VoteSweep::new(Colouring::Checkerboard, 0, 1, 0).unwrap();
    let cpu_op = VoteSweepOp::new("vote", sweep);
    let gpu_op = stencil::accelerated(VoteSweepOp::new("vote", sweep), DevicePolicy::Gpu);
    let at = Anchor::whole(shape);
    let mut cpu_out = Voxels::zeros(Dtype::U32, shape).unwrap();
    let mut gpu_out = Voxels::zeros(Dtype::U32, shape).unwrap();
    let cpu = seconds(|| Combine::apply(&cpu_op, &[&labels, &field], &mut cpu_out, &at).unwrap());
    let _ = Combine::apply(&gpu_op, &[&labels, &field], &mut gpu_out, &at);
    let gpu = (0..3)
        .map(|_| {
            seconds(|| Combine::apply(&gpu_op, &[&labels, &field], &mut gpu_out, &at).unwrap())
        })
        .fold(f64::INFINITY, f64::min);
    assert_eq!(
        cpu_out.view::<u32>().unwrap(),
        gpu_out.view::<u32>().unwrap()
    );
    // Calibrate the unit against the smoothing's declared cost on this machine.
    let small = [32, 128, 128];
    let probe = Voxels::F32(ramp(small));
    let smooth = SmoothOp::new("smooth", Gaussian::new([1.0, 2.0, 2.0], 4.0).unwrap());
    let mut out = Voxels::zeros(Dtype::F64, small).unwrap();
    let t = seconds(|| {
        smooth
            .apply(&probe, &mut out, &Anchor::whole(small))
            .unwrap()
    });
    let unit_ns = t * 1e9 / small.iter().product::<usize>() as f64 / smooth.cost_per_voxel();
    println!("vote sweep, one checkerboard class, {shape:?}");
    println!(
        "  cpu  {:8.3} s  {:7.2} ns/voxel  {:6.1} units",
        cpu,
        cpu * 1e9 / voxels,
        cpu * 1e9 / voxels / unit_ns
    );
    println!(
        "  gpu  {:8.3} s  {:7.2} ns/voxel  {:6.1} units  speed-up {:.1}x",
        gpu,
        gpu * 1e9 / voxels,
        gpu * 1e9 / voxels / unit_ns,
        cpu / gpu
    );
}
