// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The label-energy sweep (`ops::label_energy`) on the GPU.
//
// One kernel launch per parity class, eight per sweep in `SWEEP_ORDER`, each
// over the voxels of that class only. The labels are updated **in place** on
// the device: no voxel of a class is in the 3x3x3 window of another, so a class
// reads only labels the launch does not write. Labels, quantised `p` and `f`
// are uploaded once and stay on the device across every class and every sweep
// of a call; only the labels (and two counters per sweep) come back.
//
// Parity: **bit-identical to the CPU reference, by construction.** The CPU
// decision is taken in integer energy units (`LabelEnergy::quantum`): boundary
// weights come from a table indexed by quantised `p` pairs, the smoothing bias
// is an integer, the data cost is `min((f - mu)^2 * scale, cap)` in three `f32`
// operations (a subtraction and two multiplications, which CUDA and Vulkan both
// round correctly, and which no compiler can contract into an FMA because no
// addition follows a product) truncated to an integer, and Metropolis accepts
// against an integer threshold table built on the host per sweep. The kernel
// receives the same tables and computes the same integers; the counter hash is
// the CPU's splitmix64, emulated in pairs of `u32` so that it needs no 64-bit
// integer support from the device (wgpu does not guarantee any). The
// simple-point test is two lookups in the same 8 MiB table on both sides.
//
// What it does not report: `SweepStats::delta_energy` (a per-sweep sum of
// signed 64-bit deltas has no portable atomic); it is `NaN`, and a caller that
// wants it computes `label_energy` before and after.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use cubecl::prelude::*;
use cubecl::server::Handle;
use ndarray::{Array3, ArrayView3};

use crate::dtype::Dtype;
use crate::error::{ensure, Error, Result};
use crate::gpu::device::{check_indexable, device, launch_shape, Device, DeviceBlock, Rt};
use crate::gpu::{admit, Accelerated, DevicePolicy, GpuKernel};
use crate::op::{Anchor, SourceInputs};
use crate::ops::label_energy::{
    check_inputs, class_prefix, mean_f32, quantise_p, simple_point_table, Acceptance, LabelEnergy,
    LabelEnergySweepOp, SweepStats, OUTSIDE, PAIR_LEVELS, SECOND_WORD_SALT, SWEEP_ORDER,
};
use crate::ops::skeleton::Adjacency;
use crate::voxels::Voxels;

// ------------------------------------------------------------- kernel --
//
// Nested `if`s are kept where the outer test guards the inner one's index or is
// a comptime switch: the kernel source then reads as the code it compiles to.

/// `a * b` modulo 2^64, as `(hi, lo)` words.
#[cube]
fn mul64(a_hi: u32, a_lo: u32, b_hi: u32, b_lo: u32) -> (u32, u32) {
    let lo = a_lo * b_lo;
    let hi = a_lo.mul_hi(b_lo) + a_lo * b_hi + a_hi * b_lo;
    (hi, lo)
}

/// `a + b` modulo 2^64, as `(hi, lo)` words.
#[cube]
fn add64(a_hi: u32, a_lo: u32, b_hi: u32, b_lo: u32) -> (u32, u32) {
    let lo = a_lo + b_lo;
    let mut carry = 0u32;
    if lo < a_lo {
        carry = 1u32;
    }
    (a_hi + b_hi + carry, lo)
}

/// `z ^ (z >> s)` for `0 < s < 32`.
#[cube]
fn xorshift64(hi: u32, lo: u32, #[comptime] s: u32) -> (u32, u32) {
    (hi ^ (hi >> s), lo ^ ((lo >> s) | (hi << (32 - s))))
}

/// `label_energy::splitmix` on `(hi, lo)` words.
#[cube]
fn splitmix(hi: u32, lo: u32) -> (u32, u32) {
    let (h1, l1) = add64(hi, lo, 0x9e37_79b9u32, 0x7f4a_7c15u32);
    let (h2, l2) = xorshift64(h1, l1, 30u32);
    let (h3, l3) = mul64(h2, l2, 0xbf58_476du32, 0x1ce4_e5b9u32);
    let (h4, l4) = xorshift64(h3, l3, 27u32);
    let (h5, l5) = mul64(h4, l4, 0x94d0_49bbu32, 0x1331_11ebu32);
    xorshift64(h5, l5, 31u32)
}

/// Is `shifted - 1` (a neighbour of local `i` at offset `d - 1`) inside an
/// axis of `n`, and where?
#[cube]
fn along(i: u32, d: u32, n: u32) -> (bool, u32) {
    let shifted = i + d;
    let inside = shifted >= 1u32 && shifted - 1u32 < n;
    let mut index = 0u32;
    if inside {
        index = shifted - 1u32;
    }
    (inside, index)
}

/// The mean of `label` from the sorted table, `(found, mean)`.
#[cube]
#[allow(clippy::collapsible_if)]
fn lookup(keys: &[u32], values: &[f32], n: u32, label: u32) -> (bool, f32) {
    let mut lo = 0u32;
    let mut hi = n;
    while lo < hi {
        let mid = (lo + hi) / 2u32;
        if keys[mid as usize] < label {
            lo = mid + 1u32;
        } else {
            hi = mid;
        }
    }
    let mut found = false;
    let mut value = 0.0f32;
    if lo < n {
        if keys[lo as usize] == label {
            found = true;
            value = values[lo as usize];
        }
    }
    (found, value)
}

/// `LabelEnergy::data_units`.
#[cube]
fn data_units(f: f32, mu: f32, scale: f32) -> i32 {
    let d = f - mu;
    let mut cost = (d * d) * scale;
    if cost > 536_870_912.0f32 {
        cost = 536_870_912.0f32; // DATA_CAP
    }
    i32::cast_from(cost)
}

/// `max(0, threshold - like)`.
#[cube]
fn hinge(threshold: u32, like: u32) -> i32 {
    let mut h = 0i32;
    if threshold > like {
        h = i32::cast_from(threshold - like);
    }
    h
}

/// Can the centre leave `{label == x}`? `label_energy::removable`.
#[cube]
fn removable(window: &Array<u32>, x: u32, simple: &[u32], #[comptime] six: bool) -> bool {
    let mut config = 0u32;
    for k in 0..27usize {
        if k != 13usize {
            let label = window[k];
            let mut member = label == x;
            if six {
                member = label != x;
            }
            if member {
                let mut bit = k as u32;
                if k > 13usize {
                    bit -= 1u32;
                }
                config |= 1u32 << bit;
            }
        }
    }
    ((simple[(config >> 5u32) as usize] >> (config & 31u32)) & 1u32) == 1u32
}

/// `label_energy::is_simple_label_flip_with` (`topology`: 0 none, 1 six, 2
/// twenty-six).
#[cube]
fn allowed(window: &Array<u32>, a: u32, b: u32, simple: &[u32], #[comptime] topology: u32) -> bool {
    let mut ok = true;
    if topology == 1 {
        ok = removable(window, a, simple, true) && removable(window, b, simple, true);
    }
    if topology == 2 {
        ok = removable(window, a, simple, false) && removable(window, b, simple, false);
    }
    ok
}

/// One parity class of one sweep. Threads cover the class's sub-lattice
/// `[m0, m1, m2]`, whose first voxel is at local `first*`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
fn sweep_class_kernel(
    labels: &mut [u32],
    pq: &[u32],
    f: &[f32],
    mean_keys: &[u32],
    mean_values: &[f32],
    edges: &[i32],
    simple: &[u32],
    thresholds: &[i32],
    stats: &[Atomic<u32>],
    n0: u32,
    n1: u32,
    n2: u32,
    m0: u32,
    m1: u32,
    m2: u32,
    first0: u32,
    first1: u32,
    first2: u32,
    o0: u32,
    o1: u32,
    o2: u32,
    volume1: u32,
    volume2: u32,
    prefix_hi: u32,
    prefix_lo: u32,
    n_means: u32,
    smooth_units: i32,
    smooth_threshold: u32,
    data_scale: f32,
    stats_at: u32,
    #[comptime] topology: u32,
    #[comptime] metropolis: bool,
    #[comptime] data: bool,
) {
    let total = (m0 * m1 * m2) as usize;
    if ABSOLUTE_POS < total {
        let t = ABSOLUTE_POS as u32;
        let i2 = first2 + 2u32 * (t % m2);
        let i1 = first1 + 2u32 * ((t / m2) % m1);
        let i0 = first0 + 2u32 * (t / (m2 * m1));
        let centre = ((i0 * n1 + i1) * n2 + i2) as usize;
        let a = labels[centre];

        let mut window = Array::<u32>::new(27usize);
        let mut at = Array::<u32>::new(27usize);
        let mut differs = false;
        for k in 0..27usize {
            let kk = k as u32;
            let (in0, a0) = along(i0, kk / 9u32, n0);
            let (in1, a1) = along(i1, (kk / 3u32) % 3u32, n1);
            let (in2, a2) = along(i2, kk % 3u32, n2);
            if in0 && in1 && in2 {
                let q = (a0 * n1 + a1) * n2 + a2;
                let label = labels[q as usize];
                window[k] = label;
                at[k] = q;
                if label != a {
                    differs = true;
                }
            } else {
                window[k] = 0xffff_ffffu32; // OUTSIDE
                at[k] = 0u32;
            }
        }

        if differs {
            // candidates in order of first appearance, as the CPU lists them
            let mut candidates = Array::<u32>::new(26usize);
            let mut count = 0u32;
            let mut outside = 0u32;
            for k in 0..27usize {
                let label = window[k];
                if k != 13usize && label != a {
                    if label == 0xffff_ffffu32 {
                        outside += 1u32;
                    } else {
                        let mut seen = false;
                        for c in 0..count {
                            if candidates[c as usize] == label {
                                seen = true;
                            }
                        }
                        if !seen {
                            candidates[count as usize] = label;
                            count += 1u32;
                        }
                    }
                }
            }

            let qc = pq[centre];
            let mut weight = Array::<i32>::new(27usize);
            let mut keep_cost = 0i32;
            let mut like_a = 0u32;
            for k in 0..27usize {
                weight[k] = 0i32;
                let label = window[k];
                if k != 13usize && label != 0xffff_ffffu32 {
                    let mut row = k as u32;
                    if k > 13usize {
                        row = 26u32 - row;
                    }
                    let w = edges[(row * PAIR_LEVELS_U32 + qc + pq[at[k] as usize]) as usize];
                    weight[k] = w;
                    if label == a {
                        keep_cost += w;
                        like_a += 1u32;
                    }
                }
            }

            let mut data_a_on = false;
            let mut data_a = 0i32;
            let mut fv = 0.0f32;
            if data {
                fv = f[centre];
                let bits = u32::reinterpret(fv);
                if (bits & 0x7f80_0000u32) != 0x7f80_0000u32 {
                    let (found, mu) = lookup(mean_keys, mean_values, n_means, a);
                    if found {
                        data_a_on = true;
                        data_a = data_units(fv, mu, data_scale);
                    }
                }
            }
            let hinge_a = hinge(smooth_threshold, like_a + outside);

            let mut chosen = a;
            let mut blocked = false;
            if metropolis {
                // the voxel's global linear index, in 64 bits
                let g0 = o0 + i0;
                let g1 = o1 + i1;
                let g2 = o2 + i2;
                let (x_hi, x_lo) = (g0.mul_hi(volume1), g0 * volume1);
                let (y_hi, y_lo) = add64(x_hi, x_lo, 0u32, g1);
                let (z_hi, z_lo) = mul64(y_hi, y_lo, 0u32, volume2);
                let (index_hi, index_lo) = add64(z_hi, z_lo, 0u32, g2);
                let (h_hi, h_lo) = splitmix(prefix_hi ^ index_hi, prefix_lo ^ index_lo);
                let (g_hi, _g_lo) = splitmix(h_hi ^ SALT_HI, h_lo ^ SALT_LO);
                let pick = h_hi.mul_hi(count);
                let level = g_hi >> 16u32;
                let b = candidates[pick as usize];
                let (score, _delta) = evaluate(
                    &window,
                    &weight,
                    b,
                    keep_cost,
                    hinge_a,
                    outside,
                    data_a_on,
                    data_a,
                    fv,
                    mean_keys,
                    mean_values,
                    n_means,
                    data_scale,
                    smooth_units,
                    smooth_threshold,
                    data,
                );
                if score <= 0i32 || score < thresholds[level as usize] {
                    if allowed(&window, a, b, simple, topology) {
                        chosen = b;
                    } else {
                        blocked = true;
                    }
                }
            } else {
                let mut scores = Array::<i32>::new(26usize);
                for c in 0..count {
                    let (score, _delta) = evaluate(
                        &window,
                        &weight,
                        candidates[c as usize],
                        keep_cost,
                        hinge_a,
                        outside,
                        data_a_on,
                        data_a,
                        fv,
                        mean_keys,
                        mean_values,
                        n_means,
                        data_scale,
                        smooth_units,
                        smooth_threshold,
                        data,
                    );
                    scores[c as usize] = score;
                }
                // negative scores in ascending (score, label) order, until one
                // the topology allows: the CPU's sorted walk
                let mut done = false;
                let mut has_previous = false;
                let mut previous_score = 0i32;
                let mut previous_label = 0u32;
                for _ in 0..count {
                    if !done {
                        let mut found = false;
                        let mut best_score = 0i32;
                        let mut best_label = 0u32;
                        for c in 0..count {
                            let score = scores[c as usize];
                            let label = candidates[c as usize];
                            let after = !has_previous
                                || score > previous_score
                                || (score == previous_score && label > previous_label);
                            if score < 0i32 && after {
                                if !found
                                    || score < best_score
                                    || (score == best_score && label < best_label)
                                {
                                    found = true;
                                    best_score = score;
                                    best_label = label;
                                }
                            }
                        }
                        if found {
                            has_previous = true;
                            previous_score = best_score;
                            previous_label = best_label;
                            if allowed(&window, a, best_label, simple, topology) {
                                chosen = best_label;
                                done = true;
                            } else {
                                blocked = true;
                            }
                        } else {
                            done = true;
                        }
                    }
                }
            }

            if chosen != a {
                labels[centre] = chosen;
                stats[stats_at as usize].fetch_add(1u32);
            }
            if blocked {
                stats[(stats_at + 1u32) as usize].fetch_add(1u32);
            }
        }
    }
}

/// `(score, delta)` of flipping the centre to `b`: `decide`'s `evaluate`.
#[cube]
#[allow(clippy::collapsible_if)]
#[allow(clippy::too_many_arguments)]
fn evaluate(
    window: &Array<u32>,
    weight: &Array<i32>,
    b: u32,
    keep_cost: i32,
    hinge_a: i32,
    outside: u32,
    data_a_on: bool,
    data_a: i32,
    fv: f32,
    mean_keys: &[u32],
    mean_values: &[f32],
    n_means: u32,
    data_scale: f32,
    smooth_units: i32,
    smooth_threshold: u32,
    #[comptime] data: bool,
) -> (i32, i32) {
    let mut gain = 0i32;
    let mut like_b = 0u32;
    for k in 0..27usize {
        if k != 13usize && window[k] == b {
            gain += weight[k];
            like_b += 1u32;
        }
    }
    let mut delta = keep_cost - gain;
    if data {
        if data_a_on {
            let (found, mu_b) = lookup(mean_keys, mean_values, n_means, b);
            if found {
                delta += data_units(fv, mu_b, data_scale) - data_a;
            }
        }
    }
    let mut bias = 0i32;
    if smooth_units > 0i32 {
        bias = smooth_units * (hinge(smooth_threshold, like_b + outside) - hinge_a);
    }
    (delta + bias, delta)
}

const PAIR_LEVELS_U32: u32 = PAIR_LEVELS as u32;
const SALT_HI: u32 = (SECOND_WORD_SALT >> 32) as u32;
const SALT_LO: u32 = SECOND_WORD_SALT as u32;

// --------------------------------------------------------------- host --

/// The simple-point table on the device: uploaded once per process.
fn simple_table_on(gpu: &Device) -> Handle {
    static TABLE: OnceLock<Handle> = OnceLock::new();
    TABLE
        .get_or_init(|| {
            gpu.client()
                .create_from_slice(u32::as_bytes(simple_point_table()))
        })
        .clone()
}

/// Device bytes of a call on a block of `shape`: labels, quantised `p` and `f`
/// (four bytes a voxel each) and the tables.
pub fn device_bytes(shape: [usize; 3], data: bool) -> u64 {
    let voxels = shape.iter().product::<usize>() as u64;
    voxels * if data { 12 } else { 8 } + (12 << 20)
}

/// Run `sweeps` — `(sweep index, temperature)` pairs, in order — over a buffer
/// at `at` in the volume, on the GPU: the answer
/// `label_energy_sweep_into` gives applied to each in turn, to the bit, with the
/// means table fixed across them.
///
/// The labels stay on the device across all the sweeps' 8 classes; the returned
/// statistics have `flips` and `topology_rejections` exact and `delta_energy`
/// `NaN` (see the module header).
#[allow(clippy::too_many_arguments)]
pub fn label_energy_sweeps_gpu(
    energy: &LabelEnergy,
    labels: ArrayView3<'_, u32>,
    p: ArrayView3<'_, f64>,
    f: Option<ArrayView3<'_, f64>>,
    means: &HashMap<u32, f64>,
    sweeps: &[(usize, f64)],
    at: &Anchor,
) -> Result<(Array3<u32>, Vec<SweepStats>)> {
    check_inputs(&labels, &p, f.as_ref(), energy)?;
    for &(_, temperature) in sweeps {
        ensure!(
            temperature.is_finite(),
            "label energy: temperature must be finite, got {temperature}"
        );
    }
    let gpu = device().ok_or_else(|| Error::InvalidArgument("no GPU device".into()))?;
    let shape = [labels.shape()[0], labels.shape()[1], labels.shape()[2]];
    check_indexable(shape, "gpu label energy")?;
    ensure!(
        at.volume.iter().all(|&v| v < u32::MAX as usize),
        "gpu label energy: a volume of {:?} has an axis longer than the kernels index",
        at.volume
    );
    let data = energy.data_on() && f.is_some();
    let _admission = admit(device_bytes(shape, data));
    let mut stats: Vec<SweepStats> = sweeps
        .iter()
        .map(|&(sweep, temperature)| SweepStats {
            sweep,
            temperature,
            delta_energy: f64::NAN,
            energy_before: f64::NAN,
            energy_after: f64::NAN,
            ..SweepStats::default()
        })
        .collect();
    if shape.contains(&0) || sweeps.is_empty() {
        return Ok((labels.to_owned(), stats));
    }

    let contiguous = |view: ArrayView3<'_, f64>| -> Vec<f64> {
        view.as_slice()
            .map(<[f64]>::to_vec)
            .unwrap_or_else(|| view.iter().copied().collect())
    };
    let device_labels = DeviceBlock {
        handle: match labels.as_slice() {
            Some(slice) => gpu.upload_elements(slice, |v: u32| v),
            None => gpu.upload_elements(&labels.iter().copied().collect::<Vec<_>>(), |v: u32| v),
        },
        shape,
        dtype: Dtype::U32,
    };
    let pq = match p.as_slice() {
        Some(slice) => gpu.upload_elements(slice, quantise_p),
        None => gpu.upload_elements(&contiguous(p), quantise_p),
    };
    let client = gpu.client();
    let fv = match (data, f) {
        (true, Some(f)) => match f.as_slice() {
            Some(slice) => gpu.upload_elements(slice, |v: f64| v as f32),
            None => gpu.upload_elements(&contiguous(f), |v: f64| v as f32),
        },
        _ => client.create_from_slice(f32::as_bytes(&[0.0])),
    };
    // the means as the CPU uses them, sorted by label
    let mut table: Vec<(u32, f32)> = if data {
        means
            .keys()
            .filter_map(|&label| mean_f32(means, label).map(|mu| (label, mu)))
            .collect()
    } else {
        Vec::new()
    };
    table.sort_unstable_by_key(|&(label, _)| label);
    let n_means = table.len() as u32;
    let mut keys: Vec<u32> = table.iter().map(|&(label, _)| label).collect();
    let mut values: Vec<f32> = table.iter().map(|&(_, mu)| mu).collect();
    if keys.is_empty() {
        keys.push(OUTSIDE);
        values.push(0.0);
    }
    let mean_keys = client.create_from_slice(u32::as_bytes(&keys));
    let mean_values = client.create_from_slice(f32::as_bytes(&values));
    let edges = client.create_from_slice(i32::as_bytes(energy.edge_table()));
    let simple = simple_table_on(gpu);
    let counters = client.create_from_slice(u32::as_bytes(&vec![0u32; 2 * sweeps.len()]));
    let no_thresholds = client.create_from_slice(i32::as_bytes(&[0]));

    let topology = match energy.params().topology {
        None => 0u32,
        Some(Adjacency::Six) => 1,
        Some(Adjacency::TwentySix) => 2,
    };
    let voxels = shape.iter().product::<usize>();
    let arg_u32 = |handle: &Handle, len: usize| unsafe {
        // SAFETY: each handle was allocated for `len` four-byte elements.
        BufferArg::<Rt>::from_raw_parts(handle.clone(), len)
    };
    let [n0, n1, n2] = shape.map(|n| n as u32);
    let origin = at.offset.map(|o| o as u32);
    let params = energy.params();
    for (slot, &(sweep, temperature)) in sweeps.iter().enumerate() {
        let acceptance = Acceptance::new(energy, temperature);
        let thresholds = acceptance
            .thresholds()
            .map(|t| (client.create_from_slice(i32::as_bytes(t)), t.len()));
        let (threshold_handle, threshold_len) = match &thresholds {
            Some((handle, len)) => (handle.clone(), *len),
            None => (no_thresholds.clone(), 1),
        };
        for &class in &SWEEP_ORDER {
            let parity = [(class >> 2) & 1, (class >> 1) & 1, class & 1];
            let first: [usize; 3] =
                std::array::from_fn(|axis| (parity[axis] + (at.offset[axis] & 1)) & 1);
            let sub: [usize; 3] =
                std::array::from_fn(|axis| shape[axis].saturating_sub(first[axis]).div_ceil(2));
            let threads = sub.iter().product::<usize>();
            if threads == 0 {
                continue;
            }
            let prefix = class_prefix(params.seed, sweep as u64, class as u64);
            let (count, dim) = launch_shape(threads);
            // SAFETY: every buffer holds the length passed with it; the
            // kernel indexes the voxel buffers below `n0 * n1 * n2`, the
            // tables below their lengths (edge rows and `p` levels are in
            // range by construction), and the counters at `2 * slot + 1`.
            unsafe {
                sweep_class_kernel::launch_unchecked::<Rt>(
                    client,
                    count,
                    dim,
                    device_labels.arg(),
                    arg_u32(&pq, voxels),
                    arg_u32(&fv, if data { voxels } else { 1 }),
                    arg_u32(&mean_keys, keys.len()),
                    arg_u32(&mean_values, values.len()),
                    arg_u32(&edges, energy.edge_table().len()),
                    arg_u32(&simple, simple_point_table().len()),
                    arg_u32(&threshold_handle, threshold_len),
                    arg_u32(&counters, 2 * sweeps.len()),
                    n0,
                    n1,
                    n2,
                    sub[0] as u32,
                    sub[1] as u32,
                    sub[2] as u32,
                    first[0] as u32,
                    first[1] as u32,
                    first[2] as u32,
                    origin[0],
                    origin[1],
                    origin[2],
                    at.volume[1] as u32,
                    at.volume[2] as u32,
                    (prefix >> 32) as u32,
                    prefix as u32,
                    n_means,
                    energy.smooth_units(),
                    params.smooth_threshold,
                    energy.data_scale(),
                    (2 * slot) as u32,
                    topology,
                    thresholds.is_some(),
                    data,
                );
            }
        }
    }
    let mut out = Voxels::zeros(Dtype::U32, shape)?;
    gpu.download_into(device_labels, &mut out)?;
    let bytes = client
        .read_one(counters)
        .map_err(|error| Error::InvalidArgument(format!("device read failed: {error:?}")))?;
    let counts = u32::from_bytes(&bytes);
    for (slot, stat) in stats.iter_mut().enumerate() {
        stat.flips = u64::from(counts[2 * slot]);
        stat.topology_rejections = u64::from(counts[2 * slot + 1]);
    }
    let Voxels::U32(out) = out else {
        unreachable!()
    };
    Ok((out, stats))
}

// ------------------------------------------------------------ the op --

/// The GPU implementation of one [`LabelEnergySweepOp`].
pub struct GpuLabelEnergySweep {
    op: LabelEnergySweepOp,
}

impl GpuLabelEnergySweep {
    pub fn new(op: LabelEnergySweepOp) -> Self {
        Self { op }
    }
}

impl GpuKernel for GpuLabelEnergySweep {
    fn device_bytes(&self, block: [usize; 3]) -> u64 {
        device_bytes(block, self.op.energy().data_on())
    }

    fn cost_per_voxel(&self) -> f64 {
        LABEL_ENERGY_SWEEP_GPU_COST
    }

    fn apply(&self, _inputs: &[&Voxels], _out: &mut Voxels, _at: &Anchor) -> Result<()> {
        Err(Error::InvalidArgument(
            "gpu label-energy sweep: reads its boundary evidence as a source image, so it is \
             applied through `apply_with`"
                .into(),
        ))
    }

    fn apply_with(
        &self,
        input: &Voxels,
        sources: SourceInputs<'_>,
        out: &mut Voxels,
        at: &Anchor,
    ) -> Result<()> {
        let (p_image, f_image) = self.op.images();
        let p = sources.get(p_image)?.view::<f64>()?;
        let f = match f_image {
            Some(image) => Some(sources.get(image)?.view::<f64>()?),
            None => None,
        };
        let (labels, _) = label_energy_sweeps_gpu(
            self.op.energy(),
            input.view::<u32>()?,
            p,
            f,
            self.op.means(),
            &[(self.op.sweep(), self.op.temperature())],
            at,
        )?;
        out.view_mut::<u32>()?.assign(&labels);
        Ok(())
    }
}

/// One sweep per call with its transfers, scaled to the CPU placeholder
/// [`crate::ops::label_energy::LABEL_ENERGY_SWEEP_COST`] by the ratio
/// `benchmark_label_energy` measures: ~12 ns a voxel on the GPU (transfer
/// dominated) against ~500 ns a voxel on one CPU thread.
pub const LABEL_ENERGY_SWEEP_GPU_COST: f64 = 1.5;

/// [`crate::ops::label_energy::refine_labels`] on the GPU: the same sweeps,
/// the same labels to the bit.
///
/// Without the data term (or without `f`) the means are never read, so every
/// sweep runs in one call with the labels resident on the device. With it,
/// the means move with the labels, so each sweep is its own call with the
/// table refreshed on the host in between, as the CPU driver does. The
/// statistics carry exact `flips` and `topology_rejections`; the energies are
/// `NaN` (see [`label_energy_sweeps_gpu`]).
pub fn refine_labels_gpu(
    energy: &LabelEnergy,
    labels: ArrayView3<'_, u32>,
    p: ArrayView3<'_, f64>,
    f: Option<ArrayView3<'_, f64>>,
    schedule: &crate::ops::label_energy::AnnealingSchedule,
) -> Result<(Array3<u32>, Vec<SweepStats>)> {
    use crate::ops::label_energy::label_means;
    let shape = [labels.shape()[0], labels.shape()[1], labels.shape()[2]];
    let at = Anchor::whole(shape);
    let sweeps: Vec<(usize, f64)> = schedule.temperatures().into_iter().enumerate().collect();
    match f {
        Some(f) if energy.data_on() => {
            let mut current = labels.to_owned();
            let mut stats = Vec::with_capacity(sweeps.len());
            for sweep in sweeps {
                let means = label_means(current.view(), f)?;
                let (next, s) =
                    label_energy_sweeps_gpu(energy, current.view(), p, Some(f), &means, &[sweep], &at)?;
                current = next;
                stats.extend(s);
            }
            Ok((current, stats))
        }
        _ => label_energy_sweeps_gpu(energy, labels, p, None, &HashMap::new(), &sweeps, &at),
    }
}

/// [`LabelEnergySweepOp`] with its GPU implementation attached, under `policy`.
pub fn accelerated(
    op: LabelEnergySweepOp,
    policy: DevicePolicy,
) -> Accelerated<LabelEnergySweepOp> {
    let kernel = Arc::new(GpuLabelEnergySweep::new(op.clone()));
    Accelerated::with_kernel(op, kernel, policy)
}

// ---------------------------------------------------------------- tests --

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use ndarray::s;

    use super::*;
    use crate::assemble::ImageId;
    use crate::gpu::available;
    use crate::op::BlockOp;
    use crate::ops::label_energy::{
        counter_draws, label_energy_sweep_into, label_means, LabelEnergyParams,
    };

    fn device_or_skip(test: &str) -> bool {
        if !available() {
            eprintln!("{test}: no GPU reachable, skipped");
        }
        available()
    }

    fn hash01(i: usize, j: usize, k: usize, salt: u64) -> f64 {
        counter_draws(salt, i as u64, j as u64, k as u64)[0]
    }

    /// Jittered blocks of `cell` voxels with label noise, a membrane map high
    /// on the label boundaries, and an appearance field per label.
    /// `smooth`: boundaries jittered by a smooth wave rather than per voxel.
    fn fixture(
        n: [usize; 3],
        cell: usize,
        noise: f64,
        smooth: bool,
    ) -> (Array3<u32>, Array3<f64>, Array3<f64>) {
        let labels = Array3::from_shape_fn(n, |(i, j, k)| {
            let jitter = |salt: u64| {
                let amplitude = cell as f64 * 0.3;
                if smooth {
                    let phase = salt as f64;
                    let wave = ((i as f64 * 0.21 + phase).sin()
                        + (j as f64 * 0.17 + 2.0 * phase).sin()
                        + (k as f64 * 0.13 + 3.0 * phase).sin())
                        / 3.0;
                    (amplitude * (1.0 + wave)) as usize
                } else {
                    (hash01(i, j, k, salt) * amplitude * 2.0) as usize
                }
            };
            let c = [
                (i + jitter(1)) / cell,
                (j + jitter(2)) / cell,
                (k + jitter(3)) / cell,
            ];
            let base = 1 + (c[0] * 7 + c[1] * 13 + c[2] * 29) as u32 % 11;
            if hash01(i, j, k, 4) < noise {
                1 + (hash01(i, j, k, 5) * 11.0) as u32
            } else {
                base
            }
        });
        let p = Array3::from_shape_fn(n, |(i, j, k)| {
            let l = labels[[i, j, k]];
            let edge = (i + 1 < n[0] && labels[[i + 1, j, k]] != l)
                || (j + 1 < n[1] && labels[[i, j + 1, k]] != l)
                || (k + 1 < n[2] && labels[[i, j, k + 1]] != l);
            let noise = hash01(i, j, k, 6);
            if edge {
                0.6 + 0.4 * noise
            } else {
                0.3 * noise
            }
        });
        let f = Array3::from_shape_fn(n, |(i, j, k)| {
            f64::from(labels[[i, j, k]] % 4) * 0.4 + hash01(i, j, k, 7)
        });
        (labels, p, f)
    }

    #[allow(clippy::too_many_arguments)]
    fn cpu_sweeps(
        energy: &LabelEnergy,
        labels: &Array3<u32>,
        p: &Array3<f64>,
        f: Option<&Array3<f64>>,
        means: &HashMap<u32, f64>,
        sweeps: &[(usize, f64)],
        at: &Anchor,
    ) -> (Array3<u32>, Vec<SweepStats>) {
        let mut current = labels.clone();
        let mut next = Array3::zeros(labels.raw_dim());
        let mut stats = Vec::new();
        for &(sweep, t) in sweeps {
            stats.push(
                label_energy_sweep_into(
                    energy,
                    current.view(),
                    p.view(),
                    f.map(|f| f.view()),
                    means,
                    sweep,
                    t,
                    at,
                    next.view_mut(),
                )
                .unwrap(),
            );
            std::mem::swap(&mut current, &mut next);
        }
        (current, stats)
    }

    fn assert_same(
        what: &str,
        cpu: &(Array3<u32>, Vec<SweepStats>),
        gpu: &(Array3<u32>, Vec<SweepStats>),
    ) {
        let differing = cpu.0.iter().zip(&gpu.0).filter(|(a, b)| a != b).count();
        assert_eq!(differing, 0, "{what}: {differing} labels differ");
        for (c, g) in cpu.1.iter().zip(&gpu.1) {
            assert_eq!(
                (c.flips, c.topology_rejections),
                (g.flips, g.topology_rejections),
                "{what}: sweep {}",
                c.sweep
            );
        }
    }

    #[test]
    fn gpu_sweeps_match_the_cpu_reference_to_the_bit() {
        if !device_or_skip("gpu_sweeps_match_the_cpu_reference_to_the_bit") {
            return;
        }
        let n = [19, 34, 29];
        let (labels, p, f) = fixture(n, 6, 0.05, false);
        let means = label_means(labels.view(), f.view()).unwrap();
        let schedules: [&[(usize, f64)]; 3] = [
            &[(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)],
            &[(0, 2.0), (1, 1.0), (2, 0.5), (3, 0.25)],
            &[(5, 0.3), (6, 0.0), (7, 4.0)],
        ];
        let mut total_flips = 0;
        for topology in [None, Some(Adjacency::Six), Some(Adjacency::TwentySix)] {
            for data in [false, true] {
                for smooth in [false, true] {
                    let mut params = LabelEnergyParams::new([2.0, 1.0, 1.0], 1.0, 2.0, 0.05)
                        .with_topology(topology)
                        .with_seed(17);
                    if data {
                        params = params.with_data(0.4, 0.5);
                    }
                    if smooth {
                        params = params.with_smoothing(0.3, 10);
                    }
                    let energy = LabelEnergy::new(params).unwrap();
                    let fv = data.then_some(&f);
                    for sweeps in schedules {
                        let at = Anchor::whole(n);
                        let cpu = cpu_sweeps(&energy, &labels, &p, fv, &means, sweeps, &at);
                        let gpu = label_energy_sweeps_gpu(
                            &energy,
                            labels.view(),
                            p.view(),
                            fv.map(|f| f.view()),
                            &means,
                            sweeps,
                            &at,
                        )
                        .unwrap();
                        let what = format!("{topology:?} data {data} smooth {smooth} {sweeps:?}");
                        assert_same(&what, &cpu, &gpu);
                        total_flips += cpu.1.iter().map(|s| s.flips).sum::<u64>();
                        if topology.is_some() && sweeps[0].1 > 1.0 {
                            assert!(cpu.1[0].topology_rejections > 0, "{what}: exercised");
                        }
                    }
                }
            }
        }
        assert!(total_flips > 10_000, "the fixture moves: {total_flips}");
    }

    /// The whole-volume driver gives `refine_labels`' labels and flip counts,
    /// with the data term (means refreshed per sweep) and without.
    #[test]
    fn refine_labels_gpu_matches_refine_labels() {
        if !device_or_skip("refine_labels_gpu_matches_refine_labels") {
            return;
        }
        use crate::ops::label_energy::{refine_labels, AnnealingSchedule};
        let n = [17, 30, 26];
        let (labels, p, f) = fixture(n, 6, 0.05, false);
        let schedule = AnnealingSchedule::new(1, 1.0, 0.5, 3, 1).unwrap();
        for data in [false, true] {
            let mut params = LabelEnergyParams::new([1.5, 1.0, 1.0], 1.0, 2.0, 0.05)
                .with_topology(Some(Adjacency::Six))
                .with_seed(3);
            if data {
                params = params.with_data(0.4, 0.5);
            }
            let energy = LabelEnergy::new(params).unwrap();
            let fv = data.then(|| f.view());
            let (cpu, cpu_stats) = refine_labels(&energy, labels.view(), p.view(), fv, &schedule).unwrap();
            let (gpu, gpu_stats) = refine_labels_gpu(&energy, labels.view(), p.view(), fv, &schedule).unwrap();
            assert_eq!(cpu, gpu, "data {data}: labels");
            let flips = |s: &[SweepStats]| s.iter().map(|s| s.flips).collect::<Vec<_>>();
            assert_eq!(flips(&cpu_stats), flips(&gpu_stats), "data {data}: flips per sweep");
            assert!(cpu_stats.iter().map(|s| s.flips).sum::<u64>() > 0, "the fixture moves");
        }
    }

    /// A block inside a larger volume, at an odd offset: the parity classes
    /// and the counter index come from global coordinates on the device too.
    #[test]
    fn gpu_sweeps_match_at_an_offset_and_through_the_op() {
        if !device_or_skip("gpu_sweeps_match_at_an_offset_and_through_the_op") {
            return;
        }
        let n = [21, 26, 31];
        let (labels, p, f) = fixture(n, 5, 0.05, false);
        let means = label_means(labels.view(), f.view()).unwrap();
        let energy = LabelEnergy::new(
            LabelEnergyParams::new([1.0; 3], 1.0, 1.0, 0.02)
                .with_data(0.3, 0.5)
                .with_smoothing(0.2, 10)
                .with_seed(5),
        )
        .unwrap();
        let cut = s![3..17, 5..22, 1..30];
        let (lb, pb, fb) = (
            labels.slice(cut).to_owned(),
            p.slice(cut).to_owned(),
            f.slice(cut).to_owned(),
        );
        let at = Anchor::new([3, 5, 1], [40, 50, 60]);
        let sweeps = [(4, 1.5), (5, 0.0)];
        let cpu = cpu_sweeps(&energy, &lb, &pb, Some(&fb), &means, &sweeps, &at);
        let gpu = label_energy_sweeps_gpu(
            &energy,
            lb.view(),
            pb.view(),
            Some(fb.view()),
            &means,
            &sweeps,
            &at,
        )
        .unwrap();
        assert_same("offset", &cpu, &gpu);

        // the accelerated op, through `apply_with`, against the CPU op
        for (sweep, t) in sweeps {
            let op = LabelEnergySweepOp::new(
                energy.clone(),
                1usize,
                Some(ImageId::from(2usize)),
                Arc::new(means.clone()),
                sweep,
                t,
            )
            .unwrap();
            let fast = accelerated(op.clone(), DevicePolicy::Gpu);
            assert!(fast.runs_on_gpu());
            assert_eq!(fast.source_inputs(at.volume), op.source_inputs(at.volume));
            let input = Voxels::U32(lb.clone());
            let (pv, fvv) = (Voxels::F64(pb.clone()), Voxels::F64(fb.clone()));
            let entries = [(ImageId::from(1usize), &pv), (ImageId::from(2usize), &fvv)];
            let mut want = Voxels::zeros(Dtype::U32, input.shape()).unwrap();
            let mut got = Voxels::zeros(Dtype::U32, input.shape()).unwrap();
            op.apply_with(&input, SourceInputs::new(&entries), &mut want, &at)
                .unwrap();
            fast.apply_with(&input, SourceInputs::new(&entries), &mut got, &at)
                .unwrap();
            assert_eq!(got.view::<u32>().unwrap(), want.view::<u32>().unwrap());
            assert_ne!(got.view::<u32>().unwrap(), lb.view(), "sweep {sweep} moved");
        }
    }

    fn seconds(f: impl FnOnce()) -> f64 {
        let start = Instant::now();
        f();
        start.elapsed().as_secs_f64()
    }

    /// 64x512x512 labels, 10 sweeps (5 Metropolis, 5 ICM), topology and data
    /// on: the CPU reference (rayon, all threads) against the GPU, transfers
    /// included. `cargo test --release --features gpu-cuda --lib
    /// benchmark_label_energy -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn benchmark_label_energy() {
        if !device_or_skip("benchmark_label_energy") {
            return;
        }
        let n = [64, 512, 512];
        let (labels, p, f) = fixture(n, 24, 0.005, true);
        let means = label_means(labels.view(), f.view()).unwrap();
        let energy = LabelEnergy::new(
            LabelEnergyParams::new([2.0, 1.0, 1.0], 1.0, 2.0, 0.05)
                .with_data(0.3, 0.5)
                .with_smoothing(0.2, 10)
                .with_seed(1),
        )
        .unwrap();
        let sweeps: Vec<(usize, f64)> = (0..10)
            .map(|s| {
                (
                    s,
                    if s < 5 {
                        0.5 * 0.7f64.powi(s as i32)
                    } else {
                        0.0
                    },
                )
            })
            .collect();
        let at = Anchor::whole(n);
        let table = seconds(|| {
            simple_point_table();
        });
        // compile the kernels
        let _ = label_energy_sweeps_gpu(
            &energy,
            labels.slice(s![..8, ..64, ..64]).view(),
            p.slice(s![..8, ..64, ..64]).view(),
            Some(f.slice(s![..8, ..64, ..64]).view()),
            &means,
            &[(0, 0.5), (1, 0.0)],
            &Anchor::new([0, 0, 0], n),
        )
        .unwrap();
        let one = || {
            label_energy_sweeps_gpu(
                &energy,
                labels.view(),
                p.view(),
                Some(f.view()),
                &means,
                &sweeps[..1],
                &at,
            )
            .unwrap();
        };
        one(); // the first full-size call also grows the device pool
        let one_sweep = seconds(one);
        let mut cpu = None;
        let cpu_time = seconds(|| {
            cpu = Some(cpu_sweeps(
                &energy,
                &labels,
                &p,
                Some(&f),
                &means,
                &sweeps,
                &at,
            ))
        });
        let mut gpu = None;
        let gpu_time = seconds(|| {
            gpu = Some(
                label_energy_sweeps_gpu(
                    &energy,
                    labels.view(),
                    p.view(),
                    Some(f.view()),
                    &means,
                    &sweeps,
                    &at,
                )
                .unwrap(),
            )
        });
        let (cpu, gpu) = (cpu.unwrap(), gpu.unwrap());
        assert_same("benchmark", &cpu, &gpu);
        // one call per sweep, as the op runs: every sweep uploads and downloads
        let mut current = labels.clone();
        let per_sweep_time = seconds(|| {
            for &sweep in &sweeps {
                current = label_energy_sweeps_gpu(
                    &energy,
                    current.view(),
                    p.view(),
                    Some(f.view()),
                    &means,
                    &[sweep],
                    &at,
                )
                .unwrap()
                .0;
            }
        });
        assert_eq!(current, cpu.0);
        let flips: u64 = cpu.1.iter().map(|s| s.flips).sum();
        let boundary = labels
            .indexed_iter()
            .filter(|&((i, j, k), &l)| {
                let lo = [
                    i.saturating_sub(1),
                    j.saturating_sub(1),
                    k.saturating_sub(1),
                ];
                labels
                    .slice(s![
                        lo[0]..(i + 2).min(n[0]),
                        lo[1]..(j + 2).min(n[1]),
                        lo[2]..(k + 2).min(n[2])
                    ])
                    .iter()
                    .any(|&m| m != l)
            })
            .count() as f64
            / (n[0] * n[1] * n[2]) as f64;
        let voxels = (n[0] * n[1] * n[2]) as f64;
        eprintln!(
            "label energy {n:?} ({:.0}% boundary voxels), 10 sweeps, {flips} flips, {} rayon threads: simple-point table \
             {table:.2} s (once per process); CPU {cpu_time:.2} s; GPU {gpu_time:.3} s resident \
             ({:.1}x), {per_sweep_time:.3} s with a round trip per sweep ({:.1}x); per voxel and \
             sweep: CPU {:.1} ns, GPU {:.2} ns; one resident sweep {one_sweep:.3} s, so the \
             kernels take {:.3} s a sweep and the transfers {:.3} s a call",
            boundary * 100.0,
            rayon::current_num_threads(),
            cpu_time / gpu_time,
            cpu_time / per_sweep_time,
            cpu_time / voxels / 10.0 * 1e9,
            per_sweep_time / voxels / 10.0 * 1e9,
            (gpu_time - one_sweep) / 9.0,
            one_sweep - (gpu_time - one_sweep) / 9.0,
        );
    }
}
