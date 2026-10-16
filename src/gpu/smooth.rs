// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// Separable Gaussian smoothing on the GPU: the device twin of
// `ops::smooth::SmoothOp`, which stays the definition.
//
// Same kernels (the `f64` weights `ops::ridge::gaussian_weights` builds, rounded
// to `f32`), same pass order (axis 0, 1, 2), same boundary rule
// (`ridge::Boundary`, resolved per tap exactly as `Boundary::index` does). What
// differs is the arithmetic: `f32` accumulation, with fused multiply-adds, on
// the device, against `f64` on the host. **Parity tolerance:** relative to the
// input's largest magnitude, `|gpu - cpu| <= 1e-5 * max|x|` — three passes of
// up to a few dozen `f32` multiply-adds each, every intermediate a convex
// combination of the input, so the error is a small multiple of `f32`'s
// `6e-8` per pass and nowhere near the bound. The test states it.
//
// One thread per output voxel, one launch per blurred axis; an axis whose sigma
// is zero (the one-tap kernel `[1.0]`) is skipped rather than launched, which is
// exact. Two device buffers ping-pong: the uploaded input is overwritten by the
// second pass, so a block needs `2 * 4` bytes a voxel on the device.

use std::sync::Arc;

use cubecl::prelude::*;

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::op::Anchor;
use crate::ops::ridge::Boundary;
use crate::ops::smooth::{Gaussian, SmoothOp};
use crate::voxels::Voxels;

use super::device::{check_indexable, device, launch_shape, Rt};
use super::{admit, Accelerated, DevicePolicy, GpuKernel};

/// `Boundary::index` for one tap. `reflect` is comptime, so each convention is
/// its own compiled kernel with no branch on it.
#[cube]
fn resolve(position: i32, extent: i32, #[comptime] reflect: bool) -> i32 {
    let mut index = 0i32;
    if extent > 1 {
        if reflect {
            let period = 2 * extent;
            let mut folded = position % period;
            if folded < 0 {
                folded += period;
            }
            if folded >= extent {
                folded = period - folded - 1;
            }
            index = folded;
        } else if position < 0 {
            index = 0;
        } else if position >= extent {
            index = extent - 1;
        } else {
            index = position;
        }
    }
    index
}

/// One separable pass along `axis` (comptime: 0, 1 or 2) of a `[n0, n1, n2]`
/// C-order block.
#[cube(launch_unchecked)]
fn convolve_axis(
    src: &[f32],
    dst: &mut [f32],
    weights: &[f32],
    n0: u32,
    n1: u32,
    n2: u32,
    #[comptime] axis: u32,
    #[comptime] reflect: bool,
) {
    let total = (n0 * n1 * n2) as usize;
    if ABSOLUTE_POS < total {
        let p = ABSOLUTE_POS as u32;
        let i2 = p % n2;
        let i1 = (p / n2) % n1;
        let i0 = p / (n2 * n1);
        let mut centre = i2;
        let mut extent = n2;
        let mut stride = 1u32;
        if axis == 0 {
            centre = i0;
            extent = n0;
            stride = n1 * n2;
        }
        if axis == 1 {
            centre = i1;
            extent = n1;
            stride = n2;
        }
        let base = p - centre * stride;
        let taps = weights.len() as u32;
        let radius = ((taps - 1) / 2) as i32;
        let mut acc = 0.0f32;
        for t in 0..taps {
            let at = centre as i32 + t as i32 - radius;
            let index = resolve(at, extent as i32, reflect) as u32;
            acc += weights[t as usize] * src[(base + index * stride) as usize];
        }
        dst[ABSOLUTE_POS] = acc;
    }
}

/// The GPU implementation of a [`Gaussian`].
pub struct GpuSmooth {
    weights: [Vec<f32>; 3],
    reflect: bool,
}

impl GpuSmooth {
    pub fn new(gaussian: &Gaussian) -> Self {
        let narrow = |k: &Vec<f64>| k.iter().map(|&w| w as f32).collect::<Vec<f32>>();
        let kernels = gaussian.kernels();
        Self {
            weights: [
                narrow(&kernels[0]),
                narrow(&kernels[1]),
                narrow(&kernels[2]),
            ],
            reflect: gaussian.boundary() == Boundary::Reflect,
        }
    }

    /// Smooth an `f32` array on the device: upload, three passes, download.
    /// The data-level entry point; [`GpuKernel::apply`] is this over `Voxels`.
    pub fn smooth_f32(&self, input: &Voxels, out: &mut Voxels) -> Result<()> {
        let gpu = device().ok_or_else(|| Error::InvalidArgument("no GPU device".into()))?;
        let shape = input.shape();
        check_indexable(shape, "gpu smooth")?;
        let _admission = admit(self.device_bytes(shape));
        if shape.contains(&0) {
            return Ok(());
        }
        let client = gpu.client();
        let mut src = gpu.upload_f32(input)?;
        let mut dst = gpu.empty(shape, Dtype::F32);
        let (count, dim) = launch_shape(src.len());
        let [n0, n1, n2] = shape.map(|n| n as u32);
        for axis in 0..3u32 {
            let weights = &self.weights[axis as usize];
            if weights.len() == 1 {
                continue; // `[1.0]`: the identity, exactly.
            }
            let handle = gpu.upload_slice_f32(weights);
            // SAFETY: every buffer holds `n0 * n1 * n2` (or `taps`) elements and
            // the kernel indexes them only below that, guarded by `total`.
            unsafe {
                convolve_axis::launch_unchecked::<Rt>(
                    client,
                    count.clone(),
                    dim,
                    src.arg(),
                    dst.arg(),
                    BufferArg::from_raw_parts(handle, weights.len()),
                    n0,
                    n1,
                    n2,
                    axis,
                    self.reflect,
                );
            }
            std::mem::swap(&mut src, &mut dst);
        }
        gpu.download_into(src, out)
    }
}

impl GpuKernel for GpuSmooth {
    fn device_bytes(&self, block: [usize; 3]) -> u64 {
        // Two `f32` ping-pong buffers, plus a megabyte for the weights and
        // CubeCL's pool rounding.
        8 * block.iter().product::<usize>() as u64 + (1 << 20)
    }

    fn cost_per_voxel(&self) -> f64 {
        GPU_SMOOTH_COST_PER_VOXEL
            + GPU_SMOOTH_COST_PER_TAP * self.weights.iter().map(Vec::len).sum::<usize>() as f64
    }

    fn apply(&self, inputs: &[&Voxels], out: &mut Voxels, _at: &Anchor) -> Result<()> {
        let [input] = inputs else {
            return Err(Error::InvalidArgument("gpu smooth takes one input".into()));
        };
        self.smooth_f32(input, out)
    }
}

/// Measured with `benchmark_gaussian_cpu_against_gpu` in `src/gpu/tests.rs` on
/// a Quadro RTX 5000 (PCIe 3 x16), in the units `ops::smooth` uses (one
/// voxelwise map, calibrated by the CPU op's own declared cost in the same
/// run). The per-voxel term is the transfer (4 bytes up, 4 down, plus the host
/// conversion to the `f64` the op produces) and the launch overhead amortised
/// over a 16 Mvoxel block; the per-tap term is the kernel.
pub(crate) const GPU_SMOOTH_COST_PER_VOXEL: f64 = 4.3;
pub(crate) const GPU_SMOOTH_COST_PER_TAP: f64 = 0.008;

/// [`SmoothOp`] with its GPU implementation attached, under `policy`.
pub fn accelerated(op: SmoothOp, policy: DevicePolicy) -> Accelerated<SmoothOp> {
    let kernel = Arc::new(GpuSmooth::new(op.gaussian()));
    Accelerated::with_kernel(op, kernel, policy)
}
