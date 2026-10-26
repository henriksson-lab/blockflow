// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The vessel-filling network on a GPU (burn's CUDA or wgpu backend, the
//! `deep-fill-cuda` / `deep-fill-wgpu` features), the whole forward pass on
//! the device.
//!
//! The same network as [`super::network`], the same weights and the same
//! lengths; what differs is where each stage runs and how:
//!
//! * the pools, the encoder and both trilinear upsamplings are burn tensor
//!   ops on the device (the upsampling as edge-padded shifts, stacked and
//!   interleaved, which is the host's arithmetic term for term);
//! * `conv3`, transposed, is the same polyphase convolution as on the host,
//!   as eight ordinary convolutions, one per output parity (3 or 4 taps per
//!   axis, so no product is by zero), interleaved by a reshape and a
//!   permutation: 0.24 s on the 128 x 512 x 404 crop, against 0.64 s as one
//!   4^3 convolution to 128 channels;
//! * `conv4` and the trilinear upsampling before it, the only stage at full
//!   resolution with 17 channels, are composed: an upsampling by 2 followed
//!   by a 3-tap convolution is, per output parity, a 3-tap convolution of the
//!   half-resolution features ([`upsampled_conv4`]), exactly so with both
//!   edge-padded. conv4 pads with zeros, so the outermost output plane of
//!   each face is recomputed the direct way (upsampled, concatenated,
//!   convolved). 0.11 s on the crop, against 0.47 s for the direct way over
//!   the whole volume;
//! * at resample 1 a filled mask is thresholded on the device and copied
//!   back as a byte a voxel.
//!
//! **The first pass compiles.** The runtime compiles kernels on first use,
//! specialised on how their lengths divide: seconds per process. The GPU
//! thread warms a few small shapes when it starts, while the run's earlier
//! phases are on the host; a `cubecl.toml` with `[compilation] cache = true`
//! in the working directory keeps compiled kernels across processes.
//!
//! The probability agrees with the host network's to floating-point
//! rounding (the device sums in another order); the README gives the
//! measured difference.

use blockflow::{Error, Result};
use burn::prelude::{Device, Tensor, TensorData};
use burn::tensor::activation::{relu, sigmoid};
use burn::tensor::module::conv3d;
use burn::tensor::ops::{ConvOptions, PadMode};

use super::network::{
    encode, encoder, host, max_pool2, output_length, polyphase_matrix, reverse_spatial, Conv,
    Stored, Weights,
};

#[cfg(not(any(feature = "deep-fill-cuda", feature = "deep-fill-wgpu")))]
compile_error!(
    "the `deep-fill-gpu` feature needs a runtime: enable `deep-fill-cuda` or `deep-fill-wgpu`"
);

/// Device bytes per network-input voxel a forward pass holds at its peak:
/// measured as the device's memory in use (`nvidia-smi`, the runtime's pool
/// included) over the idle figure, 4.7 GB for a 320 x 512 x 256 input and
/// 8.3 GB for 512 x 512 x 256 (112 and 124 bytes a voxel, 144 between the
/// two), rounded up.
const BYTES_PER_VOXEL: u64 = 144;

/// The burn device of this build's GPU runtime.
pub(super) fn burn_device() -> Device {
    #[cfg(feature = "deep-fill-cuda")]
    {
        Device::cuda(burn::tensor::DeviceIndex::Default)
    }
    #[cfg(all(feature = "deep-fill-wgpu", not(feature = "deep-fill-cuda")))]
    {
        Device::wgpu(burn::tensor::DeviceKind::DefaultDevice)
    }
}

/// The device bytes the process may lease (`blockflow::gpu`'s budget:
/// three quarters of the device by default, `BLOCKFLOW_GPU_MEMORY` to set
/// it); `None` without a device.
pub(super) fn device_budget() -> Option<u64> {
    blockflow::gpu::budgets().map(|(_, memory)| memory.total())
}

/// The trained network on the GPU.
pub struct GpuNetwork {
    device: Device,
    encoder: [Conv; 5],
    /// `conv3` as eight ordinary convolutions, one per output parity
    /// (`4 pz + 2 py + px`), `[16, 32, k, k, k]` with `k` 3 along an axis of
    /// parity 0 and 4 along one of parity 1 (a third of the 4^3 window's
    /// products are by zero).
    conv3: Vec<Tensor<5>>,
    conv3_bias: Tensor<5>,
    /// `conv4`, `[2, 17, 3, 3, 3]`, zero padding 1.
    conv4: Conv,
    /// `conv4`'s skip-channel part, `[2, 1, 3, 3, 3]` with the bias.
    conv4_skip: Conv,
    /// `conv4`'s feature part composed with the upsampling before it, at
    /// half resolution: `[2 * 8, 16, 3, 3, 3]`, output channel `8 o +
    /// parity` over the edge-padded half-resolution features
    /// ([`upsampled_conv4`]).
    conv4_features: Tensor<5>,
    budget: u64,
}

impl GpuNetwork {
    /// The network of `weights` on this build's GPU; an error when there is
    /// none to open.
    pub(super) fn new(weights: &Weights) -> Result<Self> {
        let budget = device_budget().ok_or_else(|| {
            Error::invalid(
                "deep fill: no GPU is reachable (no device or driver, or BLOCKFLOW_GPU=off)",
            )
        })?;
        let device = burn_device();
        let matrix = polyphase_matrix(&weights.conv3.0);
        // The matrix is `[tap][ci][8 c + parity]`, tap `16 tz + 4 ty + tx`;
        // a convolution weight is `[c][ci][tz][ty][tx]`.
        let (c_in, c_out) = (32, 16);
        let conv3 = (0..8)
            .map(|parity| {
                let k = [parity >> 2, (parity >> 1) & 1, parity & 1].map(|p| 3 + p);
                let mut kernel = vec![0.0f32; c_out * c_in * k[0] * k[1] * k[2]];
                for c in 0..c_out {
                    for ci in 0..c_in {
                        for tz in 0..k[0] {
                            for ty in 0..k[1] {
                                for tx in 0..k[2] {
                                    let tap = 16 * tz + 4 * ty + tx;
                                    kernel
                                        [(((c * c_in + ci) * k[0] + tz) * k[1] + ty) * k[2] + tx] =
                                        matrix[(tap * c_in + ci) * c_out * 8 + 8 * c + parity];
                                }
                            }
                        }
                    }
                }
                Tensor::<5>::from_data(
                    TensorData::new(kernel, [c_out, c_in, k[0], k[1], k[2]]),
                    &device,
                )
            })
            .collect();
        let conv3_bias = Tensor::<5>::from_data(
            TensorData::new(weights.conv3.1.data.clone(), [1, 16, 1, 1, 1]),
            &device,
        );
        Ok(Self {
            encoder: encoder(weights, &device),
            conv3,
            conv3_bias,
            conv4: Conv::new(&weights.conv4, 1, 1, 1, &device),
            conv4_skip: Conv::new(
                &(
                    Stored {
                        shape: vec![2, 1, 3, 3, 3],
                        data: (0..2)
                            .flat_map(|o| weights.conv4.0.data[o * 17 * 27..][..27].to_vec())
                            .collect(),
                    },
                    Stored {
                        shape: vec![2],
                        data: weights.conv4.1.data.clone(),
                    },
                ),
                1,
                1,
                1,
                &device,
            ),
            conv4_features: Tensor::<5>::from_data(
                TensorData::new(upsampled_conv4(&weights.conv4.0), [16, 16, 3, 3, 3]),
                &device,
            ),
            device,
            budget,
        })
    }

    /// The device budget blocks are sized against.
    pub fn budget(&self) -> u64 {
        self.budget
    }

    /// Device bytes a forward pass over `shape` holds at its peak.
    pub fn device_bytes(shape: [usize; 3]) -> u64 {
        BYTES_PER_VOXEL * shape.iter().product::<usize>() as u64
    }

    /// [`super::network::Network::forward`] on the device.
    pub fn forward(&self, input: Vec<f32>, shape: [usize; 3]) -> Result<(Vec<f32>, [usize; 3])> {
        let (p, shape) = self.probability(input, shape)?;
        Ok((host(p)?, shape))
    }

    /// Whether [`Self::forward`]'s probability reaches `threshold`, compared
    /// on the device: a byte a voxel to copy back instead of four.
    pub fn above(
        &self,
        input: Vec<f32>,
        shape: [usize; 3],
        threshold: f32,
    ) -> Result<(Vec<bool>, [usize; 3])> {
        let (p, shape) = self.probability(input, shape)?;
        let data = p.greater_equal_elem(threshold).into_data();
        // The runtime stores a bool in a byte or in a 32-bit word.
        let n = data.num_elements();
        let bytes = data.as_bytes();
        let set = match bytes.len() / n.max(1) {
            1 => bytes.iter().map(|&b| b != 0).collect(),
            4 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| *w != [0; 4])
                .collect(),
            _ => {
                return Err(Error::backend(format!(
                    "vessel filling: a mask of {n} voxels came back as {} bytes",
                    bytes.len()
                )))
            }
        };
        Ok((set, shape))
    }

    /// The probability as a device tensor `[1, 1, ...]`, and its shape.
    fn probability(&self, input: Vec<f32>, shape: [usize; 3]) -> Result<(Tensor<5>, [usize; 3])> {
        if shape.iter().any(|&n| n < 6) {
            return Err(Error::invalid(format!(
                "vessel filling: the network takes at least 6 voxels per axis, got {shape:?}"
            )));
        }
        if input.len() != shape.iter().product::<usize>() {
            return Err(Error::invalid("vessel filling: input length"));
        }
        // With `VESSEL_GRAPH_PROFILE`, each stage is waited for and timed.
        let profile = crate::profile::enabled();
        let mut last = std::time::Instant::now();
        let mut mark = |stage: &str| {
            if profile {
                let _ = self.device.sync();
                eprintln!(
                    "deep fill gpu {shape:?}: {stage} {:.3}s",
                    last.elapsed().as_secs_f64()
                );
                last = std::time::Instant::now();
            }
        };
        let x = Tensor::<5>::from_data(
            TensorData::new(input, [1, 1, shape[0], shape[1], shape[2]]),
            &self.device,
        );
        mark("upload");
        let coarse = encode(&self.encoder, max_pool2(x.clone()));
        mark("encoder");
        let up = upsample2(upsample2(upsample2(coarse, 2), 3), 4);
        let [_, _, n0, n1, n2] = up.dims();
        // conv3: a window starting at m - 1 reaches m + 2.
        let padded = up.pad([(1, 2), (1, 2), (1, 2)], PadMode::Constant(0.0));
        // Parity 0 along an axis reads a window of 3 starting at m - 1.
        let parities = self
            .conv3
            .iter()
            .enumerate()
            .map(|(parity, kernel)| {
                let mut x = padded.clone();
                for (axis, n) in [n0, n1, n2].into_iter().enumerate() {
                    if (parity >> (2 - axis)) & 1 == 0 {
                        x = x.narrow(2 + axis, 0, n + 2);
                    }
                }
                conv3d(
                    x,
                    kernel.clone(),
                    None,
                    ConvOptions::new([1; 3], [0; 3], [1; 3], 1),
                )
            })
            .collect();
        let poly = Tensor::stack::<6>(parities, 2);
        // Channel c at parity 4 pz + 2 py + px of window m is output 2 m + p.
        let l = [2 * n0 - 1, 2 * n1 - 1, 2 * n2 - 1];
        let half = poly
            .reshape([16, 2, 2, 2, n0, n1, n2])
            .permute([0, 4, 1, 5, 2, 6, 3])
            .reshape([1, 16, 2 * n0, 2 * n1, 2 * n2])
            .narrow(2, 0, l[0])
            .narrow(3, 0, l[1])
            .narrow(4, 0, l[2]);
        let half = relu(half + self.conv3_bias.clone());
        mark("conv3");
        let [d, h, w] = l.map(|n| 2 * n);
        debug_assert_eq!([d, h, w], shape.map(output_length));
        // The skip connection: the input cropped to the decoder's length.
        let skip = x.narrow(2, 0, d).narrow(3, 0, h).narrow(4, 0, w);
        // conv4 of the upsampled features with edge padding, at half
        // resolution (one 3^3 kernel per output parity), plus conv4 of the
        // skip channel at full resolution.
        let features = conv3d(
            half.clone().pad([(1, 1), (1, 1), (1, 1)], PadMode::Edge),
            self.conv4_features.clone(),
            None,
            ConvOptions::new([1; 3], [0; 3], [1; 3], 1),
        )
        .reshape([2, 2, 2, 2, l[0], l[1], l[2]])
        .permute([0, 4, 1, 5, 2, 6, 3])
        .reshape([1, 2, d, h, w]);
        let mut c = features + self.conv4_skip.apply(skip.clone());
        mark("conv4");
        // The outermost plane of each face, where conv4's zero padding
        // differs from the edge padding above: recomputed as conv4 of the
        // upsampled features.
        let full = [d, h, w];
        for axis in 0..3 {
            let n = full[axis];
            let mut faces = vec![(0, 1)];
            if n > 1 {
                faces.push((n - 1, n));
            }
            for (a, b) in faces {
                let exact = self.exact_planes(&half, &skip, l, full, axis, a, b);
                let mut ranges = [0..1, 0..2, 0..d, 0..h, 0..w];
                ranges[2 + axis] = a..b;
                c = c.slice_assign(ranges, exact);
            }
        }
        mark("faces");
        // sigmoid(4 (c - 0.5)) per channel, then the two-way softmax's
        // channel 0.
        let squashed = sigmoid(c.sub_scalar(0.5).mul_scalar(4.0));
        let a = squashed.clone().narrow(1, 0, 1);
        let b = squashed.narrow(1, 1, 1);
        let p = sigmoid(a - b);
        mark("squash");
        Ok((p, [d, h, w]))
    }

    /// conv4 (zero padding) of the upsampled features and the skip channel at
    /// output planes `a..b` along `axis`: `[1, 2, ...]`, `b - a` along `axis`
    /// and the whole volume along the others.
    #[allow(clippy::too_many_arguments)]
    fn exact_planes(
        &self,
        half: &Tensor<5>,
        skip: &Tensor<5>,
        l: [usize; 3],
        full: [usize; 3],
        axis: usize,
        a: usize,
        b: usize,
    ) -> Tensor<5> {
        let dim = 2 + axis;
        // The upsampled planes the 3^3 convolution reads (zero padding past
        // the volume, which the convolution's own padding supplies), and the
        // half-resolution planes those interpolate.
        let (ua, ub) = (a.saturating_sub(1), (b + 1).min(full[axis]));
        let s0 = (ua / 2).saturating_sub(1);
        let s1 = ((ub - 1) / 2 + 2).min(l[axis]);
        let mut features =
            upsample2(half.clone().narrow(dim, s0, s1 - s0), dim).narrow(dim, ua - 2 * s0, ub - ua);
        for other in (0..3).filter(|&other| other != axis) {
            features = upsample2(features, 2 + other);
        }
        let input = Tensor::cat(vec![skip.clone().narrow(dim, ua, ub - ua), features], 1);
        self.conv4.apply(input).narrow(dim, a - ua, b - a)
    }
}

/// `conv4`'s 16 feature channels (`[2, 17, 3, 3, 3]`, the reference's axis
/// order) composed with the trilinear upsampling by 2 before it, as kernels
/// over the half-resolution features: `[2 * 8, 16, 3, 3, 3]`, our axis order,
/// output channel `8 o + 4 pz + 2 py + px`.
///
/// Per axis, upsampled `u[2 m] = 0.25 h[m - 1] + 0.75 h[m]` and `u[2 m + 1]
/// = 0.75 h[m] + 0.25 h[m + 1]`, so a 3-tap `k` over `u` at output `2 m + p`
/// is a 3-tap over `h[m - 1 ..= m + 1]`: [`PARITY_TAPS`]`[p] k`. The
/// identity holds with `u` and `h` edge-padded alike (`u[-1] = u[0] = h[0]`),
/// which is not conv4's zero padding: the outermost output plane of each face
/// is recomputed.
fn upsampled_conv4(weight: &Stored) -> Vec<f32> {
    let k = reverse_spatial(weight);
    let at = |o: usize, c: usize, z: usize, y: usize, x: usize| {
        k[(((o * 17 + c) * 3 + z) * 3 + y) * 3 + x] as f64
    };
    let mut out = vec![0.0f32; 16 * 16 * 27];
    for o in 0..2 {
        for parity in 0..8 {
            let p = [parity >> 2, (parity >> 1) & 1, parity & 1];
            for c in 0..16 {
                for a in 0..27 {
                    let (az, ay, ax) = (a / 9, (a / 3) % 3, a % 3);
                    let mut sum = 0.0f64;
                    for i in 0..27 {
                        let (iz, iy, ix) = (i / 9, (i / 3) % 3, i % 3);
                        sum += PARITY_TAPS[p[0]][az][iz]
                            * PARITY_TAPS[p[1]][ay][iy]
                            * PARITY_TAPS[p[2]][ax][ix]
                            * at(o, c + 1, iz, iy, ix);
                    }
                    out[((8 * o + parity) * 16 + c) * 27 + a] = sum as f32;
                }
            }
        }
    }
    out
}

/// `PARITY_TAPS[p][j][i]`: the weight of `h[m - 1 + j]` in `u[2 m + p - 1 +
/// i]`, the upsampled voxel a 3-tap kernel's tap `i` reads at output `2 m +
/// p`.
const PARITY_TAPS: [[[f64; 3]; 3]; 2] = [
    [[0.75, 0.25, 0.0], [0.25, 0.75, 0.75], [0.0, 0.0, 0.25]],
    [[0.25, 0.0, 0.0], [0.75, 0.75, 0.25], [0.0, 0.25, 0.75]],
];

/// Linear upsampling by 2 along `dim`, `align_corners=False`, the
/// neighbours clamped at the ends: output `2 i` is `0.25 x[i - 1] + 0.75
/// x[i]`, `2 i + 1` is `0.75 x[i] + 0.25 x[i + 1]` (`up_taps` on the host).
fn upsample2(x: Tensor<5>, dim: usize) -> Tensor<5> {
    let dims = x.dims();
    let n = dims[dim];
    let mut padding = [(0, 0); 5];
    padding[dim] = (1, 1);
    let padded = x.clone().pad(padding, PadMode::Edge);
    let previous = padded.clone().narrow(dim, 0, n);
    let next = padded.narrow(dim, 2, n);
    let centre = x.mul_scalar(0.75);
    let even = previous.mul_scalar(0.25) + centre.clone();
    let odd = centre + next.mul_scalar(0.25);
    let mut out = dims;
    out[dim] = 2 * n;
    Tensor::stack::<6>(vec![even, odd], dim + 1).reshape(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// In one dimension: a 3-tap kernel over the edge-padded upsampling of
    /// `h` is [`PARITY_TAPS`]' 3-tap over the edge-padded `h`, at every output.
    #[test]
    fn upsampled_convolution_is_a_convolution_per_parity() {
        let h = [0.3, -1.2, 2.5, 0.7, -0.4];
        let k = [0.9, -0.35, 1.6];
        let l = h.len();
        let at = |i: isize| h[i.clamp(0, l as isize - 1) as usize];
        let u = |o: isize| {
            let o = o.clamp(0, 2 * l as isize - 1);
            let i = o / 2;
            if o % 2 == 0 {
                0.25 * at(i - 1) + 0.75 * at(i)
            } else {
                0.75 * at(i) + 0.25 * at(i + 1)
            }
        };
        for o in 0..2 * l as isize {
            let direct: f64 = (0..3).map(|t| k[t] * u(o + t as isize - 1)).sum();
            let (m, p) = (o / 2, (o % 2) as usize);
            let composed: f64 = (0..3)
                .map(|j| {
                    let weight: f64 = (0..3).map(|i| PARITY_TAPS[p][j][i] * k[i]).sum();
                    weight * at(m - 1 + j as isize)
                })
                .sum();
            assert!(
                (direct - composed).abs() < 1e-12,
                "output {o}: {direct} {composed}"
            );
        }
    }
}
