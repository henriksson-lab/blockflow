// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The vessel-filling network on burn's CPU backend (`flex`), with the
//! reference's trained weights read from a safetensors file.
//!
//! The architecture, in the reference's own layer names (`encode` in its
//! `vessel_filling_network.py`):
//!
//! ```text
//! skip = x                                          1 ch, n
//! max pool 2 -> conv1 7^3 stride 2 -> relu          16 ch, n/4
//! max pool 2 -> conv2 5^3 -> relu                   32 ch, n/8
//! desepconv4 (depthwise 3^3, pointwise 1^3)         32 ch
//! convbin1 3^3 -> relu                              32 ch
//! trilinear x2 -> conv3 (transposed 7^3, stride 2)  16 ch, n/2
//!   -> relu -> trilinear x2                         16 ch, n
//! cat(skip, .) -> conv4 3^3 -> sigmoid(4 (. - 0.5)) 2 ch
//! softmax over the two channels, channel 0          the probability
//! ```
//!
//! Dropout layers sit between these in the reference; they are identity at
//! inference and are left out (see the module documentation of [`super`] for
//! the reference running them in training mode).
//!
//! **Axis order.** The reference hands its network `[x, y, z]` arrays and this
//! crate holds `[z, y, x]`; every kernel's spatial axes are reversed at load
//! time, so the network applied to a `[z, y, x]` buffer is the reference's
//! applied to the same voxels in its order. Pooling and upsampling treat the
//! axes alike.
//!
//! **Lengths.** An input of `n >= 6` voxels per axis gives an output of
//! [`output_length`]`(n) = 8 floor((n + 2) / 8) - 2`, the first voxels of the
//! input's: the reference's pooling floors, and it crops the skip connection
//! to the decoder's length.
//!
//! **What runs where.** The encoder (`conv1` to `convbin1`, at a quarter and
//! an eighth of the resolution) is burn's `conv3d`. The decoder's two layers
//! are most of the arithmetic and burn's convolutions are a poor fit for both,
//! so they are assembled here; timings on a 128 x 512 x 404 crop, 40 cores:
//!
//! * `conv3`, transposed. Burn's transposed 3-D convolution materialises a
//!   column buffer of `out_ch * k^3` floats per input voxel (9.7 GB for a 192 x
//!   384 x 384 block) and scatters it on one thread. Its polyphase equivalent
//!   (one ordinary convolution with a 4^3 kernel and 8 x 16 output channels,
//!   one per output parity) took 7.5 s through burn's `conv3d`, and as 64
//!   per-tap products summed, 9.8 s. Here each batch of output rows gathers
//!   its column (64 taps x 32 channels, each tap a contiguous run of the
//!   channels-last padded input) and is one burn `matmul` with the
//!   `[2048, 128]` kernel: 4.3 s, 107 GMAC of which a third multiply zeros.
//! * `conv4` with the upsampling before it, the only stage at full
//!   resolution with 17 channels: 73 s through burn's `conv3d` (an im2col of
//!   459 floats per voxel feeding a two-row matrix product), and 17 floats per
//!   voxel resident. Here it is a direct convolution over z planes, each
//!   upsampled when first needed, three held at a time: 0.6 s.
//!
//! Burn has no 3-D max pool or trilinear interpolation; the pools are
//! reshapes and maxima, the upsampling is written out.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use blockflow::{Error, Result};
use burn::prelude::{Device, Tensor, TensorData};
use burn::tensor::activation::relu;
use burn::tensor::module::conv3d;
use burn::tensor::ops::ConvOptions;
use rayon::prelude::*;

/// Output rows of `conv3` per matrix product: its column buffer is 8 KB a
/// row.
const CONV3_ROWS: usize = 1 << 14;

/// One convolution's weight and bias, already on the device.
pub(super) struct Conv {
    weight: Tensor<5>,
    bias: Tensor<1>,
    options: ConvOptions<3>,
}

impl Conv {
    pub(super) fn apply(&self, x: Tensor<5>) -> Tensor<5> {
        conv3d(
            x,
            self.weight.clone(),
            Some(self.bias.clone()),
            self.options.clone(),
        )
    }
}

/// A float32 tensor of a safetensors file.
pub(super) struct Stored {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Read every tensor of a safetensors file. Only `F32` tensors are accepted:
/// the reference's weights are float32, and `reference-clearmap/convert_weights.py`
/// writes them so.
pub(super) fn read_safetensors(path: &Path) -> Result<BTreeMap<String, Stored>> {
    let bytes = fs::read(path).map_err(|error| {
        Error::invalid(format!(
            "reading the vessel-filling weights {}: {error}",
            path.display()
        ))
    })?;
    let bad = |what: &str| {
        Error::invalid(format!(
            "{} is not a safetensors file ({what}); convert the reference's \
             `cnn_binary_vessel_filling.pth` with reference-clearmap/convert_weights.py",
            path.display()
        ))
    };
    if bytes.len() < 8 {
        return Err(bad("too short"));
    }
    let header_len = u64::from_le_bytes(bytes[..8].try_into().expect("eight bytes")) as usize;
    let body_start = 8usize
        .checked_add(header_len)
        .filter(|&end| end <= bytes.len())
        .ok_or_else(|| bad("header length"))?;
    let header: serde_json::Value =
        serde_json::from_slice(&bytes[8..body_start]).map_err(|_| bad("header"))?;
    let entries = header.as_object().ok_or_else(|| bad("header"))?;
    let body = &bytes[body_start..];
    let mut out = BTreeMap::new();
    for (name, entry) in entries {
        if name == "__metadata__" {
            continue;
        }
        let dtype = entry["dtype"].as_str().ok_or_else(|| bad("dtype"))?;
        if dtype != "F32" {
            return Err(Error::invalid(format!(
                "{}: tensor `{name}` is {dtype}; the network reads F32 weights",
                path.display()
            )));
        }
        let shape = entry["shape"]
            .as_array()
            .ok_or_else(|| bad("shape"))?
            .iter()
            .map(|d| d.as_u64().map(|d| d as usize))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| bad("shape"))?;
        let offsets = entry["data_offsets"]
            .as_array()
            .ok_or_else(|| bad("offsets"))?;
        let (lo, hi) = match offsets.as_slice() {
            [lo, hi] => (
                lo.as_u64().ok_or_else(|| bad("offsets"))? as usize,
                hi.as_u64().ok_or_else(|| bad("offsets"))? as usize,
            ),
            _ => return Err(bad("offsets")),
        };
        let count: usize = shape.iter().product();
        if hi > body.len() || lo > hi || hi - lo != 4 * count {
            return Err(bad("offsets"));
        }
        let data = body[lo..hi]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        out.insert(name.clone(), Stored { shape, data });
    }
    Ok(out)
}

/// A `[a, b, k, k, k]` kernel with its three spatial axes reversed, from the
/// reference's `[x, y, z]` order to this crate's `[z, y, x]`.
pub(super) fn reverse_spatial(stored: &Stored) -> Vec<f32> {
    let [a, b, k0, k1, k2] = stored.shape[..] else {
        unreachable!("checked to be five-dimensional");
    };
    let mut out = vec![0.0; stored.data.len()];
    for i in 0..a * b {
        for z in 0..k0 {
            for y in 0..k1 {
                for x in 0..k2 {
                    let from = ((i * k0 + z) * k1 + y) * k2 + x;
                    let to = ((i * k2 + x) * k1 + y) * k0 + z;
                    out[to] = stored.data[from];
                }
            }
        }
    }
    out
}

/// `conv3` as one matrix product per batch of output rows; see the module
/// documentation.
struct Conv3 {
    /// `[64 * 32, 128]`: row `32 t + ci` for tap `t = (tz, ty, tx)` (`16 tz +
    /// 4 ty + tx`) and input channel `ci`; column `8 c + 4 pz + 2 py + px`
    /// for output channel `c` at parity `(pz, py, px)`.
    kernel: Tensor<2>,
    bias: Vec<f32>,
}

/// The conv4 stage's weights, `[2][17][3][3][3]`, spatial axes reversed.
struct Conv4 {
    weight: Vec<f32>,
    bias: [f32; 2],
}

/// The trained vessel-filling network.
pub struct Network {
    device: Device,
    encoder: [Conv; 5],
    conv3: Conv3,
    conv4: Conv4,
}

/// The network's tensors as read from a safetensors file, every shape
/// checked; what [`Network`] and the GPU network are built from.
pub(super) struct Weights {
    pub conv1: (Stored, Stored),
    pub conv2: (Stored, Stored),
    pub depthwise: (Stored, Stored),
    pub pointwise: (Stored, Stored),
    pub convbin1: (Stored, Stored),
    pub conv3: (Stored, Stored),
    pub conv4: (Stored, Stored),
}

impl Weights {
    /// Read the reference's trained weights (its state-dict names) from a
    /// safetensors file.
    pub fn read(path: &Path) -> Result<Self> {
        let mut tensors = read_safetensors(path)?;
        let mut take = |name: &str, shape: &[usize]| -> Result<Stored> {
            let stored = tensors.remove(name).ok_or_else(|| {
                Error::invalid(format!(
                    "{}: no tensor `{name}`; these are not the vessel-filling network's weights",
                    path.display()
                ))
            })?;
            if stored.shape != shape {
                return Err(Error::invalid(format!(
                    "{}: `{name}` is {:?}, the network's is {shape:?}",
                    path.display(),
                    stored.shape
                )));
            }
            Ok(stored)
        };
        Ok(Self {
            conv1: (
                take("conv1.weight", &[16, 1, 7, 7, 7])?,
                take("conv1.bias", &[16])?,
            ),
            conv2: (
                take("conv2.weight", &[32, 16, 5, 5, 5])?,
                take("conv2.bias", &[32])?,
            ),
            depthwise: (
                take("desepconv4.depthwise.weight", &[32, 1, 3, 3, 3])?,
                take("desepconv4.depthwise.bias", &[32])?,
            ),
            pointwise: (
                take("desepconv4.pointwise.weight", &[32, 32, 1, 1, 1])?,
                take("desepconv4.pointwise.bias", &[32])?,
            ),
            convbin1: (
                take("convbin1.weight", &[32, 32, 3, 3, 3])?,
                take("convbin1.bias", &[32])?,
            ),
            conv3: (
                take("conv3.weight", &[32, 16, 7, 7, 7])?,
                take("conv3.bias", &[16])?,
            ),
            conv4: (
                take("conv4.weight", &[2, 17, 3, 3, 3])?,
                take("conv4.bias", &[2])?,
            ),
        })
    }
}

impl Conv {
    /// A convolution of `weight` (spatial axes reversed) and `bias` on
    /// `device`.
    pub(super) fn new(
        (weight, bias): &(Stored, Stored),
        stride: usize,
        padding: usize,
        groups: usize,
        device: &Device,
    ) -> Self {
        let shape: [usize; 5] = weight.shape[..].try_into().expect("checked");
        Conv {
            weight: Tensor::<5>::from_data(TensorData::new(reverse_spatial(weight), shape), device),
            bias: Tensor::<1>::from_data(TensorData::new(bias.data.clone(), [shape[0]]), device),
            options: ConvOptions::new([stride; 3], [padding; 3], [1; 3], groups),
        }
    }
}

/// The encoder's five convolutions on `device`: conv1, conv2, depthwise,
/// pointwise, convbin1.
pub(super) fn encoder(weights: &Weights, device: &Device) -> [Conv; 5] {
    [
        Conv::new(&weights.conv1, 2, 3, 1, device),
        Conv::new(&weights.conv2, 1, 2, 1, device),
        Conv::new(&weights.depthwise, 1, 1, 32, device),
        Conv::new(&weights.pointwise, 1, 0, 1, device),
        Conv::new(&weights.convbin1, 1, 1, 1, device),
    ]
}

/// `conv1` to `convbin1` with their relus: the coarse features, `[1, 32, n /
/// 8]`, of the max-pooled input `x` (`[1, 1, n / 2]`).
pub(super) fn encode(convs: &[Conv; 5], x: Tensor<5>) -> Tensor<5> {
    let [conv1, conv2, depthwise, pointwise, convbin1] = convs;
    let x = relu(conv1.apply(x));
    let x = max_pool2(x);
    let x = relu(conv2.apply(x));
    let x = pointwise.apply(depthwise.apply(x));
    relu(convbin1.apply(x))
}

impl Network {
    /// Load the reference's trained weights (its state-dict names) from a
    /// safetensors file.
    pub fn load(path: &Path) -> Result<Self> {
        Ok(Self::new(&Weights::read(path)?))
    }

    /// The network of `weights`, on burn's CPU backend.
    pub(super) fn new(weights: &Weights) -> Self {
        let device = Device::flex();
        let conv4_weight = &weights.conv4.0;
        let conv4_bias = &weights.conv4.1;
        Self {
            encoder: encoder(weights, &device),
            conv3: Conv3::new(&weights.conv3.0, &weights.conv3.1, &device),
            conv4: Conv4 {
                weight: reverse_spatial(conv4_weight),
                bias: [conv4_bias.data[0], conv4_bias.data[1]],
            },
            device,
        }
    }

    /// The foreground probability of a binary volume `shape` (`[z, y, x]`,
    /// row-major, 0 or 1), every length at least 6. Returns the probabilities
    /// of the first [`output_length`] voxels per axis, in the same order, and
    /// their shape.
    pub fn forward(&self, input: &[f32], shape: [usize; 3]) -> Result<(Vec<f32>, [usize; 3])> {
        if shape.iter().any(|&n| n < 6) {
            return Err(Error::invalid(format!(
                "vessel filling: the network takes at least 6 voxels per axis, got {shape:?}"
            )));
        }
        if input.len() != shape.iter().product::<usize>() {
            return Err(Error::invalid("vessel filling: input length"));
        }
        let pooled_shape = shape.map(|n| n / 2);
        let pooled = max_pool2_host(input, shape);
        let x = Tensor::<5>::from_data(
            TensorData::new(
                pooled,
                [1, 1, pooled_shape[0], pooled_shape[1], pooled_shape[2]],
            ),
            &self.device,
        );
        let x = encode(&self.encoder, x);
        let [_, channels, c0, c1, c2] = x.dims();
        let coarse = host(x)?;
        let up = upsample2_host(&coarse, channels, [c0, c1, c2]);
        let n = [2 * c0, 2 * c1, 2 * c2];
        let half = self.conv3.apply(&up, n, &self.device)?;
        let half_shape = n.map(|n| 2 * n - 1);
        let out_shape = half_shape.map(|n| 2 * n);
        debug_assert_eq!(out_shape, shape.map(output_length));
        let out = self.conv4.apply(&half, half_shape, input, shape);
        Ok((out, out_shape))
    }
}

/// The network's output length for an input of `n >= 6` voxels.
pub fn output_length(n: usize) -> usize {
    8 * ((n + 2) / 8) - 2
}

pub(super) fn host<const D: usize>(tensor: Tensor<D>) -> Result<Vec<f32>> {
    tensor
        .into_data()
        .try_into_vec::<f32>()
        .map_err(|error| Error::backend(format!("vessel filling: {error:?}")))
}

/// `conv3`'s weight (`[32, 16, 7, 7, 7]`, the reference's axis order) as the
/// polyphase matrix `[64 * 32, 16 * 8]` of [`Conv3`]: row `32 t + ci` for tap
/// `t` and input channel `ci`, column `8 c + parity`. Per axis, output `o = 2
/// m + p` collects the inputs `i` with `o = 2 i - 3 + k`: for parity `p = 0`
/// the taps `k = 5, 3, 1` at `i = m - 1, m, m + 1`, for `p = 1` the taps `k =
/// 6, 4, 2, 0` at `i = m - 1 ..= m + 2`. So a 4-tap window starting at `m - 1`
/// with kernel `k = p + 5 - 2 t` (zero where that is negative) gives either
/// parity.
pub(super) fn polyphase_matrix(weight: &Stored) -> Vec<f32> {
    let reversed = reverse_spatial(weight);
    let (c_in, c_out, k) = (32, 16, 7);
    let at = |ci: usize, co: usize, z: usize, y: usize, x: usize| {
        reversed[(((ci * c_out + co) * k + z) * k + y) * k + x]
    };
    let kernel = |p: usize, t: usize| (p + 5).checked_sub(2 * t);
    let mut matrix = vec![0.0f32; 64 * c_in * c_out * 8];
    for tap in 0..64 {
        let t = [tap / 16, (tap / 4) % 4, tap % 4];
        let matrix = &mut matrix[tap * c_in * c_out * 8..][..c_in * c_out * 8];
        for ci in 0..c_in {
            for co in 0..c_out {
                for parity in 0..8 {
                    let p = [parity >> 2, (parity >> 1) & 1, parity & 1];
                    if let (Some(kz), Some(ky), Some(kx)) =
                        (kernel(p[0], t[0]), kernel(p[1], t[1]), kernel(p[2], t[2]))
                    {
                        matrix[ci * c_out * 8 + co * 8 + parity] = at(ci, co, kz, ky, kx);
                    }
                }
            }
        }
    }
    matrix
}

impl Conv3 {
    /// `ConvTranspose3d(32, 16, 7, stride 2, padding 3)` as a polyphase
    /// convolution ([`polyphase_matrix`]).
    fn new(weight: &Stored, bias: &Stored, device: &Device) -> Self {
        Self {
            kernel: Tensor::<2>::from_data(
                TensorData::new(polyphase_matrix(weight), [64 * 32, 16 * 8]),
                device,
            ),
            bias: bias.data.clone(),
        }
    }

    /// Apply to `input` (`[32, n0, n1, n2]`), with the relu after it: `[2 n0 -
    /// 1][16][2 n1 - 1][2 n2 - 1]`, plane by plane.
    fn apply(&self, input: &[f32], n: [usize; 3], device: &Device) -> Result<Vec<f32>> {
        const C_IN: usize = 32;
        // Channels last, padded by 1 below and 2 above (a window starting at
        // m - 1 reaches m + 2), and one more zero plane so the rows past the
        // last window (which are discarded) still read inside the buffer.
        let p = n.map(|n| n + 3);
        let plane = p[1] * p[2];
        let mut padded = vec![0.0f32; (p[0] + 1) * plane * C_IN];
        let volume = n[0] * n[1] * n[2];
        padded
            .par_chunks_mut(plane * C_IN)
            .skip(1)
            .take(n[0])
            .enumerate()
            .for_each(|(z, out)| {
                for y in 0..n[1] {
                    for x in 0..n[2] {
                        let at = ((y + 1) * p[2] + x + 1) * C_IN;
                        let from = (z * n[1] + y) * n[2] + x;
                        for c in 0..C_IN {
                            out[at + c] = input[c * volume + from];
                        }
                    }
                }
            });
        let offsets: Vec<usize> = (0..64)
            .map(|tap| ((tap / 16) * p[1] + (tap / 4) % 4) * p[2] + tap % 4)
            .collect();

        let out_shape = n.map(|n| 2 * n - 1);
        let out_plane = 16 * out_shape[1] * out_shape[2];
        let mut out = vec![0.0f32; out_shape[0] * out_plane];
        // Window m (padded coordinates of its first voxel) is row
        // (m0 p1 + m1) p2 + m2; batches of whole z planes.
        // Each output row's column is the 64 taps' 32 channels, each tap a
        // contiguous run of the padded buffer at the tap's offset.
        let planes = (CONV3_ROWS / plane).max(1);
        for first in (0..n[0]).step_by(planes) {
            let last = (first + planes).min(n[0]);
            let start = first * plane;
            let len = (last - first) * plane;
            let mut columns = vec![0.0f32; len * 64 * C_IN];
            columns
                .par_chunks_mut(64 * C_IN)
                .enumerate()
                .for_each(|(row, column)| {
                    for (tap, offset) in offsets.iter().enumerate() {
                        let from = (start + row + offset) * C_IN;
                        column[tap * C_IN..][..C_IN].copy_from_slice(&padded[from..][..C_IN]);
                    }
                });
            let columns =
                Tensor::<2>::from_data(TensorData::new(columns, [len, 64 * C_IN]), device);
            let sum = host(columns.matmul(self.kernel.clone()))?;
            let o_first = 2 * first;
            let o_last = (2 * last).min(out_shape[0]);
            out[o_first * out_plane..o_last * out_plane]
                .par_chunks_mut(out_plane)
                .enumerate()
                .for_each(|(index, target)| {
                    let o0 = o_first + index;
                    let (m0, p0) = (o0 / 2 - first, o0 % 2);
                    for c in 0..16 {
                        let bias = self.bias[c];
                        for o1 in 0..out_shape[1] {
                            let (m1, p1) = (o1 / 2, o1 % 2);
                            let row = &mut target[(c * out_shape[1] + o1) * out_shape[2]..]
                                [..out_shape[2]];
                            for (o2, value) in row.iter_mut().enumerate() {
                                let (m2, p2) = (o2 / 2, o2 % 2);
                                let r = (m0 * p[1] + m1) * p[2] + m2;
                                let v = sum[r * 128 + c * 8 + p0 * 4 + p1 * 2 + p2] + bias;
                                *value = v.max(0.0);
                            }
                        }
                    }
                });
        }
        Ok(out)
    }
}

/// Linear upsampling by 2 along one axis of `n` voxels, `align_corners=False`:
/// output `2 i` is `0.25 x[i - 1] + 0.75 x[i]`, `2 i + 1` is `0.75 x[i] +
/// 0.25 x[i + 1]`, the neighbours clamped at the ends. Per output voxel, the
/// two inputs and their weights.
fn up_taps(n: usize) -> Vec<(usize, usize, f32, f32)> {
    (0..2 * n)
        .map(|o| {
            let i = o / 2;
            if o % 2 == 0 {
                (i.saturating_sub(1), i, 0.25, 0.75)
            } else {
                (i, (i + 1).min(n - 1), 0.75, 0.25)
            }
        })
        .collect()
}

/// Upsample a `[ny][nx]` plane by 2 along both axes into `[2 ny][2 nx]`.
fn upsample_plane(plane: &[f32], ny: usize, nx: usize, out: &mut [f32]) {
    let ty = up_taps(ny);
    let tx = up_taps(nx);
    let mut rows = vec![0.0f32; 2 * ny * nx];
    for (o, &(a, b, wa, wb)) in ty.iter().enumerate() {
        let (ra, rb) = (&plane[a * nx..][..nx], &plane[b * nx..][..nx]);
        for ((t, &va), &vb) in rows[o * nx..][..nx].iter_mut().zip(ra).zip(rb) {
            *t = wa * va + wb * vb;
        }
    }
    for y in 0..2 * ny {
        let row = &rows[y * nx..][..nx];
        let target = &mut out[y * 2 * nx..][..2 * nx];
        for (t, &(a, b, wa, wb)) in target.iter_mut().zip(&tx) {
            *t = wa * row[a] + wb * row[b];
        }
    }
}

/// Trilinear upsampling by 2 of `[channels, n0, n1, n2]`.
fn upsample2_host(input: &[f32], channels: usize, n: [usize; 3]) -> Vec<f32> {
    let plane = n[1] * n[2];
    let out_plane = 4 * plane;
    let tz = up_taps(n[0]);
    let mut out = vec![0.0f32; channels * 2 * n[0] * out_plane];
    out.par_chunks_mut(out_plane)
        .enumerate()
        .for_each(|(index, target)| {
            let (c, o) = (index / (2 * n[0]), index % (2 * n[0]));
            let (a, b, wa, wb) = tz[o];
            let base = c * n[0] * plane;
            let mixed: Vec<f32> = input[base + a * plane..][..plane]
                .iter()
                .zip(&input[base + b * plane..][..plane])
                .map(|(&va, &vb)| wa * va + wb * vb)
                .collect();
            upsample_plane(&mixed, n[1], n[2], target);
        });
    out
}

impl Conv4 {
    /// Upsample `half` (`[l0][16][l1][l2]`, plane-major) by 2, prepend the
    /// input (`input_shape`, at least `2 l` per axis, cropped to it) as
    /// channel 0, convolve (3^3, zero padding) and take the two-way softmax of
    /// `sigmoid(4 (c - 0.5))`, channel 0: `[2 l0][2 l1][2 l2]`. One output z
    /// plane at a time, the three upsampled planes it reads held in a ring.
    fn apply(
        &self,
        half: &[f32],
        l: [usize; 3],
        input: &[f32],
        input_shape: [usize; 3],
    ) -> Vec<f32> {
        let [d, h, w] = l.map(|n| 2 * n);
        let half_plane = 16 * l[1] * l[2];
        let plane = h * w;
        let input_plane = input_shape[1] * input_shape[2];
        let input_row = |z: usize, y: usize| &input[z * input_plane + y * input_shape[2]..][..w];
        let tz = up_taps(l[0]);
        let upsampled = |z: usize| -> Vec<f32> {
            let (a, b, wa, wb) = tz[z];
            let mut out = vec![0.0f32; 16 * plane];
            out.par_chunks_mut(plane)
                .enumerate()
                .for_each(|(c, target)| {
                    let at = |i: usize| &half[i * half_plane + c * l[1] * l[2]..][..l[1] * l[2]];
                    let mixed: Vec<f32> = at(a)
                        .iter()
                        .zip(at(b))
                        .map(|(&va, &vb)| wa * va + wb * vb)
                        .collect();
                    upsample_plane(&mixed, l[1], l[2], target);
                });
            out
        };
        let squash = |c: f32| 1.0 / (1.0 + (-4.0 * (c - 0.5)).exp());
        let mut out = vec![0.0f32; d * plane];
        // ring[k]: the upsampled plane z - 1 + k, if inside the volume.
        let mut ring: [Option<Vec<f32>>; 3] = [None, Some(upsampled(0)), None];
        for z in 0..d {
            ring.rotate_left(1);
            ring[1] = ring[1].take().or_else(|| Some(upsampled(z)));
            ring[2] = (z + 1 < d).then(|| upsampled(z + 1));
            let ring = &ring;
            out[z * plane..(z + 1) * plane]
                .par_chunks_mut(w)
                .enumerate()
                .for_each(|(y, target)| {
                    let mut acc = [vec![self.bias[0]; w], vec![self.bias[1]; w]];
                    for dz in 0..3 {
                        let Some(zz) = (z + dz).checked_sub(1).filter(|&zz| zz < d) else {
                            continue;
                        };
                        let features = ring[dz].as_deref().expect("planes inside the volume");
                        for ch in 0..17 {
                            for dy in 0..3 {
                                let Some(yy) = (y + dy).checked_sub(1).filter(|&yy| yy < h) else {
                                    continue;
                                };
                                let row = match ch {
                                    0 => input_row(zz, yy),
                                    _ => &features[(ch - 1) * plane + yy * w..][..w],
                                };
                                for dx in 0..3 {
                                    let tap = ((ch * 3 + dz) * 3 + dy) * 3 + dx;
                                    let w0 = self.weight[tap];
                                    let w1 = self.weight[17 * 27 + tap];
                                    let (a0, a1) = acc.split_at_mut(1);
                                    let (a0, a1) = (&mut a0[0], &mut a1[0]);
                                    // Output x reads row[x + dx - 1].
                                    let (out_range, in_range) = match dx {
                                        0 => (1..w, 0..w - 1),
                                        1 => (0..w, 0..w),
                                        _ => (0..w - 1, 1..w),
                                    };
                                    for ((s0, s1), &v) in a0[out_range.clone()]
                                        .iter_mut()
                                        .zip(&mut a1[out_range])
                                        .zip(&row[in_range])
                                    {
                                        *s0 += w0 * v;
                                        *s1 += w1 * v;
                                    }
                                }
                            }
                        }
                    }
                    for ((t, &c0), &c1) in target.iter_mut().zip(&acc[0]).zip(&acc[1]) {
                        let (a, b) = (squash(c0), squash(c1));
                        *t = 1.0 / (1.0 + (b - a).exp());
                    }
                });
        }
        out
    }
}

/// `MaxPool3d(2)` on the host: `[z, y, x]` to half (floor) per axis.
fn max_pool2_host(input: &[f32], shape: [usize; 3]) -> Vec<f32> {
    let [_, h, w] = shape;
    let [od, oh, ow] = shape.map(|n| n / 2);
    let mut out = vec![0.0f32; od * oh * ow];
    out.par_chunks_mut(oh * ow)
        .enumerate()
        .for_each(|(z, target)| {
            for y in 0..oh {
                for x in 0..ow {
                    let mut best = f32::NEG_INFINITY;
                    for dz in 0..2 {
                        for dy in 0..2 {
                            let row = ((2 * z + dz) * h + 2 * y + dy) * w + 2 * x;
                            best = best.max(input[row]).max(input[row + 1]);
                        }
                    }
                    target[y * ow + x] = best;
                }
            }
        });
    out
}

/// `MaxPool3d(2)`: floor, as the reference's.
pub(super) fn max_pool2(x: Tensor<5>) -> Tensor<5> {
    let mut x = x;
    for dim in 2..5 {
        let dims = x.dims();
        let half = dims[dim] / 2;
        // The pairs along `dim` become an axis of their own after it.
        let mut pairs = [0usize; 6];
        pairs[..dim].copy_from_slice(&dims[..dim]);
        pairs[dim] = half;
        pairs[dim + 1] = 2;
        pairs[dim + 2..].copy_from_slice(&dims[dim + 1..]);
        let mut back = dims;
        back[dim] = half;
        x = x
            .narrow(dim, 0, 2 * half)
            .reshape(pairs)
            .max_dim(dim + 1)
            .reshape(back);
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(values: Vec<f32>, shape: [usize; 5]) -> Tensor<5> {
        Tensor::<5>::from_data(TensorData::new(values, shape), &Device::flex())
    }

    #[test]
    fn upsampling_matches_align_corners_false() {
        let got = upsample2_host(&[1.0, 2.0, 4.0], 1, [3, 1, 1]);
        // Along z; y and x are single voxels, doubled.
        let want = [1.0, 1.25, 1.75, 2.5, 3.5, 4.0];
        for (o, w) in want.iter().enumerate() {
            for k in 0..4 {
                assert!((got[o * 4 + k] - w).abs() < 1e-6, "{got:?}");
            }
        }
        let got = upsample2_host(&[1.0, 2.0, 4.0], 1, [1, 1, 3]);
        for (o, w) in want.iter().enumerate() {
            assert!((got[o] - w).abs() < 1e-6, "{got:?}");
        }
    }

    /// `ConvTranspose3d(32, 16, 7, stride 2, padding 3)` and a relu, written
    /// out, in this crate's axis order (the kernel's spatial axes reversed);
    /// plane-major like [`Conv3::apply`].
    fn transposed_reference(x: &[f32], n: [usize; 3], weight: &Stored, bias: &Stored) -> Vec<f32> {
        let w = reverse_spatial(weight);
        let out_n = n.map(|n| 2 * n - 1);
        let at = |c: usize, o: [usize; 3]| ((o[0] * 16 + c) * out_n[1] + o[1]) * out_n[2] + o[2];
        let mut out = vec![0.0f32; 16 * out_n.iter().product::<usize>()];
        for o0 in 0..out_n[0] {
            for o1 in 0..out_n[1] {
                for o2 in 0..out_n[2] {
                    for c in 0..16 {
                        out[at(c, [o0, o1, o2])] = bias.data[c];
                    }
                }
            }
        }
        for ci in 0..32 {
            for i0 in 0..n[0] {
                for i1 in 0..n[1] {
                    for i2 in 0..n[2] {
                        let v = x[((ci * n[0] + i0) * n[1] + i1) * n[2] + i2];
                        for k0 in 0..7 {
                            for k1 in 0..7 {
                                for k2 in 0..7 {
                                    let o = [2 * i0 + k0, 2 * i1 + k1, 2 * i2 + k2];
                                    if o.iter().zip(out_n).any(|(&o, n)| o < 3 || o - 3 >= n) {
                                        continue;
                                    }
                                    let o = o.map(|o| o - 3);
                                    for co in 0..16 {
                                        let k = (((ci * 16 + co) * 7 + k0) * 7 + k1) * 7 + k2;
                                        out[at(co, o)] += v * w[k];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        out.iter().map(|v| v.max(0.0)).collect()
    }

    #[test]
    fn the_polyphase_convolution_is_the_transposed_one() {
        let mut state = 1u64;
        let mut next = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        };
        let weight = Stored {
            shape: vec![32, 16, 7, 7, 7],
            data: (0..32 * 16 * 343).map(|_| next()).collect(),
        };
        let bias = Stored {
            shape: vec![16],
            data: (0..16).map(|_| next()).collect(),
        };
        let n = [3, 4, 5];
        let x: Vec<f32> = (0..32 * 60).map(|_| next()).collect();
        let want = transposed_reference(&x, n, &weight, &bias);
        let device = Device::flex();
        let got = Conv3::new(&weight, &bias, &device)
            .apply(&x, n, &device)
            .unwrap();
        assert_eq!(got.len(), want.len());
        let max = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max < 1e-4, "max difference {max}");
        assert!(want.iter().any(|v| *v > 0.1));
    }

    #[test]
    fn max_pool_floors_odd_lengths() {
        let x = tensor((0..20).map(|v| v as f32).collect(), [1, 1, 2, 2, 5]);
        assert_eq!(host(max_pool2(x)).unwrap(), vec![16.0, 18.0]);
        let pooled = max_pool2_host(&(0..8).map(|v| v as f32).collect::<Vec<_>>(), [2, 2, 2]);
        assert_eq!(pooled, vec![7.0]);
    }
}
