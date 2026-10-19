// SPDX-License-Identifier: MIT
// Original work for this crate.

//! [`DeepFillOp`]: the network over one block's read extent.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use blockflow::op::{Anchor, BlockOp, Slicing};
use blockflow::voxels::Voxels;
use blockflow::{Dtype, Error, Result};
use rayon::prelude::*;

use super::network::Network;
use super::{DeepFillOutput, DeepFillParams};

/// Measured: wall-clock nanoseconds per read-extent voxel at resample 1, a
/// 128 x 512 x 404 block on 40 cores (7.2-7.7 s; the op runs one block at a
/// time on every core). Resample `r` runs the network on `1 / r^3` of the
/// voxels; the host-side pooling and upsampling are not counted.
const NS_PER_VOXEL: f64 = 280.0;

/// A loaded network and the thread pool it runs in.
pub struct DeepFillModel {
    network: Mutex<Network>,
    pool: rayon::ThreadPool,
    /// The highest probability the network gives anywhere in an all-zero
    /// volume (its biases alone; measured at load over the small shapes where
    /// the edges dominate).
    zero_probability: f32,
}

impl DeepFillModel {
    /// Load the weights at `path`.
    pub fn load(path: &Path) -> Result<Self> {
        let network = Network::load(path)?;
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("deep-fill-{i}"))
            .build()
            .map_err(|error| Error::backend(format!("deep fill: thread pool: {error}")))?;
        // The edges dominate in small volumes, where every length modulo the
        // network's stride shows up; past 24 voxels the maximum no longer
        // moves (0.4536 with the trained weights, 0.4992 at 8 voxels).
        let mut zero_probability = 0.0f32;
        for n in 6..=24 {
            let zeros = vec![0.0f32; n * n * n];
            let (p, _) = pool.install(|| network.forward(&zeros, [n; 3]))?;
            zero_probability = p.into_iter().fold(zero_probability, f32::max);
        }
        Ok(Self {
            network: Mutex::new(network),
            pool,
            zero_probability,
        })
    }

    /// One model per weights file for the whole process: every channel and
    /// every run that names the same file shares it.
    pub fn shared(path: &Path) -> Result<Arc<Self>> {
        static MODELS: OnceLock<Mutex<BTreeMap<PathBuf, Arc<DeepFillModel>>>> = OnceLock::new();
        let key = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        let mut models = MODELS
            .get_or_init(Default::default)
            .lock()
            .map_err(|_| Error::backend("deep fill: a model load panicked"))?;
        if let Some(model) = models.get(&key) {
            return Ok(model.clone());
        }
        let model = Arc::new(Self::load(path)?);
        models.insert(key, model.clone());
        Ok(model)
    }

    pub fn zero_probability(&self) -> f32 {
        self.zero_probability
    }

    /// The probability of `input` (`[z, y, x]`, lengths at least 6) and its
    /// shape ([`Network::forward`]), one block at a time across the process.
    ///
    /// The forward pass runs on a plain thread inside this model's own pool.
    /// The caller is an executor worker, itself a rayon thread: had it entered
    /// the pool (or held the lock while waiting in a join), rayon could hand
    /// it another block's task meanwhile, and that task would wait for the
    /// lock its own thread holds. A plain thread's join blocks without
    /// stealing.
    pub fn probability(&self, input: &[f32], shape: [usize; 3]) -> Result<(Vec<f32>, [usize; 3])> {
        let network = self
            .network
            .lock()
            .map_err(|_| Error::backend("deep fill: a forward pass panicked"))?;
        std::thread::scope(|scope| {
            scope
                .spawn(|| self.pool.install(|| network.forward(input, shape)))
                .join()
                .map_err(|_| Error::backend("deep fill: the forward pass panicked"))?
        })
    }
}

/// The deep filling of one block: see the module documentation of
/// [`super`].
pub struct DeepFillOp {
    model: Arc<DeepFillModel>,
    params: DeepFillParams,
    output: DeepFillOutput,
}

impl DeepFillOp {
    pub fn new(model: Arc<DeepFillModel>, params: DeepFillParams, output: DeepFillOutput) -> Self {
        Self {
            model,
            params,
            output,
        }
    }
}

/// The probability over `mask[start..]` per axis (`mask` a row-major `[z, y,
/// x]` buffer of `shape`), max-pooled by `r` and upsampled back: the
/// probabilities and their shape, which starts at `start` and is at most
/// `shape - start` per axis. The network's output is shorter than its input
/// by up to `8 r - 1` voxels per axis (see [`super::network`]). An axis
/// shorter than the network's smallest input is padded with zeros.
fn region_probability(
    model: &DeepFillModel,
    mask: &[bool],
    shape: [usize; 3],
    start: [usize; 3],
    r: usize,
) -> Result<(Vec<f32>, [usize; 3])> {
    let region = [0, 1, 2].map(|a| shape[a] - start[a]);
    let net = region.map(|n| (n / r).max(6));
    // Max pooling by r: every voxel of the r^3 cell sets its cell.
    let pooled = region.map(|n| n / r);
    let mut input = vec![0.0f32; net.iter().product()];
    input
        .par_chunks_mut(net[1] * net[2])
        .take(pooled[0])
        .enumerate()
        .for_each(|(zp, plane)| {
            for z in zp * r..(zp + 1) * r {
                for y in 0..pooled[1] * r {
                    let from = ((start[0] + z) * shape[1] + start[1] + y) * shape[2] + start[2];
                    let row = &mask[from..][..pooled[2] * r];
                    let target = &mut plane[(y / r) * net[2]..][..pooled[2]];
                    for (x, &set) in row.iter().enumerate() {
                        if set {
                            target[x / r] = 1.0;
                        }
                    }
                }
            }
        });
    let (p, out) = model.probability(&input, net)?;
    Ok(if r == 1 {
        (p, out)
    } else {
        upsample(&p, out, r)
    })
}

/// Trilinear upsampling of a row-major `[z, y, x]` volume by an integer
/// factor, `align_corners=False` (`torch.nn.Upsample(mode="trilinear")`),
/// axis by axis.
fn upsample(p: &[f32], shape: [usize; 3], r: usize) -> (Vec<f32>, [usize; 3]) {
    let taps = |n: usize| -> Vec<(usize, usize, f32)> {
        (0..n * r)
            .map(|o| {
                let source = ((o as f64 + 0.5) / r as f64 - 0.5).max(0.0);
                let i0 = (source.floor() as usize).min(n - 1);
                let i1 = (i0 + 1).min(n - 1);
                (i0, i1, (source - i0 as f64) as f32)
            })
            .collect()
    };
    let [d, h, w] = shape;
    // x: every row of `w` to `r w`.
    let tx = taps(w);
    let mut a = vec![0.0f32; d * h * w * r];
    a.par_chunks_mut(w * r)
        .zip(p.par_chunks(w))
        .for_each(|(out, row)| {
            for (o, &(i0, i1, t)) in out.iter_mut().zip(&tx) {
                *o = (1.0 - t) * row[i0] + t * row[i1];
            }
        });
    // y: every plane of `h` rows to `r h`.
    let row = w * r;
    let ty = taps(h);
    let mut b = vec![0.0f32; d * h * r * row];
    b.par_chunks_mut(h * r * row)
        .zip(a.par_chunks(h * row))
        .for_each(|(out, plane)| {
            for (o, &(i0, i1, t)) in out.chunks_mut(row).zip(&ty) {
                let (r0, r1) = (&plane[i0 * row..][..row], &plane[i1 * row..][..row]);
                for ((v, &v0), &v1) in o.iter_mut().zip(r0).zip(r1) {
                    *v = (1.0 - t) * v0 + t * v1;
                }
            }
        });
    // z: every plane to `r d` planes.
    let plane = h * r * row;
    let tz = taps(d);
    let mut c = vec![0.0f32; d * r * plane];
    c.par_chunks_mut(plane)
        .zip(&tz)
        .for_each(|(out, &(i0, i1, t))| {
            let (p0, p1) = (&b[i0 * plane..][..plane], &b[i1 * plane..][..plane]);
            for ((v, &v0), &v1) in out.iter_mut().zip(p0).zip(p1) {
                *v = (1.0 - t) * v0 + t * v1;
            }
        });
    (c, shape.map(|n| n * r))
}

/// Row `(z, y)` of a buffer in `p`, which starts at `start` and has
/// `p_shape`; `None` where `p` does not reach.
fn probability_row(
    p: &[f32],
    p_shape: [usize; 3],
    start: [usize; 3],
    z: usize,
    y: usize,
) -> Option<&[f32]> {
    let (z, y) = (z.checked_sub(start[0])?, y.checked_sub(start[1])?);
    (z < p_shape[0] && y < p_shape[1])
        .then(|| &p[(z * p_shape[1] + y) * p_shape[2]..][..p_shape[2]])
}

impl BlockOp for DeepFillOp {
    fn name(&self) -> &'static str {
        "deep fill"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        self.params.reach()
    }

    fn slicing(&self) -> Slicing {
        Slicing::Whole(
            "the network's three stride-2 stages tie every voxel to the block's aligned start, \
             and one block is one forward pass",
        )
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        dtype == Dtype::Bool
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        match self.output {
            DeepFillOutput::Filled => Dtype::Bool,
            DeepFillOutput::Probability => Dtype::F32,
        }
    }

    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        match self.output {
            // A set voxel stays set; an empty extent stays empty when even the
            // network's edge response to nothing is below the threshold.
            DeepFillOutput::Filled if value != 0.0 => Some(1.0),
            DeepFillOutput::Filled
                if (self.model.zero_probability as f64) < self.params.threshold =>
            {
                Some(0.0)
            }
            _ => None,
        }
    }

    fn cost_per_voxel(&self) -> f64 {
        NS_PER_VOXEL / (self.params.resample.pow(3) as f64)
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, at: &Anchor) -> Result<()> {
        let mask = input.view::<bool>()?;
        let alignment = self.params.alignment();
        // The first multiple of the alignment inside the extent: where a pass
        // over the whole volume would have the network's strides fall.
        let start = [0, 1, 2].map(|a| {
            let offset = at.offset[a];
            offset.next_multiple_of(alignment) - offset
        });
        let shape = input.shape();
        if (0..3).any(|a| start[a] >= shape[a]) {
            return Err(Error::invalid(format!(
                "deep fill: a read extent {shape:?} at {:?} holds no aligned start",
                at.offset
            )));
        }
        let owned;
        let mask = match mask.as_slice() {
            Some(mask) => mask,
            None => {
                owned = mask.as_standard_layout().to_owned();
                owned.as_slice().expect("standard layout")
            }
        };
        let (p, p_shape) =
            region_probability(&self.model, mask, shape, start, self.params.resample)?;
        // Voxel (z, y, x) of the buffer is p's (z, y, x) - start, where p
        // reaches; nothing before `start` is read back (it is halo).
        let p_at = |z: usize, y: usize| probability_row(&p, p_shape, start, z, y);
        let plane = shape[1] * shape[2];
        let threshold = self.params.threshold as f32;
        let x0 = start[2];
        let covered = (shape[2] - x0).min(p_shape[2]);
        match self.output {
            DeepFillOutput::Filled => {
                let mut target = out.view_mut::<bool>()?;
                let target = target.as_slice_mut().ok_or_else(|| {
                    Error::backend("deep fill: an output block not in standard layout")
                })?;
                target
                    .par_chunks_mut(plane)
                    .zip(mask.par_chunks(plane))
                    .enumerate()
                    .for_each(|(z, (out, mask))| {
                        out.copy_from_slice(mask);
                        for y in 0..shape[1] {
                            if let Some(p) = p_at(z, y) {
                                let row = &mut out[y * shape[2] + x0..][..covered];
                                for (t, &q) in row.iter_mut().zip(p) {
                                    *t |= q >= threshold;
                                }
                            }
                        }
                    });
            }
            DeepFillOutput::Probability => {
                let mut target = out.view_mut::<f32>()?;
                let target = target.as_slice_mut().ok_or_else(|| {
                    Error::backend("deep fill: an output block not in standard layout")
                })?;
                target
                    .par_chunks_mut(plane)
                    .enumerate()
                    .for_each(|(z, out)| {
                        out.fill(0.0);
                        for y in 0..shape[1] {
                            if let Some(p) = p_at(z, y) {
                                out[y * shape[2] + x0..][..covered].copy_from_slice(&p[..covered]);
                            }
                        }
                    });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upsampling_matches_align_corners_false() {
        let want = [1.0, 1.25, 1.75, 2.5, 3.5, 4.0];
        for axis in 0..3 {
            let mut shape = [1, 1, 1];
            shape[axis] = 3;
            let (got, out) = upsample(&[1.0f32, 2.0, 4.0], shape, 2);
            assert_eq!(out, shape.map(|n| 2 * n));
            for z in 0..out[0] {
                for y in 0..out[1] {
                    for x in 0..out[2] {
                        let g = got[(z * out[1] + y) * out[2] + x];
                        let w = want[[z, y, x][axis]];
                        assert!((g - w).abs() < 1e-6, "axis {axis}: {got:?}");
                    }
                }
            }
        }
    }
}
