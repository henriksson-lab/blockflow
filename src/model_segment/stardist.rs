//! StarDist as a [`SegmentBackend`].
//!
//! The second backend, and the reason [`SegmentBackend`] exists: swapping this
//! for [`crate::model_segment::cellpose`] is a one-word change at the call site, and
//! `blockflow::agreement::compare_labels` will then score one against the other
//! on the same tile — matched, split, merged, missed, spurious — rather than
//! against a single number that hides which.
//!
//! # Inference backend
//!
//! `stardist-rs` offers both Burn and Candle. This adapter uses Candle CUDA
//! when `stardist-candle-cuda` is enabled, because upstream's refreshed
//! end-to-end benchmarks show that is the path that reaches speed parity and
//! then beats the Python reference. Without that feature it keeps the Burn CPU
//! path, which is a useful fallback but is not the fast path for large runs.
//!
//! # Burn fallback
//!
//! `stardist-rs`'s own benchmarks use `burn::backend::Flex`, and copying that
//! was a mistake worth recording: **`Flex` is burn's pure-Rust CPU backend** —
//! its own documentation says so, "Flex: Pure-Rust CPU backend (std, no_std,
//! WebAssembly)" — and not, as this file previously claimed, a thing that picks
//! an accelerator at run time. StarDist was therefore running on the CPU
//! however much GPU was present, which is what a 1% utilisation measurement
//! eventually said out loud.
//!
//! If someone explicitly enables `stardist-cuda` without `stardist-candle-cuda`,
//! the Burn device is CUDA. Otherwise the Burn fallback is Flex, which remains
//! the honest choice for a machine with no GPU.
//!
//! # What this file has to do that the cellpose one does not
//!
//! StarDist's `predict_instances` does not own the network — it takes a closure
//! that runs one forward pass and hands back `prob` and `dist` in *its* layout.
//! So the layout conversion is here, and it is the part to read twice: the
//! network emits `NCHW` and the predictor wants `YXC`, on a grid that is the
//! image downsampled by `config.grid`. Both conversions are lifted from
//! `stardist-rs`'s own `bench_burn_real_data` example rather than derived, so
//! that a disagreement with upstream is a disagreement about one copied block
//! and not about an interpretation.

use std::path::Path;
use std::sync::Mutex;

use crate::error::{Error, Result};
use ndarray::{Array3, ArrayView3};
use stardist_rs::{Config2D, StarDist2D, StarDistDirectPrediction, StarDistPredictError};

use crate::model_segment::SegmentBackend;

#[cfg(feature = "stardist-candle-cuda")]
use candle_core::{Device as CandleDevice, Tensor as CandleTensor};
#[cfg(feature = "stardist-candle-cuda")]
use stardist_rs::model::candle::StarDist2D as Network;

#[cfg(not(feature = "stardist-candle-cuda"))]
use burn::tensor::{Tensor, TensorData};
#[cfg(not(feature = "stardist-candle-cuda"))]
use stardist_rs::model::burn::StarDist2D as Network;

#[cfg(all(feature = "stardist-cuda", not(feature = "stardist-candle-cuda")))]
fn network_device() -> burn::prelude::Device {
    burn::prelude::Device::cuda(burn::tensor::DeviceIndex::Default)
}
#[cfg(all(not(feature = "stardist-cuda"), not(feature = "stardist-candle-cuda")))]
fn network_device() -> burn::prelude::Device {
    burn::prelude::Device::flex()
}

/// StarDist, held for the length of a run.
pub struct StardistBackend {
    /// The predictor — thresholds, NMS, the star-convex polygon rasteriser —
    /// and the network, together behind one lock.
    ///
    /// One lock rather than two because they are used together and never apart:
    /// `predict_instances` calls the network through a closure, so a second
    /// lock would only be a second thing to take in the same order.
    inner: Mutex<Inner>,
    config: Config2D,
    prob_threshold: Option<f32>,
    nms_threshold: Option<f32>,
    /// Fixed intensity bounds for the network's input.
    ///
    /// **StarDist does not normalise for you.** Cellpose's `eval` does it
    /// internally, so the cellpose adapter can hand over raw values; StarDist's
    /// `predict_instances` takes whatever it is given and its training assumed
    /// roughly `[0, 1]`. Handing it raw 8-bit values under-detects massively —
    /// measured at 34 objects against cellpose's ~80 on the same tile.
    ///
    /// Fixed bounds rather than the percentiles StarDist's own examples use,
    /// for the same reason the anchored cellpose backend uses them: percentiles
    /// of a buffer are a function of the buffer, and this pipeline hands
    /// different buffers to different blocks.
    range: (f32, f32),
    cost: f64,
}

struct Inner {
    predictor: StarDist2D,
    network: Network,
    #[cfg(feature = "stardist-candle-cuda")]
    device: CandleDevice,
}

impl StardistBackend {
    /// Load a StarDist 2-D model from a model directory — the layout
    /// `stardist-rs` calls a *model dir*: a `config.json`, a `thresholds.json`
    /// and Keras `.h5` weights.
    ///
    /// `weights` is the `.h5` file, given separately because a model directory
    /// may hold several — `weights_best.h5`, `weights_last.h5` — and which one
    /// to run is the caller's decision, not a convention this crate should bake
    /// in.
    ///
    /// The thresholds default to the model's own, which is what the file is
    /// for; pass `Some` to override either.
    pub fn new(
        model_dir: &Path,
        weights: &Path,
        prob_threshold: Option<f32>,
        nms_threshold: Option<f32>,
        range: (f32, f32),
    ) -> Result<Self> {
        let predictor = StarDist2D::from_model_dir(model_dir)
            .map_err(|error| Error::backend(format!("stardist: {error:?}")))?;
        let config = predictor.config.clone();

        #[cfg(feature = "stardist-candle-cuda")]
        let (network, device) = {
            let device = CandleDevice::new_cuda(0)
                .map_err(|error| Error::backend(format!("stardist cuda: {error:?}")))?;
            let keras = stardist_rs::weights::load_keras_hdf5_weights(weights)
                .map_err(|error| Error::backend(format!("stardist weights: {error:?}")))?;
            let network = Network::init(config.clone(), &device)
                .load_keras_weights(&keras, &device)
                .map_err(|error| Error::backend(format!("stardist weights: {error:?}")))?;
            (network, device)
        };

        #[cfg(not(feature = "stardist-candle-cuda"))]
        let network = {
            let device = network_device();
            let keras = stardist_rs::weights::load_keras_hdf5_weights(weights)
                .map_err(|error| Error::backend(format!("stardist weights: {error:?}")))?;
            Network::init(config.clone(), &device)
                .load_keras_weights(&keras, &device)
                .map_err(|error| Error::backend(format!("stardist weights: {error:?}")))?
        };

        Ok(Self {
            inner: Mutex::new(Inner {
                predictor,
                network,
                #[cfg(feature = "stardist-candle-cuda")]
                device,
            }),
            config,
            prob_threshold,
            nms_threshold,
            range,
            // Not measured. Unlike the cellpose figure, which comes off this
            // hardware, nothing here has been timed yet — so this is a
            // placeholder that says "a network, not a memcpy" and is meant to be
            // replaced by `with_cost_per_voxel` once there is a number.
            cost: 3.0,
        })
    }

    /// Say what a tile costs per voxel, once it has been measured.
    #[must_use]
    pub fn with_cost_per_voxel(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }
}

impl SegmentBackend for StardistBackend {
    fn name(&self) -> &'static str {
        "stardist"
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }

    fn segment(&self, tile: ArrayView3<'_, f32>, _at: &crate::Anchor) -> Result<Array3<u32>> {
        let (depth, height, width) = tile.dim();
        if depth != 1 {
            return Err(Error::InvalidArgument(format!(
                "the stardist backend here is two-dimensional and was handed a tile {depth} \
                 deep. A 2-D image is a volume one voxel deep in this crate's convention; \
                 `stardist-rs` has a 3-D model and it wants a different adapter, not a deeper \
                 tile through this one."
            )));
        }

        // Into the units the network was trained on. A fixed linear map, so a
        // pixel's input is a function of that pixel and not of its block.
        let (low, high) = self.range;
        let span = (high - low).max(1e-6);
        let values: Vec<f32> = tile
            .iter()
            .map(|value| ((value - low) / span).clamp(0.0, 1.0))
            .collect();
        let config = self.config.clone();

        let instances = {
            let inner = self
                .inner
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let predictor = &inner.predictor;

            predictor
                .predict_instances(
                    &values,
                    &[height, width],
                    Some("YX"),
                    // Sparse. The dense path materialises `prob` and `dist` over
                    // the whole tile; the sparse one keeps only the candidates
                    // above threshold, which is what a tile mostly is not.
                    true,
                    self.prob_threshold,
                    self.nms_threshold,
                    None,
                    // `n_tiles`: none. `blockflow` is doing the tiling, and a
                    // second tiling inside it would be a second set of seams
                    // with a different owner rule — see the cellpose adapter's
                    // note about `stitch_threshold`, which is the same argument.
                    None,
                    // The label image is what this backend is for.
                    true,
                    None,
                    false,
                    // `b`: the border objects are **not** excluded, and this is
                    // the load-bearing argument of the call. StarDist's `b`
                    // drops objects near the tile edge, which is exactly what
                    // this crate's ownership rule is for — and doing both would
                    // delete the objects in every halo, which is to say the
                    // objects at every seam of the slide.
                    0,
                    true,
                    true,
                    |x, x_shape, axes| forward_inner(&inner, &config, x, x_shape, axes),
                )
                .map_err(|error| Error::backend(format!("stardist: {error:?}")))?
                .instances
        };

        let Some(labels) = instances.labels else {
            return Err(Error::backend(
                "stardist returned no label image although one was asked for".to_string(),
            ));
        };

        let mut out = Array3::<u32>::zeros((1, height, width));
        for y in 0..height {
            for x in 0..width {
                out[[0, y, x]] = labels[[y, x]];
            }
        }
        Ok(out)
    }
}

fn forward_inner(
    inner: &Inner,
    config: &Config2D,
    x: &[f32],
    x_shape: &[usize],
    axes: &str,
) -> std::result::Result<StarDistDirectPrediction, StarDistPredictError> {
    #[cfg(feature = "stardist-candle-cuda")]
    {
        return forward(&inner.network, config, &inner.device, x, x_shape, axes);
    }
    #[cfg(not(feature = "stardist-candle-cuda"))]
    {
        return forward(&inner.network, config, x, x_shape, axes);
    }
}

/// One forward pass, with the layout conversion StarDist's predictor expects.
///
/// The network emits `prob` and `dist` as `NCHW` on the grid `config.grid`
/// downsamples the image onto; the predictor reads them as `YXC`. Both
/// conversions are `stardist-rs`'s own, copied from its benchmark
/// example so that a disagreement with upstream is a disagreement about a
/// copied block rather than about an interpretation of a layout.
#[cfg(feature = "stardist-candle-cuda")]
fn forward(
    network: &Network,
    config: &Config2D,
    device: &CandleDevice,
    x: &[f32],
    x_shape: &[usize],
    axes: &str,
) -> std::result::Result<StarDistDirectPrediction, StarDistPredictError> {
    if axes != "YXC" || x_shape.len() != 3 || x_shape[2] != 1 {
        return Err(StarDistPredictError::OutputShapeMismatch);
    }
    let height = x_shape[0];
    let width = x_shape[1];

    let mut nchw = vec![0.0f32; height * width];
    for y in 0..height {
        for column in 0..width {
            nchw[y * width + column] = x[(y * width + column) * x_shape[2]];
        }
    }

    let input = CandleTensor::from_vec(nchw, (1, 1, height, width), device)
        .map_err(|_| StarDistPredictError::OutputShapeMismatch)?;
    let outputs = network
        .forward(&input)
        .map_err(|_| StarDistPredictError::OutputShapeMismatch)?;
    let prob_nchw =
        tensor_to_vec(outputs.prob).map_err(|_| StarDistPredictError::OutputShapeMismatch)?;
    let dist_nchw =
        tensor_to_vec(outputs.dist).map_err(|_| StarDistPredictError::OutputShapeMismatch)?;

    let prob_h = height / config.grid[0];
    let prob_w = width / config.grid[1];
    Ok(StarDistDirectPrediction {
        prob: prob_nchw[..prob_h * prob_w].to_vec(),
        prob_shape: vec![prob_h, prob_w, 1],
        dist: dist_to_yxc(&dist_nchw, config.n_rays, prob_h, prob_w),
        dist_shape: vec![prob_h, prob_w, config.n_rays],
        prob_class: None,
        prob_class_shape: None,
    })
}

#[cfg(feature = "stardist-candle-cuda")]
fn tensor_to_vec(tensor: CandleTensor) -> candle_core::Result<Vec<f32>> {
    tensor
        .to_device(&CandleDevice::Cpu)?
        .flatten_all()?
        .to_vec1()
}

#[cfg(not(feature = "stardist-candle-cuda"))]
fn forward(
    network: &Network,
    config: &Config2D,
    x: &[f32],
    x_shape: &[usize],
    axes: &str,
) -> std::result::Result<StarDistDirectPrediction, StarDistPredictError> {
    if axes != "YXC" || x_shape.len() != 3 || x_shape[2] != 1 {
        return Err(StarDistPredictError::OutputShapeMismatch);
    }
    let height = x_shape[0];
    let width = x_shape[1];

    let mut nchw = vec![0.0f32; height * width];
    for y in 0..height {
        for column in 0..width {
            nchw[y * width + column] = x[(y * width + column) * x_shape[2]];
        }
    }

    let device = network_device();
    let input = Tensor::<4>::from_data(TensorData::new(nchw, [1, 1, height, width]), &device);
    let outputs = network.forward(input);
    let prob_data = outputs.prob.into_data();
    let dist_data = outputs.dist.into_data();
    let prob_nchw = prob_data
        .as_slice::<f32>()
        .map_err(|_| StarDistPredictError::OutputShapeMismatch)?;
    let dist_nchw = dist_data
        .as_slice::<f32>()
        .map_err(|_| StarDistPredictError::OutputShapeMismatch)?;

    let prob_h = height / config.grid[0];
    let prob_w = width / config.grid[1];
    Ok(StarDistDirectPrediction {
        prob: prob_nchw[..prob_h * prob_w].to_vec(),
        prob_shape: vec![prob_h, prob_w, 1],
        dist: dist_to_yxc(dist_nchw, config.n_rays, prob_h, prob_w),
        dist_shape: vec![prob_h, prob_w, config.n_rays],
        prob_class: None,
        prob_class_shape: None,
    })
}

/// `dist` from `[ray][y][x]` to `[y][x][ray]`.
fn dist_to_yxc(dist: &[f32], rays: usize, height: usize, width: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; height * width * rays];
    for ray in 0..rays {
        for y in 0..height {
            for x in 0..width {
                out[(y * width + x) * rays + ray] = dist[(ray * height + y) * width + x];
            }
        }
    }
    out
}
