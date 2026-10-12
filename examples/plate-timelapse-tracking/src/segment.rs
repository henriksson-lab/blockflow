// SPDX-License-Identifier: MIT

//! Segmenter construction and intensity normalisation.
//!
//! A network backend is loaded **once** and shared by every frame job. Its
//! input range is fixed at load time, so per-movie and per-frame
//! normalisation are applied by [`Normalized`], a thin adapter that maps each
//! tile to `[0, 1]` with that frame's `low`/`high` before the shared backend
//! sees it. The backend itself is configured for `[0, 1]` input.

use std::path::Path;
use std::sync::Arc;

use blockflow::model_segment::stub::ThresholdBackend;
use blockflow::model_segment::SegmentBackend;
use blockflow::time_series::TimeSeriesImage;
use blockflow::zarr_env::ZarrEnvironment;
use blockflow::{Anchor, Error, Result};
use ndarray::{Array3, ArrayView3};

use crate::params::{Normalization, Segmenter};

/// Load the segmenter.
pub fn load_backend(segmenter: &Segmenter) -> Result<Arc<dyn SegmentBackend>> {
    match segmenter {
        Segmenter::Threshold { threshold } => Ok(Arc::new(ThresholdBackend::new(*threshold))),
        #[cfg(feature = "stardist")]
        Segmenter::Stardist {
            model,
            weights,
            prob_threshold,
            nms_threshold,
        } => {
            let weights = weights
                .clone()
                .unwrap_or_else(|| model.join("weights_best.h5"));
            Ok(Arc::new(
                blockflow::model_segment::stardist::StardistBackend::new(
                    model,
                    &weights,
                    *prob_threshold,
                    *nms_threshold,
                    (0.0, 1.0),
                )?,
            ))
        }
        #[cfg(not(feature = "stardist"))]
        Segmenter::Stardist { .. } => Err(Error::invalid(
            "this binary was built without StarDist; rebuild with --features stardist",
        )),
        #[cfg(feature = "cellpose")]
        Segmenter::Cellpose {
            model,
            cuda_device,
            diameter,
            flow_threshold,
            cellprob_threshold,
            min_size,
        } => {
            use blockflow::model_segment::cellpose::{CellposeBackend, Device};
            let device = cuda_device.map_or(Device::Cpu, Device::Cuda);
            let params = cellpose::EvalParams {
                diameter: *diameter,
                flow_threshold: *flow_threshold,
                cellprob_threshold: *cellprob_threshold,
                min_size: *min_size,
                // The adapter has already mapped the frame to [0, 1]; a
                // per-tile percentile normalisation here would make the
                // answer depend on the block cut.
                normalize_params: Some(cellpose::NormalizeParams {
                    lowhigh: Some(vec![0.0, 1.0]),
                    ..cellpose::NormalizeParams::default()
                }),
                ..cellpose::EvalParams::default()
            };
            Ok(Arc::new(CellposeBackend::new(model, device, params)?))
        }
        #[cfg(not(feature = "cellpose"))]
        Segmenter::Cellpose { .. } => Err(Error::invalid(
            "this binary was built without Cellpose; rebuild with --features cellpose",
        )),
    }
}

/// Maps a tile to `(v - low) / (high - low)` and hands it to `inner`.
pub struct Normalized {
    pub inner: Arc<dyn SegmentBackend>,
    pub low: f32,
    pub high: f32,
}

impl SegmentBackend for Normalized {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn segment(&self, tile: ArrayView3<'_, f32>, at: &Anchor) -> Result<Array3<u32>> {
        let scale = 1.0 / (self.high - self.low);
        let low = self.low;
        let mapped = tile.mapv(|v| (v - low) * scale);
        self.inner.segment(mapped.view(), at)
    }

    fn cost_per_voxel(&self) -> f64 {
        self.inner.cost_per_voxel()
    }
}

/// Read one frame of one channel whole, as `f64`. Plate frames are a few
/// megapixels; this is the only whole-frame read in the pipeline and it is
/// used only for percentile normalisation.
pub fn read_frame(
    image: &TimeSeriesImage,
    channel: usize,
    stored_t: usize,
    scratch: &Path,
) -> Result<Array3<f64>> {
    read_frame_at(image, 0, channel, stored_t, scratch)
}

/// [`read_frame`] at a pyramid level.
pub fn read_frame_at(
    image: &TimeSeriesImage,
    level: usize,
    channel: usize,
    stored_t: usize,
    scratch: &Path,
) -> Result<Array3<f64>> {
    let frame = image.frame(level, channel, stored_t)?;
    let env = ZarrEnvironment::attach(scratch, &[frame])?.without_cache();
    let values = env.image(0)?.widened();
    drop(env);
    let _ = std::fs::remove_dir_all(scratch);
    Ok(values)
}

/// `(low, high)` percentiles of `values`, by nearest rank on the sorted
/// sample. Non-finite values are ignored. Refuses a sample with no spread,
/// which would make the normalisation divide by zero.
pub fn percentiles(mut values: Vec<f64>, low: f64, high: f64) -> Result<(f32, f32)> {
    values.retain(|v| v.is_finite());
    if values.is_empty() {
        return Err(Error::invalid("cannot normalise: no finite pixels"));
    }
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let rank = |p: f64| -> f64 {
        let index = ((p / 100.0) * (values.len() - 1) as f64).round() as usize;
        values[index.min(values.len() - 1)]
    };
    let (lo, hi) = (rank(low), rank(high));
    if !(hi > lo) {
        return Err(Error::invalid(format!(
            "cannot normalise: percentiles {low} and {high} are both {lo}"
        )));
    }
    Ok((lo as f32, hi as f32))
}

/// A deterministic sample of every frame of `channel`: every 7th pixel in
/// raster order. Used for movie-wide percentiles.
pub fn movie_sample(image: &TimeSeriesImage, channel: usize, scratch: &Path) -> Result<Vec<f64>> {
    let mut sample = Vec::new();
    for t in 0..image.n_frames() {
        let frame = read_frame(image, channel, t, &scratch.join(format!("frame-{t}")))?;
        sample.extend(frame.iter().step_by(7).copied());
    }
    Ok(sample)
}

/// The `(low, high)` a frame is segmented with, or `None` for raw values.
pub fn frame_range(
    normalization: Normalization,
    movie: Option<(f32, f32)>,
    image: &TimeSeriesImage,
    channel: usize,
    stored_t: usize,
    scratch: &Path,
) -> Result<Option<(f32, f32)>> {
    Ok(match normalization {
        Normalization::None => None,
        Normalization::Fixed { low, high } => Some((low, high)),
        Normalization::MoviePercentile { .. } => Some(movie.ok_or_else(|| {
            Error::invalid("movie-wide normalisation was not computed before the frame stage")
        })?),
        Normalization::FramePercentile { low, high } => {
            let frame = read_frame(image, channel, stored_t, scratch)?;
            Some(percentiles(frame.into_raw_vec_and_offset().0, low, high)?)
        }
    })
}

/// Per-pair stage-drift shifts `[dy, dx]` in level-0 pixels between
/// consecutive stored frames: phase correlation of `channel` at `level` (the
/// middle `z` slice for 3-D frames), scaled to level 0. The series stage
/// refines each with the detections and accumulates them.
pub fn estimate_drift(
    image: &TimeSeriesImage,
    channel: usize,
    level: usize,
    scratch: &Path,
) -> Result<Vec<[f64; 2]>> {
    use blockflow::track::drift::phase_correlation;
    let full = image.frame_shape(0)?;
    let coarse = image.frame_shape(level)?;
    let scale = [
        full[1] as f64 / coarse[1] as f64,
        full[2] as f64 / coarse[2] as f64,
    ];
    let slice = |t: usize| -> Result<ndarray::Array2<f64>> {
        let frame = read_frame_at(
            image,
            level,
            channel,
            t,
            &scratch.join(format!("drift-{t}")),
        )?;
        let z = frame.dim().0 / 2;
        Ok(frame.index_axis(ndarray::Axis(0), z).to_owned())
    };
    let mut shifts = Vec::new();
    let mut previous = slice(0)?;
    for t in 1..image.n_frames() {
        let current = slice(t)?;
        let [dy, dx] = phase_correlation(previous.view(), current.view())?;
        shifts.push([dy * scale[0], dx * scale[1]]);
        previous = current;
    }
    Ok(shifts)
}

/// The finest level whose larger frame side is at most 1536 pixels.
pub fn default_drift_level(image: &TimeSeriesImage) -> Result<usize> {
    for level in 0..image.levels().len() {
        let [_, y, x] = image.frame_shape(level)?;
        if y.max(x) <= 1536 {
            return Ok(level);
        }
    }
    Ok(image.levels().len() - 1)
}
