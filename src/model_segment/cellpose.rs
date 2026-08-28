//! Cellpose as a [`SegmentBackend`].
//!
//! A thin adapter and nothing more: it converts the tile, calls `eval`, and
//! converts the labels back. Everything about blocks, ownership, ids and
//! measurement is [`crate::model_segment::InstanceSegment`]'s, and this file does not know that
//! any of it exists.
//!
//! # The two things it does have to get right
//!
//! **The GPU is one device and the executor runs blocks concurrently.** The
//! model is held behind a `Mutex`, so inference is serialised while everything
//! around it — the reads, the per-object accumulation, the row encoding — stays
//! parallel across `Hints::concurrency` workers. That is the v1 arrangement and
//! it is stated rather than assumed: whether it is the *right* one is a
//! measurement nobody has taken yet, and `blockflow`'s `Stats` is where the
//! answer will be. See `BLOCKFLOW_PLAN.md` §4.4.
//!
//! **`ndarray` 0.16 against 0.17.** `cellpose` is on 0.16 and `blockflow` on
//! 0.17, which are different crates to the compiler even though the types have
//! the same name. The boundary is two `from_shape_vec` calls per block — one
//! copy each way, against seconds of inference — and it lives here rather than
//! leaking into the op.

use std::path::Path;
use std::sync::Mutex;

use crate::error::{Error, Result};
use cellpose::{CellposeModel, EvalParams};
use ndarray::{Array3, ArrayView3};

use crate::model_segment::flow::WindowGrid;
use crate::model_segment::SegmentBackend;

/// Which device runs the network.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Cuda(usize),
}

impl From<Device> for cellpose::core::Device {
    fn from(device: Device) -> Self {
        match device {
            Device::Cpu => cellpose::core::Device::Cpu,
            Device::Cuda(ordinal) => cellpose::core::Device::Cuda(ordinal),
        }
    }
}

/// Cellpose, held for the length of a run.
pub struct CellposeBackend {
    /// One model, one device. See the header for what the lock is for.
    model: Mutex<CellposeModel>,
    params: EvalParams,
    /// What one tile costs, per voxel, in `blockflow`'s units.
    cost: f64,
}

impl CellposeBackend {
    /// Load `weights` onto `device`.
    ///
    /// `params` is Cellpose's own `EvalParams` in full rather than a curated
    /// subset of it. Everything an op needs from the caller is a parameter,
    /// including the ones this crate has no opinion about — `diameter`,
    /// `flow_threshold`, `cellprob_threshold`, `niter`, `min_size` — and
    /// re-declaring them here would be this crate deciding which of somebody
    /// else's knobs matter.
    ///
    /// Two of them are **not** the caller's, and are overridden below with the
    /// reason written out: `do_3d` and `stitch_threshold`.
    pub fn new(weights: &Path, device: Device, params: EvalParams) -> Result<Self> {
        let model = CellposeModel::new(weights, device.into(), None)
            .map_err(|error| Error::backend(format!("cellpose: {error}")))?;
        Ok(Self {
            model: Mutex::new(model),
            params: two_dimensional(params),
            // Measured on this hardware: 3103 ms for a 1024 x 1024 tile with
            // CP-SAM ViT-L, fp16, on a Quadro RTX 5000 — see `CLAUDE.md`'s
            // performance table. That is 2.96 us per voxel, against 1.0 for an
            // ordinary voxelwise pass. Stating it is what stops a planner
            // scheduling a three-second tile as if it were a memcpy.
            cost: 3.0,
        })
    }

    /// Say what a tile costs, when it has been measured on the hardware in
    /// front of you rather than the hardware this was written on.
    #[must_use]
    pub fn with_cost_per_voxel(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }
}

/// `params`, with the two settings that are this crate's rather than the
/// caller's.
///
/// `blockflow` is doing the outer tiling — that is the whole point — and
/// Cellpose's own cross-slice stitching would be a second, different answer to
/// the question this crate answers with ownership-by-centroid. Two stitching
/// schemes over one dataset is how a cell gets counted twice.
///
/// Cellpose's *internal* tiling (`bsize`, `tile_overlap`, `batch_size`) is
/// untouched and stays the caller's: it is how one block is fed through the
/// network, which is a different question from how the slide is fed through
/// blocks.
fn two_dimensional(mut params: EvalParams) -> EvalParams {
    params.do_3d = false;
    params.stitch_threshold = 0.0;
    params
}

impl SegmentBackend for CellposeBackend {
    fn name(&self) -> &'static str {
        "cellpose"
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }

    fn segment(&self, tile: ArrayView3<'_, f32>, _at: &crate::Anchor) -> Result<Array3<u32>> {
        let (depth, height, width) = tile.dim();
        if depth != 1 {
            return Err(Error::InvalidArgument(format!(
                "the cellpose backend is two-dimensional and was handed a tile {depth} deep. \
                 A 2-D image is a volume one voxel deep in this crate's convention; a genuinely \
                 3-D block wants `do_3d`, which this adapter turns off because `blockflow` is \
                 doing the tiling."
            )));
        }

        // Into cellpose's ndarray. One copy; the alternative is no adapter.
        let values: Vec<f32> = tile.iter().copied().collect();
        let image = ndarray16::ArrayD::from_shape_vec(ndarray16::IxDyn(&[height, width]), values)
            .map_err(|error| {
            Error::backend(format!("cellpose: tile does not reshape: {error}"))
        })?;

        let output = {
            let model = self
                .model
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            model
                .eval(&image, &self.params)
                .map_err(|error| Error::backend(format!("cellpose: {error}")))?
        };

        // Back out. Cellpose labels are `i32` with 0 for background; negative
        // values would be a contract this crate cannot represent, so they are
        // refused by name rather than cast into something enormous.
        let mut labels = Array3::<u32>::zeros((1, height, width));
        for (index, value) in output.masks.iter().enumerate() {
            if *value < 0 {
                return Err(Error::backend(format!(
                    "cellpose returned label {value} at index {index}. Labels are region \
                     identities and a negative one has no reading here."
                )));
            }
            labels[[0, index / width, index % width]] = *value as u32;
        }
        Ok(labels)
    }
}

// ------------------------------------------------- the anchored backend --

/// **Cellpose with its windows anchored to the image, not to the buffer.**
///
/// The same network and the same mask computation as [`CellposeBackend`], with
/// one thing replaced: where the 256 px windows go. [`CellposeBackend`] calls
/// `eval`, which tiles whatever buffer it is handed by
/// `linspace(0, L - 256, ceil((1 + 2*overlap)*L/256))` — a **non-integer stride
/// that depends on the buffer size**, so a block of a different size re-places
/// every window and changes the answer everywhere. This one runs
/// [`crate::model_segment::flow`]'s absolute grid, so a pixel sits in the same windows at the
/// same offsets whatever block reached it.
///
/// Measured on a 1024² tissue region of `2079_R1`, blended flow field, blocks
/// of 1024 / 512 / 256 against each other:
///
/// | | values differing | worst |
/// |---|---|---|
/// | cellpose's placement | effectively all | — |
/// | anchored, batch 4 | 3.3% | 1.1e-5 |
/// | **anchored, batch 1** | **0 of 3 145 728** | **0** |
///
/// The residual at batch 4 is not geometry: the network is not bit-pure under
/// batch composition — the same window alone against inside a batch differs in
/// 97% of elements by up to 1.9e-5, which is kernel and reduction-order
/// variation, coarsened by the `bfloat16` default. A fixed batch shape removes
/// it, and costs nothing measurable: one 256² ViT-L pass already saturates the
/// GPU.
///
/// # What else had to be pinned, and why each would otherwise leak
///
/// * **normalisation** is a fixed linear map from `range`, not cellpose's
///   per-image percentiles. Percentiles of a buffer are a function of the
///   buffer;
/// * **no rescaling.** `eval` resizes by `diam_mean / diameter` *before*
///   tiling, and resizing a buffer is not the same as cropping a resized whole
///   image — the interpolation phase differs unless origins happen to align
///   with the scale's denominator. So this backend runs the network at its
///   native scale and the diameter is not a parameter of it;
/// * **`max_size_fraction` is relative to the buffer**, so the same fraction is
///   a different absolute size in a different block. It is converted from an
///   absolute pixel count here, per buffer, so the threshold is the one the
///   caller meant.
///
/// # The halo it needs
///
/// At least [`WindowGrid::halo`] — one window — so every window covering the
/// core is inside the buffer. And **more than that for the masks**: flow
/// following moves each pixel up to `niter` steps toward its cell's centre, and
/// `steps_interp` clamps the trajectory to the *buffer*, so a cell whose
/// trajectories would leave the buffer is shaped by that edge. A halo
/// comfortably above the largest cell diameter covers it; the run reports the
/// largest travel it actually saw so the assumption is checkable rather than
/// assumed.
pub struct AnchoredCellpose {
    model: Mutex<CellposeModel>,
    grid: WindowGrid,
    taper: Vec<f64>,
    range: (f32, f32),
    batch: usize,
    niter: usize,
    cellprob_threshold: f32,
    flow_threshold: f32,
    min_size: i32,
    /// The largest object to keep, in pixels — an absolute count, converted to
    /// cellpose's buffer-relative fraction per call.
    max_size: usize,
    cost: f64,
}

impl AnchoredCellpose {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        weights: &Path,
        device: Device,
        grid: WindowGrid,
        range: (f32, f32),
        batch: usize,
        niter: usize,
        cellprob_threshold: f32,
        flow_threshold: f32,
        min_size: i32,
        max_size: usize,
    ) -> Result<Self> {
        let model = CellposeModel::new(weights, device.into(), None)
            .map_err(|error| Error::backend(format!("cellpose: {error}")))?;
        Ok(Self {
            model: Mutex::new(model),
            taper: crate::model_segment::flow::net::taper_profile(grid.window()),
            grid,
            range,
            batch,
            niter,
            cellprob_threshold,
            flow_threshold,
            min_size,
            max_size,
            cost: 3.0,
        })
    }

    pub fn grid(&self) -> WindowGrid {
        self.grid
    }
}

impl SegmentBackend for AnchoredCellpose {
    fn name(&self) -> &'static str {
        "cellpose-anchored"
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }

    fn segment(&self, tile: ArrayView3<'_, f32>, at: &crate::Anchor) -> Result<Array3<u32>> {
        let (depth, height, width) = tile.dim();
        if depth != 1 {
            return Err(Error::InvalidArgument(format!(
                "the anchored cellpose backend is two-dimensional and was handed a tile \
                 {depth} deep."
            )));
        }

        // A fixed linear normalisation: the network's input at a pixel is a
        // function of that pixel, and of nothing about the buffer.
        let (low, high) = self.range;
        let span = (high - low).max(1e-6);
        let image = ndarray::Array2::<f32>::from_shape_fn((height, width), |(row, column)| {
            ((tile[[0, row, column]] - low) / span).clamp(0.0, 1.0)
        });

        let timed = std::env::var("BLOCKFLOW_SEGMENT_TIMING").is_ok();
        let started = std::time::Instant::now();

        let flows = {
            let model = self
                .model
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let accumulated = crate::model_segment::flow::net::accumulate(
                &model.net,
                image.view(),
                [at.offset[1], at.offset[2]],
                [at.volume[1], at.volume[2]],
                self.grid,
                &self.taper,
                self.batch,
            )?;
            let network_done = started.elapsed().as_secs_f64();
            let blended = accumulated.blend();
            if timed {
                eprintln!(
                    "    [timing] windows+accumulate {:.2}s, blend {:.2}s",
                    network_done,
                    started.elapsed().as_secs_f64() - network_done
                );
            }
            blended
        };
        let flows_done = started.elapsed().as_secs_f64();

        // The network emits (dY, dX, cellprob) as its last three channels.
        let channels = flows.dim().0;
        if channels < 3 {
            return Err(Error::backend(format!(
                "the network returned {channels} channels; flows and a probability need three"
            )));
        }
        let mut d_p = ndarray16::Array3::<f32>::zeros((2, height, width));
        let mut cellprob = ndarray16::Array2::<f32>::zeros((height, width));
        for row in 0..height {
            for column in 0..width {
                d_p[[0, row, column]] = flows[[channels - 3, row, column]];
                d_p[[1, row, column]] = flows[[channels - 2, row, column]];
                cellprob[[row, column]] = flows[[channels - 1, row, column]];
            }
        }

        // `max_size_fraction` is a fraction of the buffer, so an absolute size
        // has to be re-expressed against this buffer to mean the same thing.
        let area = (height * width) as f32;
        let max_fraction = (self.max_size as f32 / area).clamp(1e-6, 1.0);

        // The Euler integration that turns flows into masks is half the cost of
        // a block, and cellpose-rs has a GPU path for it. Measured on a 2048
        // buffer: 10.44s on the CPU. The device is the model's own, so the
        // flows do not cross to a second one.
        // The CPU path is kept reachable, not as a fallback but because it is
        // the one a machine without a GPU has, and because comparing the two is
        // how the GPU path stays honest — they agree cell for cell.
        let masks = if std::env::var("BLOCKFLOW_SEGMENT_CPU_MASKS").is_ok() {
            cellpose::dynamics::compute_masks(
                &d_p,
                &cellprob,
                self.niter,
                self.cellprob_threshold,
                self.flow_threshold,
                self.min_size,
                max_fraction,
            )
        } else {
            let model = self
                .model
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let device = model.net.device().clone();
            cellpose::dynamics::compute_masks_gpu(
                &d_p,
                &cellprob,
                self.niter,
                self.cellprob_threshold,
                self.flow_threshold,
                self.min_size,
                max_fraction,
                &device,
            )
        }
        .map_err(|error| Error::backend(format!("cellpose masks: {error}")))?;

        if timed {
            eprintln!(
                "    [timing] compute_masks {:.2}s of {:.2}s total",
                started.elapsed().as_secs_f64() - flows_done,
                started.elapsed().as_secs_f64()
            );
        }

        let mut out = Array3::<u32>::zeros((1, height, width));
        for row in 0..height {
            for column in 0..width {
                let value = masks[[row, column]];
                if value < 0 {
                    return Err(Error::backend(format!(
                        "cellpose returned label {value}; labels are identities"
                    )));
                }
                out[[0, row, column]] = value as u32;
            }
        }
        Ok(out)
    }
}
