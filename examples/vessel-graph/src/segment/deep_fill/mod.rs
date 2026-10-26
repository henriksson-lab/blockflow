// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Deep vessel filling: a trained 3-D CNN that fills the hollow lumens of
//! large vessels in a binary mask, the reference pipeline's `deep_fill` step
//! (on by default in its 3.1 vasculature configuration, after each channel's
//! binarise, smooth and fill).
//!
//! **The answer** is the mask OR (network probability >= threshold), as the
//! reference's `fill_vessels` takes `max(cnn, mask) >= threshold`. The
//! network can be run on the mask max-pooled by `resample` and its
//! probability upsampled back (trilinear): the reference's configuration
//! resamples the all-vessel channel by 1 and the large-vessel channels by 2.
//!
//! **Planned per block.** [`DeepFillOp`] is a [`BlockOp`] with a halo of
//! `64 r` voxels per side, `r` the resampling. The network's output at a
//! voxel depends on the input within 55 voxels of it (measured with the
//! trained weights by perturbing single voxels: -55..55, and -112..112 at
//! resample 2), and its three stride-2 stages make the answer depend on a
//! voxel's position modulo `8 r`; each block starts the network at the first
//! multiple of `8 r` inside its read extent, so up to `8 r - 1` halo voxels go
//! unused, and `55 + 7 <= 64` (`112 + 15 <= 128`). Every block then computes
//! what one pass over the whole volume would, whatever the grid. The
//! reference instead processes overlapping blocks of up to 500 (1000) voxels
//! with an overlap of 50 (100), less than the network's reach, and ORs the
//! blocks' answers where they overlap.
//!
//! **At the volume's far faces** the network's output is shorter than its
//! input by up to `8 r - 1` voxels per axis (its pooling floors, as
//! the reference's does); those voxels get no network answer and keep the
//! mask. The reference leaves them at zero in its sink, mask included.
//!
//! **Dropout.** The reference never switches its network to evaluation mode,
//! so its three dropout layers (p 0.15) stay active at inference and its filled
//! mask changes from run to run. This port is the network in evaluation mode,
//! deterministic. The README gives the measured difference.
//!
//! **One block at a time.** The network's intermediates are several times the
//! block (the column buffer and the 16-channel half-resolution features are
//! the largest), so blocks queue for the network and each one runs on every
//! core, in a thread pool of its own.
//!
//! **On a GPU** (the `deep-fill-cuda` or `deep-fill-wgpu` feature, chosen at
//! run time by [`DeepFillDevice`]; the `gpu` module has the network) blocks
//! queue for one thread that drives the device, so one block is on it at a
//! time. Its
//! memory is its own cap: `blockflow::gpu`'s device budget (three quarters
//! of the device, `BLOCKFLOW_GPU_MEMORY`), which a block's forward pass
//! leases while it runs and which the deep-fill block is halved to fit
//! (`DeepFillModel::fit_block`). Host memory is the planner's as on the
//! CPU: a block's buffers, a fraction of the CPU network's.
//!
//! The network itself is behind the `deep-fill` cargo feature (it needs burn);
//! without it [`DeepFillParams`] and the CLI options exist but running one is
//! refused. The weights are not part of this crate: they are the reference's
//! `cnn_binary_vessel_filling.pth`, converted to safetensors by
//! `reference-clearmap/convert_weights.py`, and read from a path given at run
//! time (see the README for why).
//!
//! [`BlockOp`]: blockflow::op::BlockOp

use std::path::PathBuf;

use blockflow::{Error, Result};

#[cfg(feature = "deep-fill-gpu")]
pub mod gpu;
#[cfg(feature = "deep-fill")]
pub mod network;
#[cfg(feature = "deep-fill")]
mod op;

#[cfg(feature = "deep-fill-gpu")]
pub use gpu::GpuNetwork;
#[cfg(feature = "deep-fill")]
pub use op::{DeepFillModel, DeepFillOp};

use super::plan::{Cursor, Image};

/// The environment variable a run reads the weights' path from when none is
/// given.
pub const WEIGHTS_VARIABLE: &str = "VESSEL_FILL_WEIGHTS";

/// One channel's deep filling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeepFillParams {
    /// Max-pool the mask by this factor before the network and upsample the
    /// probability back; 1 runs at full resolution.
    pub resample: usize,
    /// Probability at or above which a voxel is filled.
    pub threshold: f64,
}

impl Default for DeepFillParams {
    fn default() -> Self {
        Self {
            resample: 1,
            threshold: 0.5,
        }
    }
}

impl DeepFillParams {
    /// Resampled by `resample`, threshold 0.5.
    pub fn resampled(resample: usize) -> Self {
        Self {
            resample,
            ..Self::default()
        }
    }

    pub fn validate(&self) -> Result<()> {
        if !(1..=8).contains(&self.resample) {
            return Err(Error::invalid(format!(
                "deep fill: resample {} is outside 1..=8",
                self.resample
            )));
        }
        if !(self.threshold > 0.0 && self.threshold <= 1.0) {
            return Err(Error::invalid(format!(
                "deep fill: threshold {} is outside (0, 1]",
                self.threshold
            )));
        }
        Ok(())
    }

    /// The network's stride at this resampling: blocks start it at multiples
    /// of this.
    pub fn alignment(&self) -> usize {
        8 * self.resample
    }

    /// Halo per side: the network's reach (55 voxels, 112 at resample 2) plus
    /// the alignment slack, rounded up to `64 * resample`.
    pub fn reach(&self) -> usize {
        64 * self.resample
    }
}

/// What a deep-fill op writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepFillOutput {
    /// `Bool`: the mask OR probability >= threshold.
    Filled,
    /// `F32`: the network's probability alone.
    Probability,
}

/// Where the network runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeepFillDevice {
    /// burn's CPU backend, every core.
    Cpu,
    /// The GPU, and an error when this build has no GPU runtime or none is
    /// reachable.
    Gpu,
    /// The GPU when this build has a runtime and a device is reachable,
    /// otherwise the CPU. The default.
    #[default]
    Auto,
}

impl DeepFillDevice {
    pub fn parse(text: &str) -> Result<Self> {
        match text.trim() {
            "cpu" => Ok(Self::Cpu),
            "gpu" => Ok(Self::Gpu),
            "auto" => Ok(Self::Auto),
            other => Err(Error::invalid(format!(
                "deep fill: device `{other}` is not one of cpu, gpu, auto"
            ))),
        }
    }

    /// `Cpu` or `Gpu`: where a run with this choice puts the network, or why
    /// it cannot.
    pub fn resolve(self) -> Result<Self> {
        let reachable = GPU_BUILD && blockflow::gpu::available();
        match self {
            Self::Cpu => Ok(Self::Cpu),
            Self::Auto => Ok(if reachable { Self::Gpu } else { Self::Cpu }),
            Self::Gpu if reachable => Ok(Self::Gpu),
            Self::Gpu if !GPU_BUILD => Err(Error::invalid(
                "deep fill: --deep-fill-device gpu needs a build with the `deep-fill-cuda` \
                 (or `deep-fill-wgpu`) feature",
            )),
            Self::Gpu => Err(Error::invalid(
                "deep fill: --deep-fill-device gpu, and no GPU is reachable (no device or \
                 driver, or BLOCKFLOW_GPU=off); use cpu or auto",
            )),
        }
    }
}

/// Where the weights are, where the network runs and how the deep-fill
/// phases are cut; shared by every channel of a run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepFillSetup {
    /// The weights (safetensors); `None` reads [`WEIGHTS_VARIABLE`].
    pub weights: Option<PathBuf>,
    /// Block edge of the deep-fill phases, z, y, x (clamped to the volume).
    /// Each block is read with its halo and run through the network whole.
    /// On a GPU the block is halved along its longest axis until a block's
    /// forward pass fits the device budget.
    pub block: [usize; 3],
    pub device: DeepFillDevice,
}

impl Default for DeepFillSetup {
    fn default() -> Self {
        Self {
            weights: None,
            block: [256, 512, 512],
            device: DeepFillDevice::Auto,
        }
    }
}

impl DeepFillSetup {
    /// The weights' path: the one given, else [`WEIGHTS_VARIABLE`].
    pub fn weights_path(&self) -> Result<PathBuf> {
        if let Some(path) = &self.weights {
            return Ok(path.clone());
        }
        match std::env::var_os(WEIGHTS_VARIABLE) {
            Some(path) if !path.is_empty() => Ok(PathBuf::from(path)),
            _ => Err(Error::invalid(format!(
                "deep vessel filling needs the network's weights: give \
                 --deep-fill-weights <file.safetensors> (or set {WEIGHTS_VARIABLE}), \
                 or switch it off with --no-deep-fill. The weights are the reference's \
                 cnn_binary_vessel_filling.pth, converted by \
                 reference-clearmap/convert_weights.py"
            ))),
        }
    }
}

/// Whether this build can run the network.
pub const AVAILABLE: bool = cfg!(feature = "deep-fill");

/// Whether this build can run the network on a GPU (`deep-fill-cuda` or
/// `deep-fill-wgpu`).
pub const GPU_BUILD: bool = cfg!(feature = "deep-fill-gpu");

/// Append the deep filling of `mask` (`Bool`) as one pixel phase on its own
/// block grid; the grid is restored afterwards.
pub fn append_deep_fill(
    c: &mut Cursor<'_>,
    mask: Image,
    params: &DeepFillParams,
    setup: &DeepFillSetup,
    output: DeepFillOutput,
) -> Result<Image> {
    params.validate()?;
    #[cfg(feature = "deep-fill")]
    {
        use blockflow::geometry::BlockGrid;
        use blockflow::op::Chain;

        let model = DeepFillModel::shared(&setup.weights_path()?, setup.device.resolve()?)?;
        let grid = c.plan.grid().clone();
        let volume = grid.volume();
        let block = model.fit_block(setup.block, volume, params);
        c.plan.regrid(BlockGrid::new(volume, block)?);
        let op = DeepFillOp::new(model, *params, output);
        let filled = c.pixels(c.on(mask, vec![Chain::op(op)]));
        c.plan.regrid(grid);
        filled
    }
    #[cfg(not(feature = "deep-fill"))]
    {
        let _ = (c, mask, setup, output);
        Err(Error::invalid(
            "deep vessel filling: this vessel-graph was built without the `deep-fill` \
             feature; rebuild with `--features deep-fill`, or pass --no-deep-fill",
        ))
    }
}

/// Deep-fill a `Bool` mask held in memory, cut into `setup.block`. Returns
/// what `output` says.
pub fn deep_fill(
    mask: blockflow::voxels::Voxels,
    params: &DeepFillParams,
    setup: &DeepFillSetup,
    output: DeepFillOutput,
) -> Result<blockflow::voxels::Voxels> {
    use blockflow::assemble::PlanBuilder;
    use blockflow::geometry::BlockGrid;
    use blockflow::Dtype;

    use super::plan::{read_back, run_in_memory};

    if mask.dtype() != Dtype::Bool {
        return Err(Error::invalid(format!(
            "deep fill: the input holds {} and deep filling takes a mask",
            mask.dtype().numpy_name()
        )));
    }
    let shape = mask.shape();
    let source = Image::new(0usize, Dtype::Bool);
    let mut plan = PlanBuilder::new(shape, Dtype::Bool, BlockGrid::new(shape, setup.block)?);
    let mut c = Cursor::new(&mut plan, source);
    let filled = append_deep_fill(&mut c, source, params, setup, output)?;
    let assembly = plan.finish()?;
    let (_, env) = run_in_memory(&assembly, vec![mask], &[filled.id])?;
    read_back(&env, filled)
}
