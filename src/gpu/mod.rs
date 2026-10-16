// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// GPU compute for non-ML block ops.
//
// Before this module the GPU existed only inside model backends, and the
// scheduler knew nothing about it. This is the smallest layer that makes a GPU
// a place an ordinary `BlockOp` can run: one process-wide device, a byte budget
// on it, transfers of `Voxels` blocks, and a wrapper that picks the GPU or the
// CPU implementation of an op at run time. **The CPU implementation stays the
// definition**: a GPU kernel is checked against it by parity tests with a stated
// tolerance, and a build without `gpu-cuda` / `gpu-wgpu` compiles none of the
// device code and runs everything on the host.
//
// Survey: what was there
// ----------------------
// * **Burn / CubeCL at the pinned rev.** blockflow's Burn fork
//   (`e20ada6`, 0.22.0-pre.2) pins CubeCL git rev `4834896` (0.11.0-pre.2). In
//   this CubeCL a kernel is a Rust function under `#[cube(launch_unchecked)]`
//   taking `&[T]` / `&mut [T]` buffers, plain scalars, and `#[comptime]`
//   arguments that specialise it (one compiled kernel per value). It is launched
//   as `kernel::launch_unchecked::<R>(&client, CubeCount, CubeDim,
//   BufferArg::from_raw_parts(handle, len), scalars…)`, with `client =
//   R::client(&device)` and buffers from `client.create_from_slice` /
//   `client.empty` / `client.read_one`. Compilation is lazy and cached per
//   kernel and comptime key: CUDA through NVRTC (cudarc, loaded dynamically),
//   wgpu through WGSL/SPIR-V. Burn's "custom kernel" route (`burn-cubecl`'s
//   `CubeTensor`) is the same machinery wrapped in tensor types we do not need.
// * **How blockflow's model backends pick devices.** `yolo3d` takes an
//   `InferenceDevice::{Cpu, Cuda(index)}` at construction and builds a Burn
//   `Device::flex()` / `Device::cuda(..)`; the model sits behind a `Mutex`, so
//   blocks are serialised on it. `stardist-cuda`, `sam-cuda` and `cellpose-cuda`
//   do the same through Burn or candle. Each holds its own device, nothing
//   accounts for device memory, and the planner prices them like any CPU op.
//
// The decision: CubeCL, not cudarc + NVRTC
// ----------------------------------------
// CubeCL is already in the dependency graph at Burn's rev (the `cubecl`
// dependency here is that exact source, so nothing new is fetched and, with
// `burn/cuda` on too, one CubeCL and one CUDA context serve both). The same
// kernel source runs on CUDA and on wgpu (Vulkan / Metal / DX12), and HIP; the
// kernels are typed Rust with comptime specialisation rather than C strings.
// cudarc + NVRTC would give more control (explicit streams, pinned host memory,
// textures) but is CUDA-only, means a second CUDA context beside Burn's, and
// moves kernels into strings no compiler checks. The price of CubeCL is a
// pre-release API that moves between revs, and that host<->device copies go
// through its pageable path.
//
// What is here
// ------------
// * [`Accelerated`] wraps a CPU op (a [`BlockOp`] or a [`Combine`]) with an
//   optional [`GpuKernel`] and a [`DevicePolicy`]. Reach, slicing, dtypes and
//   constants are the CPU op's; only `apply`, `cost_per_voxel` and
//   `device_bytes` depend on where the block runs, and that is decided per call
//   from whether a device is reachable.
// * [`admit`]: a lease on the device, held while a block is on it. Two budgets,
//   both [`MemoryBudget`]s: device bytes and concurrent GPU tasks. The chain
//   walk in `op.rs` takes it from [`BlockOp::device_bytes`]; GPU kernels take it
//   too, so a direct call is bounded, and it is reentrant per thread so the two
//   do not count one block twice.
// * Kernels (behind the runtime features): `smooth` (separable Gaussian, the
//   GPU twin of `ops::smooth::SmoothOp`) and [`stencil`] (a 26-neighbourhood
//   colour-class update on `u32` labels and an `f32` field — the template for a
//   checkerboard label-energy sweep; its CPU reference is compiled always).
//
// What the planner still ignores
// ------------------------------
// * The device is priced by `cost_per_voxel` only: transfer is folded into that
//   per-voxel figure, so the plan sees GPU throughput but not that transfer is a
//   fixed latency per block plus bandwidth, nor PCIe contention between blocks.
// * Device memory is admitted at run time, not planned: block candidates are
//   still chosen against the host budget, and a block too large for the device
//   runs alone (oversubscribed) rather than being split.
// * Every GPU op uploads its input and downloads its output. Two GPU ops in a
//   sequence round-trip through the host between them; keeping blocks resident
//   on the device across a chain is the next step, and needs a `Voxels`-like
//   device buffer the chain walk can hand from op to op.
// * Host conversion: inputs are widened to `f32` on the host and outputs are
//   converted to the CPU op's dtype on the host (`f64` for smoothing), so a
//   `u16` block crosses PCIe at 4 bytes a voxel, not 2.
// * The CPU alternative is priced per host thread and the GPU per block, while
//   the GPU is one shared device: `expected_concurrency` workers each believe
//   they get the GPU figure, which overstates GPU throughput by up to the task
//   slot count.
//
// Measured (64x512x512 `f32`, sigma [1, 2, 2], see `tests.rs`): CPU op 0.6 s
// on one thread; GPU op 0.045 s on CUDA including transfers, of which the
// kernels are ~0.007 s — **transfer is ~85% of a GPU block**, which is why
// device residency across chained ops is the next thing worth building. wgpu
// (Vulkan) runs the same kernels at 0.08 s.

use std::cell::Cell;
use std::sync::Arc;

use crate::budget::{Lease, MemoryBudget};
use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::op::{Anchor, BlockConstraint, BlockOp, Combine, Slicing};
use crate::reach::Reach;
use crate::voxels::Voxels;

pub mod stencil;

#[cfg(feature = "gpu")]
pub mod device;
#[cfg(feature = "gpu")]
pub mod smooth;
#[cfg(feature = "gpu")]
pub use device::{device, Device};

#[cfg(all(test, feature = "gpu"))]
mod tests;

/// Where an [`Accelerated`] op runs its blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DevicePolicy {
    /// Always the CPU reference.
    Cpu,
    /// The GPU when one is reachable, otherwise the CPU. The default.
    #[default]
    Auto,
    /// The GPU, and an error at `apply` when there is none: for a run that
    /// must not silently fall back to a slower plan.
    Gpu,
}

/// Is a GPU reachable from this process? `false` in a build without a runtime
/// feature, when the driver or device is missing, or when `BLOCKFLOW_GPU=off`.
pub fn available() -> bool {
    #[cfg(feature = "gpu")]
    {
        device::device().is_some()
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

// ------------------------------------------------------------- admission --

thread_local! {
    /// Device bytes this thread already holds a lease for. Makes [`admit`]
    /// reentrant: the chain walk admits a block, then the kernel it calls admits
    /// the same block again, and the second must not wait on the first.
    static HELD: Cell<u64> = const { Cell::new(0) };
}

/// A block's admission to the device: one task slot and its device bytes.
/// Dropping it releases both.
#[derive(Debug)]
#[must_use = "the admission is released when dropped"]
pub struct DeviceAdmission {
    _memory: Option<Lease>,
    _slot: Option<Lease>,
    restore: Option<u64>,
}

impl Drop for DeviceAdmission {
    fn drop(&mut self) {
        if let Some(previous) = self.restore {
            HELD.with(|held| held.set(previous));
        }
    }
}

/// Wait until a block needing `bytes` of device memory may run on the device.
///
/// Blocks on two budgets in a fixed order — a task slot, then the bytes — so a
/// thread waiting for bytes always holds a slot and the two cannot deadlock. A
/// request larger than the whole device budget is admitted alone (the
/// [`MemoryBudget::acquire`] rule): pressure lowers concurrency, it does not
/// fail the run. Free when there is no device, or when this thread already
/// holds at least `bytes`.
pub fn admit(bytes: u64) -> DeviceAdmission {
    let none = DeviceAdmission {
        _memory: None,
        _slot: None,
        restore: None,
    };
    #[cfg(feature = "gpu")]
    {
        let Some(gpu) = device::device() else {
            return none;
        };
        let previous = HELD.with(Cell::get);
        if previous >= bytes {
            return none;
        }
        let (slots, memory) = gpu.budgets();
        let slot = (previous == 0).then(|| slots.acquire(1));
        let memory = memory.acquire(bytes - previous);
        HELD.with(|held| held.set(bytes));
        DeviceAdmission {
            _memory: Some(memory),
            _slot: slot,
            restore: Some(previous),
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = (bytes, &HELD);
        none
    }
}

/// The two budgets [`admit`] draws on, for a caller that wants to inspect them
/// (tests, progress reporting): `(task slots, device bytes)`.
pub fn budgets() -> Option<(MemoryBudget, MemoryBudget)> {
    #[cfg(feature = "gpu")]
    {
        device::device().map(|gpu| {
            let (slots, memory) = gpu.budgets();
            (slots.clone(), memory.clone())
        })
    }
    #[cfg(not(feature = "gpu"))]
    {
        None
    }
}

// --------------------------------------------------------------- kernels --

/// A GPU implementation of one op, for [`Accelerated`] to select.
///
/// `inputs` is one buffer for a [`BlockOp`] and one per branch for a
/// [`Combine`], with the same dtypes, shapes and anchor the CPU op is handed;
/// `out` has the CPU op's output dtype and shape. The kernel must compute the
/// CPU op's function to within its stated tolerance — it is not a second
/// definition.
pub trait GpuKernel: Send + Sync {
    /// Device bytes one block of output shape `block` needs while it runs:
    /// every device buffer alive at once, inputs and scratch included.
    fn device_bytes(&self, block: [usize; 3]) -> u64;

    /// Cost per voxel in the units of [`BlockOp::cost_per_voxel`], **including
    /// the transfer** to and from the device. Measured; see the benchmark in
    /// `src/gpu/tests.rs`.
    fn cost_per_voxel(&self) -> f64;

    fn apply(&self, inputs: &[&Voxels], out: &mut Voxels, at: &Anchor) -> Result<()>;
}

/// A CPU op with an optional GPU implementation, chosen per call.
///
/// Everything that shapes a plan — name, reach, slicing, dtypes, output shape,
/// constants, block constraints — is the CPU op's, so a plan does not change
/// with the device. What changes is `apply`, `cost_per_voxel` and
/// `device_bytes`, each answered from [`Self::runs_on_gpu`].
///
/// **Only ops whose whole behaviour is `apply`.** The wrapper does not forward
/// source inputs, placement or side outputs, so an op that overrides
/// `apply_with` / `apply_placed` / `side_outputs` must not be wrapped.
pub struct Accelerated<O> {
    cpu: O,
    gpu: Option<Arc<dyn GpuKernel>>,
    policy: DevicePolicy,
}

impl<O> Accelerated<O> {
    /// The CPU op alone: what a build without a GPU runtime constructs.
    pub fn cpu_only(cpu: O) -> Self {
        Self {
            cpu,
            gpu: None,
            policy: DevicePolicy::Cpu,
        }
    }

    pub fn with_kernel(cpu: O, gpu: Arc<dyn GpuKernel>, policy: DevicePolicy) -> Self {
        Self {
            cpu,
            gpu: Some(gpu),
            policy,
        }
    }

    pub fn with_policy(mut self, policy: DevicePolicy) -> Self {
        self.policy = policy;
        self
    }

    /// The CPU implementation, which is the definition.
    pub fn reference(&self) -> &O {
        &self.cpu
    }

    pub fn policy(&self) -> DevicePolicy {
        self.policy
    }

    /// Whether the next block will run on the GPU.
    pub fn runs_on_gpu(&self) -> bool {
        self.policy != DevicePolicy::Cpu && self.gpu.is_some() && available()
    }

    /// The kernel to run, `None` for the CPU, or the refusal [`DevicePolicy::Gpu`]
    /// asks for.
    fn kernel(&self, name: &str) -> Result<Option<&dyn GpuKernel>> {
        if self.runs_on_gpu() {
            return Ok(self.gpu.as_deref());
        }
        if self.policy == DevicePolicy::Gpu {
            return Err(Error::InvalidArgument(format!(
                "{name}: the device policy requires a GPU, and {}",
                if self.gpu.is_none() {
                    "this op has no GPU implementation"
                } else {
                    "none is reachable (no `gpu-cuda`/`gpu-wgpu` feature, no device, or BLOCKFLOW_GPU=off)"
                }
            )));
        }
        Ok(None)
    }
}

impl<O: BlockOp> BlockOp for Accelerated<O> {
    fn name(&self) -> &'static str {
        self.cpu.name()
    }
    fn reach(&self, axis: usize, volume_len: usize) -> usize {
        self.cpu.reach(axis, volume_len)
    }
    fn reach_spec(&self, volume: [usize; 3]) -> Reach {
        self.cpu.reach_spec(volume)
    }
    fn slicing(&self) -> Slicing {
        self.cpu.slicing()
    }
    fn accepts(&self, dtype: Dtype) -> bool {
        self.cpu.accepts(dtype)
    }
    fn produces(&self, input: Dtype) -> Dtype {
        self.cpu.produces(input)
    }
    fn output_shape(&self, input: [usize; 3]) -> [usize; 3] {
        self.cpu.output_shape(input)
    }
    fn preferred_iteration(&self) -> Option<[usize; 3]> {
        self.cpu.preferred_iteration()
    }
    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        self.cpu.constant_maps_to(value)
    }
    fn block_constraint(&self, volume: [usize; 3]) -> Option<BlockConstraint> {
        self.cpu.block_constraint(volume)
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, at: &Anchor) -> Result<()> {
        match self.kernel(self.cpu.name())? {
            Some(kernel) => kernel.apply(&[input], out, at),
            None => self.cpu.apply(input, out, at),
        }
    }

    /// The GPU figure, transfer included, when the block will run there.
    fn cost_per_voxel(&self) -> f64 {
        match &self.gpu {
            Some(kernel) if self.runs_on_gpu() => kernel.cost_per_voxel(),
            _ => self.cpu.cost_per_voxel(),
        }
    }
    fn cost_per_voxel_in(&self, block: [usize; 3]) -> f64 {
        match &self.gpu {
            Some(kernel) if self.runs_on_gpu() => kernel.cost_per_voxel(),
            _ => self.cpu.cost_per_voxel_in(block),
        }
    }

    fn device_bytes(&self, block: [usize; 3]) -> Option<u64> {
        let kernel = self.gpu.as_ref().filter(|_| self.runs_on_gpu())?;
        Some(kernel.device_bytes(self.cpu.output_shape(block)))
    }
}

impl<O: Combine> Combine for Accelerated<O> {
    fn name(&self) -> &'static str {
        self.cpu.name()
    }
    fn reach(&self, axis: usize, volume_len: usize) -> usize {
        self.cpu.reach(axis, volume_len)
    }
    fn reach_spec(&self, volume: [usize; 3]) -> Reach {
        self.cpu.reach_spec(volume)
    }
    fn slicing(&self) -> Slicing {
        self.cpu.slicing()
    }
    fn accepts(&self, inputs: &[Dtype]) -> bool {
        self.cpu.accepts(inputs)
    }
    fn produces(&self, inputs: &[Dtype]) -> Dtype {
        self.cpu.produces(inputs)
    }
    fn output_shape(&self, inputs: &[[usize; 3]]) -> Result<[usize; 3]> {
        self.cpu.output_shape(inputs)
    }
    fn constant_maps_to(&self, values: &[f64]) -> Option<f64> {
        self.cpu.constant_maps_to(values)
    }

    fn apply(&self, inputs: &[&Voxels], out: &mut Voxels, at: &Anchor) -> Result<()> {
        match self.kernel(self.cpu.name())? {
            Some(kernel) => kernel.apply(inputs, out, at),
            None => self.cpu.apply(inputs, out, at),
        }
    }

    fn cost_per_voxel(&self, branches: usize) -> f64 {
        match &self.gpu {
            Some(kernel) if self.runs_on_gpu() => kernel.cost_per_voxel(),
            _ => self.cpu.cost_per_voxel(branches),
        }
    }

    fn device_bytes(&self, block: [usize; 3]) -> Option<u64> {
        let kernel = self.gpu.as_ref().filter(|_| self.runs_on_gpu())?;
        Some(kernel.device_bytes(block))
    }
}
