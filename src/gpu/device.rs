// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The process-wide device, its two budgets, and host<->device transfer of
// blocks.
//
// One device per process, opened on first use and never closed: CubeCL caches a
// client per device itself, and opening a CUDA context per op or per block would
// cost hundreds of milliseconds each and multiply the context's memory. The
// device is the runtime's default (CUDA ordinal 0, or wgpu's default adapter).
//
// Environment, read once at first use:
// * `BLOCKFLOW_GPU=off` (or `0`) — behave as if there were no device.
// * `BLOCKFLOW_GPU_MEMORY` — device bytes [`super::admit`] may lease. Default:
//   three quarters of the device's total memory, leaving room for the context,
//   CubeCL's pool rounding, and whatever else shares the card.
// * `BLOCKFLOW_GPU_TASKS` — blocks on the device at once. Default 2: one
//   transferring while another computes is the overlap worth having; beyond
//   that, blocks only queue on the same device.

use std::sync::OnceLock;

use cubecl::bytes::Bytes;
use cubecl::prelude::*;
use cubecl::server::Handle;
use ndarray::{Array3, ArrayView3};

use crate::budget::MemoryBudget;
use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::voxels::Voxels;

#[cfg(not(any(feature = "gpu-cuda", feature = "gpu-wgpu")))]
compile_error!("the `gpu` feature needs a runtime: enable `gpu-cuda` or `gpu-wgpu`");

/// The CubeCL runtime this build launches on: CUDA when `gpu-cuda` is on,
/// otherwise wgpu.
#[cfg(feature = "gpu-cuda")]
pub type Rt = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "gpu-wgpu", not(feature = "gpu-cuda")))]
pub type Rt = cubecl::wgpu::WgpuRuntime;

/// The process's GPU: a CubeCL client and the budgets blocks are admitted on.
pub struct Device {
    client: ComputeClient<Rt>,
    total_bytes: u64,
    slots: MemoryBudget,
    memory: MemoryBudget,
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("runtime", &Rt::name(&self.client))
            .field("total_bytes", &self.total_bytes)
            .field("budget_bytes", &self.memory.total())
            .field("tasks", &self.slots.total())
            .finish()
    }
}

static DEVICE: OnceLock<Option<Device>> = OnceLock::new();

/// The process's GPU, opened on first call; `None` when there is none to open.
///
/// Opening is guarded against the runtime panicking (CubeCL unwraps a missing
/// driver), so a machine without a GPU gets `None` and the CPU path, not a
/// crash.
pub fn device() -> Option<&'static Device> {
    DEVICE.get_or_init(open).as_ref()
}

fn open() -> Option<Device> {
    let off = std::env::var("BLOCKFLOW_GPU")
        .map(|value| matches!(value.trim(), "off" | "0" | "false" | "no"))
        .unwrap_or(false);
    if off {
        return None;
    }
    let client = std::panic::catch_unwind(|| Rt::client(&Default::default())).ok()?;
    // CUDA's `max_page_size` is a quarter of `cuDeviceTotalMem`; on wgpu it is
    // the largest buffer, and four of them is a guess the variable overrides.
    let total_bytes = client.properties().memory.max_page_size.saturating_mul(4);
    let budget = env_u64("BLOCKFLOW_GPU_MEMORY").unwrap_or(total_bytes / 4 * 3);
    let tasks = env_u64("BLOCKFLOW_GPU_TASKS").unwrap_or(2).max(1);
    Some(Device {
        client,
        total_bytes,
        slots: MemoryBudget::new(tasks),
        memory: MemoryBudget::new(budget),
    })
}

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok()?.trim().parse().ok()
}

impl Device {
    pub fn client(&self) -> &ComputeClient<Rt> {
        &self.client
    }

    /// The device's total memory as the runtime reports it.
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// `(task slots, device bytes)`, as [`super::admit`] draws on them.
    pub(crate) fn budgets(&self) -> (&MemoryBudget, &MemoryBudget) {
        (&self.slots, &self.memory)
    }

    /// Upload a host block as `f32`, converting on the host (chunk by chunk,
    /// with no whole-block intermediate) when it is not already `f32`.
    pub fn upload_f32(&self, voxels: &Voxels) -> Result<DeviceBlock> {
        let handle = match voxels {
            Voxels::Bool(a) => self.upload_as_f32(a, |v| f32::from(u8::from(v))),
            Voxels::U8(a) => self.upload_as_f32(a, f32::from),
            Voxels::U16(a) => self.upload_as_f32(a, f32::from),
            Voxels::U32(a) => self.upload_as_f32(a, |v| v as f32),
            Voxels::U64(a) => self.upload_as_f32(a, |v| v as f32),
            Voxels::I8(a) => self.upload_as_f32(a, f32::from),
            Voxels::I16(a) => self.upload_as_f32(a, f32::from),
            Voxels::I32(a) => self.upload_as_f32(a, |v| v as f32),
            Voxels::I64(a) => self.upload_as_f32(a, |v| v as f32),
            Voxels::F32(a) => self.upload_as_f32(a, |v| v),
            Voxels::F64(a) => self.upload_as_f32(a, |v| v as f32),
        };
        Ok(DeviceBlock {
            handle,
            shape: voxels.shape(),
            dtype: Dtype::F32,
        })
    }

    /// Upload an `f32` array.
    pub fn upload_array_f32(&self, array: ArrayView3<'_, f32>) -> DeviceBlock {
        let shape = [array.shape()[0], array.shape()[1], array.shape()[2]];
        let handle = match array.as_slice() {
            Some(slice) => self.upload_elements(slice, |v| v),
            None => self.upload_elements(&array.iter().copied().collect::<Vec<_>>(), |v| v),
        };
        DeviceBlock {
            handle,
            shape,
            dtype: Dtype::F32,
        }
    }

    /// Upload a `u32` block (labels) as it is. Any other dtype is refused: a
    /// label is not something to convert silently.
    pub fn upload_u32(&self, voxels: &Voxels) -> Result<DeviceBlock> {
        let Voxels::U32(array) = voxels else {
            return Err(Error::InvalidArgument(format!(
                "upload_u32: a label block must be uint32, got {}",
                voxels.dtype().numpy_name()
            )));
        };
        let handle = match array.as_slice() {
            Some(slice) => self.upload_elements(slice, |v| v),
            None => self.upload_elements(&array.iter().copied().collect::<Vec<_>>(), |v| v),
        };
        Ok(DeviceBlock {
            handle,
            shape: voxels.shape(),
            dtype: Dtype::U32,
        })
    }

    /// An uninitialised device block.
    pub fn empty(&self, shape: [usize; 3], dtype: Dtype) -> DeviceBlock {
        let bytes = shape.iter().product::<usize>() * dtype.size_of();
        DeviceBlock {
            handle: self.client.empty(bytes.max(4)),
            shape,
            dtype,
        }
    }

    /// Upload a small `f32` vector (kernel weights).
    pub fn upload_slice_f32(&self, values: &[f32]) -> Handle {
        self.client.create_from_slice(f32::as_bytes(values))
    }

    fn upload_as_f32<T: Copy>(&self, array: &Array3<T>, convert: impl Fn(T) -> f32) -> Handle {
        match array.as_slice() {
            Some(slice) => self.upload_elements(slice, convert),
            None => self.upload_elements(&array.iter().copied().collect::<Vec<_>>(), convert),
        }
    }

    /// Upload `source` converted element by element, in chunks of
    /// [`UPLOAD_CHUNK`] bytes.
    ///
    /// **Chunked because it is eight times faster here, measured.** CubeCL's
    /// CUDA write stages host data through its pinned pool; one 64 MiB write
    /// measured 0.21 s (`create_from_slice`, which also copies the slice twice)
    /// or 0.14 s (`create`), and the same bytes as 1 MiB writes into one
    /// allocation 0.026 s, about 2.6 GB/s. The chunks are written in order on
    /// this thread's stream, so a kernel launched after them sees them all.
    fn upload_elements<T: Copy, E: DeviceElement>(
        &self,
        source: &[T],
        convert: impl Fn(T) -> E,
    ) -> Handle {
        let element = std::mem::size_of::<E>();
        let total = source.len() * element;
        let handle = self.client.empty(total.max(4));
        let per_chunk = (UPLOAD_CHUNK / element).max(1);
        let mut written = 0usize;
        for piece in source.chunks(per_chunk) {
            let len = piece.len() * element;
            let values: Vec<E> = piece.iter().map(|&value| convert(value)).collect();
            let target = handle
                .clone()
                .offset_start(written as u64)
                .offset_end((total - written - len) as u64);
            self.client.write(&target, E::into_bytes(values));
            written += len;
        }
        handle
    }

    /// Download a block into `out`, which must have its shape. An `f32` block
    /// may land in an `f32` or `f64` buffer; a `u32` block in a `u32` one.
    /// Waits for every kernel queued before it.
    pub fn download_into(&self, block: DeviceBlock, out: &mut Voxels) -> Result<()> {
        if out.shape() != block.shape {
            return Err(Error::ShapeMismatch {
                expected: block.shape.to_vec(),
                got: out.shape().to_vec(),
            });
        }
        let bytes = self
            .client
            .read_one(block.handle)
            .map_err(|error| Error::InvalidArgument(format!("device read failed: {error:?}")))?;
        match (block.dtype, out) {
            (Dtype::F32, Voxels::F32(array)) => fill(array, f32::from_bytes(&bytes), |v| v),
            (Dtype::F32, Voxels::F64(array)) => fill(array, f32::from_bytes(&bytes), f64::from),
            (Dtype::U32, Voxels::U32(array)) => fill(array, u32::from_bytes(&bytes), |v| v),
            (dtype, out) => {
                return Err(Error::InvalidArgument(format!(
                    "download_into: a {} device block cannot land in a {} buffer",
                    dtype.numpy_name(),
                    out.dtype().numpy_name()
                )))
            }
        }
        Ok(())
    }

    /// Download an `f32` block as an array.
    pub fn download_f32(&self, block: DeviceBlock) -> Result<Array3<f32>> {
        let mut out = Voxels::zeros(Dtype::F32, block.shape)?;
        self.download_into(block, &mut out)?;
        let Voxels::F32(array) = out else {
            unreachable!()
        };
        Ok(array)
    }
}

fn fill<S: Copy, T>(array: &mut Array3<T>, values: &[S], convert: impl Fn(S) -> T) {
    match array.as_slice_mut() {
        Some(slice) => {
            for (dst, &src) in slice.iter_mut().zip(values) {
                *dst = convert(src);
            }
        }
        None => {
            for (dst, &src) in array.iter_mut().zip(values) {
                *dst = convert(src);
            }
        }
    }
}

/// The size of one host-to-device write; see `Device::upload_elements`.
pub const UPLOAD_CHUNK: usize = 1 << 20;

/// An element type a device block holds.
trait DeviceElement: Copy {
    fn into_bytes(values: Vec<Self>) -> Bytes;
}

impl DeviceElement for f32 {
    fn into_bytes(values: Vec<Self>) -> Bytes {
        Bytes::from_elems(values)
    }
}

impl DeviceElement for u32 {
    fn into_bytes(values: Vec<Self>) -> Bytes {
        Bytes::from_elems(values)
    }
}

/// A block resident on the device: a buffer, its shape and its element type
/// (`f32` or `u32`). Consumed by a download; dropping it returns the memory to
/// CubeCL's pool.
#[derive(Debug, Clone)]
pub struct DeviceBlock {
    pub(crate) handle: Handle,
    pub shape: [usize; 3],
    pub dtype: Dtype,
}

impl DeviceBlock {
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn arg(&self) -> BufferArg<Rt> {
        // SAFETY: the handle was allocated for exactly `len` elements of this
        // block's dtype, and the kernels index it below `len`.
        unsafe { BufferArg::from_raw_parts(self.handle.clone(), self.len()) }
    }
}

/// A one-dimensional launch over `elements` threads, split onto two grid axes
/// so it stays under the 65535-per-axis limit wgpu has (CUDA's x is larger, but
/// the kernels are written once). Kernels index by `ABSOLUTE_POS` and guard
/// against the tail.
pub(crate) fn launch_shape(elements: usize) -> (CubeCount, CubeDim) {
    const THREADS: u32 = 256;
    let cubes = elements.div_ceil(THREADS as usize).max(1) as u64;
    let x = cubes.min(65_535) as u32;
    let y = cubes.div_ceil(x as u64) as u32;
    (CubeCount::Static(x, y, 1), CubeDim::new_1d(THREADS))
}

/// Refuse a block whose linear index would not fit the kernels' `u32`
/// arithmetic.
pub(crate) fn check_indexable(shape: [usize; 3], what: &str) -> Result<()> {
    let elements = shape
        .iter()
        .try_fold(1u64, |acc, &n| acc.checked_mul(n as u64));
    match elements {
        Some(n) if n < u32::MAX as u64 => Ok(()),
        _ => Err(Error::InvalidArgument(format!(
            "{what}: a block of {shape:?} has more voxels than the GPU kernels index (2^32 - 1)"
        ))),
    }
}
