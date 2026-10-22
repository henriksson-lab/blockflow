//! Bounded physical point sampling over typed, chunk-backed sources.
//!
//! This is a consumer of a source, not another cache. Sources may use
//! `cache::CachingSource`; they must declare their complete additional read
//! workspace (including decoder scratch and cache) in `SourceLayout`.
//! The limit here covers sampler allocations and that declaration, not RSS or
//! unrelated caller-owned images, coordinates, results and allocator metadata.
use crate::{
    dtype::Dtype,
    error::{Error, Result},
    voxels::Voxels,
};

#[derive(Clone, Debug)]
pub struct SourceLayout {
    pub shape_zyx: [usize; 3],
    pub chunk_zyx: [usize; 3],
    pub dtype: Dtype,
    /// Physical pixel-centre map, XYZ. Positive diagonal transforms only.
    pub origin_xyz: [f64; 3],
    pub spacing_xyz: [f64; 3],
    /// Maximum source-owned workspace/cache alive during a read, excluding the
    /// returned data and validity buffer, which the sampler prices itself.
    /// This is a source contract, not a measurement or a codec-specific bound.
    pub additional_workspace_bytes: usize,
}
impl SourceLayout {
    pub fn validate(&self) -> Result<()> {
        if self.shape_zyx.contains(&0)
            || self.chunk_zyx.contains(&0)
            || self.dtype == Dtype::F16
            || self.origin_xyz.iter().any(|x| !x.is_finite())
            || self.spacing_xyz.iter().any(|x| !x.is_finite() || *x <= 0.)
        {
            return Err(Error::invalid("invalid sampling source layout"));
        }
        Ok(())
    }
}

pub struct SampleChunk {
    pub values: Voxels,
    /// None means all values valid. A returned mask is one byte per voxel in
    /// the accounting (Vec<bool> uses no more than that).
    pub valid: Option<Vec<bool>>,
}
/// One immutable selected image/level, already calibrated. `read_chunk` must
/// return exactly the requested clipped chunk and respect its declared source
/// workspace. Missing acquisition data is represented by validity, not an IO
/// error; IO/decoder/budget errors must propagate.
pub trait SampleSource {
    fn layout(&self) -> &SourceLayout;
    fn read_chunk(&mut self, start_zyx: [usize; 3], shape_zyx: [usize; 3]) -> Result<SampleChunk>;
}
/// Adapter for existing storage environments, preserving their typed read and
/// cache path. Calibration, image geometry and source-workspace reservation
/// must describe the selected image, not necessarily environment image zero.
/// Accounting-only environments are refused rather than fabricating samples.
pub struct EnvironmentSampleSource<'a> {
    environment: &'a dyn crate::env::Environment,
    image: usize,
    layout: SourceLayout,
}
impl<'a> EnvironmentSampleSource<'a> {
    pub fn new(
        environment: &'a dyn crate::env::Environment,
        image: usize,
        layout: SourceLayout,
    ) -> Result<Self> {
        layout.validate()?;
        Ok(Self {
            environment,
            image,
            layout,
        })
    }
}
impl SampleSource for EnvironmentSampleSource<'_> {
    fn layout(&self) -> &SourceLayout {
        &self.layout
    }
    fn read_chunk(&mut self, start: [usize; 3], shape: [usize; 3]) -> Result<SampleChunk> {
        let region = crate::region::Region::new(&start, &shape);
        match self.environment.read(self.image, &region)? {
            crate::env::BlockBuf::Array(values) => Ok(SampleChunk {
                values,
                valid: None,
            }),
            crate::env::BlockBuf::Accounted { .. } => Err(Error::invalid(
                "cannot sample an accounting-only environment",
            )),
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct SamplingBudget {
    pub max_working_bytes: usize,
    pub max_chunk_reads: usize,
    /// Returned decoded image+validity bytes, cumulatively across this batch.
    /// Compressed transport bytes and read amplification belong to the source.
    pub max_decoded_bytes: usize,
}
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Sample {
    pub value: f64,
    pub gradient_xyz: [f64; 3],
    pub valid: bool,
    pub gradient_valid: bool,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct SamplingStats {
    pub requested_points: usize,
    pub outside_points: usize,
    pub invalid_points: usize,
    pub invalid_gradients: usize,
    pub chunk_reads: usize,
    /// Conservative IO charge: returned typed image plus a reserved mask byte
    /// per voxel, even when the source omits the optional validity mask.
    pub decoded_bytes: usize,
    pub peak_planned_working_bytes: usize,
}
#[derive(Clone, Copy)]
struct Corner {
    chunk: [usize; 3],
    index: [usize; 3],
    point: usize,
    weight: f64,
    gradient: [f64; 3],
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| Error::invalid("sampling size overflow"))
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| Error::invalid("sampling size overflow"))
}
fn count(shape: [usize; 3]) -> Result<usize> {
    shape.into_iter().try_fold(1, mul)
}
fn numeric(v: &Voxels, p: [usize; 3]) -> f64 {
    match v {
        Voxels::Bool(a) => {
            if a[p] {
                1.
            } else {
                0.
            }
        }
        Voxels::U8(a) => a[p] as f64,
        Voxels::U16(a) => a[p] as f64,
        Voxels::U32(a) => a[p] as f64,
        Voxels::U64(a) => a[p] as f64,
        Voxels::I8(a) => a[p] as f64,
        Voxels::I16(a) => a[p] as f64,
        Voxels::I32(a) => a[p] as f64,
        Voxels::I64(a) => a[p] as f64,
        Voxels::F32(a) => a[p] as f64,
        Voxels::F64(a) => a[p],
    }
}
/// Fill caller-owned results. Integer values are numerically converted to f64;
/// U64/I64 above its exact integer range therefore lose precision. At the upper
/// acquisition boundary the gradient is the inward linear-cell derivative.
/// Singleton axes have zero gradient. Indices within eight relative machine
/// epsilons of an integer snap to that centre; interior knots use the forward
/// cell derivative. A masked/nonfinite contributing value
/// invalidates the value only for nonzero value weight. Gradient validity is
/// separate, so an isolated valid pixel keeps its value at its exact centre.
/// On Err the output must be discarded: never treat partial work as missing
/// tissue. Per-batch IO is preflighted before the first read.
pub fn sample_points(
    source: &mut impl SampleSource,
    positions_xyz: &[[f64; 3]],
    output: &mut [Sample],
    budget: SamplingBudget,
) -> Result<SamplingStats> {
    let layout = source.layout().clone();
    layout.validate()?;
    if output.len() != positions_xyz.len() {
        return Err(Error::invalid("sampling output length differs"));
    }
    let n = output.len();
    if n == 0 {
        return Ok(SamplingStats::default());
    }
    let capacity = mul(n, 8)?;
    let request_bytes = mul(capacity, std::mem::size_of::<Corner>())?;
    let fixed = add(request_bytes, layout.additional_workspace_bytes)?;
    let max_chunk_shape = std::array::from_fn(|i| layout.chunk_zyx[i].min(layout.shape_zyx[i]));
    let max_chunk_bytes = mul(count(max_chunk_shape)?, add(layout.dtype.size_of(), 1)?)?;
    let planned = add(fixed, max_chunk_bytes)?;
    if planned > budget.max_working_bytes {
        return Err(Error::invalid("sampling_memory_budget_exceeded"));
    }
    // No growth/reallocation, and unstable sort is in-place.
    let mut requests = Vec::with_capacity(capacity);
    output.fill(Sample::default());
    let mut stats = SamplingStats {
        requested_points: n,
        peak_planned_working_bytes: planned,
        ..SamplingStats::default()
    };
    for (point, p) in positions_xyz.iter().enumerate() {
        let q: [f64; 3] =
            std::array::from_fn(|i| (p[i] - layout.origin_xyz[i]) / layout.spacing_xyz[i]);
        // Roundoff in the physical->index inverse must not turn an exact
        // observed centre into a tiny masked interpolation across a tear.
        let q = q.map(|v| {
            let nearest = v.round();
            if (v - nearest).abs() <= 8. * f64::EPSILON * v.abs().max(1.) {
                nearest
            } else {
                v
            }
        });
        if (0..3)
            .any(|i| !q[i].is_finite() || q[i] < 0. || q[i] > (layout.shape_zyx[2 - i] - 1) as f64)
        {
            stats.outside_points += 1;
            continue;
        }
        output[point].valid = true;
        output[point].gradient_valid = true;
        let lo: [usize; 3] = std::array::from_fn(|i| {
            (q[i].floor() as usize).min(layout.shape_zyx[2 - i].saturating_sub(2))
        });
        let f: [f64; 3] = std::array::from_fn(|i| q[i] - lo[i] as f64);
        let lengths: [usize; 3] = std::array::from_fn(|i| layout.shape_zyx[2 - i].min(2));
        for z in 0..lengths[2] {
            for y in 0..lengths[1] {
                for x in 0..lengths[0] {
                    let bits = [x, y, z];
                    let weights: [f64; 3] = std::array::from_fn(|i| {
                        if lengths[i] == 1 {
                            1.
                        } else if bits[i] == 0 {
                            1. - f[i]
                        } else {
                            f[i]
                        }
                    });
                    let weight = weights.iter().product();
                    let gradient = std::array::from_fn(|i| {
                        if lengths[i] == 1 {
                            0.
                        } else {
                            (if bits[i] == 0 { -1. } else { 1. }) / layout.spacing_xyz[i]
                                * (0..3)
                                    .filter(|&j| j != i)
                                    .map(|j| weights[j])
                                    .product::<f64>()
                        }
                    });
                    // Derivatives need some corners with zero interpolation weight.
                    if weight == 0. && gradient == [0.; 3] {
                        continue;
                    }
                    let index = [lo[2] + z, lo[1] + y, lo[0] + x];
                    requests.push(Corner {
                        chunk: std::array::from_fn(|i| index[i] / layout.chunk_zyx[i]),
                        index,
                        point,
                        weight,
                        gradient,
                    });
                }
            }
        }
    }
    requests.sort_unstable_by_key(|r| (r.chunk, r.point, r.index));
    let geometry = |key: [usize; 3]| {
        let start = std::array::from_fn(|i| key[i] * layout.chunk_zyx[i]);
        let shape =
            std::array::from_fn(|i| layout.chunk_zyx[i].min(layout.shape_zyx[i] - start[i]));
        (start, shape)
    };
    let mut last = None;
    for r in &requests {
        if last != Some(r.chunk) {
            let (_, shape) = geometry(r.chunk);
            stats.chunk_reads = add(stats.chunk_reads, 1)?;
            stats.decoded_bytes = add(
                stats.decoded_bytes,
                mul(count(shape)?, add(layout.dtype.size_of(), 1)?)?,
            )?;
            last = Some(r.chunk);
        }
    }
    if stats.chunk_reads > budget.max_chunk_reads || stats.decoded_bytes > budget.max_decoded_bytes
    {
        return Err(Error::invalid("sampling_io_budget_exceeded"));
    }
    let mut offset = 0;
    while offset < requests.len() {
        let key = requests[offset].chunk;
        let (start, shape) = geometry(key);
        let chunk = source.read_chunk(start, shape)?;
        if chunk.values.shape() != shape
            || chunk.values.dtype() != layout.dtype
            || chunk
                .valid
                .as_ref()
                .is_some_and(|v| v.len() != chunk.values.len())
        {
            return Err(Error::invalid(
                "sampling source violated chunk shape/dtype/mask contract",
            ));
        }
        while offset < requests.len() && requests[offset].chunk == key {
            let r = &requests[offset];
            let p = std::array::from_fn(|i| r.index[i] - start[i]);
            let at = (p[0] * shape[1] + p[1]) * shape[2] + p[2];
            let v = numeric(&chunk.values, p);
            let dst = &mut output[r.point];
            if !v.is_finite() || chunk.valid.as_ref().is_some_and(|m| !m[at]) {
                if r.weight != 0. {
                    dst.valid = false;
                }
                if r.gradient != [0.; 3] {
                    dst.gradient_valid = false;
                }
            } else {
                dst.value += r.weight * v;
                for i in 0..3 {
                    dst.gradient_xyz[i] += r.gradient[i] * v;
                }
            }
            offset += 1;
        }
    }
    for sample in output {
        if sample.valid && !sample.value.is_finite() {
            return Err(Error::invalid("sampling_numeric_overflow"));
        }
        if !sample.valid {
            *sample = Sample::default();
            stats.invalid_points += 1;
        } else if sample.gradient_valid && sample.gradient_xyz.iter().any(|v| !v.is_finite()) {
            return Err(Error::invalid("sampling_numeric_overflow"));
        }
        if !sample.gradient_valid {
            sample.gradient_xyz = [0.; 3];
            stats.invalid_gradients += 1;
        }
    }
    Ok(stats)
}
