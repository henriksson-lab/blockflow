// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// A windowed statistic sampled on a global lattice, **evaluated once per
// sample** rather than once per block that brackets it.
//
// Why this exists beside `local` and `lattice`
// --------------------------------------------
// `local::LocalStatisticOp` is the fused form: each block evaluates every
// sample its buffer brackets and interpolates them back. Its halo is the
// lattice distance *plus* the element, and every sample inside that halo is
// evaluated again by each block whose halo holds it. At a spacing comparable to
// the block edge that is most of the work done two or three times — a z slab of
// eight planes under a window five planes thick, sampled every five, evaluates
// each sample about three times (`LocalStatistic::evaluations_per_sample`).
//
// `lattice` splits the op into two pixel phases, and that cannot be fused into
// a consumer: the interpolation's phase is cut over the fine volume but its
// input is the coarse image, and a source leaf in that phase is read at the
// coarse fetch, so nothing fine can be combined with the interpolated value in
// the same phase. The only way out there is a full-resolution `f64` image per
// statistic.
//
// This module takes the fragment route instead, which is the one CLAHE's tile
// maps already take (`exposure`):
//
// | phase | reads | writes |
// |---|---|---|
// | [`LatticeSamplesOp`] | the image and its population, at the **element's** reach | one fragment per block: the samples whose centres are in its core |
// | [`GatherSamplesOp`] | the fragments of the few blocks around it; no pixels | one fragment per block: the samples its core brackets |
// | [`InterpolateSamplesOp`] | its own gathered fragment, and its source images voxelwise | the combined voxel |
//
// Each sample is owned by the one block whose core holds its centre, so it is
// evaluated once. Its window is the element around that centre, clamped to the
// volume, and the block's read extent is its core widened by the element's own
// sides — so every window a block evaluates is whole. The samples are small
// (one `f64` each), so the gather moves them between blocks for the price of a
// few kilobytes, and every block interpolates from the box it was handed.
//
// Why the answer is the fused op's, bit for bit
// ---------------------------------------------
// The fused op's samples that a core voxel reads are whole windows too — that
// is what its halo is for — gathered in the element's order by the same code
// (`local::sample_value`) and reduced by the same reducer, so they are the same
// numbers. The interpolation is the same expression in the same order: the
// brackets come from the same `Alignment::bracket` on the same global
// coordinate, and the blend is `a + t * (b - a)` along `x`, then `y`, then `z`.
// The narrowing sites are the same two. And the second phase hands the
// combining function the same `f64` values a `FnCombine` over the fused op's
// output would have been handed. `tests/lattice_samples.rs` pins all of it
// against the fused op on the bits, under several decompositions.
//
// Why there is no barrier
// -----------------------
// The interpolation could assemble the whole lattice once, after a barrier —
// but a barrier waits for every earlier phase of the plan, not only for the
// samples, and in a plan with a slow phase before this one (the vessel
// example's deep fill of the first channel, before the second channel's
// binarisation) that serialises what would otherwise overlap. The gather phase
// instead declares a fragment reach of the blocks a core's samples can be owned
// by, which is a scheduling dependency on those blocks and nothing more.

use std::sync::Arc;

use ndarray::Axis;

use crate::assemble::{ImageId, Phase, PlanBuilder};
use crate::dtype::Dtype;
use crate::env::BlockBuf;
use crate::error::{Error, Result};
use crate::fragment::{
    BlockOutput, BlockView, Coverage, FragmentInput, FragmentOp, FragmentOutput, SeamFold,
    SidecarSize, SourceBlocks,
};
use crate::op::SourceInput;
use crate::reach::Reach;
use crate::sidecar::Lifecycle;
use crate::voxels::{VoxelElement, Voxels};

use super::element::{StructuringElement, Total};
use super::local::{
    sample_value, Alignment, EmptyPopulation, LatticeNarrowing, LocalStatistic, Population,
    SampleLattice, Sampling, Statistic, INTERPOLATION_COST, SAMPLE_COST_PER_ELEMENT_VOXEL,
};
use super::rank::{ExcludedCentre, MASK_COST_FACTOR};
use super::voxelwise::COMBINE_COST;

const SAMPLES_MAGIC: u64 = 0x4c41_5453_414d_5031;

/// A box of samples carried no data: the run was
/// simulated and there were no pixels to evaluate.
const SIMULATED: u8 = 1;
const EVALUATED: u8 = 0;

/// What [`InterpolateSamplesOp`] hands its combining function per voxel: the
/// source images' values in the order they were named, then the interpolated
/// statistics in the order they were stated.
pub type SampleCombine = Arc<dyn Fn(&[f64]) -> f64 + Send + Sync>;

/// The statistics one gather serves: they must share the element and the
/// sampling, and may differ in everything else.
fn check_statistics(name: &str, statistics: &[LocalStatistic]) -> Result<()> {
    let Some(first) = statistics.first() else {
        return Err(Error::InvalidArgument(format!(
            "{name}: a lattice of samples needs at least one statistic"
        )));
    };
    for (index, other) in statistics.iter().enumerate().skip(1) {
        if other.element() != first.element() || other.sampling() != first.sampling() {
            return Err(Error::InvalidArgument(format!(
                "{name}: statistic {index} has a different element or sampling from statistic \
                 0. One phase gathers one window per sample and hands it to every statistic, so \
                 they have to agree on the window and on where the samples are; state two \
                 phases for two lattices."
            )));
        }
    }
    Ok(())
}

/// The samples whose centres lie in `[start, end)` on `axis`: a contiguous run
/// of indices, because the positions are increasing.
fn owned(lattice: &SampleLattice, axis: usize, start: usize, end: usize) -> (usize, usize) {
    let positions = lattice.positions(axis);
    let low = positions.partition_point(|&at| at < start);
    let high = positions.partition_point(|&at| at < end);
    (low, high.max(low))
}

fn put_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
    what: &'static str,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8], what: &'static str) -> Self {
        Self { bytes, at: 0, what }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .at
            .checked_add(n)
            .filter(|&end| end <= self.bytes.len());
        let Some(end) = end else {
            return Err(Error::InvalidArgument(format!(
                "{}: truncated at byte {} of {}",
                self.what,
                self.at,
                self.bytes.len()
            )));
        };
        let slice = &self.bytes[self.at..end];
        self.at = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("eight bytes"),
        ))
    }

    fn usize(&mut self) -> Result<usize> {
        usize::try_from(self.u64()?).map_err(|_| {
            Error::InvalidArgument(format!("{}: a count does not fit this machine", self.what))
        })
    }

    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.u64()?))
    }

    fn finish(&self) -> Result<()> {
        if self.at != self.bytes.len() {
            return Err(Error::InvalidArgument(format!(
                "{}: {} trailing byte(s)",
                self.what,
                self.bytes.len() - self.at
            )));
        }
        Ok(())
    }
}

// ------------------------------------------------------------- samples --

/// Evaluate one or more statistics over a window at every lattice point whose
/// centre lies in this block's core, and emit them as the block's fragment.
///
/// The phase reads no pixels of its own input: the image the statistics are
/// over and the population are declared source inputs, read at the element's
/// reach — not the lattice distance plus the element the fused op needs. See
/// the module header for why that is the whole saving.
///
/// Several statistics over **one** window — an equalisation's low and high
/// percentiles — share the gather. Each is handed its own copy of the window,
/// in the element's order, because a selection may permute what it is handed
/// and a summation's bits depend on the order.
pub struct LatticeSamplesOp {
    name: &'static str,
    stream: String,
    image: ImageId,
    mask: Option<ImageId>,
    population: Population,
    statistics: Vec<LocalStatistic>,
    lifecycle: Lifecycle,
    cost: f64,
}

impl LatticeSamplesOp {
    /// Sample `statistics` of `image`. They must share the element and the
    /// sampling; their alignments and narrowings may differ.
    pub fn new(
        name: &'static str,
        stream: impl Into<String>,
        image: impl Into<ImageId>,
        statistics: Vec<LocalStatistic>,
    ) -> Result<Self> {
        check_statistics(name, &statistics)?;
        let cost = statistics
            .iter()
            .map(|statistic| {
                let window = statistic.element().len();
                let per_sample = SAMPLE_COST_PER_ELEMENT_VOXEL * window as f64
                    + statistic.statistic().cost_per_sample(window);
                per_sample * statistic.sampling().samples_per_voxel().unwrap_or(1.0)
            })
            .sum();
        Ok(Self {
            name,
            stream: stream.into(),
            image: image.into(),
            mask: None,
            population: Population::new(),
            statistics,
            lifecycle: Lifecycle::DeleteOnExit,
            cost,
        })
    }

    /// Read each window's population from `mask`, a `Bool` image; see
    /// `LocalStatisticOp::masked_by`, whose rule this is.
    pub fn masked_by(mut self, mask: impl Into<ImageId>) -> Self {
        self.mask = Some(mask.into());
        self.cost *= MASK_COST_FACTOR;
        self
    }

    /// See [`Population`]. Meaningful only alongside [`Self::masked_by`].
    pub fn with_population(mut self, population: Population) -> Self {
        self.population = population;
        self
    }

    /// Keep the fragments after the run rather than discarding them.
    pub fn with_lifecycle(mut self, lifecycle: Lifecycle) -> Self {
        self.lifecycle = lifecycle;
        self
    }

    pub fn statistics(&self) -> &[LocalStatistic] {
        &self.statistics
    }

    pub fn stream(&self) -> &str {
        &self.stream
    }

    fn element(&self) -> &StructuringElement {
        self.statistics[0].element()
    }

    fn sampling(&self) -> &Sampling {
        self.statistics[0].sampling()
    }

    /// The element's own sides, which is all a sample owned by the core reads
    /// beyond it.
    fn window(&self) -> Reach {
        let element = self.element();
        Reach::asymmetric([element.sides(0), element.sides(1), element.sides(2)])
    }

    fn evaluate(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<Vec<u8>> {
        let volume = at.volume();
        let lattice = SampleLattice::of(self.sampling(), volume)?;
        let mut ranges = [(0usize, 0usize); 3];
        for (axis, range) in ranges.iter_mut().enumerate() {
            *range = owned(
                &lattice,
                axis,
                at.core.start[axis],
                at.core.start[axis] + at.core.shape[axis],
            );
        }
        let counts = ranges.map(|(low, high)| high - low);
        let k = self.statistics.len();

        let image = sources.get(self.image.index())?;
        let mask = match self.mask {
            Some(mask) => Some(sources.get(mask.index())?),
            None => None,
        };
        let simulated = !matches!(image, BlockBuf::Array(_))
            || mask.is_some_and(|mask| !matches!(mask, BlockBuf::Array(_)));

        let mut bytes =
            Vec::with_capacity(8 + 1 + 8 * 7 + 8 * k * counts.iter().product::<usize>());
        put_u64(&mut bytes, SAMPLES_MAGIC);
        bytes.push(if simulated { SIMULATED } else { EVALUATED });
        put_u64(&mut bytes, k as u64);
        for axis in 0..3 {
            put_u64(&mut bytes, ranges[axis].0 as u64);
        }
        for count in counts {
            put_u64(&mut bytes, count as u64);
        }
        if simulated || counts.contains(&0) {
            return Ok(bytes);
        }

        // The source buffers were read at the block's read extent.
        let input = image.as_array()?.widened().mapv(Total);
        let mask_array = match mask {
            Some(mask) => Some(mask.as_array()?.view::<bool>()?),
            None => None,
        };
        if let Some(mask) = &mask_array {
            if mask.shape() != input.shape() {
                return Err(Error::InvalidArgument(format!(
                    "op {:?}: the population is {:?} and the image is {:?}; both are read at \
                     the block's own read extent",
                    self.name,
                    mask.shape(),
                    input.shape()
                )));
            }
        }
        let input_view = input.view();
        let element = self.element();
        let full = element.len();
        let needs_centre = mask_array.is_some()
            && (matches!(self.population.centre, ExcludedCentre::Fill(_))
                || self.population.empty == EmptyPopulation::Centre);
        let statistics: Vec<Statistic> = self.statistics.iter().map(|s| s.statistic()).collect();
        let narrowings: Vec<LatticeNarrowing> =
            self.statistics.iter().map(|s| s.narrowing()).collect();
        let mut window: Vec<Total> = Vec::with_capacity(full);
        let mut copy: Vec<Total> = Vec::with_capacity(if k > 1 { full } else { 0 });
        let mut offsets: Vec<[isize; 3]> = Vec::with_capacity(full);
        let mut scratch: Vec<f64> = Vec::new();
        let mut values = vec![0.0f64; k];
        let widen = |value: Total| -> f64 { value.into() };
        for p in ranges[0].0..ranges[0].1 {
            for q in ranges[1].0..ranges[1].1 {
                for r in ranges[2].0..ranges[2].1 {
                    let centre = [
                        lattice.centre(0, p) as isize,
                        lattice.centre(1, q) as isize,
                        lattice.centre(2, r) as isize,
                    ];
                    // The first statistic gathers; the others reduce a copy of
                    // what it gathered, taken before it reduced. The gather is
                    // the shared kernel's, so a sample is the fused op's sample.
                    let mut gathered = false;
                    let first = sample_value(
                        &input_view,
                        mask_array.as_ref(),
                        &at.at,
                        element,
                        centre,
                        self.population,
                        needs_centre,
                        &widen,
                        &mut |window: &mut [Total]| {
                            gathered = true;
                            if k > 1 {
                                copy.clear();
                                copy.extend_from_slice(window);
                            }
                            statistics[0].reduce_with(window, full, &mut scratch)
                        },
                        &mut window,
                        &mut offsets,
                    );
                    values[0] = first;
                    for (index, statistic) in statistics.iter().enumerate().skip(1) {
                        values[index] = if gathered {
                            window.clear();
                            window.extend_from_slice(&copy);
                            statistic.reduce_with(&mut window, full, &mut scratch)
                        } else {
                            // Not reduced at all: a filled centre or a carried
                            // one, which is the same value for every statistic,
                            // exactly as each fused op would have written it.
                            first
                        };
                    }
                    for (value, narrowing) in values.iter().zip(&narrowings) {
                        let value = match narrowing.at_samples {
                            Some(narrowing) => narrowing.apply(*value),
                            None => *value,
                        };
                        put_u64(&mut bytes, value.to_bits());
                    }
                }
            }
        }
        Ok(bytes)
    }
}

impl FragmentOp for LatticeSamplesOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// Nothing: the phase writes no voxels, and what a sample reads beyond the
    /// core is stated as the source inputs' reach.
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        // Not tight: one sample per core voxel is always true and bounds
        // nothing a spacing of more than one would write.
        let per_sample = 8 * self.statistics.len() as u64;
        vec![
            FragmentOutput::new(self.stream.clone(), self.lifecycle, Coverage::EveryBlock).sized(
                SidecarSize::Terms {
                    fixed: 8 + 1 + 8 * 7,
                    per_core_voxel: per_sample as f64,
                    per_read_voxel: 0.0,
                    per_face_voxel: 0.0,
                    tight: false,
                },
            ),
        ]
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        let mut inputs = vec![SourceInput::new(self.image, self.window())];
        if let Some(mask) = self.mask {
            inputs.push(SourceInput::new(mask, self.window()).holding(Dtype::Bool));
        }
        inputs
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::PerBlock)
    }

    /// The image widened to `f64` and then copied as [`Total`] — both alive
    /// while the copy is made — over the read extent the element widens the
    /// core to; one window's gather (its values, a copy for a second
    /// statistic, its offsets and the reducer's scratch); and the fragment.
    fn scratch_bytes(&self, read: [usize; 3]) -> u64 {
        let element = self.element();
        let voxels: u64 = (0..3)
            .map(|axis| {
                let (lo, hi) = element.sides(axis);
                (read[axis] + lo + hi) as u64
            })
            .product();
        let window = element.len() as u64 * (8 + 8 + 24 + 8);
        let core = read.iter().map(|&len| len as f64).product::<f64>();
        let per_voxel = self.sampling().samples_per_voxel().unwrap_or(1.0);
        let fragment = (8.0 * self.statistics.len() as f64 * core * per_voxel).ceil() as u64;
        voxels * 16 + window + fragment
    }

    fn apply(&self, _at: &BlockView<'_>) -> Result<BlockOutput> {
        Err(Error::InvalidArgument(format!(
            "fragment op {:?} reads its image as a declared source and must be applied through \
             `apply_with`",
            self.name
        )))
    }

    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        Ok(BlockOutput::fragment(
            self.stream.clone(),
            self.evaluate(at, sources)?,
        ))
    }
}

// ------------------------------------------------------------- boxes --

/// A box of lattice samples, `k` statistics each, in sample order: what a
/// samples block writes for the samples it owns, and what a gather block writes
/// for the samples its core brackets. One encoding for both.
struct SampleBox {
    simulated: bool,
    k: usize,
    start: [usize; 3],
    count: [usize; 3],
    values: Vec<f64>,
}

impl SampleBox {
    fn simulated(k: usize) -> Self {
        Self {
            simulated: true,
            k,
            start: [0; 3],
            count: [0; 3],
            values: Vec::new(),
        }
    }

    fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(8 + 1 + 8 * 7 + 8 * self.values.len());
        put_u64(&mut bytes, SAMPLES_MAGIC);
        bytes.push(if self.simulated { SIMULATED } else { EVALUATED });
        put_u64(&mut bytes, self.k as u64);
        for value in self.start.iter().chain(&self.count) {
            put_u64(&mut bytes, *value as u64);
        }
        if !self.simulated {
            for value in &self.values {
                put_u64(&mut bytes, value.to_bits());
            }
        }
        bytes
    }

    fn decode(bytes: &[u8], what: &'static str) -> Result<Self> {
        let mut reader = Reader::new(bytes, what);
        if reader.u64()? != SAMPLES_MAGIC {
            return Err(Error::InvalidArgument(format!(
                "{what}: wrong magic; this is not a lattice-samples fragment"
            )));
        }
        let simulated = reader.u8()? == SIMULATED;
        let k = reader.usize()?;
        let start = [reader.usize()?, reader.usize()?, reader.usize()?];
        let count = [reader.usize()?, reader.usize()?, reader.usize()?];
        let mut values = Vec::new();
        if !simulated {
            let n = k * count.iter().product::<usize>();
            values.reserve(n);
            for _ in 0..n {
                values.push(reader.f64()?);
            }
        }
        reader.finish()?;
        Ok(Self {
            simulated,
            k,
            start,
            count,
            values,
        })
    }

    /// Statistic `stat` of the sample at local index `at` in the box.
    fn get(&self, at: [usize; 3], stat: usize) -> f64 {
        let flat = (at[0] * self.count[1] + at[1]) * self.count[2] + at[2];
        self.values[flat * self.k + stat]
    }
}

/// The samples the voxels `[start, end)` of `axis` bracket between, as a
/// half-open index range: the same brackets the fused kernel takes.
fn bracketed(
    lattice: &SampleLattice,
    alignment: Alignment,
    axis: usize,
    start: usize,
    end: usize,
) -> (usize, usize) {
    if end <= start {
        return (0, 0);
    }
    let low = alignment.bracket(lattice, axis, start).0;
    let high = alignment.bracket(lattice, axis, end - 1).1;
    (low, high + 1)
}

/// What the gather and interpolation need of a samples op: where the samples
/// are, and per statistic how a voxel is blended from them and narrowed after.
/// Not the statistics themselves — a window of `5 x 200 x 200` offsets is
/// megabytes, and a planner builds these ops once per candidate grid.
#[derive(Clone)]
struct Blend {
    stream: String,
    sampling: Sampling,
    statistics: Vec<(Alignment, LatticeNarrowing)>,
}

impl Blend {
    fn of(samples: &LatticeSamplesOp) -> Self {
        Self {
            stream: samples.stream.clone(),
            sampling: samples.sampling().clone(),
            statistics: samples
                .statistics
                .iter()
                .map(|statistic| (statistic.alignment(), statistic.narrowing()))
                .collect(),
        }
    }

    /// The samples any statistic's voxels in `core` bracket: the union over
    /// the alignments, which are the only thing that differs.
    fn needed(&self, lattice: &SampleLattice, core: &crate::region::Region) -> [(usize, usize); 3] {
        let mut ranges = [(usize::MAX, 0usize); 3];
        for (axis, range) in ranges.iter_mut().enumerate() {
            for &(alignment, _) in &self.statistics {
                let (low, high) = bracketed(
                    lattice,
                    alignment,
                    axis,
                    core.start[axis],
                    core.start[axis] + core.shape[axis],
                );
                if high > low {
                    range.0 = range.0.min(low);
                    range.1 = range.1.max(high);
                }
            }
            if range.0 > range.1 {
                *range = (0, 0);
            }
        }
        ranges
    }

    /// The most samples a box for a block whose read extent is `read` holds,
    /// times the bytes a sample takes: the samples it brackets reach at most a
    /// gap past it on each side. Every voxel, on a lattice that is not evenly
    /// spaced, which is the bound the fragment's own size states.
    fn box_bytes(&self, read: [usize; 3]) -> u64 {
        let per_sample = 8 * self.statistics.len() as u64;
        let samples: u64 = match &self.sampling {
            Sampling::Centred { spacing } => (0..3)
                .map(|axis| (read[axis].div_ceil(spacing[axis].max(1)) + 2) as u64)
                .product(),
            _ => read.iter().map(|&len| len as u64).product(),
        };
        // The fixed header and the encoded box, beside the decoded one.
        2 * (8 + 1 + 8 * 7 + per_sample * samples)
    }

    /// How many blocks away, per axis, the samples a block brackets can be
    /// owned on `grid`: the fragment reach the gather declares.
    fn reach(&self, grid: &crate::geometry::BlockGrid) -> Result<[usize; 3]> {
        let volume = grid.volume();
        let edge = grid.block();
        let lattice = SampleLattice::of(&self.sampling, volume)?;
        let mut reach = [0usize; 3];
        for axis in 0..3 {
            let mut start = 0;
            let mut index: usize = 0;
            while start < volume[axis] {
                let end = (start + edge[axis]).min(volume[axis]);
                for &(alignment, _) in &self.statistics {
                    let (low, high) = bracketed(&lattice, alignment, axis, start, end);
                    if high > low {
                        let first = lattice.centre(axis, low) / edge[axis];
                        let last = lattice.centre(axis, high - 1) / edge[axis];
                        reach[axis] = reach[axis]
                            .max(index.saturating_sub(first))
                            .max(last.saturating_sub(index));
                    }
                }
                start = end;
                index += 1;
            }
        }
        Ok(reach)
    }
}

// ------------------------------------------------------------ gather --

/// Hand each block the samples its own voxels bracket: a `fragments ->
/// fragments` step reading the samples phase's fragments over the few blocks
/// around it, and writing one box per block.
///
/// It reads no pixels, so its fragment reach costs scheduling and nothing
/// else; and it is what lets the interpolation read only its own fragment,
/// which needs no barrier — the interpolation of a block waits for the samples
/// around it, not for every phase before it.
pub struct GatherSamplesOp {
    name: &'static str,
    output: String,
    samples_phase: usize,
    blend: Blend,
    reach: [usize; 3],
}

impl FragmentOp for GatherSamplesOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own(&self.blend.stream, self.samples_phase).with_reach(self.reach)]
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        let per_sample = 8 * self.blend.statistics.len() as u64;
        vec![FragmentOutput::new(
            self.output.clone(),
            Lifecycle::DeleteOnExit,
            Coverage::EveryBlock,
        )
        .sized(SidecarSize::Terms {
            fixed: 8 + 1 + 8 * 7,
            // Not tight: the samples a block brackets reach a gap past
            // its core, so one per core voxel is not a bound on a block
            // thinner than the gap — but a lattice holds no more
            // samples than the volume, which the read extent's term
            // states.
            per_core_voxel: 0.0,
            per_read_voxel: per_sample as f64,
            per_face_voxel: 0.0,
            tight: false,
        })]
    }

    /// The box it writes and the per-sample tally, beside the decoded boxes of
    /// the neighbours it reads, one at a time.
    fn scratch_bytes(&self, read: [usize; 3]) -> u64 {
        3 * self.blend.box_bytes(read)
    }

    /// A function of the set of fragments: each sample is placed by its own
    /// index. The executor checks it by gathering the neighbourhood reversed.
    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::Unordered)
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let k = self.blend.statistics.len();
        let lattice = SampleLattice::of(&self.blend.sampling, at.volume())?;
        let ranges = self.blend.needed(&lattice, at.core);
        let start = ranges.map(|(low, _)| low);
        let count = ranges.map(|(low, high)| high - low);
        let n: usize = count.iter().product();
        let mut values = vec![0.0f64; n * k];
        let mut written = vec![0u8; n];
        let mut simulated = false;
        for (key, bytes) in at.fragments(&self.blend.stream) {
            let owned = SampleBox::decode(bytes, "lattice samples fragment")?;
            if owned.simulated {
                simulated = true;
                continue;
            }
            if owned.k != k {
                return Err(Error::InvalidArgument(format!(
                    "{}: block {:?} carries {} statistic(s) and this gather {k}",
                    self.name, key.block, owned.k
                )));
            }
            // The overlap of what the neighbour owns with what this block needs.
            let mut low = [0usize; 3];
            let mut high = [0usize; 3];
            for axis in 0..3 {
                low[axis] = owned.start[axis].max(start[axis]);
                high[axis] = (owned.start[axis] + owned.count[axis]).min(start[axis] + count[axis]);
            }
            if (0..3).any(|axis| high[axis] <= low[axis]) {
                continue;
            }
            for p in low[0]..high[0] {
                for q in low[1]..high[1] {
                    for r in low[2]..high[2] {
                        let local = ((p - start[0]) * count[1] + (q - start[1])) * count[2]
                            + (r - start[2]);
                        let theirs = [p - owned.start[0], q - owned.start[1], r - owned.start[2]];
                        for stat in 0..k {
                            values[local * k + stat] = owned.get(theirs, stat);
                        }
                        written[local] += 1;
                    }
                }
            }
        }
        let gathered = if simulated {
            SampleBox::simulated(k)
        } else {
            if let Some(missing) = written.iter().position(|&times| times != 1) {
                return Err(Error::InvalidArgument(format!(
                    "{}: block {:?} brackets sample {missing} of its box {start:?}+{count:?}, and \
                     it arrived {} time(s) from the blocks within a reach of {:?}. Every sample \
                     is owned by the one block whose core holds its centre; the reach is \
                     derived from the grid the plan was built on, so the samples phase and this \
                     one must be cut on that grid.",
                    self.name, at.index, written[missing], self.reach
                )));
            }
            SampleBox {
                simulated: false,
                k,
                start,
                count,
                values,
            }
        };
        Ok(BlockOutput::fragment(
            self.output.clone(),
            gathered.encode(),
        ))
    }
}

// -------------------------------------------------------- interpolation --

/// Interpolate the samples [`LatticeSamplesOp`] wrote back to every voxel, and
/// combine them there with the voxels of stated source images.
///
/// Each block reads its own box of samples — the ones its voxels bracket,
/// gathered by [`GatherSamplesOp`] — so its reach is zero, it reads its sources
/// voxelwise, and it waits for nothing but the gather blocks it needs. It then
/// writes `combine(sources..., statistics...)` at every voxel.
///
/// The combining function is what this adds to a lattice interpolation, and it
/// is the reason the phase is a fragment phase: the voxels the statistic is
/// combined with are read in the same pass, so no full-resolution copy of the
/// interpolated statistic is ever written.
pub struct InterpolateSamplesOp {
    name: &'static str,
    gathered: String,
    gather_phase: usize,
    blend: Blend,
    sources: Vec<ImageId>,
    combine: SampleCombine,
    output: Dtype,
    cost: f64,
}

impl InterpolateSamplesOp {
    /// Write `dtype` instead of `f64`, converting each combined value with that
    /// type's own `from_f64` — the conversion `FnCombine` writes with.
    pub fn producing(mut self, dtype: Dtype) -> Result<Self> {
        if dtype == Dtype::F16 {
            return Err(Error::InvalidArgument(format!(
                "{}: cannot write f16 voxels",
                self.name
            )));
        }
        self.output = dtype;
        Ok(self)
    }

    fn write(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let mut buffer = at.output_buffer(0.0)?;
        let Some(out) = buffer.as_array_mut() else {
            return Ok(BlockOutput::nothing().with_pixels(buffer));
        };
        let own = at.own(&self.gathered).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "{}: block {:?} has no gathered samples",
                self.name, at.index
            ))
        })?;
        let samples = SampleBox::decode(own, "gathered lattice samples")?;
        if samples.simulated {
            return Err(Error::InvalidArgument(format!(
                "{}: the samples were simulated and this block holds data",
                self.name
            )));
        }
        let lattice = SampleLattice::of(&self.blend.sampling, at.volume())?;
        let shape = out.shape();
        let origin = at.at.offset;
        let held: Vec<&Voxels> = self
            .sources
            .iter()
            .map(|image| -> Result<&Voxels> {
                let voxels = sources.get(image.index())?.as_array()?;
                if voxels.shape() != shape {
                    return Err(Error::InvalidArgument(format!(
                        "{}: source image {} is {:?} and the block is {shape:?}; a source is \
                         read voxelwise",
                        self.name,
                        image.index(),
                        voxels.shape()
                    )));
                }
                Ok(voxels)
            })
            .collect::<Result<_>>()?;
        let brackets: Vec<Brackets> = self
            .blend
            .statistics
            .iter()
            .map(|&(alignment, _)| brackets(&samples, &lattice, alignment, origin, shape))
            .collect::<Result<_>>()?;
        // **One plane at a time**, so that what a block holds beyond its own
        // buffers is a plane of each widened source, statistic and combined
        // value rather than a block of each ([`Self::scratch_bytes`]). Every
        // voxel is still computed from the same numbers in the same order.
        let plane = shape[1] * shape[2];
        let mut columns: Vec<Vec<f64>> = vec![Vec::with_capacity(plane); held.len()];
        let mut interpolated: Vec<Vec<f64>> = vec![Vec::with_capacity(plane); brackets.len()];
        let mut values: Vec<f64> = Vec::with_capacity(plane);
        let mut arguments = vec![0.0; held.len() + brackets.len()];
        for i in 0..shape[0] {
            for (column, voxels) in columns.iter_mut().zip(&held) {
                widen_plane(voxels, i, column)?;
            }
            for (stat, (column, brackets)) in interpolated.iter_mut().zip(&brackets).enumerate() {
                interpolate_plane(
                    &samples,
                    stat,
                    brackets,
                    self.blend.statistics[stat].1,
                    i,
                    column,
                );
            }
            values.clear();
            values.extend((0..plane).map(|v| {
                for (slot, column) in arguments.iter_mut().zip(&columns) {
                    *slot = column[v];
                }
                for (slot, column) in arguments[columns.len()..].iter_mut().zip(&interpolated) {
                    *slot = column[v];
                }
                (self.combine)(&arguments)
            }));
            write_plane(out, i, &values, self.name)?;
        }
        Ok(BlockOutput::nothing().with_pixels(buffer))
    }
}

/// Per axis, each voxel's two bracketing samples as indices into a block's box,
/// and the blend weight between them.
type Brackets = [Vec<(usize, usize, f64)>; 3];

/// The fused kernel's brackets at the fine voxels `origin .. origin + shape`:
/// the same brackets from the same global coordinates.
fn brackets(
    samples: &SampleBox,
    lattice: &SampleLattice,
    alignment: Alignment,
    origin: [usize; 3],
    shape: [usize; 3],
) -> Result<Brackets> {
    let mut brackets: Brackets = Default::default();
    for (axis, along) in brackets.iter_mut().enumerate() {
        along.reserve(shape[axis]);
        for step in 0..shape[axis] {
            let (a, b, t) = alignment.bracket(lattice, axis, origin[axis] + step);
            let low = samples.start[axis];
            let high = low + samples.count[axis];
            if a < low || b >= high {
                return Err(Error::InvalidArgument(format!(
                    "lattice interpolation: voxel {} of axis {axis} brackets samples {a}..={b} \
                     and the block holds {low}..{high}",
                    origin[axis] + step
                )));
            }
            along.push((a - low, b - low, t));
        }
    }
    Ok(brackets)
}

/// The fused kernel's interpolation at plane `i` of a block, in row-major
/// order: blended along `x`, then `y`, then `z`, then narrowed.
fn interpolate_plane(
    samples: &SampleBox,
    stat: usize,
    brackets: &Brackets,
    narrowing: LatticeNarrowing,
    i: usize,
    out: &mut Vec<f64>,
) {
    out.clear();
    let (a0, b0, t0) = brackets[0][i];
    for &(a1, b1, t1) in &brackets[1] {
        for &(a2, b2, t2) in &brackets[2] {
            let at_edge = |p: usize, q: usize| {
                lerp(
                    samples.get([p, q, a2], stat),
                    samples.get([p, q, b2], stat),
                    t2,
                )
            };
            let at_face = |p: usize| lerp(at_edge(p, a1), at_edge(p, b1), t1);
            out.push(lerp(at_face(a0), at_face(b0), t0));
        }
    }
    if let Some(narrowing) = narrowing.after_interpolation {
        for value in out.iter_mut() {
            *value = narrowing.apply(*value);
        }
    }
}

/// `a + t * (b - a)`, the fused kernel's blend.
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

fn widen_plane_as<T: VoxelElement>(voxels: &Voxels, i: usize, out: &mut Vec<f64>) -> Result<()> {
    let view = voxels.view::<T>()?;
    out.clear();
    out.extend(
        view.index_axis(Axis(0), i)
            .iter()
            .map(|&value| value.into_f64()),
    );
    Ok(())
}

/// Plane `i` of `voxels`, widened to `f64` the way [`Voxels::widened`] widens.
fn widen_plane(voxels: &Voxels, i: usize, out: &mut Vec<f64>) -> Result<()> {
    match voxels.dtype() {
        Dtype::Bool => widen_plane_as::<bool>(voxels, i, out),
        Dtype::U8 => widen_plane_as::<u8>(voxels, i, out),
        Dtype::U16 => widen_plane_as::<u16>(voxels, i, out),
        Dtype::U32 => widen_plane_as::<u32>(voxels, i, out),
        Dtype::U64 => widen_plane_as::<u64>(voxels, i, out),
        Dtype::I8 => widen_plane_as::<i8>(voxels, i, out),
        Dtype::I16 => widen_plane_as::<i16>(voxels, i, out),
        Dtype::I32 => widen_plane_as::<i32>(voxels, i, out),
        Dtype::I64 => widen_plane_as::<i64>(voxels, i, out),
        Dtype::F32 => widen_plane_as::<f32>(voxels, i, out),
        Dtype::F64 => widen_plane_as::<f64>(voxels, i, out),
        Dtype::F16 => Err(Error::InvalidArgument(
            "lattice interpolation: cannot read f16 voxels".into(),
        )),
    }
}

fn write_plane_as<T: VoxelElement>(out: &mut Voxels, i: usize, values: &[f64]) -> Result<()> {
    let mut view = out.view_mut::<T>()?;
    for (slot, &value) in view.index_axis_mut(Axis(0), i).iter_mut().zip(values) {
        *slot = T::from_f64(value);
    }
    Ok(())
}

/// Write plane `i` of `out` from `values`, converting each with the element
/// type's own `from_f64` — the conversion `FnCombine` writes with.
fn write_plane(out: &mut Voxels, i: usize, values: &[f64], name: &str) -> Result<()> {
    match out.dtype() {
        Dtype::Bool => write_plane_as::<bool>(out, i, values),
        Dtype::U8 => write_plane_as::<u8>(out, i, values),
        Dtype::U16 => write_plane_as::<u16>(out, i, values),
        Dtype::U32 => write_plane_as::<u32>(out, i, values),
        Dtype::U64 => write_plane_as::<u64>(out, i, values),
        Dtype::I8 => write_plane_as::<i8>(out, i, values),
        Dtype::I16 => write_plane_as::<i16>(out, i, values),
        Dtype::I32 => write_plane_as::<i32>(out, i, values),
        Dtype::I64 => write_plane_as::<i64>(out, i, values),
        Dtype::F32 => write_plane_as::<f32>(out, i, values),
        Dtype::F64 => write_plane_as::<f64>(out, i, values),
        Dtype::F16 => Err(Error::InvalidArgument(format!(
            "{name}: cannot write f16 voxels"
        ))),
    }
}

impl FragmentOp for InterpolateSamplesOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// Nothing: a block's voxels need its own box of samples and its own
    /// sources' voxels, and nothing of a neighbour's.
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }

    fn writes_pixels(&self) -> bool {
        true
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        self.output
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own(&self.gathered, self.gather_phase)]
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        self.sources
            .iter()
            .map(|&image| SourceInput::voxelwise(image))
            .collect()
    }

    /// One plane of each widened source, each interpolated statistic and the
    /// combined values; the brackets; and its box of samples, encoded and
    /// decoded. The block's sources and output are the phase's own buffers.
    fn scratch_bytes(&self, read: [usize; 3]) -> u64 {
        let plane = (read[1] * read[2]) as u64;
        let columns = (self.sources.len() + self.blend.statistics.len() + 1) as u64;
        let brackets = 24 * self.blend.statistics.len() as u64 * read.iter().sum::<usize>() as u64;
        plane * 8 * columns + brackets + self.blend.box_bytes(read)
    }

    /// A function of this block's own fragment and nothing else.
    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::Unordered)
    }

    fn apply(&self, _at: &BlockView<'_>) -> Result<BlockOutput> {
        Err(Error::InvalidArgument(format!(
            "fragment op {:?} reads its operands as declared sources and must be applied \
             through `apply_with`",
            self.name
        )))
    }

    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        self.write(at, sources)
    }
}

/// Append the three phases — the samples, the gather, and the interpolation
/// combined with `sources` — and return the last, which writes the image.
///
/// All three are cut on the builder's current grid, which the gather's reach is
/// derived from: the second and third read the first's fragments by block
/// index.
///
/// `combine` is handed `sources.len() + statistics` values per voxel: the
/// sources in the order named, widened to `f64`, then the statistics in the
/// order `samples` states them. The image holds `output`.
pub fn append_lattice_sample_phases(
    plan: &mut PlanBuilder,
    samples: LatticeSamplesOp,
    interpolate_name: &'static str,
    sources: Vec<ImageId>,
    combine: SampleCombine,
    output: Dtype,
) -> Result<Phase> {
    // What the later phases need of the samples op, taken before the builder
    // takes the op itself.
    let blend = Blend::of(&samples);
    let reach = blend.reach(plan.grid())?;
    let k = blend.statistics.len();
    let gathered = format!("{}.gathered", blend.stream);
    let samples_phase = plan.fragments(samples)?;
    let gather = plan.fragments(GatherSamplesOp {
        name: "gather lattice samples",
        output: gathered.clone(),
        samples_phase: samples_phase.index(),
        blend: blend.clone(),
        reach,
    })?;
    let interpolate = InterpolateSamplesOp {
        name: interpolate_name,
        gathered,
        gather_phase: gather.index(),
        blend,
        sources,
        combine,
        output: Dtype::F64,
        cost: INTERPOLATION_COST * k as f64 + COMBINE_COST,
    }
    .producing(output)?;
    plan.fragments(interpolate)
}
