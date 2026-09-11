// SPDX-License-Identifier: MIT
//
// Original work for this crate.

//! Global exposure and contrast helpers.
//!
//! These functions are data-level rules: they build one global histogram and
//! return either a transformed vector or a scalar decision. A future
//! planner-visible histogram phase should call these same rules rather than
//! carry a second definition of equalization or matching.

use ndarray::{ArrayView3, ArrayViewMut3};

use crate::assemble::{Phase, PlanBuilder};
use crate::dtype::Dtype;
use crate::env::BlockBuf;
use crate::error::{Error, Result};
use crate::fragment::{
    BlockOutput, BlockView, FragmentInput, FragmentOp, PhaseView, SeamFold, SourceBlocks,
};
use crate::op::SourceInput;
use crate::sidecar::Lifecycle;
use crate::voxels::Voxels;

use super::histogram::{decode_finite_samples, BytesCursor, FiniteHistogram, FiniteSamplesOp};
use super::shapes_agree;

const EQUALIZATION_MAP_MAGIC: u64 = 0x4551_4849_5354_4d50;
const CLAHE_SAMPLES_MAGIC: u64 = 0x434c_4148_4553_414d;
const CLAHE_MAPS_MAGIC: u64 = 0x434c_4148_454d_4150;

/// Histogram-equalize finite values to `[0, 1]`.
///
/// Non-finite values are copied through unchanged. A constant finite image maps
/// to `0.0`, because no contrast can be recovered from it and inventing a high
/// value would make an all-constant image look meaningful to later thresholds.
pub fn equalize_histogram(values: &[f64], bins: usize) -> Result<Vec<f64>> {
    let map = EqualizationMap::from_values(values, bins)?;
    Ok(values.iter().map(|&value| map.apply(value)).collect())
}

/// Phase 1 of planner-visible global histogram equalization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EqualizeHistogramOp {
    name: &'static str,
    stream: String,
    samples_phase: usize,
    image: usize,
    bins: usize,
}

impl EqualizeHistogramOp {
    pub fn new(
        name: &'static str,
        stream: impl Into<String>,
        samples_phase: usize,
        bins: usize,
    ) -> Result<Self> {
        if bins == 0 {
            return Err(Error::InvalidArgument(
                "histogram equalization needs at least one bin".to_string(),
            ));
        }
        Ok(Self {
            name,
            stream: stream.into(),
            samples_phase,
            image: 0,
            bins,
        })
    }

    pub fn reading_image(mut self, image: usize) -> Self {
        self.image = image;
        self
    }

    pub fn bins(&self) -> usize {
        self.bins
    }
}

impl FragmentOp for EqualizeHistogramOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn writes_pixels(&self) -> bool {
        true
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::F64
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own(&self.stream, self.samples_phase)]
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        vec![SourceInput::voxelwise(self.image)]
    }

    fn barrier(&self) -> bool {
        true
    }

    fn gathers(&self) -> bool {
        false
    }

    fn reduce(&self, at: &PhaseView<'_>) -> Result<Vec<u8>> {
        let mut values = Vec::new();
        at.stream_fragments(&self.stream, &mut |_key, bytes| {
            values.extend(decode_finite_samples(
                bytes,
                "global histogram equalization samples",
            )?);
            Ok(())
        })?;
        EqualizationMap::from_values(&values, self.bins)?.encode()
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::Unordered)
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let declared = self.source_inputs(at.volume());
        Err(Error::InvalidArgument(format!(
            "fragment op {:?} declares source image {} and must be applied through \
             `apply_with`; the phase input is not the image being equalized",
            self.name(),
            declared[0].image.index()
        )))
    }

    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let map = EqualizationMap::decode(at.reduced)?;
        let pixels = sources.get(self.image)?.as_array()?;
        let mut buffer = at.output_buffer(0.0)?;
        if let Some(out) = buffer.as_array_mut() {
            apply_equalization_map(pixels, &map, out)?;
        }
        Ok(BlockOutput::nothing().with_pixels(buffer))
    }
}

/// Append finite-sample collection and global histogram equalization.
pub fn append_equalize_histogram_phases(
    plan: &mut PlanBuilder,
    stream: impl Into<String>,
    bins: usize,
) -> Result<Phase> {
    let stream = stream.into();
    let samples = plan.fragments(FiniteSamplesOp::new(
        "global histogram samples",
        stream.clone(),
        Lifecycle::DeleteOnExit,
    ))?;
    plan.fragments(
        EqualizeHistogramOp::new("equalize histogram", stream, samples.index(), bins)?
            .reading_image(samples.index()),
    )
}

pub struct ClaheTileSamplesOp {
    name: &'static str,
    stream: String,
    lifecycle: Lifecycle,
}

impl ClaheTileSamplesOp {
    pub fn new(name: &'static str, stream: impl Into<String>, lifecycle: Lifecycle) -> Self {
        Self {
            name,
            stream: stream.into(),
            lifecycle,
        }
    }
}

impl FragmentOp for ClaheTileSamplesOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn reads_pixels(&self) -> bool {
        true
    }

    fn writes_pixels(&self) -> bool {
        false
    }

    fn outputs(&self) -> Vec<crate::fragment::FragmentOutput> {
        vec![crate::fragment::FragmentOutput::new(
            self.stream.clone(),
            self.lifecycle,
            crate::fragment::Coverage::EveryBlock,
        )
        .sized(crate::fragment::SidecarSize::per_read_voxel(16, 32))]
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::PerBlock)
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let BlockBuf::Array(pixels) = at.pixels()? else {
            return Ok(BlockOutput::fragment(
                self.stream.clone(),
                encode_clahe_samples(&[]),
            ));
        };
        let values = pixels.widened();
        let mut samples = Vec::new();
        for ((i, j, k), &value) in values.indexed_iter() {
            if value.is_finite() {
                samples.push((
                    [
                        at.at.offset[0] + i,
                        at.at.offset[1] + j,
                        at.at.offset[2] + k,
                    ],
                    value,
                ));
            }
        }
        Ok(BlockOutput::fragment(
            self.stream.clone(),
            encode_clahe_samples(&samples),
        ))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EqualizeAdapthistOp {
    name: &'static str,
    stream: String,
    samples_phase: usize,
    image: usize,
    tile_shape: [usize; 3],
    bins: usize,
    clip_limit: f64,
}

impl EqualizeAdapthistOp {
    pub fn new(
        name: &'static str,
        stream: impl Into<String>,
        samples_phase: usize,
        tile_shape: [usize; 3],
        bins: usize,
        clip_limit: f64,
    ) -> Result<Self> {
        validate_adapthist(tile_shape, bins, clip_limit)?;
        Ok(Self {
            name,
            stream: stream.into(),
            samples_phase,
            image: 0,
            tile_shape,
            bins,
            clip_limit,
        })
    }

    pub fn reading_image(mut self, image: usize) -> Self {
        self.image = image;
        self
    }
}

impl FragmentOp for EqualizeAdapthistOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn writes_pixels(&self) -> bool {
        true
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::F64
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own(&self.stream, self.samples_phase)]
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        vec![SourceInput::voxelwise(self.image)]
    }

    fn barrier(&self) -> bool {
        true
    }

    fn gathers(&self) -> bool {
        false
    }

    fn reduce(&self, at: &PhaseView<'_>) -> Result<Vec<u8>> {
        let mut samples = Vec::new();
        at.stream_fragments(&self.stream, &mut |_key, bytes| {
            samples.extend(decode_clahe_samples(bytes)?);
            Ok(())
        })?;
        ClaheMapSet::from_samples(
            at.grid.volume(),
            self.tile_shape,
            self.bins,
            self.clip_limit,
            &samples,
        )?
        .encode()
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::Unordered)
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let declared = self.source_inputs(at.volume());
        Err(Error::InvalidArgument(format!(
            "fragment op {:?} declares source image {} and must be applied through \
             `apply_with`; the phase input is not the image being adaptive-equalized",
            self.name(),
            declared[0].image.index()
        )))
    }

    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let maps = ClaheMapSet::decode(at.reduced)?;
        let pixels = sources.get(self.image)?.as_array()?.widened();
        let mut buffer = at.output_buffer(0.0)?;
        let out = buffer.as_array_mut().ok_or_else(|| {
            Error::InvalidArgument(
                "adaptive histogram equalization expected an array output".into(),
            )
        })?;
        let mut out = out.view_mut::<f64>()?;
        for ((i, j, k), slot) in out.indexed_iter_mut() {
            let value = pixels[[i, j, k]];
            *slot = maps.apply(
                [
                    at.at.offset[0] + i,
                    at.at.offset[1] + j,
                    at.at.offset[2] + k,
                ],
                value,
            );
        }
        Ok(BlockOutput::nothing().with_pixels(buffer))
    }
}

pub fn append_equalize_adapthist_phases(
    plan: &mut PlanBuilder,
    stream: impl Into<String>,
    lifecycle: Lifecycle,
    tile_shape: [usize; 3],
    bins: usize,
    clip_limit: f64,
) -> Result<Phase> {
    let stream = stream.into();
    let samples = plan.fragments(ClaheTileSamplesOp::new(
        "adaptive histogram samples",
        stream.clone(),
        lifecycle,
    ))?;
    plan.fragments(
        EqualizeAdapthistOp::new(
            "adaptive histogram equalization",
            stream,
            samples.index(),
            tile_shape,
            bins,
            clip_limit,
        )?
        .reading_image(samples.index()),
    )
}

#[derive(Debug, Clone, PartialEq)]
struct EqualizationMap {
    minimum: f64,
    width: f64,
    cdf: Vec<f64>,
}

impl EqualizationMap {
    fn from_values(values: &[f64], bins: usize) -> Result<Self> {
        let histogram = FiniteHistogram::new(values, bins, "histogram equalization")?;
        if histogram.constant {
            return Ok(Self {
                minimum: histogram.minimum,
                width: 0.0,
                cdf: vec![0.0],
            });
        }
        Ok(Self {
            minimum: histogram.minimum,
            width: histogram.width(),
            cdf: histogram
                .cumulative_count
                .iter()
                .map(|count| count / histogram.total_count)
                .collect(),
        })
    }

    fn apply(&self, value: f64) -> f64 {
        if !value.is_finite() {
            return value;
        }
        if self.width == 0.0 {
            return 0.0;
        }
        self.cdf[bin_for(value, self.minimum, self.width, self.cdf.len())]
    }

    fn encode(&self) -> Result<Vec<u8>> {
        if self.cdf.is_empty() {
            return Err(Error::InvalidArgument(
                "histogram equalization map needs at least one CDF entry".to_string(),
            ));
        }
        let mut bytes = Vec::with_capacity(32 + self.cdf.len() * 8);
        bytes.extend_from_slice(&EQUALIZATION_MAP_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&(self.cdf.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&self.minimum.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.width.to_bits().to_le_bytes());
        for &value in &self.cdf {
            if !value.is_finite() {
                return Err(Error::InvalidArgument(
                    "histogram equalization map CDF values must be finite".to_string(),
                ));
            }
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        Ok(bytes)
    }

    fn decode(bytes: &[u8]) -> Result<Self> {
        let mut cursor = BytesCursor::new(bytes, "histogram equalization map");
        cursor.expect_magic(EQUALIZATION_MAP_MAGIC)?;
        let bins = cursor.take_u64()? as usize;
        if bins == 0 {
            return Err(Error::InvalidArgument(
                "histogram equalization map carried no CDF entries".to_string(),
            ));
        }
        let minimum = f64::from_bits(cursor.take_u64()?);
        let width = f64::from_bits(cursor.take_u64()?);
        if !minimum.is_finite() || !width.is_finite() || width < 0.0 {
            return Err(Error::InvalidArgument(format!(
                "histogram equalization map has invalid minimum={minimum} width={width}"
            )));
        }
        let mut cdf = Vec::with_capacity(bins);
        for _ in 0..bins {
            let value = f64::from_bits(cursor.take_u64()?);
            if !value.is_finite() {
                return Err(Error::InvalidArgument(
                    "histogram equalization map CDF values must be finite".to_string(),
                ));
            }
            cdf.push(value);
        }
        cursor.expect_end()?;
        Ok(Self {
            minimum,
            width,
            cdf,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
struct ClaheMapSet {
    volume: [usize; 3],
    tile_shape: [usize; 3],
    tiles: [usize; 3],
    minimum: f64,
    width: f64,
    maps: Vec<Vec<f64>>,
}

impl ClaheMapSet {
    fn from_samples(
        volume: [usize; 3],
        tile_shape: [usize; 3],
        bins: usize,
        clip_limit: f64,
        samples: &[([usize; 3], f64)],
    ) -> Result<Self> {
        validate_adapthist(tile_shape, bins, clip_limit)?;
        if samples.is_empty() {
            return Err(Error::InvalidArgument(
                "adaptive histogram equalization needs at least one finite value".to_string(),
            ));
        }
        let minimum = samples
            .iter()
            .map(|(_, value)| *value)
            .fold(f64::INFINITY, f64::min);
        let maximum = samples
            .iter()
            .map(|(_, value)| *value)
            .fold(f64::NEG_INFINITY, f64::max);
        let tiles = [
            volume[0].div_ceil(tile_shape[0]),
            volume[1].div_ceil(tile_shape[1]),
            volume[2].div_ceil(tile_shape[2]),
        ];
        if minimum == maximum {
            return Ok(Self {
                volume,
                tile_shape,
                tiles,
                minimum,
                width: 0.0,
                maps: vec![vec![0.0; bins]; tiles[0] * tiles[1] * tiles[2]],
            });
        }
        let width = (maximum - minimum) / bins as f64;
        let mut counts = vec![vec![0.0f64; bins]; tiles[0] * tiles[1] * tiles[2]];
        for &(at, value) in samples {
            let tile = [
                (at[0] / tile_shape[0]).min(tiles[0] - 1),
                (at[1] / tile_shape[1]).min(tiles[1] - 1),
                (at[2] / tile_shape[2]).min(tiles[2] - 1),
            ];
            let flat = (tile[0] * tiles[1] + tile[1]) * tiles[2] + tile[2];
            counts[flat][bin_for(value, minimum, width, bins)] += 1.0;
        }
        let maps = counts
            .into_iter()
            .map(|counts| tile_mapping(counts, clip_limit))
            .collect();
        Ok(Self {
            volume,
            tile_shape,
            tiles,
            minimum,
            width,
            maps,
        })
    }

    fn apply(&self, at: [usize; 3], value: f64) -> f64 {
        if !value.is_finite() {
            return value;
        }
        if self.width == 0.0 {
            return 0.0;
        }
        let bin = bin_for(value, self.minimum, self.width, self.maps[0].len());
        interpolate_mapping(&self.maps, self.tiles, self.tile_shape, at, bin)
    }

    fn encode(&self) -> Result<Vec<u8>> {
        let bins = self.maps.first().map(|map| map.len()).unwrap_or(0);
        if bins == 0 {
            return Err(Error::InvalidArgument(
                "adaptive histogram map set needs at least one bin".to_string(),
            ));
        }
        let mut bytes = Vec::with_capacity(88 + self.maps.len() * bins * 8);
        bytes.extend_from_slice(&CLAHE_MAPS_MAGIC.to_le_bytes());
        for value in self.volume {
            bytes.extend_from_slice(&(value as u64).to_le_bytes());
        }
        for value in self.tile_shape {
            bytes.extend_from_slice(&(value as u64).to_le_bytes());
        }
        for value in self.tiles {
            bytes.extend_from_slice(&(value as u64).to_le_bytes());
        }
        bytes.extend_from_slice(&(bins as u64).to_le_bytes());
        bytes.extend_from_slice(&self.minimum.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.width.to_bits().to_le_bytes());
        for map in &self.maps {
            if map.len() != bins {
                return Err(Error::InvalidArgument(
                    "adaptive histogram maps disagree on bin count".to_string(),
                ));
            }
            for &value in map {
                if !value.is_finite() {
                    return Err(Error::InvalidArgument(
                        "adaptive histogram map values must be finite".to_string(),
                    ));
                }
                bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
        Ok(bytes)
    }

    fn decode(bytes: &[u8]) -> Result<Self> {
        let mut cursor = BytesCursor::new(bytes, "adaptive histogram map set");
        cursor.expect_magic(CLAHE_MAPS_MAGIC)?;
        let volume = [
            cursor.take_u64()? as usize,
            cursor.take_u64()? as usize,
            cursor.take_u64()? as usize,
        ];
        let tile_shape = [
            cursor.take_u64()? as usize,
            cursor.take_u64()? as usize,
            cursor.take_u64()? as usize,
        ];
        let tiles = [
            cursor.take_u64()? as usize,
            cursor.take_u64()? as usize,
            cursor.take_u64()? as usize,
        ];
        let bins = cursor.take_u64()? as usize;
        let minimum = f64::from_bits(cursor.take_u64()?);
        let width = f64::from_bits(cursor.take_u64()?);
        validate_adapthist(tile_shape, bins, f64::INFINITY)?;
        if !minimum.is_finite() || !width.is_finite() || width < 0.0 {
            return Err(Error::InvalidArgument(format!(
                "adaptive histogram map set has invalid minimum={minimum} width={width}"
            )));
        }
        let maps_len = tiles[0] * tiles[1] * tiles[2];
        let mut maps = Vec::with_capacity(maps_len);
        for _ in 0..maps_len {
            let mut map = Vec::with_capacity(bins);
            for _ in 0..bins {
                let value = f64::from_bits(cursor.take_u64()?);
                if !value.is_finite() {
                    return Err(Error::InvalidArgument(
                        "adaptive histogram map values must be finite".to_string(),
                    ));
                }
                map.push(value);
            }
            maps.push(map);
        }
        cursor.expect_end()?;
        Ok(Self {
            volume,
            tile_shape,
            tiles,
            minimum,
            width,
            maps,
        })
    }
}

fn encode_clahe_samples(samples: &[([usize; 3], f64)]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(16 + samples.len() * 32);
    bytes.extend_from_slice(&CLAHE_SAMPLES_MAGIC.to_le_bytes());
    bytes.extend_from_slice(&(samples.len() as u64).to_le_bytes());
    for &(at, value) in samples {
        for coordinate in at {
            bytes.extend_from_slice(&(coordinate as u64).to_le_bytes());
        }
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    bytes
}

fn decode_clahe_samples(bytes: &[u8]) -> Result<Vec<([usize; 3], f64)>> {
    let mut cursor = BytesCursor::new(bytes, "adaptive histogram samples");
    cursor.expect_magic(CLAHE_SAMPLES_MAGIC)?;
    let count = cursor.take_u64()? as usize;
    let mut samples = Vec::with_capacity(count);
    for _ in 0..count {
        let at = [
            cursor.take_u64()? as usize,
            cursor.take_u64()? as usize,
            cursor.take_u64()? as usize,
        ];
        let value = f64::from_bits(cursor.take_u64()?);
        if !value.is_finite() {
            return Err(Error::InvalidArgument(
                "adaptive histogram samples must be finite".to_string(),
            ));
        }
        samples.push((at, value));
    }
    cursor.expect_end()?;
    Ok(samples)
}

fn apply_equalization_map(input: &Voxels, map: &EqualizationMap, out: &mut Voxels) -> Result<()> {
    let input = input.widened();
    let mut out = out.view_mut::<f64>()?;
    ndarray::Zip::from(&input)
        .and(&mut out)
        .for_each(|&value, slot| {
            *slot = map.apply(value);
        });
    Ok(())
}

/// Match the finite values' histogram to a finite reference distribution.
///
/// The source and reference use their own equal-width histograms. Each source
/// bin is mapped to the first reference bin whose CDF reaches the source bin's
/// CDF, and the reference bin centre is written. Non-finite source values are
/// copied through unchanged.
pub fn match_histogram(values: &[f64], reference: &[f64], bins: usize) -> Result<Vec<f64>> {
    let source = FiniteHistogram::new(values, bins, "histogram matching source")?;
    let target = FiniteHistogram::new(reference, bins, "histogram matching reference")?;
    Ok(values
        .iter()
        .map(|&value| {
            if !value.is_finite() {
                value
            } else if target.constant {
                target.centres[0]
            } else {
                let quantile = source.cdf_at_bin(source.index(value));
                target.centres[target.first_bin_at_or_above(quantile)]
            }
        })
        .collect())
}

/// Whether the central percentile span is small relative to the finite range.
pub fn is_low_contrast(
    values: &[f64],
    lower_percentile: f64,
    upper_percentile: f64,
    fraction_threshold: f64,
) -> Result<bool> {
    if !(lower_percentile.is_finite()
        && upper_percentile.is_finite()
        && fraction_threshold.is_finite())
    {
        return Err(Error::InvalidArgument(
            "low-contrast percentiles and threshold must be finite".to_string(),
        ));
    }
    if lower_percentile < 0.0 || upper_percentile > 100.0 || lower_percentile >= upper_percentile {
        return Err(Error::InvalidArgument(format!(
            "low-contrast percentiles must satisfy 0 <= lower < upper <= 100, got \
             {lower_percentile} and {upper_percentile}"
        )));
    }
    if fraction_threshold < 0.0 {
        return Err(Error::InvalidArgument(format!(
            "a low-contrast fraction threshold must be non-negative, got {fraction_threshold}"
        )));
    }
    let mut finite: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.is_empty() {
        return Err(Error::InvalidArgument(
            "low-contrast detection needs at least one finite value".to_string(),
        ));
    }
    finite.sort_by(|left, right| left.total_cmp(right));
    let range = finite[finite.len() - 1] - finite[0];
    if range == 0.0 {
        return Ok(true);
    }
    let low = percentile(&finite, lower_percentile);
    let high = percentile(&finite, upper_percentile);
    Ok(high - low <= fraction_threshold * range)
}

/// Contrast-limited adaptive histogram equalization over a grayscale volume.
///
/// The histogram range is global, each tile builds a clipped local CDF, and
/// each voxel interpolates the surrounding tile mappings in tile-centre space.
/// Non-finite values are copied through unchanged.
pub fn equalize_adapthist_into(
    input: ArrayView3<'_, f64>,
    tile_shape: [usize; 3],
    bins: usize,
    clip_limit: f64,
    mut out: ArrayViewMut3<'_, f64>,
) -> Result<()> {
    shapes_agree(input.shape(), out.shape(), "equalize_adapthist_into")?;
    validate_adapthist(tile_shape, bins, clip_limit)?;

    let shape = [input.shape()[0], input.shape()[1], input.shape()[2]];
    let finite: Vec<f64> = input
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.is_empty() {
        return Err(Error::InvalidArgument(
            "adaptive histogram equalization needs at least one finite value".to_string(),
        ));
    }
    let minimum = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if minimum == maximum {
        for (slot, &value) in out.iter_mut().zip(input.iter()) {
            *slot = if value.is_finite() { 0.0 } else { value };
        }
        return Ok(());
    }

    let width = (maximum - minimum) / bins as f64;
    let tiles = [
        shape[0].div_ceil(tile_shape[0]),
        shape[1].div_ceil(tile_shape[1]),
        shape[2].div_ceil(tile_shape[2]),
    ];
    let mut maps = Vec::with_capacity(tiles[0] * tiles[1] * tiles[2]);
    for ti in 0..tiles[0] {
        for tj in 0..tiles[1] {
            for tk in 0..tiles[2] {
                let lo = [ti * tile_shape[0], tj * tile_shape[1], tk * tile_shape[2]];
                let hi = [
                    ((ti + 1) * tile_shape[0]).min(shape[0]),
                    ((tj + 1) * tile_shape[1]).min(shape[1]),
                    ((tk + 1) * tile_shape[2]).min(shape[2]),
                ];
                let mut counts = vec![0.0f64; bins];
                for i in lo[0]..hi[0] {
                    for j in lo[1]..hi[1] {
                        for k in lo[2]..hi[2] {
                            let value = input[[i, j, k]];
                            if value.is_finite() {
                                counts[bin_for(value, minimum, width, bins)] += 1.0;
                            }
                        }
                    }
                }
                maps.push(tile_mapping(counts, clip_limit));
            }
        }
    }

    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let value = input[[i, j, k]];
                if !value.is_finite() {
                    out[[i, j, k]] = value;
                    continue;
                }
                let bin = bin_for(value, minimum, width, bins);
                out[[i, j, k]] = interpolate_mapping(&maps, tiles, tile_shape, [i, j, k], bin);
            }
        }
    }
    Ok(())
}

fn tile_mapping(mut counts: Vec<f64>, clip_limit: f64) -> Vec<f64> {
    if clip_limit.is_finite() {
        let mut excess = 0.0;
        for count in &mut counts {
            if *count > clip_limit {
                excess += *count - clip_limit;
                *count = clip_limit;
            }
        }
        let spread = excess / counts.len() as f64;
        for count in &mut counts {
            *count += spread;
        }
    }

    let total = counts.iter().sum::<f64>();
    if total == 0.0 {
        return vec![0.0; counts.len()];
    }
    let mut running = 0.0;
    counts
        .into_iter()
        .map(|count| {
            running += count;
            running / total
        })
        .collect()
}

fn interpolate_mapping(
    maps: &[Vec<f64>],
    tiles: [usize; 3],
    tile_shape: [usize; 3],
    at: [usize; 3],
    bin: usize,
) -> f64 {
    let mut low = [0usize; 3];
    let mut high = [0usize; 3];
    let mut fraction = [0.0f64; 3];
    for axis in 0..3 {
        let position = (at[axis] as f64 + 0.5) / tile_shape[axis] as f64 - 0.5;
        let lower = position.floor();
        if lower < 0.0 {
            low[axis] = 0;
            high[axis] = 0;
            fraction[axis] = 0.0;
        } else {
            low[axis] = (lower as usize).min(tiles[axis] - 1);
            high[axis] = (low[axis] + 1).min(tiles[axis] - 1);
            fraction[axis] = if low[axis] == high[axis] {
                0.0
            } else {
                position - lower
            };
        }
    }

    let mut out = 0.0;
    for da in 0..=1 {
        for db in 0..=1 {
            for dc in 0..=1 {
                let index = [
                    if da == 0 { low[0] } else { high[0] },
                    if db == 0 { low[1] } else { high[1] },
                    if dc == 0 { low[2] } else { high[2] },
                ];
                let weight = if da == 0 {
                    1.0 - fraction[0]
                } else {
                    fraction[0]
                } * if db == 0 {
                    1.0 - fraction[1]
                } else {
                    fraction[1]
                } * if dc == 0 {
                    1.0 - fraction[2]
                } else {
                    fraction[2]
                };
                let flat = (index[0] * tiles[1] + index[1]) * tiles[2] + index[2];
                out += weight * maps[flat][bin];
            }
        }
    }
    out
}

fn bin_for(value: f64, minimum: f64, width: f64, bins: usize) -> usize {
    (((value - minimum) / width) as usize).min(bins - 1)
}

fn validate_adapthist(tile_shape: [usize; 3], bins: usize, clip_limit: f64) -> Result<()> {
    if bins == 0 {
        return Err(Error::InvalidArgument(
            "adaptive histogram equalization needs at least one bin".to_string(),
        ));
    }
    if tile_shape.contains(&0) {
        return Err(Error::InvalidArgument(format!(
            "adaptive histogram equalization tile shape must be non-zero on every axis; got \
             {tile_shape:?}"
        )));
    }
    if (!clip_limit.is_finite() && !clip_limit.is_infinite()) || clip_limit <= 0.0 {
        return Err(Error::InvalidArgument(format!(
            "adaptive histogram equalization clip limit must be positive; got {clip_limit}"
        )));
    }
    Ok(())
}

fn percentile(sorted: &[f64], percentile: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let position = percentile / 100.0 * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let weight = position - lower as f64;
        sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::Array3;

    #[test]
    fn equalization_walks_the_global_cdf_and_preserves_non_finite_values() {
        let values = [0.0, 0.0, 10.0, 20.0, f64::NAN];
        let equalized = equalize_histogram(&values, 3).unwrap();
        assert_eq!(&equalized[..4], &[0.5, 0.5, 0.75, 1.0]);
        assert!(equalized[4].is_nan());
        assert_eq!(
            equalize_histogram(&[7.0, 7.0], 256).unwrap(),
            vec![0.0, 0.0]
        );
    }

    #[test]
    fn equalization_map_is_magic_counted_and_compact() {
        let map = EqualizationMap::from_values(&[0.0, 0.0, 10.0, 20.0], 3).unwrap();
        assert_eq!(map.cdf, vec![0.5, 0.75, 1.0]);
        assert_eq!(map.apply(10.0), 0.75);
        assert!(map.apply(f64::NAN).is_nan());
        let bytes = map.encode().unwrap();
        assert_eq!(bytes.len(), 32 + 3 * 8);
        assert_eq!(EqualizationMap::decode(&bytes).unwrap(), map);

        let mut trailing = bytes.clone();
        trailing.push(1);
        assert!(EqualizationMap::decode(&trailing).is_err());
        assert!(EqualizeHistogramOp::new("equalize", "samples", 0, 0).is_err());
    }

    #[test]
    fn equalize_histogram_op_states_its_planner_contract() {
        let op = EqualizeHistogramOp::new("equalize", "samples", 4, 32).unwrap();
        assert_eq!(op.name(), "equalize");
        assert!(!op.reads_pixels());
        assert!(op.writes_pixels());
        assert!(op.barrier());
        assert!(!op.gathers());
        assert_eq!(op.produces(Dtype::U16), Dtype::F64);
        assert_eq!(op.inputs()[0].phase, 4);
        assert_eq!(op.source_inputs([5, 5, 5])[0].image.index(), 0);
        assert_eq!(op.seam_fold(), Some(SeamFold::Unordered));
    }

    #[test]
    fn histogram_matching_maps_source_quantiles_to_reference_centres() {
        let source = [0.0, 0.0, 10.0, 20.0];
        let reference = [100.0, 100.0, 200.0, 300.0];
        let matched = match_histogram(&source, &reference, 3).unwrap();
        assert_eq!(
            matched,
            vec![
                133.33333333333334,
                133.33333333333334,
                200.0,
                266.6666666666667
            ]
        );
        assert_eq!(
            match_histogram(&source, &[5.0, 5.0], 3).unwrap(),
            vec![5.0, 5.0, 5.0, 5.0]
        );
    }

    #[test]
    fn low_contrast_uses_percentile_span_over_the_finite_range() {
        assert!(is_low_contrast(&[0.0, 0.49, 0.5, 1.0], 25.0, 75.0, 0.26).unwrap());
        assert!(!is_low_contrast(&[0.0, 0.25, 0.75, 1.0], 25.0, 75.0, 0.25).unwrap());
        assert!(is_low_contrast(&[3.0, 3.0, f64::INFINITY], 1.0, 99.0, 0.05).unwrap());
        assert!(is_low_contrast(&[f64::NAN], 1.0, 99.0, 0.05).is_err());
        assert!(is_low_contrast(&[0.0, 1.0], 75.0, 25.0, 0.05).is_err());
    }

    #[test]
    fn adaptive_equalization_matches_global_equalization_for_one_tile() {
        let input = Array3::from_shape_vec((1, 4, 1), vec![0.0, 0.0, 10.0, 20.0]).unwrap();
        let mut out = Array3::<f64>::zeros((1, 4, 1));

        equalize_adapthist_into(input.view(), [1, 4, 1], 3, f64::INFINITY, out.view_mut()).unwrap();

        assert_eq!(
            out.iter().copied().collect::<Vec<_>>(),
            equalize_histogram(input.as_slice().unwrap(), 3).unwrap()
        );
    }

    #[test]
    fn adaptive_equalization_interpolates_between_tile_mappings() {
        let input = Array3::from_shape_vec((1, 4, 1), vec![0.0, 0.0, 10.0, 10.0]).unwrap();
        let mut out = Array3::<f64>::zeros((1, 4, 1));

        equalize_adapthist_into(input.view(), [1, 2, 1], 2, f64::INFINITY, out.view_mut()).unwrap();

        assert_eq!(
            out.iter().copied().collect::<Vec<_>>(),
            vec![1.0, 0.75, 1.0, 1.0]
        );
    }

    #[test]
    fn adaptive_equalization_clips_dominant_bins_and_preserves_non_finite_values() {
        let input = Array3::from_shape_vec((1, 4, 1), vec![0.0, 0.0, 0.0, f64::NAN]).unwrap();
        let mut out = Array3::<f64>::zeros((1, 4, 1));

        equalize_adapthist_into(input.view(), [1, 4, 1], 4, 1.0, out.view_mut()).unwrap();

        assert_eq!(out[[0, 0, 0]], 0.0);
        assert!(out[[0, 3, 0]].is_nan());
        assert!(equalize_adapthist_into(input.view(), [0, 4, 1], 4, 1.0, out.view_mut()).is_err());
    }
}
