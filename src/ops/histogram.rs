// SPDX-License-Identifier: MIT
//
// Original work for this crate.

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::fragment::{
    BlockOutput, BlockView, Coverage, FragmentOp, FragmentOutput, SeamFold, SidecarSize,
};
use crate::sidecar::Lifecycle;
use crate::voxels::Voxels;

pub(crate) const FINITE_SAMPLE_HEADER_BYTES: u64 = 16;
const FINITE_SAMPLES_MAGIC: u64 = 0x4649_4e49_5445_4653;

/// Finite values from a block, encoded losslessly as `f64` bits.
pub(crate) fn finite_sample_values(input: &Voxels, what: &str) -> Result<Vec<f64>> {
    match input.dtype() {
        Dtype::F16 | Dtype::U64 | Dtype::I64 => Err(Error::InvalidArgument(format!(
            "{what} refuses {} input because finite sample fragments store samples as f64 and \
             would not preserve every value exactly",
            input.dtype().numpy_name()
        ))),
        _ => Ok(input
            .widened()
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect()),
    }
}

pub(crate) fn encode_finite_samples(values: &[f64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(FINITE_SAMPLE_HEADER_BYTES as usize + values.len() * 8);
    bytes.extend_from_slice(&FINITE_SAMPLES_MAGIC.to_le_bytes());
    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for &value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    bytes
}

pub(crate) fn decode_finite_samples(bytes: &[u8], what: &str) -> Result<Vec<f64>> {
    let mut cursor = BytesCursor::new(bytes, what);
    cursor.expect_magic(FINITE_SAMPLES_MAGIC)?;
    let count = cursor.take_u64()? as usize;
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(f64::from_bits(cursor.take_u64()?));
    }
    cursor.expect_end()?;
    Ok(values)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FiniteSamplesOp {
    name: &'static str,
    stream: String,
    lifecycle: Lifecycle,
}

impl FiniteSamplesOp {
    pub(crate) fn new(name: &'static str, stream: impl Into<String>, lifecycle: Lifecycle) -> Self {
        Self {
            name,
            stream: stream.into(),
            lifecycle,
        }
    }
}

impl FragmentOp for FiniteSamplesOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn reads_pixels(&self) -> bool {
        true
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![
            FragmentOutput::new(&self.stream, self.lifecycle, Coverage::EveryBlock).sized(
                SidecarSize::Terms {
                    fixed: FINITE_SAMPLE_HEADER_BYTES,
                    per_core_voxel: 0.0,
                    per_read_voxel: 8.0,
                    per_face_voxel: 0.0,
                    tight: true,
                },
            ),
        ]
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::PerBlock)
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let pixels = at.pixels()?.as_array()?;
        let values = finite_sample_values(pixels, self.name)?;
        Ok(BlockOutput::fragment(
            &self.stream,
            encode_finite_samples(&values),
        ))
    }
}

pub(crate) struct BytesCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
    what: &'a str,
}

impl<'a> BytesCursor<'a> {
    pub(crate) fn new(bytes: &'a [u8], what: &'a str) -> Self {
        Self {
            bytes,
            offset: 0,
            what,
        }
    }

    pub(crate) fn expect_magic(&mut self, expected: u64) -> Result<()> {
        let got = self.take_u64()?;
        if got != expected {
            return Err(Error::InvalidArgument(format!(
                "{} has magic {got:#x}, expected {expected:#x}",
                self.what
            )));
        }
        Ok(())
    }

    pub(crate) fn take_u64(&mut self) -> Result<u64> {
        let end = self.offset.saturating_add(8);
        let bytes = self.bytes.get(self.offset..end).ok_or_else(|| {
            Error::InvalidArgument(format!("{} ended early at byte {}", self.what, self.offset))
        })?;
        self.offset = end;
        Ok(u64::from_le_bytes(
            bytes.try_into().expect("slice length checked"),
        ))
    }

    pub(crate) fn expect_end(&self) -> Result<()> {
        if self.offset != self.bytes.len() {
            return Err(Error::InvalidArgument(format!(
                "{} has {} trailing byte(s)",
                self.what,
                self.bytes.len() - self.offset
            )));
        }
        Ok(())
    }
}

pub(crate) struct FiniteHistogram {
    pub counts: Vec<f64>,
    pub centres: Vec<f64>,
    pub cumulative_count: Vec<f64>,
    pub cumulative_intensity: Vec<f64>,
    pub total_count: f64,
    pub total_intensity: f64,
    pub constant: bool,
    pub minimum: f64,
    width: f64,
}

impl FiniteHistogram {
    pub fn new(values: &[f64], bins: usize, name: &str) -> Result<Self> {
        if bins == 0 {
            return Err(Error::InvalidArgument(format!(
                "{name} needs at least one histogram bin"
            )));
        }
        if values.is_empty() {
            return Err(Error::InvalidArgument(format!(
                "{name} needs at least one finite value"
            )));
        }
        let finite: Vec<f64> = values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect();
        if finite.is_empty() {
            return Err(Error::InvalidArgument(format!(
                "{name} needs at least one finite value"
            )));
        }
        let minimum = finite.iter().copied().fold(f64::INFINITY, f64::min);
        let maximum = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let width = (maximum - minimum) / bins as f64;
        if width <= 0.0 {
            let count = finite.len() as f64;
            return Ok(Self {
                counts: vec![count],
                centres: vec![maximum],
                cumulative_count: vec![count],
                cumulative_intensity: vec![maximum * count],
                total_count: count,
                total_intensity: maximum * count,
                constant: true,
                minimum,
                width: 0.0,
            });
        }

        let mut counts = vec![0.0f64; bins];
        for value in finite {
            counts[Self::bin_for(value, minimum, width, bins)] += 1.0;
        }
        let centres: Vec<f64> = (0..bins)
            .map(|index| minimum + (index as f64 + 0.5) * width)
            .collect();
        let mut cumulative_count = Vec::with_capacity(bins);
        let mut cumulative_intensity = Vec::with_capacity(bins);
        let mut running_count = 0.0;
        let mut running_intensity = 0.0;
        for (count, centre) in counts.iter().zip(&centres) {
            running_count += count;
            running_intensity += count * centre;
            cumulative_count.push(running_count);
            cumulative_intensity.push(running_intensity);
        }
        Ok(Self {
            counts,
            centres,
            cumulative_count,
            cumulative_intensity,
            total_count: running_count,
            total_intensity: running_intensity,
            constant: false,
            minimum,
            width,
        })
    }

    pub fn index(&self, value: f64) -> usize {
        if self.constant {
            0
        } else {
            Self::bin_for(value, self.minimum, self.width, self.counts.len())
        }
    }

    pub fn cdf_at_bin(&self, index: usize) -> f64 {
        self.cumulative_count[index] / self.total_count
    }

    pub fn first_bin_at_or_above(&self, quantile: f64) -> usize {
        let want = quantile.clamp(0.0, 1.0) * self.total_count;
        self.cumulative_count
            .iter()
            .position(|&count| count >= want)
            .unwrap_or(self.cumulative_count.len() - 1)
    }

    pub fn class_stats(&self, start: usize, end: usize) -> Option<(f64, f64)> {
        if start > end || end >= self.counts.len() {
            return None;
        }
        let count = self.cumulative_count[end]
            - start
                .checked_sub(1)
                .map(|before| self.cumulative_count[before])
                .unwrap_or(0.0);
        if count == 0.0 {
            return None;
        }
        let intensity = self.cumulative_intensity[end]
            - start
                .checked_sub(1)
                .map(|before| self.cumulative_intensity[before])
                .unwrap_or(0.0);
        Some((count, intensity))
    }

    pub fn width(&self) -> f64 {
        self.width
    }

    fn bin_for(value: f64, minimum: f64, width: f64, bins: usize) -> usize {
        (((value - minimum) / width) as usize).min(bins - 1)
    }
}
