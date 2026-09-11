// SPDX-License-Identifier: MIT
//
// Original work for this crate.

//! Global threshold selection.
//!
//! This module owns the scalar rules only: a caller hands it values and gets a
//! threshold level back. Applying that level is still `ops::voxelwise`, and a
//! future planner-visible global phase should call these same rules rather than
//! carry a second definition of Otsu or mean thresholding.

use ndarray::{ArrayView3, ArrayViewMut3};

use crate::assemble::{Phase, PlanBuilder};
use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::fragment::{
    BlockOutput, BlockView, Coverage, FragmentInput, FragmentOp, FragmentOutput, PhaseView,
    SeamFold, SidecarSize, SourceBlocks,
};
use crate::op::SourceInput;
use crate::sidecar::Lifecycle;
use crate::voxels::Voxels;

use super::histogram::{
    decode_finite_samples, encode_finite_samples, finite_sample_values, BytesCursor,
    FiniteHistogram, FINITE_SAMPLE_HEADER_BYTES,
};
use super::voxelwise::ThresholdTest;

const THRESHOLD_LEVELS_MAGIC: u64 = 0x5448_5245_534c_564c;

/// Which global threshold rule to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GlobalThreshold {
    /// Arithmetic mean of all finite values.
    Mean,
    /// Otsu's between-class variance rule over `bins` equal-width bins.
    Otsu { bins: usize },
    /// Yen's maximum-correlation threshold over `bins` equal-width bins.
    Yen { bins: usize },
    /// Li's minimum cross-entropy threshold over finite values.
    Li,
    /// The triangle threshold over `bins` equal-width bins.
    Triangle { bins: usize },
    /// The minimum threshold over a smoothed `bins`-bin histogram.
    Minimum { bins: usize, max_smoothings: usize },
}

impl GlobalThreshold {
    fn checked_bins(name: &str, bins: usize) -> Result<usize> {
        if bins == 0 {
            return Err(Error::InvalidArgument(format!(
                "{name} needs at least one histogram bin"
            )));
        }
        Ok(bins)
    }

    pub fn otsu(bins: usize) -> Result<Self> {
        Self::checked_bins("an Otsu threshold", bins)?;
        Ok(GlobalThreshold::Otsu { bins })
    }

    pub fn yen(bins: usize) -> Result<Self> {
        Self::checked_bins("a Yen threshold", bins)?;
        Ok(GlobalThreshold::Yen { bins })
    }

    pub fn triangle(bins: usize) -> Result<Self> {
        Self::checked_bins("a triangle threshold", bins)?;
        Ok(GlobalThreshold::Triangle { bins })
    }

    pub fn minimum(bins: usize, max_smoothings: usize) -> Result<Self> {
        Self::checked_bins("a minimum threshold", bins)?;
        if max_smoothings == 0 {
            return Err(Error::InvalidArgument(
                "a minimum threshold needs at least one smoothing attempt".to_string(),
            ));
        }
        Ok(GlobalThreshold::Minimum {
            bins,
            max_smoothings,
        })
    }

    /// Select a threshold from `values`.
    ///
    /// Non-finite values are ignored. A threshold over no finite values is
    /// refused because any fallback would become a hidden policy choice in a
    /// global phase.
    pub fn of<I>(self, values: I) -> Result<f64>
    where
        I: IntoIterator<Item = f64>,
    {
        let values: Vec<f64> = values
            .into_iter()
            .filter(|value| value.is_finite())
            .collect();
        if values.is_empty() {
            return Err(Error::InvalidArgument(
                "a global threshold needs at least one finite value".to_string(),
            ));
        }
        match self {
            GlobalThreshold::Mean => Ok(mean_threshold(&values)),
            GlobalThreshold::Otsu { bins } => otsu_threshold(&values, bins),
            GlobalThreshold::Yen { bins } => yen_threshold(&values, bins),
            GlobalThreshold::Li => li_threshold(&values),
            GlobalThreshold::Triangle { bins } => triangle_threshold(&values, bins),
            GlobalThreshold::Minimum {
                bins,
                max_smoothings,
            } => minimum_threshold(&values, bins, max_smoothings),
        }
    }
}

/// A threshold selection whose output may need one or many levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GlobalThresholdSelection {
    Single(GlobalThreshold),
    MultiOtsu { bins: usize, classes: usize },
}

impl GlobalThresholdSelection {
    pub fn single(rule: GlobalThreshold) -> Self {
        GlobalThresholdSelection::Single(rule)
    }

    pub fn multi_otsu(bins: usize, classes: usize) -> Result<Self> {
        GlobalThreshold::checked_bins("a multi-Otsu threshold", bins)?;
        if classes < 2 {
            return Err(Error::InvalidArgument(format!(
                "multi-Otsu needs at least two classes, got {classes}"
            )));
        }
        if classes > bins {
            return Err(Error::InvalidArgument(format!(
                "multi-Otsu cannot split {bins} histogram bins into {classes} non-empty classes"
            )));
        }
        Ok(GlobalThresholdSelection::MultiOtsu { bins, classes })
    }

    pub fn thresholds_of<I>(self, values: I) -> Result<Vec<f64>>
    where
        I: IntoIterator<Item = f64>,
    {
        let mut values: Vec<f64> = values
            .into_iter()
            .filter(|value| value.is_finite())
            .collect();
        if values.is_empty() {
            return Err(Error::InvalidArgument(
                "a global threshold needs at least one finite value".to_string(),
            ));
        }
        values.sort_by(f64::total_cmp);
        match self {
            GlobalThresholdSelection::Single(rule) => Ok(vec![rule.of(values)?]),
            GlobalThresholdSelection::MultiOtsu { bins, classes } => {
                multi_otsu_thresholds(&values, bins, classes)
            }
        }
    }
}

impl From<GlobalThreshold> for GlobalThresholdSelection {
    fn from(value: GlobalThreshold) -> Self {
        GlobalThresholdSelection::Single(value)
    }
}

/// The image a global threshold application writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GlobalThresholdOutput {
    Mask { test: ThresholdTest },
    Classes,
}

impl GlobalThresholdOutput {
    fn dtype(self) -> Dtype {
        match self {
            GlobalThresholdOutput::Mask { .. } => Dtype::Bool,
            GlobalThresholdOutput::Classes => Dtype::U32,
        }
    }
}

/// Phase 0 of a planner-visible global threshold: emit finite samples.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobalThresholdSamplesOp {
    name: &'static str,
    stream: String,
}

impl GlobalThresholdSamplesOp {
    pub fn new(name: &'static str, stream: impl Into<String>) -> Self {
        Self {
            name,
            stream: stream.into(),
        }
    }

    pub fn stream(&self) -> &str {
        &self.stream
    }
}

impl FragmentOp for GlobalThresholdSamplesOp {
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
            FragmentOutput::new(&self.stream, Lifecycle::DeleteOnExit, Coverage::EveryBlock).sized(
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
        let values = finite_sample_values(pixels, "global threshold samples")?;
        Ok(BlockOutput::fragment(
            &self.stream,
            encode_finite_samples(&values),
        ))
    }
}

/// Phase 1 of a planner-visible global threshold: reduce samples and write an image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApplyGlobalThresholdOp {
    name: &'static str,
    stream: String,
    samples_phase: usize,
    image: usize,
    selection: GlobalThresholdSelection,
    output: GlobalThresholdOutput,
}

impl ApplyGlobalThresholdOp {
    pub fn new(
        name: &'static str,
        stream: impl Into<String>,
        samples_phase: usize,
        selection: impl Into<GlobalThresholdSelection>,
        output: GlobalThresholdOutput,
    ) -> Self {
        Self {
            name,
            stream: stream.into(),
            samples_phase,
            image: 0,
            selection: selection.into(),
            output,
        }
    }

    pub fn reading_image(mut self, image: usize) -> Self {
        self.image = image;
        self
    }

    pub fn selection(&self) -> GlobalThresholdSelection {
        self.selection
    }

    pub fn output(&self) -> GlobalThresholdOutput {
        self.output
    }
}

impl FragmentOp for ApplyGlobalThresholdOp {
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
        self.output.dtype()
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
            values.extend(decode_finite_samples(bytes, "global threshold samples")?);
            Ok(())
        })?;
        let levels = self.selection.thresholds_of(values)?;
        encode_threshold_levels(&levels)
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::Unordered)
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let declared = self.source_inputs(at.volume());
        Err(Error::InvalidArgument(format!(
            "fragment op {:?} declares source image {} and must be applied through \
             `apply_with`; the phase input is not the image being thresholded",
            self.name(),
            declared[0].image.index()
        )))
    }

    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let levels = decode_threshold_levels(at.reduced)?;
        let pixels = sources.get(self.image)?.as_array()?;
        let mut buffer = at.output_buffer(0.0)?;
        if let Some(out) = buffer.as_array_mut() {
            apply_threshold_levels(pixels, &levels, self.output, out)?;
        }
        Ok(BlockOutput::nothing().with_pixels(buffer))
    }
}

/// Append the two planner-visible phases for a global threshold.
///
/// The sample phase writes only fragments, so the apply phase must read the
/// image that was current before that phase was appended. That image id is the
/// sample phase's own index, and this helper binds the two together so callers
/// do not have to remember the convention.
pub fn append_global_threshold_phases(
    plan: &mut PlanBuilder,
    stream: impl Into<String>,
    selection: impl Into<GlobalThresholdSelection>,
    output: GlobalThresholdOutput,
) -> Result<Phase> {
    let stream = stream.into();
    let samples = plan.fragments(GlobalThresholdSamplesOp::new(
        "global threshold samples",
        stream.clone(),
    ))?;
    plan.fragments(
        ApplyGlobalThresholdOp::new(
            "apply global threshold",
            stream,
            samples.index(),
            selection,
            output,
        )
        .reading_image(samples.index()),
    )
}

pub fn mean_threshold(values: &[f64]) -> f64 {
    values.iter().copied().sum::<f64>() / values.len() as f64
}

/// Apply one selected threshold as a bool mask.
///
/// Non-finite values compare false, matching the scalar comparison that
/// `ThresholdMask` ultimately uses.
pub fn threshold_mask(values: &[f64], level: f64, test: ThresholdTest) -> Result<Vec<bool>> {
    if !level.is_finite() {
        return Err(Error::InvalidArgument(format!(
            "a threshold mask level must be finite, got {level}"
        )));
    }
    Ok(values
        .iter()
        .map(|&value| {
            value.is_finite()
                && match test {
                    ThresholdTest::Above => value > level,
                    ThresholdTest::AtOrAbove => value >= level,
                }
        })
        .collect())
}

/// Apply selected thresholds as class labels `0..thresholds.len()`.
///
/// Thresholds must be finite and sorted ascending. Duplicate thresholds are
/// allowed because a constant multi-Otsu result can legitimately contain them;
/// they simply create an empty class under the strict `value > threshold` rule.
pub fn threshold_classes(values: &[f64], thresholds: &[f64]) -> Result<Vec<u32>> {
    if thresholds.is_empty() {
        return Err(Error::InvalidArgument(
            "class labelling needs at least one threshold".to_string(),
        ));
    }
    for (index, &threshold) in thresholds.iter().enumerate() {
        if !threshold.is_finite() {
            return Err(Error::InvalidArgument(format!(
                "class threshold {index} is {threshold}; thresholds must be finite"
            )));
        }
        if index > 0 && threshold < thresholds[index - 1] {
            return Err(Error::InvalidArgument(format!(
                "class thresholds must be sorted ascending; threshold {index} is {threshold} \
                 after {}",
                thresholds[index - 1]
            )));
        }
    }
    values
        .iter()
        .map(|&value| {
            if !value.is_finite() {
                return Err(Error::InvalidArgument(format!(
                    "class labelling refuses non-finite value {value}; a class label has no \
                     NaN representation"
                )));
            }
            Ok(thresholds.partition_point(|&threshold| value > threshold) as u32)
        })
        .collect()
}

pub fn otsu_threshold(values: &[f64], bins: usize) -> Result<f64> {
    GlobalThreshold::checked_bins("an Otsu threshold", bins)?;
    if values.is_empty() {
        return Err(Error::InvalidArgument(
            "an Otsu threshold needs at least one finite value".to_string(),
        ));
    }
    let histogram = FiniteHistogram::new(values, bins, "an Otsu threshold")?;
    if histogram.constant {
        return Ok(histogram.centres[0]);
    }

    let mut best_score = f64::NEG_INFINITY;
    let mut best = histogram.centres[0];
    for index in 0..bins.saturating_sub(1) {
        let Some(score) = histogram.between_class_score(&[index]) else {
            continue;
        };
        if score > best_score {
            best_score = score;
            best = histogram.centres[index];
        }
    }
    Ok(best)
}

pub fn yen_threshold(values: &[f64], bins: usize) -> Result<f64> {
    GlobalThreshold::checked_bins("a Yen threshold", bins)?;
    let histogram = FiniteHistogram::new(values, bins, "a Yen threshold")?;
    if histogram.constant {
        return Ok(histogram.centres[0]);
    }
    let total = histogram.total_count;
    let probabilities: Vec<f64> = histogram.counts.iter().map(|count| count / total).collect();
    let mut p1 = Vec::with_capacity(probabilities.len());
    let mut p1_sq = Vec::with_capacity(probabilities.len());
    let mut running = 0.0;
    let mut running_sq = 0.0;
    for &p in &probabilities {
        running += p;
        running_sq += p * p;
        p1.push(running);
        p1_sq.push(running_sq);
    }
    let mut p2_sq = vec![0.0; probabilities.len()];
    running_sq = 0.0;
    for index in (0..probabilities.len()).rev() {
        running_sq += probabilities[index] * probabilities[index];
        p2_sq[index] = running_sq;
    }

    let mut best_score = f64::NEG_INFINITY;
    let mut best = None;
    for index in 0..probabilities.len().saturating_sub(1) {
        let foreground = p1[index] * (1.0 - p1[index]);
        let entropy = p1_sq[index] * p2_sq[index + 1];
        if foreground <= 0.0 || entropy <= 0.0 {
            continue;
        }
        let score = -entropy.ln() + 2.0 * foreground.ln();
        if score > best_score {
            best_score = score;
            best = Some(histogram.centres[index]);
        }
    }
    best.ok_or_else(|| {
        Error::InvalidArgument(
            "a Yen threshold could not split the populated histogram".to_string(),
        )
    })
}

pub fn li_threshold(values: &[f64]) -> Result<f64> {
    let mut finite: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.is_empty() {
        return Err(Error::InvalidArgument(
            "a Li threshold needs at least one finite value".to_string(),
        ));
    }
    let minimum = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let shift = if minimum <= 0.0 { 1.0 - minimum } else { 0.0 };
    if shift != 0.0 {
        for value in &mut finite {
            *value += shift;
        }
    }
    if finite.iter().all(|&value| value == finite[0]) {
        return Ok(finite[0] - shift);
    }
    let maximum = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let minimum = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let tolerance = ((maximum - minimum).abs() * 1.0e-12).max(1.0e-12);
    let mut threshold = mean_threshold(&finite);
    for _ in 0..10_000 {
        let mut below_sum = 0.0;
        let mut below_count = 0usize;
        let mut above_sum = 0.0;
        let mut above_count = 0usize;
        for &value in &finite {
            if value <= threshold {
                below_sum += value;
                below_count += 1;
            } else {
                above_sum += value;
                above_count += 1;
            }
        }
        if below_count == 0 || above_count == 0 {
            return Ok(threshold - shift);
        }
        let below = below_sum / below_count as f64;
        let above = above_sum / above_count as f64;
        if below <= 0.0 || above <= 0.0 || below == above {
            return Ok(threshold - shift);
        }
        let next = (below - above) / (below.ln() - above.ln());
        if (next - threshold).abs() <= tolerance {
            return Ok(next - shift);
        }
        threshold = next;
    }
    Err(Error::InvalidArgument(
        "a Li threshold did not converge within 10000 iterations".to_string(),
    ))
}

pub fn triangle_threshold(values: &[f64], bins: usize) -> Result<f64> {
    GlobalThreshold::checked_bins("a triangle threshold", bins)?;
    let histogram = FiniteHistogram::new(values, bins, "a triangle threshold")?;
    if histogram.constant {
        return Ok(histogram.centres[0]);
    }
    let (first, last) = populated_span(&histogram)?;
    let peak = histogram.counts[first..=last]
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| first + index)
        .expect("a populated span is non-empty");
    let start = if peak - first >= last - peak {
        last
    } else {
        first
    };
    let x1 = start as f64;
    let y1 = histogram.counts[start];
    let x2 = peak as f64;
    let y2 = histogram.counts[peak];
    let norm = ((y2 - y1).powi(2) + (x2 - x1).powi(2)).sqrt();
    let mut best = peak;
    let mut best_distance = f64::NEG_INFINITY;
    let range = if start <= peak {
        start..=peak
    } else {
        peak..=start
    };
    for index in range {
        let x0 = index as f64;
        let y0 = histogram.counts[index];
        let distance = ((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1).abs() / norm;
        if distance > best_distance {
            best_distance = distance;
            best = index;
        }
    }
    Ok(histogram.centres[best])
}

pub fn minimum_threshold(values: &[f64], bins: usize, max_smoothings: usize) -> Result<f64> {
    GlobalThreshold::checked_bins("a minimum threshold", bins)?;
    if max_smoothings == 0 {
        return Err(Error::InvalidArgument(
            "a minimum threshold needs at least one smoothing attempt".to_string(),
        ));
    }
    let histogram = FiniteHistogram::new(values, bins, "a minimum threshold")?;
    if histogram.constant {
        return Ok(histogram.centres[0]);
    }
    let mut counts = histogram.counts.clone();
    for _ in 0..max_smoothings {
        let peaks = local_maxima(&counts);
        if peaks.len() == 2 {
            let left = peaks[0].min(peaks[1]);
            let right = peaks[0].max(peaks[1]);
            let valley = (left..=right)
                .min_by(|&a, &b| counts[a].total_cmp(&counts[b]))
                .expect("two peaks define a non-empty interval");
            return Ok(histogram.centres[valley]);
        }
        smooth_histogram(&mut counts);
    }
    Err(Error::InvalidArgument(format!(
        "a minimum threshold did not reach two peaks after {max_smoothings} smoothings"
    )))
}

fn populated_span(histogram: &FiniteHistogram) -> Result<(usize, usize)> {
    let first = histogram.counts.iter().position(|&count| count > 0.0);
    let last = histogram.counts.iter().rposition(|&count| count > 0.0);
    match (first, last) {
        (Some(first), Some(last)) => Ok((first, last)),
        _ => Err(Error::InvalidArgument(
            "a histogram threshold found no populated bins".to_string(),
        )),
    }
}

fn local_maxima(counts: &[f64]) -> Vec<usize> {
    let mut peaks = Vec::new();
    for index in 0..counts.len() {
        let left = index
            .checked_sub(1)
            .map(|before| counts[before])
            .unwrap_or(f64::NEG_INFINITY);
        let right = counts.get(index + 1).copied().unwrap_or(f64::NEG_INFINITY);
        if counts[index] > left && counts[index] > right {
            peaks.push(index);
        }
    }
    peaks
}

fn smooth_histogram(counts: &mut [f64]) {
    if counts.len() < 2 {
        return;
    }
    let previous = counts.to_vec();
    for index in 0..counts.len() {
        let left = index
            .checked_sub(1)
            .map(|before| previous[before])
            .unwrap_or(previous[index]);
        let right = previous.get(index + 1).copied().unwrap_or(previous[index]);
        counts[index] = (left + previous[index] + right) / 3.0;
    }
}

fn apply_threshold_levels(
    input: &Voxels,
    levels: &[f64],
    output: GlobalThresholdOutput,
    out: &mut Voxels,
) -> Result<()> {
    match output {
        GlobalThresholdOutput::Mask { test } => {
            let mut out = out.view_mut::<bool>()?;
            let input = input.widened();
            apply_threshold_mask_into(input.view(), levels, test, out.view_mut())
        }
        GlobalThresholdOutput::Classes => {
            let mut out = out.view_mut::<u32>()?;
            let input = input.widened();
            apply_threshold_classes_into(input.view(), levels, out.view_mut())
        }
    }
}

fn apply_threshold_mask_into(
    input: ArrayView3<'_, f64>,
    levels: &[f64],
    test: ThresholdTest,
    mut out: ArrayViewMut3<'_, bool>,
) -> Result<()> {
    if levels.len() != 1 {
        return Err(Error::InvalidArgument(format!(
            "a threshold mask needs exactly one level, got {}",
            levels.len()
        )));
    }
    let level = levels[0];
    if !level.is_finite() {
        return Err(Error::InvalidArgument(format!(
            "a threshold mask level must be finite, got {level}"
        )));
    }
    ndarray::Zip::from(&input)
        .and(&mut out)
        .for_each(|&value, slot| {
            *slot = value.is_finite()
                && match test {
                    ThresholdTest::Above => value > level,
                    ThresholdTest::AtOrAbove => value >= level,
                };
        });
    Ok(())
}

fn apply_threshold_classes_into(
    input: ArrayView3<'_, f64>,
    levels: &[f64],
    mut out: ArrayViewMut3<'_, u32>,
) -> Result<()> {
    if levels.is_empty() {
        return Err(Error::InvalidArgument(
            "class labelling needs at least one threshold".to_string(),
        ));
    }
    validate_class_thresholds(levels)?;
    if let Some(value) = input.iter().copied().find(|value| !value.is_finite()) {
        return Err(Error::InvalidArgument(format!(
            "class labelling refuses non-finite value {value}; a class label has no \
             NaN representation"
        )));
    }
    ndarray::Zip::from(&input)
        .and(&mut out)
        .for_each(|&value, slot| {
            *slot = levels.partition_point(|&threshold| value > threshold) as u32;
        });
    Ok(())
}

fn validate_class_thresholds(thresholds: &[f64]) -> Result<()> {
    for (index, &threshold) in thresholds.iter().enumerate() {
        if !threshold.is_finite() {
            return Err(Error::InvalidArgument(format!(
                "class threshold {index} is {threshold}; thresholds must be finite"
            )));
        }
        if index > 0 && threshold < thresholds[index - 1] {
            return Err(Error::InvalidArgument(format!(
                "class thresholds must be sorted ascending; threshold {index} is {threshold} \
                 after {}",
                thresholds[index - 1]
            )));
        }
    }
    Ok(())
}

fn encode_f64_values(magic: u64, values: &[f64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(FINITE_SAMPLE_HEADER_BYTES as usize + values.len() * 8);
    bytes.extend_from_slice(&magic.to_le_bytes());
    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for &value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    bytes
}

fn decode_f64_values(magic: u64, bytes: &[u8], what: &str) -> Result<Vec<f64>> {
    let mut cursor = BytesCursor::new(bytes, what);
    cursor.expect_magic(magic)?;
    let count = cursor.take_u64()? as usize;
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(f64::from_bits(cursor.take_u64()?));
    }
    cursor.expect_end()?;
    Ok(values)
}

fn encode_threshold_levels(levels: &[f64]) -> Result<Vec<u8>> {
    if levels.is_empty() {
        return Err(Error::InvalidArgument(
            "a global threshold reduction needs at least one level".to_string(),
        ));
    }
    for (index, &level) in levels.iter().enumerate() {
        if !level.is_finite() {
            return Err(Error::InvalidArgument(format!(
                "threshold level {index} is {level}; levels must be finite"
            )));
        }
        if index > 0 && level < levels[index - 1] {
            return Err(Error::InvalidArgument(format!(
                "threshold levels must be sorted ascending; level {index} is {level} after {}",
                levels[index - 1]
            )));
        }
    }
    Ok(encode_f64_values(THRESHOLD_LEVELS_MAGIC, levels))
}

fn decode_threshold_levels(bytes: &[u8]) -> Result<Vec<f64>> {
    let levels = decode_f64_values(THRESHOLD_LEVELS_MAGIC, bytes, "global threshold reduction")?;
    if levels.is_empty() {
        return Err(Error::InvalidArgument(
            "a global threshold reduction carried no levels".to_string(),
        ));
    }
    validate_class_thresholds(&levels)?;
    Ok(levels)
}

pub fn multi_otsu_thresholds(values: &[f64], bins: usize, classes: usize) -> Result<Vec<f64>> {
    if classes < 2 {
        return Err(Error::InvalidArgument(format!(
            "multi-Otsu needs at least two classes, got {classes}"
        )));
    }
    if classes > bins {
        return Err(Error::InvalidArgument(format!(
            "multi-Otsu cannot split {bins} histogram bins into {classes} non-empty classes"
        )));
    }
    if values.is_empty() {
        return Err(Error::InvalidArgument(
            "multi-Otsu needs at least one finite value".to_string(),
        ));
    }
    let histogram = FiniteHistogram::new(values, bins, "multi-Otsu")?;
    if histogram.constant {
        return Ok(vec![histogram.centres[0]; classes - 1]);
    }

    let mut current = Vec::with_capacity(classes - 1);
    let mut best_score = f64::NEG_INFINITY;
    let mut best = Vec::new();
    search_multi_otsu(
        &histogram,
        classes,
        0,
        &mut current,
        &mut best_score,
        &mut best,
    );
    if best.is_empty() {
        return Err(Error::InvalidArgument(format!(
            "multi-Otsu found no split of {bins} bins into {classes} populated classes"
        )));
    }
    Ok(best
        .into_iter()
        .map(|index| histogram.centres[index])
        .collect())
}

fn search_multi_otsu(
    histogram: &FiniteHistogram,
    classes: usize,
    start: usize,
    current: &mut Vec<usize>,
    best_score: &mut f64,
    best: &mut Vec<usize>,
) {
    if current.len() == classes - 1 {
        let Some(score) = histogram.between_class_score(current) else {
            return;
        };
        if score > *best_score {
            *best_score = score;
            best.clear();
            best.extend(current.iter().copied());
        }
        return;
    }
    let remaining_thresholds = classes - 1 - current.len();
    let end = histogram.counts.len().saturating_sub(remaining_thresholds);
    for threshold in start..end {
        current.push(threshold);
        search_multi_otsu(histogram, classes, threshold + 1, current, best_score, best);
        current.pop();
    }
}

trait OtsuHistogram {
    fn between_class_score(&self, thresholds: &[usize]) -> Option<f64>;
}

impl OtsuHistogram for FiniteHistogram {
    fn between_class_score(&self, thresholds: &[usize]) -> Option<f64> {
        let global = self.total_intensity / self.total_count;
        let mut previous = 0usize;
        let mut score = 0.0;
        for &threshold in thresholds {
            if threshold < previous {
                return None;
            }
            let (count, intensity) = self.class_stats(previous, threshold)?;
            let mean = intensity / count;
            let delta = mean - global;
            score += count * delta * delta;
            previous = threshold + 1;
        }
        let last = self.counts.len() - 1;
        let (count, intensity) = self.class_stats(previous, last)?;
        let mean = intensity / count;
        let delta = mean - global;
        score += count * delta * delta;
        Some(score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_threshold_uses_only_finite_values() {
        let threshold = GlobalThreshold::Mean
            .of([0.0, 2.0, f64::NAN, 4.0, f64::INFINITY])
            .unwrap();
        assert_eq!(threshold, 2.0);
    }

    #[test]
    fn otsu_threshold_separates_two_exact_histogram_classes() {
        let values = [0.0, 0.0, 0.0, 10.0, 10.0, 10.0];
        let threshold = GlobalThreshold::otsu(2).unwrap().of(values).unwrap();
        assert_eq!(threshold, 2.5);
    }

    #[test]
    fn multi_otsu_thresholds_separate_three_exact_histogram_classes() {
        let values = [0.0, 0.0, 10.0, 10.0, 20.0, 20.0];
        let thresholds = multi_otsu_thresholds(&values, 3, 3).unwrap();
        assert_eq!(thresholds, vec![10.0 / 3.0, 10.0]);
    }

    #[test]
    fn selected_thresholds_have_explicit_mask_and_class_outputs() {
        let values = [0.0, 1.0, 2.0, f64::NAN];
        assert_eq!(
            threshold_mask(&values, 1.0, ThresholdTest::Above).unwrap(),
            vec![false, false, true, false]
        );
        assert_eq!(
            threshold_mask(&values, 1.0, ThresholdTest::AtOrAbove).unwrap(),
            vec![false, true, true, false]
        );
        assert!(threshold_mask(&values, f64::NAN, ThresholdTest::Above).is_err());

        assert_eq!(
            threshold_classes(&[0.0, 2.0, 5.0, 9.0], &[2.0, 5.0]).unwrap(),
            vec![0, 0, 1, 2]
        );
        assert_eq!(threshold_classes(&[7.0], &[7.0, 7.0]).unwrap(), vec![0]);
        assert!(threshold_classes(&[0.0], &[2.0, 1.0]).is_err());
        assert!(threshold_classes(&[f64::NAN], &[1.0]).is_err());
    }

    #[test]
    fn global_threshold_selection_is_decomposition_order_invariant() {
        let selection = GlobalThresholdSelection::single(GlobalThreshold::Mean);
        let first = selection
            .thresholds_of([3.0, 1.0, 4.0, 2.0, f64::NAN])
            .unwrap();
        let second = selection.thresholds_of([2.0, 4.0, 1.0, 3.0]).unwrap();
        assert_eq!(first, second);
        assert_eq!(first, vec![2.5]);

        let multi = GlobalThresholdSelection::multi_otsu(3, 3).unwrap();
        assert_eq!(
            multi
                .thresholds_of([20.0, 0.0, 10.0, 0.0, 20.0, 10.0])
                .unwrap(),
            vec![10.0 / 3.0, 10.0]
        );
        assert!(GlobalThresholdSelection::multi_otsu(0, 2).is_err());
        assert!(GlobalThresholdSelection::multi_otsu(2, 3).is_err());
    }

    #[test]
    fn threshold_fragment_codecs_are_magic_counted_and_sorted() {
        let values = vec![0.0, 1.5, f64::NAN];
        let bytes = encode_finite_samples(&values);
        let decoded = decode_finite_samples(&bytes, "test samples").unwrap();
        assert_eq!(
            decoded
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert!(decode_f64_values(THRESHOLD_LEVELS_MAGIC, &bytes, "test samples").is_err());

        let mut trailing = bytes.clone();
        trailing.push(1);
        assert!(decode_finite_samples(&trailing, "test samples").is_err());

        let levels = encode_threshold_levels(&[1.0, 1.0, 2.0]).unwrap();
        assert_eq!(
            decode_threshold_levels(&levels).unwrap(),
            vec![1.0, 1.0, 2.0]
        );
        assert!(encode_threshold_levels(&[]).is_err());
        assert!(encode_threshold_levels(&[2.0, 1.0]).is_err());
    }

    #[test]
    fn threshold_levels_apply_to_mask_and_class_images() {
        let input = Voxels::F64(
            ndarray::Array3::from_shape_vec((1, 1, 4), vec![0.0, 2.0, 5.0, f64::NAN]).unwrap(),
        );
        let mut mask = Voxels::Bool(ndarray::Array3::from_elem((1, 1, 4), false));
        apply_threshold_levels(
            &input,
            &[2.0],
            GlobalThresholdOutput::Mask {
                test: ThresholdTest::AtOrAbove,
            },
            &mut mask,
        )
        .unwrap();
        assert_eq!(
            mask.view::<bool>()
                .unwrap()
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![false, true, true, false]
        );

        let input = Voxels::F64(
            ndarray::Array3::from_shape_vec((1, 1, 4), vec![0.0, 2.0, 5.0, 9.0]).unwrap(),
        );
        let mut classes = Voxels::U32(ndarray::Array3::zeros((1, 1, 4)));
        apply_threshold_levels(
            &input,
            &[2.0, 5.0],
            GlobalThresholdOutput::Classes,
            &mut classes,
        )
        .unwrap();
        assert_eq!(
            classes
                .view::<u32>()
                .unwrap()
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![0, 0, 1, 2]
        );
    }

    #[test]
    fn threshold_fragment_ops_state_their_planner_contract() {
        let samples = GlobalThresholdSamplesOp::new("threshold samples", "threshold-values");
        assert_eq!(samples.name(), "threshold samples");
        assert!(samples.reads_pixels());
        assert!(!samples.writes_pixels());
        assert_eq!(samples.seam_fold(), Some(SeamFold::PerBlock));
        let outputs = samples.outputs();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].coverage, Coverage::EveryBlock);
        assert_eq!(outputs[0].lifecycle, Lifecycle::DeleteOnExit);

        let apply = ApplyGlobalThresholdOp::new(
            "apply threshold",
            "threshold-values",
            3,
            GlobalThreshold::Mean,
            GlobalThresholdOutput::Classes,
        );
        assert!(!apply.reads_pixels());
        assert!(apply.writes_pixels());
        assert!(apply.barrier());
        assert!(!apply.gathers());
        assert_eq!(apply.produces(Dtype::F64), Dtype::U32);
        assert_eq!(apply.inputs()[0].phase, 3);
        assert_eq!(apply.source_inputs([5, 5, 5])[0].image.index(), 0);
        assert_eq!(apply.seam_fold(), Some(SeamFold::Unordered));
    }

    #[test]
    fn yen_li_triangle_and_minimum_thresholds_have_stated_rules() {
        let values = [0.0, 0.0, 0.0, 10.0, 10.0, 10.0];
        assert_eq!(yen_threshold(&values, 2).unwrap(), 2.5);
        assert!((li_threshold(&values).unwrap() - 3.170323914242463).abs() < 1.0e-9);
        assert_eq!(triangle_threshold(&values, 2).unwrap(), 7.5);
        assert_eq!(minimum_threshold(&values, 3, 8).unwrap(), 5.0);

        assert_eq!(GlobalThreshold::yen(2).unwrap().of(values).unwrap(), 2.5);
        assert!((GlobalThreshold::Li.of(values).unwrap() - 3.170323914242463).abs() < 1.0e-9);
        assert_eq!(
            GlobalThreshold::triangle(2).unwrap().of(values).unwrap(),
            7.5
        );
        assert_eq!(
            GlobalThreshold::minimum(3, 8).unwrap().of(values).unwrap(),
            5.0
        );
    }

    #[test]
    fn otsu_threshold_has_stated_degenerate_cases() {
        assert_eq!(
            GlobalThreshold::otsu(256)
                .unwrap()
                .of([7.0, 7.0, 7.0])
                .unwrap(),
            7.0
        );
        assert_eq!(
            multi_otsu_thresholds(&[7.0, 7.0, 7.0], 4, 3).unwrap(),
            vec![7.0, 7.0]
        );
        assert!(GlobalThreshold::otsu(0).is_err());
        assert!(GlobalThreshold::yen(0).is_err());
        assert!(GlobalThreshold::triangle(0).is_err());
        assert!(GlobalThreshold::minimum(2, 0).is_err());
        assert!(GlobalThreshold::Mean.of([f64::NAN]).is_err());
        assert!(otsu_threshold(&[], 2).is_err());
        assert_eq!(yen_threshold(&[7.0, 7.0], 256).unwrap(), 7.0);
        assert_eq!(li_threshold(&[7.0, 7.0]).unwrap(), 7.0);
        assert_eq!(triangle_threshold(&[7.0, 7.0], 256).unwrap(), 7.0);
        assert_eq!(minimum_threshold(&[7.0, 7.0], 256, 8).unwrap(), 7.0);
        assert!(minimum_threshold(&[0.0, 1.0, 2.0], 3, 1).is_err());
        assert!(multi_otsu_thresholds(&[0.0, 1.0], 2, 1).is_err());
        assert!(multi_otsu_thresholds(&[0.0, 1.0], 2, 3).is_err());
    }
}
