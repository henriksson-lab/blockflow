// SPDX-License-Identifier: MIT
//
// Original work for this crate.

//! Opt-in per-object intensity statistics beyond the basic intensity row.
//!
//! [`IntensityStatistics`] is a built-in [`RegionMeasure`], added to a measurement with
//! [`super::measure::Measurements::intensity_statistics`]. Each feature family is off until
//! asked for, so a measurement that does not request one pays nothing for it:
//!
//! | family | columns | use |
//! |---|---|---|
//! | [`IntensityStatistics::with_dispersion`] | `std_dev`, `coefficient_of_variation` | spread of the signal within the object, exact rather than from a histogram |
//! | [`IntensityStatistics::with_mass_displacement`] | `mass_displacement` | distance from the object's centroid to its intensity-weighted centroid: redistribution of signal inside the object |
//! | [`IntensityStatistics::with_thresholds`] | `count_ge_<t>`, `fraction_ge_<t>` per threshold | how much of the object is above a level; with the dtype maximum, saturated voxels |
//!
//! Every row also carries `finite_count`, the voxels the statistics are over.
//!
//! With a mask that stays fixed over a time series, these change from frame to frame while
//! shape features do not. Two more time-resolved readouts come from composing existing parts
//! rather than from this measure:
//!
//! - Local background: measure the object's ring from [`super::cleanup::ring_labels_into`]
//!   (a gap, then a band) with the basic intensity or a distribution, and join by label.
//! - Compartment ratios (cytoplasm/nucleus for translocation reporters): measure a ring with
//!   `inner = 0` around the nucleus labels and divide by the nucleus row.
//!
//! **Exactness.** Values are accumulated as fixed-point integers (`fraction_bits` binary
//! places, 20 by default), so partials merge in any order to the same bytes
//! ([`FoldLaw::ExactAssociative`]). A value is rounded to that grid once, on entry; integer
//! images are exact at any setting. An accumulation that would overflow `i128` is refused
//! with an error naming the measure, rather than wrapping.

use crate::error::{Error, Result};
use crate::table::{Column, RowBuilder, Schema, Value};

use super::measure::{FoldLaw, MeasureSource, MeasureValues, MeasurementKey, RegionMeasure};

/// Default binary places of the fixed-point accumulation.
pub const DEFAULT_STATISTICS_FRACTION_BITS: u32 = 20;
const MAX_STATISTICS_FRACTION_BITS: u32 = 40;

/// Which per-object intensity statistics to compute over one intensity image.
#[derive(Debug, Clone, PartialEq)]
pub struct IntensityStatistics {
    key: String,
    source: String,
    dispersion: bool,
    mass_displacement: Option<[f64; 3]>,
    thresholds: Vec<f64>,
    fraction_bits: u32,
}

impl IntensityStatistics {
    /// No families yet: only `finite_count`. `key` names the measurement (its row stream and
    /// error messages); `source` names the intensity image it reads.
    pub fn new(key: impl Into<String>, source: impl Into<String>) -> Result<Self> {
        let key = key.into();
        let source = source.into();
        MeasurementKey::new(key.clone())?;
        MeasureSource::intensity(source.clone())?;
        Ok(Self {
            key,
            source,
            dispersion: false,
            mass_displacement: None,
            thresholds: Vec::new(),
            fraction_bits: DEFAULT_STATISTICS_FRACTION_BITS,
        })
    }

    /// Exact population standard deviation and coefficient of variation (`std_dev / |mean|`).
    pub fn with_dispersion(mut self) -> Self {
        self.dispersion = true;
        self
    }

    /// Distance between the object's centroid and its intensity-weighted centroid, in the units
    /// of `spacing` (ZYX; `[1.0; 3]` for voxels). Undefined (NaN) when the object's summed
    /// intensity is not positive.
    pub fn with_mass_displacement(mut self, spacing: [f64; 3]) -> Result<Self> {
        if spacing
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(Error::invalid(format!(
                "intensity statistics: spacing must be finite and positive, got {spacing:?}"
            )));
        }
        self.mass_displacement = Some(spacing);
        Ok(self)
    }

    /// Count and fraction of the object's finite voxels with a value `>= threshold`, one pair
    /// of columns per threshold, in the order given.
    pub fn with_thresholds(mut self, thresholds: &[f64]) -> Result<Self> {
        for &threshold in thresholds {
            if !threshold.is_finite() {
                return Err(Error::invalid(format!(
                    "intensity statistics: thresholds must be finite, got {threshold}"
                )));
            }
            if self.thresholds.contains(&threshold) {
                return Err(Error::invalid(format!(
                    "intensity statistics: threshold {threshold} is listed twice"
                )));
            }
            self.thresholds.push(threshold);
        }
        Ok(self)
    }

    /// Binary places of the fixed-point accumulation (0 to 40).
    pub fn with_fraction_bits(mut self, bits: u32) -> Result<Self> {
        if bits > MAX_STATISTICS_FRACTION_BITS {
            return Err(Error::invalid(format!(
                "intensity statistics: fraction bits must be at most \
                 {MAX_STATISTICS_FRACTION_BITS}, got {bits}"
            )));
        }
        self.fraction_bits = bits;
        Ok(self)
    }

    fn scale(&self) -> f64 {
        (1u64 << self.fraction_bits) as f64
    }

    fn quantize(&self, value: f64) -> Result<i128> {
        let scaled = (value * self.scale()).round();
        // 2^62 keeps a square and a coordinate product far inside i128.
        if scaled.abs() >= (1u64 << 62) as f64 {
            return Err(Error::invalid(format!(
                "intensity statistics {:?}: value {value} does not fit {} fraction bits",
                self.key, self.fraction_bits
            )));
        }
        Ok(scaled as i128)
    }

    fn overflow(&self) -> Error {
        Error::invalid(format!(
            "intensity statistics {:?}: accumulation overflowed; lower the fraction bits",
            self.key
        ))
    }
}

/// Column-name form of a threshold: `65535`, `0p5`, `m2` for -2.
fn threshold_name(threshold: f64) -> String {
    format!("{threshold}").replace('-', "m").replace('.', "p")
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct IntensityStatisticsPartial {
    /// The lexicographically first voxel: the row's position.
    first: Option<[usize; 3]>,
    finite: u64,
    sum: i128,
    sum_squares: i128,
    position_sum: [i128; 3],
    weighted_position_sum: [i128; 3],
    at_or_above: Vec<u64>,
}

fn checked_add(into: &mut i128, value: i128, error: impl Fn() -> Error) -> Result<()> {
    *into = into.checked_add(value).ok_or_else(error)?;
    Ok(())
}

impl RegionMeasure for IntensityStatistics {
    type Partial = IntensityStatisticsPartial;

    fn key(&self) -> MeasurementKey {
        MeasurementKey::new(self.key.clone()).expect("validated in IntensityStatistics::new")
    }

    fn schema(&self) -> Schema {
        let mut columns = vec![Column::u64("label"), Column::u64("finite_count")];
        if self.dispersion {
            columns.push(Column::f64("std_dev"));
            columns.push(Column::f64("coefficient_of_variation"));
        }
        if self.mass_displacement.is_some() {
            columns.push(Column::f64("mass_displacement"));
        }
        for &threshold in &self.thresholds {
            let name = threshold_name(threshold);
            columns.push(Column::u64(format!("count_ge_{name}")));
            columns.push(Column::f64(format!("fraction_ge_{name}")));
        }
        Schema::new(columns).expect("intensity statistics column names are distinct")
    }

    fn sources(&self) -> Vec<MeasureSource> {
        vec![MeasureSource::intensity(self.source.clone()).expect("validated in new")]
    }

    fn fold_law(&self) -> FoldLaw {
        FoldLaw::ExactAssociative
    }

    fn cost_per_voxel(&self) -> f64 {
        1.0 + self.thresholds.len() as f64 * 0.25
    }

    fn zero(&self) -> Self::Partial {
        IntensityStatisticsPartial {
            at_or_above: vec![0; self.thresholds.len()],
            ..Default::default()
        }
    }

    fn add(
        &self,
        partial: &mut Self::Partial,
        at: [usize; 3],
        _label: u64,
        values: MeasureValues<'_>,
    ) -> Result<()> {
        partial.first = Some(partial.first.map_or(at, |first| first.min(at)));
        let value = values.get(0).ok_or_else(|| {
            Error::invalid(format!(
                "intensity statistics {:?}: no intensity value",
                self.key
            ))
        })?;
        if !value.is_finite() {
            return Ok(());
        }
        partial.finite += 1;
        for (count, &threshold) in partial.at_or_above.iter_mut().zip(&self.thresholds) {
            *count += u64::from(value >= threshold);
        }
        if !self.dispersion && self.mass_displacement.is_none() {
            return Ok(());
        }
        let q = self.quantize(value)?;
        checked_add(&mut partial.sum, q, || self.overflow())?;
        if self.dispersion {
            checked_add(&mut partial.sum_squares, q * q, || self.overflow())?;
        }
        if self.mass_displacement.is_some() {
            for axis in 0..3 {
                let coordinate = at[axis] as i128;
                checked_add(&mut partial.position_sum[axis], coordinate, || {
                    self.overflow()
                })?;
                let weighted = q.checked_mul(coordinate).ok_or_else(|| self.overflow())?;
                checked_add(&mut partial.weighted_position_sum[axis], weighted, || {
                    self.overflow()
                })?;
            }
        }
        Ok(())
    }

    fn merge(&self, into: &mut Self::Partial, from: Self::Partial) -> Result<()> {
        into.first = match (into.first, from.first) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        into.finite += from.finite;
        checked_add(&mut into.sum, from.sum, || self.overflow())?;
        checked_add(&mut into.sum_squares, from.sum_squares, || self.overflow())?;
        for axis in 0..3 {
            checked_add(
                &mut into.position_sum[axis],
                from.position_sum[axis],
                || self.overflow(),
            )?;
            checked_add(
                &mut into.weighted_position_sum[axis],
                from.weighted_position_sum[axis],
                || self.overflow(),
            )?;
        }
        for (a, b) in into.at_or_above.iter_mut().zip(from.at_or_above) {
            *a += b;
        }
        Ok(())
    }

    fn finish(&self, label: u64, partial: Self::Partial, out: &mut RowBuilder) -> Result<()> {
        let (at, row) = self.row(label, partial);
        out.push(at, &row)
    }
}

impl IntensityStatistics {
    /// One object's row: its position and its values in [`RegionMeasure::schema`] order.
    pub fn row(&self, label: u64, partial: IntensityStatisticsPartial) -> ([usize; 3], Vec<Value>) {
        let n = partial.finite;
        let scale = self.scale();
        let mut row = vec![Value::U64(label), Value::U64(n)];
        if self.dispersion {
            let (std_dev, cv) = if n == 0 {
                (f64::NAN, f64::NAN)
            } else {
                // n·Σq² − (Σq)² exactly when it fits, so a constant object is exactly 0.
                let numerator = (n as i128)
                    .checked_mul(partial.sum_squares)
                    .and_then(|a| partial.sum.checked_mul(partial.sum).map(|b| (a, b)))
                    .and_then(|(a, b)| a.checked_sub(b))
                    .map(|exact| exact as f64)
                    .unwrap_or_else(|| {
                        n as f64 * partial.sum_squares as f64
                            - (partial.sum as f64) * (partial.sum as f64)
                    })
                    .max(0.0);
                let std_dev = numerator.sqrt() / n as f64 / scale;
                let mean = partial.sum as f64 / n as f64 / scale;
                let cv = if mean == 0.0 {
                    f64::NAN
                } else {
                    std_dev / mean.abs()
                };
                (std_dev, cv)
            };
            row.push(Value::F64(std_dev));
            row.push(Value::F64(cv));
        }
        if let Some(spacing) = self.mass_displacement {
            let displacement = if n == 0 || partial.sum <= 0 {
                f64::NAN
            } else {
                let mut squared = 0.0;
                for axis in 0..3 {
                    let centroid = partial.position_sum[axis] as f64 / n as f64;
                    let weighted = partial.weighted_position_sum[axis] as f64 / partial.sum as f64;
                    let delta = (weighted - centroid) * spacing[axis];
                    squared += delta * delta;
                }
                squared.sqrt()
            };
            row.push(Value::F64(displacement));
        }
        for count in &partial.at_or_above {
            row.push(Value::U64(*count));
            row.push(Value::F64(if n == 0 {
                f64::NAN
            } else {
                *count as f64 / n as f64
            }));
        }
        (partial.first.unwrap_or([0; 3]), row)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::measure::{
        assert_custom_region_builder_invariant, assert_region_measurement_invariant,
    };
    use ndarray::Array3;

    fn statistics() -> IntensityStatistics {
        IntensityStatistics::new("intensity_statistics", "signal")
            .unwrap()
            .with_dispersion()
            .with_mass_displacement([1.0, 0.5, 0.5])
            .unwrap()
            .with_thresholds(&[10.0, 0.5])
            .unwrap()
    }

    /// Rows of a resident run, in label order.
    fn rows(labels: &Array3<f64>, signal: &Array3<f64>) -> Vec<Vec<Value>> {
        let measure = statistics();
        let mut partials = std::collections::BTreeMap::new();
        for ((z, y, x), &label) in labels.indexed_iter() {
            if label == 0.0 {
                continue;
            }
            let partial = partials
                .entry(label as u64)
                .or_insert_with(|| measure.zero());
            measure
                .add(
                    partial,
                    [z, y, x],
                    label as u64,
                    MeasureValues::new(&[signal[[z, y, x]]]),
                )
                .unwrap();
        }
        partials
            .into_iter()
            .map(|(label, partial)| measure.row(label, partial).1)
            .collect()
    }

    fn fixture() -> (Array3<f64>, Array3<f64>) {
        // Object 1: a constant row of 4 voxels at value 2. Object 2: values 0, 0, 0, 20 on a row,
        // all signal at its right end, plus one non-finite voxel.
        let mut labels = Array3::<f64>::zeros((1, 3, 6));
        let mut signal = Array3::<f64>::zeros((1, 3, 6));
        for x in 0..4 {
            labels[[0, 0, x]] = 1.0;
            signal[[0, 0, x]] = 2.0;
            labels[[0, 2, x]] = 2.0;
        }
        signal[[0, 2, 3]] = 20.0;
        labels[[0, 2, 4]] = 2.0;
        signal[[0, 2, 4]] = f64::NAN;
        (labels, signal)
    }

    #[test]
    fn statistics_are_exact_and_name_their_columns() {
        let measure = statistics();
        let names: Vec<String> = measure
            .schema()
            .columns()
            .iter()
            .map(|c| c.name().to_owned())
            .collect();
        assert_eq!(
            names,
            [
                "label",
                "finite_count",
                "std_dev",
                "coefficient_of_variation",
                "mass_displacement",
                "count_ge_10",
                "fraction_ge_10",
                "count_ge_0p5",
                "fraction_ge_0p5"
            ]
        );
        let (labels, signal) = fixture();
        let rows = rows(&labels, &signal);
        let get = |row: &Vec<Value>, column: usize| match row[column] {
            Value::F64(value) => value,
            Value::U64(value) => value as f64,
        };
        let constant = &rows[0];
        assert_eq!(get(constant, 0), 1.0);
        assert_eq!(get(constant, 1), 4.0);
        assert_eq!(
            get(constant, 2),
            0.0,
            "a constant object has exactly zero spread"
        );
        assert_eq!(get(constant, 3), 0.0);
        assert_eq!(get(constant, 4), 0.0, "uniform intensity: no displacement");
        assert_eq!((get(constant, 5), get(constant, 7)), (0.0, 4.0));
        assert_eq!(get(constant, 8), 1.0);

        let skewed = &rows[1];
        assert_eq!(get(skewed, 1), 4.0, "the NaN voxel is not counted");
        // Values 0,0,0,20: mean 5, population std sqrt(75).
        assert!((get(skewed, 2) - 75f64.sqrt()).abs() < 1e-12);
        assert!((get(skewed, 3) - 75f64.sqrt() / 5.0).abs() < 1e-12);
        // Centroid x = 1.5, weighted x = 3: 1.5 voxels × 0.5 spacing.
        assert!((get(skewed, 4) - 0.75).abs() < 1e-12);
        assert_eq!((get(skewed, 5), get(skewed, 6)), (1.0, 0.25));
    }

    #[test]
    fn statistics_do_not_depend_on_the_block_decomposition() {
        let (labels, signal) = fixture();
        let measure = statistics();
        assert_region_measurement_invariant(labels.view(), &[signal.view()], &measure).unwrap();
        assert_custom_region_builder_invariant(
            labels.view(),
            &[signal.view()],
            measure,
            &[[1, 1, 2], [1, 2, 3], [1, 3, 6]],
        )
        .unwrap();
    }

    #[test]
    fn requests_are_validated() {
        let base = IntensityStatistics::new("stats", "signal").unwrap();
        assert!(base.clone().with_thresholds(&[f64::NAN]).is_err());
        assert!(base.clone().with_thresholds(&[1.0, 1.0]).is_err());
        assert!(base
            .clone()
            .with_mass_displacement([1.0, 0.0, 1.0])
            .is_err());
        assert!(base.clone().with_fraction_bits(41).is_err());
        assert!(IntensityStatistics::new("has space", "signal").is_err());
        // Nothing requested: only the count.
        assert_eq!(base.schema().len(), 2);
        let huge = base.with_dispersion().with_fraction_bits(40).unwrap();
        let mut partial = huge.zero();
        assert!(huge
            .add(&mut partial, [0; 3], 1, MeasureValues::new(&[1e10]))
            .is_err());
    }
}
