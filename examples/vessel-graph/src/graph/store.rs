// SPDX-License-Identifier: MIT
// Original work for this crate.

//! How the graph holds its numbers: each column in the narrowest type that
//! gives every value back bit for bit.
//!
//! Every column reads as `f64`, and every output (the tables, the JSON) is
//! written from those `f64`s, so a narrower store is only a store: a radius
//! of whole voxels, an artery-channel maximum of a `uint16` image, a flag or
//! a region id each fit 1 or 4 bytes and read back as the same `f64`. The
//! narrowing is decided per column by looking at every value ([`Values::of`]),
//! or up front where the source's type says it ([`Values::with_capacity`]),
//! never by rounding.

use std::sync::Arc;

use blockflow::error::{Error, Result};
use blockflow::Dtype;

use super::radius::NOT_FOUND;
use super::RadiusSearch;

/// One column's values, flat (`width` per row), as stored.
#[derive(Debug, Clone)]
pub(crate) enum Values {
    F64(Vec<f64>),
    /// Values that are each exactly an `f32`.
    F32(Vec<f32>),
    /// Integers in `i32` (not `-0.0`).
    I32(Vec<i32>),
    /// `0.0` and `1.0` (not `-0.0`).
    Flags(Vec<u8>),
}

/// How a column built value by value is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Narrow {
    F64,
    F32,
    Flags,
}

impl Narrow {
    /// The store that holds every value a voxel of `dtype` reads as
    /// (`ops::rows::value_at`), and its maximum over a ball.
    pub(crate) fn of_dtype(dtype: Dtype) -> Self {
        match dtype {
            Dtype::Bool | Dtype::U8 | Dtype::U16 | Dtype::I8 | Dtype::I16 | Dtype::F32 => Self::F32,
            _ => Self::F64,
        }
    }
}

impl Values {
    /// The narrowest store that gives back every value's bits.
    pub(crate) fn of(values: Vec<f64>) -> Self {
        let (mut flags, mut integers, mut singles) = (true, true, true);
        for value in &values {
            let bits = value.to_bits();
            flags &= bits == 0f64.to_bits() || bits == 1f64.to_bits();
            integers &= ((*value as i32) as f64).to_bits() == bits;
            singles &= ((*value as f32) as f64).to_bits() == bits;
            if !(flags || integers || singles) {
                return Self::F64(values);
            }
        }
        if flags {
            Self::Flags(values.iter().map(|value| *value as u8).collect())
        } else if integers {
            Self::I32(values.iter().map(|value| *value as i32).collect())
        } else {
            Self::F32(values.iter().map(|value| *value as f32).collect())
        }
    }

    /// An empty store of the kind `narrow` names.
    pub(crate) fn with_capacity(narrow: Narrow, capacity: usize) -> Self {
        match narrow {
            Narrow::F64 => Self::F64(Vec::with_capacity(capacity)),
            Narrow::F32 => Self::F32(Vec::with_capacity(capacity)),
            Narrow::Flags => Self::Flags(Vec::with_capacity(capacity)),
        }
    }

    /// Append `value`, which must fit the store (it does when the store was
    /// chosen from the value's source type).
    pub(crate) fn push(&mut self, value: f64) -> Result<()> {
        let fits = match self {
            Self::F64(values) => {
                values.push(value);
                true
            }
            Self::F32(values) => {
                values.push(value as f32);
                ((value as f32) as f64).to_bits() == value.to_bits()
            }
            Self::I32(values) => {
                values.push(value as i32);
                ((value as i32) as f64).to_bits() == value.to_bits()
            }
            Self::Flags(values) => {
                values.push(value as u8);
                value.to_bits() == 0f64.to_bits() || value.to_bits() == 1f64.to_bits()
            }
        };
        if fits {
            Ok(())
        } else {
            Err(Error::invalid(format!(
                "the value {value} does not fit the store its source's type chose."
            )))
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::F64(values) => values.len(),
            Self::F32(values) => values.len(),
            Self::I32(values) => values.len(),
            Self::Flags(values) => values.len(),
        }
    }

    #[inline]
    pub(crate) fn get(&self, index: usize) -> f64 {
        match self {
            Self::F64(values) => values[index],
            Self::F32(values) => f64::from(values[index]),
            Self::I32(values) => f64::from(values[index]),
            Self::Flags(values) => f64::from(values[index]),
        }
    }

    pub(crate) fn to_f64(&self) -> Vec<f64> {
        (0..self.len()).map(|index| self.get(index)).collect()
    }

    /// Rows `rows` of a store of `width` values a row, in that order.
    pub(crate) fn gather(&self, width: usize, rows: &[u32]) -> Self {
        fn pick<T: Copy>(values: &[T], width: usize, rows: &[u32]) -> Vec<T> {
            let mut out = Vec::with_capacity(rows.len() * width);
            for row in rows {
                let start = *row as usize * width;
                out.extend_from_slice(&values[start..start + width]);
            }
            out
        }
        match self {
            Self::F64(values) => Self::F64(pick(values, width, rows)),
            Self::F32(values) => Self::F32(pick(values, width, rows)),
            Self::I32(values) => Self::I32(pick(values, width, rows)),
            Self::Flags(values) => Self::Flags(pick(values, width, rows)),
        }
    }

    /// `self` then `other`, in the store both fit (`f64` when they differ).
    pub(crate) fn append(self, other: Self) -> Self {
        match (self, other) {
            (Self::F64(mut a), Self::F64(b)) => {
                a.extend(b);
                Self::F64(a)
            }
            (Self::F32(mut a), Self::F32(b)) => {
                a.extend(b);
                Self::F32(a)
            }
            (Self::I32(mut a), Self::I32(b)) => {
                a.extend(b);
                Self::I32(a)
            }
            (Self::Flags(mut a), Self::Flags(b)) => {
                a.extend(b);
                Self::Flags(a)
            }
            (a, b) => {
                let mut values = a.to_f64();
                values.extend(b.to_f64());
                Self::of(values)
            }
        }
    }

    pub(crate) fn bytes(&self) -> usize {
        match self {
            Self::F64(values) => values.len() * 8,
            Self::F32(values) => values.len() * 4,
            Self::I32(values) => values.len() * 4,
            Self::Flags(values) => values.len(),
        }
    }
}

/// A store compares by the values it gives back.
impl PartialEq for Values {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len()
            && (0..self.len()).all(|index| self.get(index) == other.get(index))
    }
}

/// Which of the four radius arrays a column of stops reads as (see
/// [`super::radius`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RadiusKind {
    /// `radii`: the norm of the per-axis voxel radius.
    Radii,
    /// `radii_axial`: `|stop[a]|`.
    RadiiAxial,
    /// `radius_units`: the norm of the per-axis µm radius.
    Units,
    /// `radius_units_axial`: `|stop[a]| * spacing[a]`.
    UnitsAxial,
}

impl RadiusKind {
    pub(crate) const ALL: [(&'static str, Self); 4] = [
        ("radii", Self::Radii),
        ("radii_axial", Self::RadiiAxial),
        ("radius_units", Self::Units),
        ("radius_units_axial", Self::UnitsAxial),
    ];

    pub(crate) fn width(self) -> usize {
        match self {
            Self::Radii | Self::Units => 1,
            Self::RadiiAxial | Self::UnitsAxial => 3,
        }
    }
}

/// The radius walk's stopping offset per vertex, one `u32` a vertex
/// (`u32::MAX` where it found no background), and what turns a stop into the
/// four radius arrays: the same arithmetic as
/// [`RadiusSearch::from_stops`].
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Stops {
    pub(crate) stops: Vec<u32>,
    pub(crate) search: Arc<RadiusSearch>,
    pub(crate) spacing: [f64; 3],
}

impl Stops {
    pub(crate) const NONE: u32 = u32::MAX;

    /// The stops as `u32`s: [`Self::NONE`] for none, else the offset's
    /// index in the search's sequence.
    pub(crate) fn new(
        search: Arc<RadiusSearch>,
        stops: Vec<u32>,
        spacing: [f64; 3],
    ) -> Result<Self> {
        let length = search.sequence().offsets().len();
        if let Some(stop) = stops
            .iter()
            .find(|stop| **stop != Self::NONE && **stop as usize >= length)
        {
            return Err(Error::invalid(format!(
                "a radius stop at offset {stop} of a search of {length} offsets."
            )));
        }
        Ok(Self {
            stops,
            search,
            spacing,
        })
    }

    /// A stop as a `u32`.
    pub(crate) fn encode(stop: Option<usize>) -> Result<u32> {
        match stop {
            None => Ok(Self::NONE),
            Some(index) => u32::try_from(index)
                .ok()
                .filter(|index| *index != Self::NONE)
                .ok_or_else(|| Error::invalid(format!("a radius stop at offset {index}"))),
        }
    }

    #[inline]
    pub(crate) fn value(&self, row: usize, kind: RadiusKind, axis: usize) -> f64 {
        let stop = self.stops[row];
        if stop == Self::NONE {
            return NOT_FOUND;
        }
        let offset = self.search.sequence().offsets()[stop as usize];
        let voxels = offset.map(|value| value.unsigned_abs() as f64);
        match kind {
            RadiusKind::RadiiAxial => voxels[axis],
            RadiusKind::UnitsAxial => voxels[axis] * self.spacing[axis],
            RadiusKind::Radii => super::columns::norm(&voxels),
            RadiusKind::Units => {
                let mut units = [0.0f64; 3];
                for axis in 0..3 {
                    units[axis] = voxels[axis] * self.spacing[axis];
                }
                super::columns::norm(&units)
            }
        }
    }
}
