// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Per-vertex vessel radius: the distance from a skeleton vertex to the
//! nearest background voxel of the vessel mask, by a fixed outward walk.
//!
//! It is not a distance transform. It is `blockflow::ops::walk`: a fixed list
//! of offsets inside an ellipsoid of half-extent [`RadiusParams::max_radius`],
//! ordered by Euclidean length, walked from each vertex until the voxel it reaches is at or
//! below [`RadiusParams::threshold`]. Offsets that leave the volume are skipped
//! rather than counted as background, so a vessel cut by the volume face is
//! measured to its nearest in-volume wall.
//!
//! **There is one search and it finds one offset**; all four arrays are that
//! offset written down four ways, in two independent splits — voxels or µm,
//! scalar or per axis:
//!
//! ```text
//! radii_axial[a]        = |stop[a]|
//! radius_units_axial[a] = |stop[a]| * spacing[a]
//! radii                 = ||radii_axial||
//! radius_units          = ||radius_units_axial||
//! ```
//!
//! The scalar voxel radius is exact. The axial arrays name one member of a tie
//! group (every permutation and sign flip of an offset has the same key), so
//! they depend on the walk's tie order, which is blockflow's ascending
//! lexicographic rule; [`RadiusMeasurement::determined`] says which rows had a
//! unique stopping offset. A vertex whose walk finds no background is reported
//! as [`NOT_FOUND`] in every array and flagged in [`RadiusMeasurement::found`].

use std::sync::Arc;

use blockflow::error::{Error, Result};
use blockflow::ops::rows::Limit;
use blockflow::ops::walk::{walk_from, OffsetSequence};
use blockflow::voxels::Voxels;

use super::columns::{norm, Column, RawColumn};
use super::store::{RadiusKind, Stops};
use super::vessel::Carried;

/// The radius reported where the walk found no background voxel.
pub const NOT_FOUND: f64 = -1.0;

#[derive(Debug, Clone, PartialEq)]
pub struct RadiusParams {
    /// Half-extent of the search, in voxels per axis. The ellipsoid's offset
    /// count grows with the cube of this, and every offset is walked from a
    /// vertex inside a solid region, so it is the cost knob.
    pub max_radius: [usize; 3],
    /// A voxel at or below this value is background.
    pub threshold: f64,
}

impl Default for RadiusParams {
    fn default() -> Self {
        Self {
            max_radius: [50, 50, 50],
            threshold: 0.0,
        }
    }
}

/// The four radius arrays, one row per point, plus two flags.
#[derive(Debug, Clone, PartialEq)]
pub struct RadiusMeasurement {
    pub radii: Vec<f64>,
    pub radii_axial: Vec<[f64; 3]>,
    pub radius_units: Vec<f64>,
    pub radius_units_axial: Vec<[f64; 3]>,
    /// False where the walk ran out of offsets.
    pub found: Vec<bool>,
    /// False where the stopping offset shared its key with another, so the
    /// axial arrays depend on the tie order.
    pub determined: Vec<bool>,
}

impl RadiusMeasurement {
    pub fn len(&self) -> usize {
        self.radii.len()
    }

    pub fn is_empty(&self) -> bool {
        self.radii.is_empty()
    }

    pub fn not_found(&self) -> usize {
        self.found.iter().filter(|found| !**found).count()
    }

    /// The four arrays as named columns, the names later stages read.
    pub fn columns(&self) -> [(&'static str, Column); 4] {
        [
            ("radii", Column::scalar(self.radii.clone())),
            ("radii_axial", Column::rows3(&self.radii_axial)),
            ("radius_units", Column::scalar(self.radius_units.clone())),
            (
                "radius_units_axial",
                Column::rows3(&self.radius_units_axial),
            ),
        ]
    }
}

/// The search, built once for a parameter point and reused for every call.
#[derive(Debug, Clone, PartialEq)]
pub struct RadiusSearch {
    params: RadiusParams,
    sequence: OffsetSequence,
}

impl RadiusSearch {
    pub fn new(params: RadiusParams) -> Result<Self> {
        // The ellipsoid's members, ordered by their Euclidean length in
        // voxels. For an isotropic maximum that is the order
        // `OffsetSequence::ellipsoid` gives; for an anisotropic one it is
        // still the nearest background voxel, which the axis-normalised order
        // is not.
        let maximum = params.max_radius;
        let divisor = maximum.map(|value| value.max(1) as f64);
        let mut members = Vec::new();
        for a in -(maximum[0] as isize)..=(maximum[0] as isize) {
            for b in -(maximum[1] as isize)..=(maximum[1] as isize) {
                for c in -(maximum[2] as isize)..=(maximum[2] as isize) {
                    let offset = [a, b, c];
                    let mut key = 0.0f64;
                    for axis in 0..3 {
                        let scaled = offset[axis] as f64 / divisor[axis];
                        key += scaled * scaled;
                    }
                    if key <= 1.0 {
                        members.push(offset);
                    }
                }
            }
        }
        let sequence = OffsetSequence::from_offsets(members, [1.0; 3])?;
        Ok(Self { params, sequence })
    }

    pub fn params(&self) -> &RadiusParams {
        &self.params
    }

    pub fn sequence(&self) -> &OffsetSequence {
        &self.sequence
    }

    /// Measure at every point of `source` (the vessel mask, or any volume
    /// whose background is at or below the threshold), with `spacing` in µm
    /// per voxel.
    pub fn measure(
        &self,
        source: &Voxels,
        points: &[[u32; 3]],
        spacing: [f64; 3],
    ) -> Result<RadiusMeasurement> {
        check_spacing(spacing)?;
        let stops = self.stops(source, points)?;
        self.from_stops(&stops, spacing)
    }

    /// The walk's stop at every point of `source` (see [`Self::stop_at`]).
    pub fn stops(&self, source: &Voxels, points: &[[u32; 3]]) -> Result<Vec<Option<usize>>> {
        let volume = source.shape();
        points
            .iter()
            .map(|at| self.stop_at(source, [0, 0, 0], volume, *at))
            .collect()
    }

    /// The walk from one point: the index of the offset it stopped at, `None`
    /// where it found no background. `window` holds the volume over a box
    /// whose lowest voxel is `origin` in a volume of shape `volume`; it must
    /// reach [`RadiusParams::max_radius`] beyond the point on every side that
    /// is inside the volume, or the walk is refused.
    pub fn stop_at(
        &self,
        window: &Voxels,
        origin: [usize; 3],
        volume: [usize; 3],
        at: [u32; 3],
    ) -> Result<Option<usize>> {
        let at = at.map(|value| value as usize);
        walk_from(
            &self.sequence,
            at,
            volume,
            window,
            origin,
            Limit::AtMost(self.params.threshold),
        )
    }

    /// The per-axis radius in µm a stop gives, [`NOT_FOUND`] where there is
    /// none: what [`RadiusMeasurement::radius_units_axial`] holds.
    pub fn units_axial(&self, stopped: Option<usize>, spacing: [f64; 3]) -> [f64; 3] {
        match stopped {
            Some(index) => {
                let offset = self.sequence.offsets()[index];
                std::array::from_fn(|axis| offset[axis].unsigned_abs() as f64 * spacing[axis])
            }
            None => [NOT_FOUND; 3],
        }
    }

    /// The four arrays from one stop per point, in point order.
    pub fn from_stops(
        &self,
        stops: &[Option<usize>],
        spacing: [f64; 3],
    ) -> Result<RadiusMeasurement> {
        check_spacing(spacing)?;
        let mut out = RadiusMeasurement {
            radii: Vec::with_capacity(stops.len()),
            radii_axial: Vec::with_capacity(stops.len()),
            radius_units: Vec::with_capacity(stops.len()),
            radius_units_axial: Vec::with_capacity(stops.len()),
            found: Vec::with_capacity(stops.len()),
            determined: Vec::with_capacity(stops.len()),
        };
        for stopped in stops {
            match *stopped {
                Some(index) => {
                    let offset = self.sequence.offsets()[index];
                    let voxels = offset.map(|value| value.unsigned_abs() as f64);
                    let mut units = [0.0f64; 3];
                    for axis in 0..3 {
                        units[axis] = voxels[axis] * spacing[axis];
                    }
                    out.radii.push(norm(&voxels));
                    out.radii_axial.push(voxels);
                    out.radius_units.push(norm(&units));
                    out.radius_units_axial.push(units);
                    out.found.push(true);
                    out.determined.push(self.sequence.stop_is_determined(index));
                }
                None => {
                    out.radii.push(NOT_FOUND);
                    out.radii_axial.push([NOT_FOUND; 3]);
                    out.radius_units.push(NOT_FOUND);
                    out.radius_units_axial.push([NOT_FOUND; 3]);
                    out.found.push(false);
                    out.determined.push(true);
                }
            }
        }
        Ok(out)
    }
}

fn check_spacing(spacing: [f64; 3]) -> Result<()> {
    for (axis, value) in spacing.iter().enumerate() {
        if !value.is_finite() || *value <= 0.0 {
            return Err(Error::invalid(format!(
                "the spacing on axis {axis} is {value}; a µm-per-voxel spacing is positive \
                 and finite."
            )));
        }
    }
    Ok(())
}

/// The four radius arrays of one stop per raw vertex, as the columns the
/// graph carries (folded by their maximum at cliques, the scalars by their
/// maximum along edges): one `u32` a vertex between the four of them, read
/// as [`RadiusSearch::from_stops`] writes them.
pub fn carried_radii(
    search: Arc<RadiusSearch>,
    stops: &[Option<usize>],
    spacing: [f64; 3],
) -> Result<Vec<Carried>> {
    let stops = stops
        .iter()
        .map(|stop| Stops::encode(*stop))
        .collect::<Result<Vec<_>>>()?;
    carried_stops(search, stops, spacing)
}

/// [`carried_radii`] of stops already encoded ([`Stops::NONE`] for none).
pub(crate) fn carried_stops(
    search: Arc<RadiusSearch>,
    stops: Vec<u32>,
    spacing: [f64; 3],
) -> Result<Vec<Carried>> {
    check_spacing(spacing)?;
    let stops = Arc::new(Stops::new(search, stops, spacing)?);
    Ok(RadiusKind::ALL
        .iter()
        .map(|(name, kind)| Carried::max(*name, RawColumn::stops(Arc::clone(&stops), *kind)))
        .collect())
}

/// One-call form of [`RadiusSearch::measure`].
pub fn radii(
    source: &Voxels,
    points: &[[u32; 3]],
    spacing: [f64; 3],
    params: &RadiusParams,
) -> Result<RadiusMeasurement> {
    RadiusSearch::new(params.clone())?.measure(source, points, spacing)
}
