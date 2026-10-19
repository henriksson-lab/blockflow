// SPDX-License-Identifier: MIT
// Original work for this crate.

//! A channel read at each vertex, over a ball as wide as the vessel there.
//!
//! The artery/vein classifier needs, per vertex, the strongest artery-channel
//! signal inside the vessel (and a little beyond it), and whether a binary
//! channel mask covers it. Both are a **bounded gather-reduce**: read every
//! voxel within the vertex's own search radius and reduce. That is not
//! `ops::walk` (which stops at the first hit and answers with a distance), and
//! it is not a fixed-window filter (`ops::rank`, `ops::sliding`), because the
//! window differs per vertex and only listed points are read.
//!
//! The search radius comes from the measured radius, shifted: for an artery
//! intensity three voxels wider than the vessel, for the binary masks and the
//! vein intensity the vessel itself. With µm per-axis radii the radius in
//! voxels is `floor(max_a (r_a + shift_vx * mean(spacing)) / spacing_a)`,
//! clamped at 0. Offsets outside the volume are skipped.

use blockflow::error::{Error, Result};
use blockflow::ops::rows::value_at;
use blockflow::voxels::Voxels;

/// How the voxels of a ball reduce to one value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduce {
    Max,
    Min,
    Mean,
}

/// Search radii in voxels from µm per-axis radii, widened by `shift_vx`
/// voxels.
pub fn search_radii(
    radius_units_axial: &[[f64; 3]],
    spacing: [f64; 3],
    shift_vx: f64,
) -> Result<Vec<f64>> {
    let shift_um = shift_um(spacing, shift_vx)?;
    Ok(radius_units_axial
        .iter()
        .map(|row| search_radius(*row, spacing, shift_um))
        .collect())
}

/// `shift_vx` voxels in µm: the mean spacing times the shift.
pub fn shift_um(spacing: [f64; 3], shift_vx: f64) -> Result<f64> {
    for (axis, value) in spacing.iter().enumerate() {
        if !value.is_finite() || *value <= 0.0 {
            return Err(Error::invalid(format!(
                "the spacing on axis {axis} is {value}; a µm-per-voxel spacing is positive and \
                 finite."
            )));
        }
    }
    Ok(shift_vx * (spacing[0] + spacing[1] + spacing[2]) / 3.0)
}

/// One row of [`search_radii`], with the shift already in µm.
pub fn search_radius(row: [f64; 3], spacing: [f64; 3], shift_um: f64) -> f64 {
    let mut best = f64::NEG_INFINITY;
    for axis in 0..3 {
        best = best.max((row[axis] + shift_um) / spacing[axis]);
    }
    let floored = best.floor();
    if floored > 0.0 {
        floored
    } else {
        0.0
    }
}

/// The largest search radius accepted; beyond it a radius is a unit mistake.
const LARGEST_RADIUS: f64 = 512.0;

/// The offsets of a ball, nearest first, for reading a channel around points.
#[derive(Debug, Clone, PartialEq)]
pub struct Ball {
    radius: isize,
    offsets: Vec<(f64, [isize; 3])>,
}

impl Ball {
    /// Every offset a search radius up to `largest` voxels takes.
    pub fn new(largest: f64) -> Result<Self> {
        if !largest.is_finite() || largest > LARGEST_RADIUS {
            return Err(Error::invalid(format!(
                "a search radius of {largest} voxels; that is a unit mistake rather than a vessel."
            )));
        }
        let radius = largest.max(0.0).ceil() as isize;
        Ok(Self {
            radius,
            offsets: ball_offsets(radius),
        })
    }

    /// The radius of the box the offsets fill, in voxels: how far beyond a
    /// point a window must reach.
    pub fn radius(&self) -> usize {
        self.radius as usize
    }

    /// Reduce `window` over the ball of radius `reach` voxels around `at`.
    /// `window` holds the volume over a box whose lowest voxel is `origin` in a
    /// volume of shape `volume`, and must hold every offset of the ball that
    /// is inside the volume. Offsets outside the volume are skipped.
    pub fn reduce_at(
        &self,
        window: &Voxels,
        origin: [usize; 3],
        volume: [usize; 3],
        at: [u32; 3],
        reach: f64,
        reduce: Reduce,
    ) -> Result<f64> {
        let at = at.map(|value| value as isize);
        let taken = self
            .offsets
            .partition_point(|(distance, _)| *distance <= reach);
        if reach > self.radius as f64 {
            return Err(Error::invalid(format!(
                "a search radius of {reach} voxels from a ball of {} voxels; the ball is built \
                 for the largest radius it is asked for.",
                self.radius
            )));
        }
        if reduce == Reduce::Max {
            if let Some(best) = self.max_inside(window, origin, volume, at, taken) {
                return Ok(best);
            }
        }
        let mut total = 0.0f64;
        let mut count = 0usize;
        let mut best: Option<f64> = None;
        for (_, offset) in &self.offsets[..taken] {
            let mut local = [0usize; 3];
            let mut inside = true;
            for axis in 0..3 {
                let target = at[axis] + offset[axis];
                if target < 0 || target >= volume[axis] as isize {
                    inside = false;
                    break;
                }
                local[axis] = (target as usize).checked_sub(origin[axis]).ok_or_else(|| {
                    Error::invalid(format!(
                        "a ball around {at:?} reached {target} on axis {axis}, below the \
                         window starting at {origin:?}"
                    ))
                })?;
            }
            if !inside {
                continue;
            }
            let value = value_at(window, local)?;
            total += value;
            count += 1;
            best = Some(match (best, reduce) {
                (None, _) => value,
                (Some(current), Reduce::Max) => current.max(value),
                (Some(current), Reduce::Min) => current.min(value),
                (Some(current), Reduce::Mean) => current,
            });
        }
        let Some(best) = best else {
            return Err(Error::invalid(format!(
                "the point {at:?} is outside the {volume:?} volume."
            )));
        };
        Ok(match reduce {
            Reduce::Mean => total / count as f64,
            _ => best,
        })
    }
}

impl Ball {
    /// The maximum over the first `taken` offsets without a check per offset,
    /// where that is the same answer: the ball's box lies inside the volume
    /// and the window, the window is contiguous, and it holds `bool`, `u8` or
    /// `u16` (every value finite, the maximum exact in `f64`). `None` where
    /// any of that does not hold, for the general loop to answer.
    fn max_inside(
        &self,
        window: &Voxels,
        origin: [usize; 3],
        volume: [usize; 3],
        at: [isize; 3],
        taken: usize,
    ) -> Option<f64> {
        if taken == 0 {
            return None;
        }
        let offsets = &self.offsets[..taken];
        // The offsets taken are within their own largest distance, so within
        // that many voxels of the point on every axis.
        let extent = offsets
            .last()
            .map_or(0, |(distance, _)| distance.floor() as isize);
        let shape = window.shape();
        for axis in 0..3 {
            let low = at[axis] - extent;
            let high = at[axis] + extent;
            if low < origin[axis] as isize
                || high >= volume[axis] as isize
                || high >= (origin[axis] + shape[axis]) as isize
            {
                return None;
            }
        }
        let strides = [(shape[1] * shape[2]) as isize, shape[2] as isize, 1];
        let base: isize = (0..3)
            .map(|axis| (at[axis] - origin[axis] as isize) * strides[axis])
            .sum();
        let linear = |offset: &[isize; 3]| {
            (base + offset[0] * strides[0] + offset[1] * strides[1] + offset[2]) as usize
        };
        match window {
            Voxels::Bool(array) => {
                let values = array.as_slice()?;
                Some(
                    match offsets.iter().any(|(_, offset)| values[linear(offset)]) {
                        true => 1.0,
                        false => 0.0,
                    },
                )
            }
            Voxels::U8(array) => {
                let values = array.as_slice()?;
                offsets
                    .iter()
                    .map(|(_, offset)| values[linear(offset)])
                    .max()
                    .map(f64::from)
            }
            Voxels::U16(array) => {
                let values = array.as_slice()?;
                offsets
                    .iter()
                    .map(|(_, offset)| values[linear(offset)])
                    .max()
                    .map(f64::from)
            }
            _ => None,
        }
    }
}

/// Reduce `source` over the ball of radius `search_radius[i]` voxels around
/// each point. A radius of 0 reads the point's own voxel.
pub fn measure(
    source: &Voxels,
    points: &[[u32; 3]],
    search_radius: &[f64],
    reduce: Reduce,
) -> Result<Vec<f64>> {
    if search_radius.len() != points.len() {
        return Err(Error::invalid(format!(
            "{} search radii for {} points.",
            search_radius.len(),
            points.len()
        )));
    }
    let largest = search_radius
        .iter()
        .copied()
        .fold(0.0f64, |best, radius| best.max(radius));
    let ball = Ball::new(largest)?;
    let volume = source.shape();
    points
        .iter()
        .zip(search_radius)
        .map(|(at, reach)| ball.reduce_at(source, [0, 0, 0], volume, *at, *reach, reduce))
        .collect()
}

/// The offsets of `[-r, r]^3` within distance `r`, ordered by distance, then
/// lexicographically. A search radius is at most `r`, so the offsets beyond
/// it are never taken; the order is total, so the sort is unique.
fn ball_offsets(radius: isize) -> Vec<(f64, [isize; 3])> {
    let mut offsets = Vec::new();
    let limit = radius as f64;
    for a in -radius..=radius {
        for b in -radius..=radius {
            for c in -radius..=radius {
                let squared = (a * a + b * b + c * c) as f64;
                let distance = squared.sqrt();
                if distance <= limit {
                    offsets.push((distance, [a, b, c]));
                }
            }
        }
    }
    offsets.sort_unstable_by(|left, right| left.0.total_cmp(&right.0).then(left.1.cmp(&right.1)));
    offsets
}
