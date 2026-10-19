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
    for (axis, value) in spacing.iter().enumerate() {
        if !value.is_finite() || *value <= 0.0 {
            return Err(Error::invalid(format!(
                "the spacing on axis {axis} is {value}; a µm-per-voxel spacing is positive and \
                 finite."
            )));
        }
    }
    let shift_um = shift_vx * (spacing[0] + spacing[1] + spacing[2]) / 3.0;
    Ok(radius_units_axial
        .iter()
        .map(|row| {
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
        })
        .collect())
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
    if !largest.is_finite() || largest > 512.0 {
        return Err(Error::invalid(format!(
            "a search radius of {largest} voxels; that is a unit mistake rather than a vessel."
        )));
    }
    let offsets = ball_offsets(largest.ceil() as isize);
    let volume = source.shape();
    let mut out = Vec::with_capacity(points.len());
    for (row, at) in points.iter().enumerate() {
        let at = at.map(|value| value as isize);
        let reach = search_radius[row];
        let taken = offsets.partition_point(|(distance, _)| *distance <= reach);
        let mut total = 0.0f64;
        let mut count = 0usize;
        let mut best: Option<f64> = None;
        for (_, offset) in &offsets[..taken] {
            let mut local = [0usize; 3];
            let mut inside = true;
            for axis in 0..3 {
                let target = at[axis] + offset[axis];
                if target < 0 || target >= volume[axis] as isize {
                    inside = false;
                    break;
                }
                local[axis] = target as usize;
            }
            if !inside {
                continue;
            }
            let value = value_at(source, local)?;
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
        out.push(match reduce {
            Reduce::Mean => total / count as f64,
            _ => best,
        });
    }
    Ok(out)
}

/// The box `[-r, r]^3` ordered by distance, then lexicographically.
fn ball_offsets(radius: isize) -> Vec<(f64, [isize; 3])> {
    let mut offsets = Vec::new();
    for a in -radius..=radius {
        for b in -radius..=radius {
            for c in -radius..=radius {
                let squared = (a * a + b * b + c * c) as f64;
                offsets.push((squared.sqrt(), [a, b, c]));
            }
        }
    }
    offsets.sort_by(|left, right| left.0.total_cmp(&right.0).then(left.1.cmp(&right.1)));
    offsets
}
