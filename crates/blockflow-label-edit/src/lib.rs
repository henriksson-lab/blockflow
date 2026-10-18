//! Deterministic, model independent primitives for editing integer label images.
//!
//! The crate deliberately knows nothing about HTTP, OME-Zarr, SAM, or a UI. Both the browser and
//! the authoritative writer use these functions so a stroke covers the same pixels at both ends.

use std::collections::BTreeSet;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Two axes in a three-dimensional level-zero voxel grid.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub enum Plane {
    Xy,
    Xz,
    Yz,
}

impl Plane {
    pub const fn axes(self) -> [usize; 3] {
        match self {
            Self::Xy => [0, 1, 2],
            Self::Xz => [0, 2, 1],
            Self::Yz => [1, 2, 0],
        }
    }

    pub const fn point_to_xyz(self, point: [u32; 2], cut: u32) -> [u32; 3] {
        match self {
            Self::Xy => [point[0], point[1], cut],
            Self::Xz => [point[0], cut, point[1]],
            Self::Yz => [cut, point[0], point[1]],
        }
    }
}

/// Policy for a stroke extending a non-background target.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub enum ConflictPolicy {
    /// Every covered voxel is assigned to the frozen target.
    #[default]
    Overwrite,
    /// Background and voxels already belonging to the target may change; other objects survive.
    PreserveExisting,
}

/// A brush path in level-zero plane coordinates. Diameter is expressed in the same coordinates.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct BrushStroke {
    pub points: Vec<[f64; 2]>,
    pub diameter: f64,
}

/// Integer bounds with an exclusive maximum.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Bounds2 {
    pub min: [u32; 2],
    pub max: [u32; 2],
}

/// Rasterize pixel centres covered by a continuous circular brush path.
///
/// Coverage is defined by distance to the entire polyline, which makes the result independent of
/// pointer event frequency and prevents holes during fast drags. Output is sorted and unique.
pub fn rasterize_brush(shape: [u32; 2], stroke: &BrushStroke) -> Vec<[u32; 2]> {
    if shape.contains(&0) || stroke.points.is_empty() || !stroke.diameter.is_finite() {
        return Vec::new();
    }
    let radius = (stroke.diameter * 0.5).max(0.0);
    let mut bounds = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for point in &stroke.points {
        if !point[0].is_finite() || !point[1].is_finite() {
            return Vec::new();
        }
        bounds[0] = bounds[0].min(point[0]);
        bounds[1] = bounds[1].min(point[1]);
        bounds[2] = bounds[2].max(point[0]);
        bounds[3] = bounds[3].max(point[1]);
    }
    let min_x = (bounds[0] - radius - 0.5).floor().max(0.0) as u32;
    let min_y = (bounds[1] - radius - 0.5).floor().max(0.0) as u32;
    let max_x = ((bounds[2] + radius + 0.5).ceil() as u32).min(shape[0]);
    let max_y = ((bounds[3] + radius + 0.5).ceil() as u32).min(shape[1]);
    let radius2 = radius * radius;
    let mut pixels = Vec::new();
    for y in min_y..max_y {
        for x in min_x..max_x {
            let p = [f64::from(x) + 0.5, f64::from(y) + 0.5];
            let covered = if stroke.points.len() == 1 {
                squared_distance(p, stroke.points[0]) <= radius2
            } else {
                stroke
                    .points
                    .windows(2)
                    .any(|line| squared_segment_distance(p, line[0], line[1]) <= radius2)
            };
            if covered {
                pixels.push([x, y]);
            }
        }
    }
    pixels
}

pub fn pixel_bounds(pixels: &[[u32; 2]]) -> Option<Bounds2> {
    let first = *pixels.first()?;
    let mut result = Bounds2 {
        min: first,
        max: [first[0] + 1, first[1] + 1],
    };
    for &[x, y] in &pixels[1..] {
        result.min[0] = result.min[0].min(x);
        result.min[1] = result.min[1].min(y);
        result.max[0] = result.max[0].max(x + 1);
        result.max[1] = result.max[1].max(y + 1);
    }
    Some(result)
}

/// Sorted tile coordinates touched by covered pixels.
pub fn dirty_tiles(pixels: &[[u32; 2]], tile_side: u32) -> Vec<[u32; 2]> {
    if tile_side == 0 {
        return Vec::new();
    }
    pixels
        .iter()
        .map(|point| [point[0] / tile_side, point[1] / tile_side])
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// Apply one frozen-target transaction to a dense row-major plane.
pub fn apply_dense(
    labels: &mut [u64],
    shape: [u32; 2],
    pixels: &[[u32; 2]],
    target: u64,
    policy: ConflictPolicy,
) -> Result<usize, &'static str> {
    let expected = usize::try_from(u64::from(shape[0]) * u64::from(shape[1]))
        .map_err(|_| "label plane is too large")?;
    if labels.len() != expected {
        return Err("label plane length does not match its shape");
    }
    let mut changed = 0;
    for &[x, y] in pixels {
        if x >= shape[0] || y >= shape[1] {
            continue;
        }
        let index = y as usize * shape[0] as usize + x as usize;
        let old = labels[index];
        let allowed =
            policy == ConflictPolicy::Overwrite || target == 0 || old == 0 || old == target;
        if allowed && old != target {
            labels[index] = target;
            changed += 1;
        }
    }
    Ok(changed)
}

fn squared_distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)
}

fn squared_segment_distance(point: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let delta = [b[0] - a[0], b[1] - a[1]];
    let length2 = delta[0] * delta[0] + delta[1] * delta[1];
    if length2 <= f64::EPSILON {
        return squared_distance(point, a);
    }
    let t =
        (((point[0] - a[0]) * delta[0] + (point[1] - a[1]) * delta[1]) / length2).clamp(0.0, 1.0);
    squared_distance(point, [a[0] + t * delta[0], a[1] + t * delta[1]])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continuous_path_has_no_event_sampling_gaps() {
        let pixels = rasterize_brush(
            [32, 8],
            &BrushStroke {
                points: vec![[1.5, 3.5], [29.5, 3.5]],
                diameter: 1.1,
            },
        );
        for x in 1..30 {
            assert!(pixels.contains(&[x, 3]), "missing x={x}");
        }
    }

    #[test]
    fn target_is_frozen_and_conflicts_are_explicit() {
        let pixels = vec![[0, 0], [1, 0], [2, 0]];
        let mut preserve = vec![0, 7, 9];
        assert_eq!(
            apply_dense(
                &mut preserve,
                [3, 1],
                &pixels,
                7,
                ConflictPolicy::PreserveExisting
            ),
            Ok(1)
        );
        assert_eq!(preserve, vec![7, 7, 9]);
        assert_eq!(
            apply_dense(&mut preserve, [3, 1], &pixels, 7, ConflictPolicy::Overwrite),
            Ok(1)
        );
        assert_eq!(preserve, vec![7, 7, 7]);
        assert_eq!(
            apply_dense(
                &mut preserve,
                [3, 1],
                &pixels,
                0,
                ConflictPolicy::PreserveExisting
            ),
            Ok(3)
        );
        assert_eq!(preserve, vec![0, 0, 0]);
    }

    #[test]
    fn axes_and_dirty_tiles_are_deterministic() {
        assert_eq!(Plane::Xy.point_to_xyz([2, 3], 4), [2, 3, 4]);
        assert_eq!(Plane::Xz.point_to_xyz([2, 3], 4), [2, 4, 3]);
        assert_eq!(Plane::Yz.point_to_xyz([2, 3], 4), [4, 2, 3]);
        assert_eq!(
            dirty_tiles(&[[511, 0], [512, 0], [512, 256]], 256),
            vec![[1, 0], [2, 0], [2, 1]]
        );
    }
}
