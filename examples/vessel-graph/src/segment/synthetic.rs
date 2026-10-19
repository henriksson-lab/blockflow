// SPDX-License-Identifier: MIT
// Original work for this crate.

//! A synthetic vessel channel for tests and demonstrations: tubes of several
//! brightnesses through a noisy background, a saturated core in one of them,
//! and a dark patch below the clip floor.

use blockflow::voxels::Voxels;
use ndarray::Array3;

/// One straight tube: from `start` to `end` (voxel coordinates, z,y,x), of
/// `radius` voxels and peak raw intensity `peak`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tube {
    pub start: [f64; 3],
    pub end: [f64; 3],
    pub radius: f64,
    pub peak: f64,
}

impl Tube {
    /// Distance from `at` to the tube's axis segment.
    pub fn distance(&self, at: [f64; 3]) -> f64 {
        let direction: [f64; 3] = std::array::from_fn(|a| self.end[a] - self.start[a]);
        let length2: f64 = direction.iter().map(|d| d * d).sum();
        let offset: [f64; 3] = std::array::from_fn(|a| at[a] - self.start[a]);
        let along = if length2 > 0.0 {
            (offset
                .iter()
                .zip(&direction)
                .map(|(o, d)| o * d)
                .sum::<f64>()
                / length2)
                .clamp(0.0, 1.0)
        } else {
            0.0
        };
        (0..3)
            .map(|a| (offset[a] - along * direction[a]).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

/// The tubes [`vessel_volume`] draws, scaled to `shape`: a bright one along
/// `x` with a saturated core, and a lattice of dimmer ones in every direction,
/// dense enough that every wide window holds some vessel.
pub fn tubes(shape: [usize; 3]) -> Vec<Tube> {
    let [z, y, x] = shape.map(|n| n as f64);
    let tube = |start: [f64; 3], end: [f64; 3], radius: f64, peak: f64| Tube {
        start: [start[0] * z, start[1] * y, start[2] * x],
        end: [end[0] * z, end[1] * y, end[2] * x],
        radius,
        peak,
    };
    vec![
        tube([0.5, 0.3, 0.05], [0.5, 0.3, 0.95], 2.5, 61_000.0),
        tube([0.4, 0.05, 0.6], [0.4, 0.95, 0.6], 2.0, 9_000.0),
        tube([0.25, 0.6, 0.15], [0.75, 0.85, 0.45], 2.0, 12_000.0),
        tube([0.6, 0.75, 0.05], [0.6, 0.75, 0.95], 1.5, 7_000.0),
        tube([0.3, 0.05, 0.2], [0.3, 0.95, 0.25], 1.5, 8_000.0),
        tube([0.7, 0.05, 0.85], [0.7, 0.95, 0.8], 2.0, 10_000.0),
        tube([0.05, 0.15, 0.4], [0.95, 0.15, 0.4], 2.0, 9_000.0),
        tube([0.05, 0.5, 0.85], [0.95, 0.55, 0.9], 1.5, 8_000.0),
    ]
}

/// A `u16` channel of `shape` with the [`tubes`] in it over a background of
/// about 1000 with deterministic noise; a small patch reads 0, below any
/// clip floor.
pub fn vessel_volume(shape: [usize; 3], seed: u64) -> Voxels {
    let tubes = tubes(shape);
    let mut state = seed | 1;
    let mut noise = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) % 121) as f64 - 60.0
    };
    let mut volume = Array3::<u16>::zeros((shape[0], shape[1], shape[2]));
    for ((z, y, x), slot) in volume.indexed_iter_mut() {
        let at = [z as f64, y as f64, x as f64];
        let mut value = 1000.0 + noise();
        for tube in &tubes {
            let d = tube.distance(at);
            if d <= tube.radius + 1.0 {
                let profile = (-(d * d) / (2.0 * (tube.radius * 0.6).powi(2))).exp();
                value = value.max(1000.0 + (tube.peak - 1000.0) * profile);
            }
        }
        if z < 3 && y < 3 && x < 3 {
            value = 0.0;
        }
        *slot = value.clamp(0.0, f64::from(u16::MAX)).round() as u16;
    }
    Voxels::U16(volume)
}

/// The voxels within `margin` of any tube's axis: what a segmentation should
/// cover.
pub fn tube_mask(shape: [usize; 3], margin: f64) -> Array3<bool> {
    let tubes = tubes(shape);
    Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(z, y, x)| {
        let at = [z as f64, y as f64, x as f64];
        tubes.iter().any(|tube| tube.distance(at) <= margin)
    })
}
