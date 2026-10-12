// SPDX-License-Identifier: MIT
//
// Original work for this crate.

//! **Stage drift between frames**, by phase correlation.
//!
//! A plate reader revisits each well between frames, and the stage does not
//! return to exactly the same place. On the first real plate this was written
//! for, the field moved about 17 µm between frames 0 and 1 — more than the
//! displacement gate — so almost every cell in frame 0 failed to link. Drift
//! is a property of the whole field, not of a cell, so it is estimated from
//! the images and removed from the registered coordinates before gating; the
//! tables keep native positions.
//!
//! [`phase_correlation`] estimates one translation between two images with a
//! sub-pixel parabolic peak fit; [`cumulative`] turns per-pair shifts into
//! per-frame offsets relative to the first frame.

use ndarray::{Array2, ArrayView2};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::error::{Error, Result};

/// The translation `d = [dy, dx]`, in pixels, such that `b(p) ≈ a(p - d)`:
/// content at `p` in `a` is at `p + d` in `b`.
///
/// Both images are mean-subtracted and Hann-windowed so the frame border does
/// not dominate the correlation. The normalised cross-power spectrum is then
/// weighted by a Gaussian low-pass (`exp(-(f / 0.1)^2)`, `f` in cycles per
/// pixel): pure phase correlation weights every frequency equally, so
/// per-pixel noise — including a static fixed pattern that does not drift —
/// competes with the scene, and the low-pass lets object-scale structure
/// decide. The integer peak is refined by a parabola through its neighbours
/// on each axis. Shifts are recovered within half the image size per axis.
pub fn phase_correlation(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Result<[f64; 2]> {
    if a.shape() != b.shape() {
        return Err(Error::invalid(format!(
            "phase correlation needs equal shapes, got {:?} and {:?}",
            a.shape(),
            b.shape()
        )));
    }
    let (h, w) = a.dim();
    if h < 4 || w < 4 {
        return Err(Error::invalid(
            "phase correlation needs images of at least 4 x 4",
        ));
    }
    let window =
        |n: usize, i: usize| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos();
    let prepare = |image: ArrayView2<'_, f64>| -> Array2<Complex<f64>> {
        let finite: Vec<f64> = image.iter().copied().filter(|v| v.is_finite()).collect();
        let mean = finite.iter().sum::<f64>() / finite.len().max(1) as f64;
        Array2::from_shape_fn((h, w), |(y, x)| {
            let v = image[[y, x]];
            let v = if v.is_finite() { v - mean } else { 0.0 };
            Complex::new(v * window(h, y) * window(w, x), 0.0)
        })
    };
    let mut fa = prepare(a);
    let mut fb = prepare(b);
    let mut planner = FftPlanner::<f64>::new();
    fft2(&mut planner, &mut fa, false);
    fft2(&mut planner, &mut fb, false);
    // Cross-power spectrum of b against a: its inverse peaks at +d.
    let frequency = |k: usize, n: usize| -> f64 {
        let k = if k > n / 2 {
            k as f64 - n as f64
        } else {
            k as f64
        };
        k / n as f64
    };
    let mut r = Array2::from_shape_fn((h, w), |(y, x)| {
        let c = fb[[y, x]] * fa[[y, x]].conj();
        let m = c.norm();
        if m > 1e-12 {
            let f2 = frequency(y, h).powi(2) + frequency(x, w).powi(2);
            c / m * (-f2 / 0.01).exp()
        } else {
            Complex::new(0.0, 0.0)
        }
    });
    fft2(&mut planner, &mut r, true);
    let mut best = (0usize, 0usize);
    let mut peak = f64::NEG_INFINITY;
    for ((y, x), v) in r.indexed_iter() {
        if v.re > peak {
            peak = v.re;
            best = (y, x);
        }
    }
    let value = |y: isize, x: isize| -> f64 {
        r[[
            y.rem_euclid(h as isize) as usize,
            x.rem_euclid(w as isize) as usize,
        ]]
        .re
    };
    let (py, px) = (best.0 as isize, best.1 as isize);
    let refine = |minus: f64, centre: f64, plus: f64| -> f64 {
        let denominator = minus - 2.0 * centre + plus;
        if denominator.abs() < 1e-12 {
            0.0
        } else {
            (0.5 * (minus - plus) / denominator).clamp(-0.5, 0.5)
        }
    };
    let sy = refine(value(py - 1, px), value(py, px), value(py + 1, px));
    let sx = refine(value(py, px - 1), value(py, px), value(py, px + 1));
    let wrap = |p: isize, n: usize| -> f64 {
        if p > n as isize / 2 {
            (p - n as isize) as f64
        } else {
            p as f64
        }
    };
    Ok([wrap(py, h) + sy, wrap(px, w) + sx])
}

/// Refine a frame-pair shift from point positions: every point of `a` votes
/// `b_j - a_i` for each point of `b` within `radius` of `a_i + prior`; votes
/// are binned at `bin`, the fullest bin wins (ties: closest to the prior), and
/// the answer is the mean of the votes within one bin of it. Points are
/// `[y, x]` in any one unit. With many cells the true drift collects a vote
/// from nearly every cell and wrong pairings scatter, so the mode is the
/// drift. Returns `prior` when no vote falls in range.
pub fn refine_with_points(
    a: &[[f64; 2]],
    b: &[[f64; 2]],
    prior: [f64; 2],
    radius: f64,
    bin: f64,
) -> [f64; 2] {
    use std::collections::HashMap;
    let mut votes: Vec<[f64; 2]> = Vec::new();
    // Grid over b for the radius search.
    let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (j, p) in b.iter().enumerate() {
        grid.entry((
            (p[0] / radius).floor() as i64,
            (p[1] / radius).floor() as i64,
        ))
        .or_default()
        .push(j);
    }
    for p in a {
        let q = [p[0] + prior[0], p[1] + prior[1]];
        let cell = (
            (q[0] / radius).floor() as i64,
            (q[1] / radius).floor() as i64,
        );
        for dy in -1..=1 {
            for dx in -1..=1 {
                for &j in grid.get(&(cell.0 + dy, cell.1 + dx)).into_iter().flatten() {
                    let d = [b[j][0] - p[0], b[j][1] - p[1]];
                    if ((d[0] - prior[0]).powi(2) + (d[1] - prior[1]).powi(2)).sqrt() <= radius {
                        votes.push(d);
                    }
                }
            }
        }
    }
    if votes.is_empty() {
        return prior;
    }
    let mut bins: HashMap<(i64, i64), usize> = HashMap::new();
    for d in &votes {
        *bins
            .entry(((d[0] / bin).round() as i64, (d[1] / bin).round() as i64))
            .or_default() += 1;
    }
    // Score each bin with its 3 x 3 neighbourhood, so a peak split across a
    // bin edge is not lost.
    let score = |k: (i64, i64)| -> usize {
        let mut total = 0;
        for dy in -1..=1 {
            for dx in -1..=1 {
                total += bins.get(&(k.0 + dy, k.1 + dx)).copied().unwrap_or(0);
            }
        }
        total
    };
    let mut keys: Vec<(i64, i64)> = bins.keys().copied().collect();
    keys.sort_unstable();
    let distance = |k: &(i64, i64)| {
        ((k.0 as f64 * bin - prior[0]).powi(2) + (k.1 as f64 * bin - prior[1]).powi(2)).sqrt()
    };
    let best = keys
        .iter()
        .copied()
        .max_by(|a, b| {
            score(*a)
                .cmp(&score(*b))
                .then_with(|| distance(b).total_cmp(&distance(a)))
        })
        .expect("non-empty");
    let centre = [best.0 as f64 * bin, best.1 as f64 * bin];
    let near: Vec<&[f64; 2]> = votes
        .iter()
        .filter(|d| (d[0] - centre[0]).abs() <= 1.5 * bin && (d[1] - centre[1]).abs() <= 1.5 * bin)
        .collect();
    let n = near.len() as f64;
    [
        near.iter().map(|d| d[0]).sum::<f64>() / n,
        near.iter().map(|d| d[1]).sum::<f64>() / n,
    ]
}

/// Per-frame offsets from per-pair shifts: `offsets[0] = 0` and
/// `offsets[i] = offsets[i - 1] + shifts[i - 1]`. Subtracting `offsets[t]`
/// from a position in frame `t` puts it in frame 0's coordinates.
pub fn cumulative(shifts: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut offsets = vec![[0.0, 0.0]];
    for shift in shifts {
        let last = *offsets.last().expect("non-empty");
        offsets.push([last[0] + shift[0], last[1] + shift[1]]);
    }
    offsets
}

fn fft2(planner: &mut FftPlanner<f64>, data: &mut Array2<Complex<f64>>, inverse: bool) {
    let (h, w) = data.dim();
    let row = if inverse {
        planner.plan_fft_inverse(w)
    } else {
        planner.plan_fft_forward(w)
    };
    let column = if inverse {
        planner.plan_fft_inverse(h)
    } else {
        planner.plan_fft_forward(h)
    };
    for mut line in data.rows_mut() {
        let mut buffer: Vec<Complex<f64>> = line.iter().copied().collect();
        row.process(&mut buffer);
        for (slot, value) in line.iter_mut().zip(buffer) {
            *slot = value;
        }
    }
    for mut line in data.columns_mut() {
        let mut buffer: Vec<Complex<f64>> = line.iter().copied().collect();
        column.process(&mut buffer);
        for (slot, value) in line.iter_mut().zip(buffer) {
            *slot = value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene(h: usize, w: usize, dy: f64, dx: f64) -> Array2<f64> {
        // A few smooth blobs, translated by (dy, dx).
        let blobs = [
            (20.0, 30.0, 6.0),
            (50.0, 70.0, 4.0),
            (80.0, 25.0, 8.0),
            (35.0, 100.0, 5.0),
        ];
        Array2::from_shape_fn((h, w), |(y, x)| {
            blobs
                .iter()
                .map(|&(by, bx, r)| {
                    let d2 = (y as f64 - by - dy).powi(2) + (x as f64 - bx - dx).powi(2);
                    (-d2 / (2.0 * r * r)).exp()
                })
                .sum()
        })
    }

    #[test]
    fn recovers_integer_and_subpixel_translations() {
        let a = scene(110, 130, 0.0, 0.0);
        for (dy, dx) in [(0.0, 0.0), (3.0, -7.0), (-12.0, 9.0), (2.4, -5.6)] {
            let b = scene(110, 130, dy, dx);
            let [sy, sx] = phase_correlation(a.view(), b.view()).unwrap();
            assert!(
                (sy - dy).abs() < 0.35 && (sx - dx).abs() < 0.35,
                "{dy},{dx} -> {sy},{sx}"
            );
        }
    }

    #[test]
    fn point_votes_find_a_shared_shift_among_random_motion() {
        // 300 points on a jittered grid, shifted by (3, -8) plus noise; a
        // prior 6 away from the truth still finds it.
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut seed = 1u64;
        let mut noise = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5
        };
        for i in 0..300 {
            let p = [
                (i / 20) as f64 * 15.0 + 2.0 * noise(),
                (i % 20) as f64 * 15.0 + 2.0 * noise(),
            ];
            a.push(p);
            b.push([p[0] + 3.0 + noise(), p[1] - 8.0 + noise()]);
        }
        let [dy, dx] = refine_with_points(&a, &b, [7.0, -4.0], 12.0, 1.0);
        assert!(
            (dy - 3.0).abs() < 0.3 && (dx + 8.0).abs() < 0.3,
            "{dy},{dx}"
        );
    }

    #[test]
    fn offsets_accumulate_from_frame_zero() {
        assert_eq!(
            cumulative(&[[1.0, -2.0], [0.5, 0.5]]),
            vec![[0.0, 0.0], [1.0, -2.0], [1.5, -1.5]]
        );
    }
}
