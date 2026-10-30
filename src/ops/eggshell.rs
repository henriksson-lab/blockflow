// SPDX-License-Identifier: MIT
//
// Original work for this crate. The ellipse fit follows Halir & Flusser (1998),
// "Numerically stable direct least squares fitting of ellipses" (WSCG'98), which
// is itself a reformulation of Fitzgibbon, Pilu & Fisher (1999), "Direct least
// square fitting of ellipses", IEEE TPAMI 21:476–480; written from the papers.

//! **Eggshell segmentation and the DIC calibration it gives.**
//!
//! A *C. elegans* egg is a prolate ellipsoid inside a thin shell, and the cells
//! never touch the shell. In DIC the shell is a thin line around the embryo
//! whose sign follows the shear: bright on the side facing the bright-edge
//! direction, dark opposite, with a smooth medium outside. That makes it a clean
//! calibration target — every edge orientation is present, in a region free of
//! cells — and its interior is the embryo mask.
//!
//! Everything here is **resident and 2-D per plane** (a plane is at most a few
//! megapixels) plus small fits across planes; it is the calibration step a
//! pipeline runs once per movie, on a few frames, and whose few numbers are then
//! parameters of the blockwise ops in `super::dic`.
//!
//! | piece | what |
//! |---|---|
//! | [`Ellipse`], [`fit_ellipse`], [`fit_ellipse_ransac`] | conics: direct least squares (Halir & Flusser) and a deterministic RANSAC around it |
//! | [`fit_shell_plane`] | one plane: texture rays → outermost line → ridge refinement, with the edge-contrast samples on the final ellipse |
//! | [`auto_seed`] | a point inside an embryo without annotations: the deepest point of the textured region |
//! | [`Ellipsoid`], [`fit_ellipsoid`] | one shell per frame from the per-plane ellipses: `ρ(z)² = R0²(1 − ((z−z0)/c)²)` |
//! | [`ellipsoid_mask`] | the interior, scaled |
//!
//! # Coordinates
//!
//! Plane points are `[x, y]` in pixels (`x` = column, `y` = row, pixel centres
//! at integers). Angles turn from `+x` towards `+y`, the convention of
//! `super::dic`. [`Ellipsoid`] is physical (micrometres, `[x, y, z]`).
//!
//! # The shell in one plane
//!
//! 1. **Initial ellipse.** The embryo is textured and the medium is not. Rays
//!    from the seed stop where the local texture (RMS of a 1 µm high-pass over
//!    1 µm) falls below the geometric mean of its 10th and 90th percentiles and
//!    stays below for `smooth_run` µm; a RANSAC ellipse through the stops
//!    ignores rays that ran into a touching neighbour.
//! 2. **Outermost line.** Around that ellipse, in elliptical polar coordinates,
//!    a thin-line response (difference of Gaussians at 1 and 2.5 px) is smoothed
//!    along the angle by 1 µm of arc — a long line parallel to the shell
//!    survives, granules do not — and each ray keeps the last above-noise
//!    sample before a `background_run` µm stretch of background. That is the
//!    shell, not the embryo boundary inside it.
//! 3. **Ridge.** Within ±1 µm of that ellipse each ray takes the extremum of
//!    the (signed, angle-smoothed) line response; a RANSAC ellipse at
//!    `ridge_tolerance` gives the final fit.
//! 4. **Edge contrast.** At the ridge points: the signed line response (the
//!    *even* contrast — the `I_edge` of `a + b·cos(φ − θ)`) and minus the radial
//!    derivative (the *odd* contrast), against the outward normal angle `φ`.

use ndarray::{Array2, Array3, ArrayView2};

use crate::error::{Error, Result};

use super::dic::{gaussian_gradient_plane, gaussian_plane, weighted_least_squares};
use super::threshold::percentile_of;

// ---------------------------------------------------------------------------
// ellipses
// ---------------------------------------------------------------------------

/// An ellipse: centre, semi-axes `a ≥ b`, and the angle of the `a` axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ellipse {
    pub cx: f64,
    pub cy: f64,
    pub a: f64,
    pub b: f64,
    pub angle: f64,
}

impl Ellipse {
    /// Normalised so that `a ≥ b` and `angle ∈ (−π/2, π/2]`.
    pub fn new(cx: f64, cy: f64, a: f64, b: f64, angle: f64) -> Self {
        let (a, b, angle) = if b > a {
            (b, a, angle + std::f64::consts::FRAC_PI_2)
        } else {
            (a, b, angle)
        };
        let mut angle = angle.rem_euclid(std::f64::consts::PI);
        if angle > std::f64::consts::FRAC_PI_2 {
            angle -= std::f64::consts::PI;
        }
        Self {
            cx,
            cy,
            a,
            b,
            angle,
        }
    }

    /// The point at parameter `t`.
    pub fn point(&self, t: f64) -> [f64; 2] {
        let (c, s) = (self.angle.cos(), self.angle.sin());
        let (u, v) = (self.a * t.cos(), self.b * t.sin());
        [self.cx + u * c - v * s, self.cy + u * s + v * c]
    }

    /// The point at parameter `t` scaled radially by `rho` (1: on the ellipse).
    pub fn point_scaled(&self, t: f64, rho: f64) -> [f64; 2] {
        let (c, s) = (self.angle.cos(), self.angle.sin());
        let (u, v) = (rho * self.a * t.cos(), rho * self.b * t.sin());
        [self.cx + u * c - v * s, self.cy + u * s + v * c]
    }

    /// The unit outward normal at parameter `t`.
    pub fn normal(&self, t: f64) -> [f64; 2] {
        let (c, s) = (self.angle.cos(), self.angle.sin());
        let (nu, nv) = (t.cos() / self.a, t.sin() / self.b);
        let (nx, ny) = (nu * c - nv * s, nu * s + nv * c);
        let n = nx.hypot(ny);
        [nx / n, ny / n]
    }

    /// The elliptical radius of `(x, y)`: 1 on the ellipse, < 1 inside.
    pub fn radius_of(&self, x: f64, y: f64) -> f64 {
        let (c, s) = (self.angle.cos(), self.angle.sin());
        let (dx, dy) = (x - self.cx, y - self.cy);
        let u = (dx * c + dy * s) / self.a;
        let v = (-dx * s + dy * c) / self.b;
        u.hypot(v)
    }

    /// Approximate distance from `(x, y)` to the ellipse along the ray from
    /// the centre (exact for a circle).
    pub fn distance(&self, x: f64, y: f64) -> f64 {
        let (c, s) = (self.angle.cos(), self.angle.sin());
        let (dx, dy) = (x - self.cx, y - self.cy);
        let u = (dx * c + dy * s) / self.a;
        let v = (-dx * s + dy * c) / self.b;
        let r = u.hypot(v);
        if r < 1e-12 {
            return self.b;
        }
        let local = ((u * self.a).powi(2) + (v * self.b).powi(2)).sqrt() / r;
        (r - 1.0).abs() * local
    }

    /// `√(ab)`.
    pub fn mean_radius(&self) -> f64 {
        (self.a * self.b).sqrt()
    }

    /// The same ellipse with every length multiplied by `factor` about the
    /// origin (e.g. pixels to micrometres).
    pub fn scaled(&self, factor: f64) -> Self {
        Self {
            cx: self.cx * factor,
            cy: self.cy * factor,
            a: self.a * factor,
            b: self.b * factor,
            angle: self.angle,
        }
    }
}

/// Real eigenvalues of a 3x3 matrix, from its characteristic cubic.
fn real_eigenvalues3(m: &[[f64; 3]; 3]) -> Vec<f64> {
    // λ³ − tr λ² + c1 λ − det = 0
    let tr = m[0][0] + m[1][1] + m[2][2];
    let c1 = m[0][0] * m[1][1] - m[0][1] * m[1][0] + m[0][0] * m[2][2] - m[0][2] * m[2][0]
        + m[1][1] * m[2][2]
        - m[1][2] * m[2][1];
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    // depressed cubic t³ + p t + q with λ = t + tr/3
    let (a2, a1, a0) = (-tr, c1, -det);
    let p = a1 - a2 * a2 / 3.0;
    let q = 2.0 * a2 * a2 * a2 / 27.0 - a2 * a1 / 3.0 + a0;
    let shift = -a2 / 3.0;
    let disc = q * q / 4.0 + p * p * p / 27.0;
    let mut roots = Vec::new();
    if disc > 0.0 {
        let sq = disc.sqrt();
        roots.push((-q / 2.0 + sq).cbrt() + (-q / 2.0 - sq).cbrt() + shift);
    } else if p.abs() < 1e-300 {
        roots.push(shift);
    } else {
        let r = (-p / 3.0).sqrt();
        let phi = ((3.0 * q) / (2.0 * p * r)).clamp(-1.0, 1.0).acos() / 3.0;
        for k in 0..3 {
            roots.push(2.0 * r * (phi - 2.0 * std::f64::consts::PI * k as f64 / 3.0).cos() + shift);
        }
    }
    roots
}

/// A null vector of `m − λ I`: the largest cross product of two of its rows.
fn null_vector3(m: &[[f64; 3]; 3], lambda: f64) -> [f64; 3] {
    let r: Vec<[f64; 3]> = (0..3)
        .map(|i| {
            let mut row = m[i];
            row[i] -= lambda;
            row
        })
        .collect();
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let candidates = [cross(r[0], r[1]), cross(r[0], r[2]), cross(r[1], r[2])];
    *candidates
        .iter()
        .max_by(|a, b| {
            let na = a.iter().map(|v| v * v).sum::<f64>();
            let nb = b.iter().map(|v| v * v).sum::<f64>();
            na.total_cmp(&nb)
        })
        .unwrap()
}

/// Direct least-squares ellipse through `points` (`[x, y]`), Halir & Flusser.
/// `None` when the points do not determine an ellipse (fewer than five, or a
/// hyperbola/parabola fits best).
pub fn fit_ellipse(points: &[[f64; 2]]) -> Option<Ellipse> {
    if points.len() < 5 {
        return None;
    }
    let n = points.len() as f64;
    let mx = points.iter().map(|p| p[0]).sum::<f64>() / n;
    let my = points.iter().map(|p| p[1]).sum::<f64>() / n;
    let sc = (points
        .iter()
        .map(|p| (p[0] - mx).powi(2) + (p[1] - my).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();
    if !(sc > 0.0) {
        return None;
    }
    let mut s1 = [[0.0; 3]; 3];
    let mut s2 = [[0.0; 3]; 3];
    let mut s3 = [[0.0; 3]; 3];
    for p in points {
        let (x, y) = ((p[0] - mx) / sc, (p[1] - my) / sc);
        let d1 = [x * x, x * y, y * y];
        let d2 = [x, y, 1.0];
        for i in 0..3 {
            for j in 0..3 {
                s1[i][j] += d1[i] * d1[j];
                s2[i][j] += d1[i] * d2[j];
                s3[i][j] += d2[i] * d2[j];
            }
        }
    }
    // T = −S3⁻¹ S2ᵀ
    let inv3 = invert3(&s3)?;
    let mut t = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut acc = 0.0;
            for k in 0..3 {
                acc += inv3[i][k] * s2[j][k];
            }
            t[i][j] = -acc;
        }
    }
    // M = S1 + S2 T, then C1⁻¹ M
    let mut m = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut acc = s1[i][j];
            for k in 0..3 {
                acc += s2[i][k] * t[k][j];
            }
            m[i][j] = acc;
        }
    }
    let m = [
        [m[2][0] / 2.0, m[2][1] / 2.0, m[2][2] / 2.0],
        [-m[1][0], -m[1][1], -m[1][2]],
        [m[0][0] / 2.0, m[0][1] / 2.0, m[0][2] / 2.0],
    ];
    let mut best: Option<[f64; 3]> = None;
    for lambda in real_eigenvalues3(&m) {
        let v = null_vector3(&m, lambda);
        let cond = 4.0 * v[0] * v[2] - v[1] * v[1];
        if cond > 0.0 {
            best = Some(v);
            break;
        }
    }
    let a1 = best?;
    let mut a2 = [0.0; 3];
    for i in 0..3 {
        a2[i] = (0..3).map(|k| t[i][k] * a1[k]).sum();
    }
    let ellipse = conic_to_ellipse([a1[0], a1[1], a1[2], a2[0], a2[1], a2[2]])?;
    Some(Ellipse::new(
        ellipse.cx * sc + mx,
        ellipse.cy * sc + my,
        ellipse.a * sc,
        ellipse.b * sc,
        ellipse.angle,
    ))
}

fn invert3(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-300 {
        return None;
    }
    let mut inv = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let (a, b) = ((j + 1) % 3, (j + 2) % 3);
            let (c, d) = ((i + 1) % 3, (i + 2) % 3);
            inv[i][j] = (m[a][c] * m[b][d] - m[a][d] * m[b][c]) / det;
        }
    }
    Some(inv)
}

/// `A x² + B xy + C y² + D x + E y + F = 0` as an [`Ellipse`].
fn conic_to_ellipse(k: [f64; 6]) -> Option<Ellipse> {
    let [a, b, c, d, e, f] = k;
    let den = 4.0 * a * c - b * b;
    if !(den > 0.0) {
        return None;
    }
    let x0 = (b * e - 2.0 * c * d) / den;
    let y0 = (b * d - 2.0 * a * e) / den;
    let f0 = a * x0 * x0 + b * x0 * y0 + c * y0 * y0 + d * x0 + e * y0 + f;
    // eigenvalues of [[a, b/2], [b/2, c]]
    let half = 0.5 * (a + c);
    let disc = (0.25 * (a - c).powi(2) + 0.25 * b * b).sqrt();
    let (l1, l2) = (half - disc, half + disc); // l1 <= l2
    let (ax1, ax2) = (-f0 / l1, -f0 / l2);
    if !(ax1 > 0.0) || !(ax2 > 0.0) {
        return None;
    }
    // the major axis (smaller |λ| for f0 of the other sign) belongs to l1 when
    // l1, l2 > 0 and f0 < 0; otherwise to l2.
    let (major, minor, lmajor) = if ax1 >= ax2 {
        (ax1.sqrt(), ax2.sqrt(), l1)
    } else {
        (ax2.sqrt(), ax1.sqrt(), l2)
    };
    // eigenvector of lmajor: (b/2, lmajor − a) or (lmajor − c, b/2)
    let (vx, vy) = if (b / 2.0).abs() > 1e-15 {
        (b / 2.0, lmajor - a)
    } else if (a - lmajor).abs() < (c - lmajor).abs() {
        (1.0, 0.0)
    } else {
        (0.0, 1.0)
    };
    Some(Ellipse::new(x0, y0, major, minor, vy.atan2(vx)))
}

/// RANSAC parameters for [`fit_ellipse_ransac`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RobustEllipse {
    /// Inlier distance, in the points' units.
    pub threshold: f64,
    pub trials: usize,
    pub seed: u64,
    /// Candidates with a semi-axis longer than this (or shorter than
    /// `min_axis`) are discarded; `f64::INFINITY` and `0` accept any.
    pub max_axis: f64,
    pub min_axis: f64,
}

impl RobustEllipse {
    pub fn new(threshold: f64, trials: usize, seed: u64) -> Self {
        Self {
            threshold,
            trials,
            seed,
            max_axis: f64::INFINITY,
            min_axis: 0.0,
        }
    }

    pub fn with_axis_bounds(mut self, min_axis: f64, max_axis: f64) -> Self {
        self.min_axis = min_axis;
        self.max_axis = max_axis;
        self
    }

    fn admits(&self, e: &Ellipse) -> bool {
        e.a.is_finite() && e.a <= self.max_axis && e.b >= self.min_axis
    }
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 17
    }
}

/// RANSAC over six-point samples, then three least-squares refits on the
/// inliers. Deterministic for a given `seed`. Returns the ellipse and the
/// inlier flags.
pub fn fit_ellipse_ransac(
    points: &[[f64; 2]],
    params: &RobustEllipse,
) -> Option<(Ellipse, Vec<bool>)> {
    if points.len() < 6 {
        return None;
    }
    let mut rng = Lcg(params.seed ^ 0x9e37_79b9_7f4a_7c15);
    let inliers_of = |e: &Ellipse| -> Vec<bool> {
        points
            .iter()
            .map(|p| e.distance(p[0], p[1]) < params.threshold)
            .collect()
    };
    let mut best: Option<(Ellipse, Vec<bool>, usize)> = None;
    let mut sample = Vec::with_capacity(6);
    for _ in 0..params.trials.max(1) {
        sample.clear();
        let mut picked = [usize::MAX; 6];
        let mut k = 0;
        while k < 6 {
            let i = (rng.next() % points.len() as u64) as usize;
            if picked[..k].contains(&i) {
                continue;
            }
            picked[k] = i;
            sample.push(points[i]);
            k += 1;
        }
        let Some(e) = fit_ellipse(&sample).filter(|e| params.admits(e)) else {
            continue;
        };
        let flags = inliers_of(&e);
        let count = flags.iter().filter(|&&f| f).count();
        if best.as_ref().is_none_or(|b| count > b.2) {
            best = Some((e, flags, count));
        }
    }
    let (mut e, mut flags, _) = best?;
    for _ in 0..3 {
        let chosen: Vec<[f64; 2]> = points
            .iter()
            .zip(&flags)
            .filter(|(_, &f)| f)
            .map(|(p, _)| *p)
            .collect();
        match fit_ellipse(&chosen).filter(|e| params.admits(e)) {
            Some(refit) => {
                e = refit;
                flags = inliers_of(&e);
            }
            None => break,
        }
    }
    Some((e, flags))
}

// ---------------------------------------------------------------------------
// sampling helpers
// ---------------------------------------------------------------------------

/// Bilinear sample at `(x, y)`; `None` outside the plane.
pub fn bilinear(plane: ArrayView2<'_, f64>, x: f64, y: f64) -> Option<f64> {
    let (rows, cols) = plane.dim();
    if !(x >= 0.0 && y >= 0.0 && x <= (cols - 1) as f64 && y <= (rows - 1) as f64) {
        return None;
    }
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(cols - 1), (y0 + 1).min(rows - 1));
    let (fx, fy) = (x - x0 as f64, y - y0 as f64);
    let top = plane[[y0, x0]] * (1.0 - fx) + plane[[y0, x1]] * fx;
    let bottom = plane[[y1, x0]] * (1.0 - fx) + plane[[y1, x1]] * fx;
    Some(top * (1.0 - fy) + bottom * fy)
}

/// Smooth every column (fixed radius index) of an angle × radius grid along
/// the angle, which wraps. `sigma` in angle samples.
fn smooth_along_angle(grid: &mut Array2<f64>, sigma: f64) {
    if !(sigma > 0.0) {
        return;
    }
    let radius = (3.0 * sigma).ceil() as isize;
    let weights: Vec<f64> = (-radius..=radius)
        .map(|k| (-0.5 * (k as f64 / sigma).powi(2)).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    let (n, m) = grid.dim();
    let source = grid.clone();
    for i in 0..n as isize {
        for j in 0..m {
            let mut acc = 0.0;
            for (k, &w) in weights.iter().enumerate() {
                let ii = (i + k as isize - radius).rem_euclid(n as isize) as usize;
                acc += w * source[[ii, j]];
            }
            grid[[i as usize, j]] = acc / total;
        }
    }
}

// ---------------------------------------------------------------------------
// the shell in one plane
// ---------------------------------------------------------------------------

/// Parameters of [`fit_shell_plane`]. Lengths in micrometres unless stated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShellParams {
    /// Pixel size in the plane, micrometres.
    pub pixel: f64,
    /// High-pass and RMS window of the texture used for the initial ellipse.
    pub texture_scale: f64,
    /// A ray's stop must be followed by this much non-texture.
    pub smooth_run: f64,
    /// Thin-line difference of Gaussians, pixels.
    pub line_sigmas: (f64, f64),
    /// Angle smoothing of the line response, as arc length.
    pub along_shell: f64,
    /// Detection threshold on the angle-smoothed line response, in noise units.
    pub line_threshold: f64,
    /// The shell is the last line before this much background.
    pub background_run: f64,
    /// Search window around the outermost-line ellipse for the ridge, inwards and outwards.
    pub ridge_window: (f64, f64),
    /// Inlier distance of the final fit.
    pub ridge_tolerance: f64,
    /// Rays per fit.
    pub rays: usize,
    /// Admissible semi-axes of the shell, µm.
    pub axis_range: (f64, f64),
}

impl ShellParams {
    /// Defaults for *C. elegans* eggs (≈ 50 × 30 µm) at `pixel` µm per pixel.
    pub fn for_pixel(pixel: f64) -> Result<Self> {
        if !(pixel > 0.0) || !pixel.is_finite() {
            return Err(Error::invalid(format!(
                "a pixel size must be positive; got {pixel}"
            )));
        }
        Ok(Self {
            pixel,
            texture_scale: 1.0,
            smooth_run: 5.0,
            line_sigmas: (1.0, 2.5),
            along_shell: 1.0,
            line_threshold: 1.8,
            background_run: 8.0,
            ridge_window: (1.0, 0.6),
            ridge_tolerance: 0.33,
            rays: 720,
            axis_range: (5.0, 60.0),
        })
    }
}

/// One edge-contrast sample on the fitted shell.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeSample {
    /// Outward normal angle, radians.
    pub phi: f64,
    /// Signed thin-line response at the ridge (raw intensity units).
    pub even: f64,
    /// Minus the radial derivative of the 1-px-smoothed plane at the ridge.
    pub odd: f64,
    /// Whether the ridge point is an inlier of the final ellipse.
    pub inlier: bool,
}

/// The shell in one plane.
#[derive(Debug, Clone, PartialEq)]
pub struct ShellPlaneFit {
    pub ellipse: Ellipse,
    /// Fraction of rays whose ridge point is within `ridge_tolerance`.
    pub inlier_fraction: f64,
    /// Fraction of rays that found a line before background in step 2.
    pub coverage: f64,
    /// Robust sigma of the thin-line response over the plane.
    pub line_noise: f64,
    /// Median of `|even|` over the inlier samples, in `line_noise` units.
    pub contrast_snr: f64,
    pub samples: Vec<EdgeSample>,
}

/// Local texture: RMS of `I − G_s * I` over `G_s`, `s = texture_scale`.
pub fn texture_rms(plane: ArrayView2<'_, f64>, sigma_px: f64) -> Result<Array2<f64>> {
    let low = gaussian_plane(plane, sigma_px)?;
    let high = &plane - &low;
    let energy = gaussian_plane(high.mapv(|v| v * v).view(), sigma_px)?;
    Ok(energy.mapv(|e| e.max(0.0).sqrt()))
}

fn texture_threshold(texture: &Array2<f64>) -> f64 {
    let mut values: Vec<f64> = texture.iter().copied().filter(|v| *v > 0.0).collect();
    if values.is_empty() {
        return 0.0;
    }
    let lo = percentile_of(&mut values, 10.0).max(1e-12);
    let hi = percentile_of(&mut values, 90.0).max(lo);
    (lo.ln() * 0.5 + hi.ln() * 0.5).exp()
}

/// A point inside an embryo: the deepest point (largest chamfer distance to
/// non-texture) of the textured region.
pub fn auto_seed<T: Copy + Into<f64>>(
    plane: ArrayView2<'_, T>,
    params: &ShellParams,
) -> Result<[f64; 2]> {
    let plane = plane.mapv(Into::into);
    let texture = texture_rms(plane.view(), params.texture_scale / params.pixel)?;
    let th = texture_threshold(&texture);
    let (rows, cols) = texture.dim();
    let big = f64::MAX / 4.0;
    let mut d = Array2::from_shape_fn(
        (rows, cols),
        |(i, j)| if texture[[i, j]] > th { big } else { 0.0 },
    );
    let diag = std::f64::consts::SQRT_2;
    for i in 0..rows {
        for j in 0..cols {
            let mut v = d[[i, j]];
            if v == 0.0 {
                continue;
            }
            if i > 0 {
                v = v.min(d[[i - 1, j]] + 1.0);
                if j > 0 {
                    v = v.min(d[[i - 1, j - 1]] + diag);
                }
                if j + 1 < cols {
                    v = v.min(d[[i - 1, j + 1]] + diag);
                }
            } else {
                v = v.min(1.0);
            }
            if j > 0 {
                v = v.min(d[[i, j - 1]] + 1.0);
            } else {
                v = v.min(1.0);
            }
            d[[i, j]] = v;
        }
    }
    for i in (0..rows).rev() {
        for j in (0..cols).rev() {
            let mut v = d[[i, j]];
            if v == 0.0 {
                continue;
            }
            if i + 1 < rows {
                v = v.min(d[[i + 1, j]] + 1.0);
                if j > 0 {
                    v = v.min(d[[i + 1, j - 1]] + diag);
                }
                if j + 1 < cols {
                    v = v.min(d[[i + 1, j + 1]] + diag);
                }
            } else {
                v = v.min(1.0);
            }
            if j + 1 < cols {
                v = v.min(d[[i, j + 1]] + 1.0);
            } else {
                v = v.min(1.0);
            }
            d[[i, j]] = v;
        }
    }
    let mut best = (0.0, [cols as f64 / 2.0, rows as f64 / 2.0]);
    for ((i, j), &v) in d.indexed_iter() {
        if v > best.0 {
            best = (v, [j as f64, i as f64]);
        }
    }
    Ok(best.1)
}

/// Step 1: rays from `seed` through the texture map.
fn initial_ellipse(texture: &Array2<f64>, seed: [f64; 2], params: &ShellParams) -> Result<Ellipse> {
    let th = texture_threshold(texture);
    let run = (params.smooth_run / params.pixel).ceil() as usize;
    let rays = 180;
    let mut stops = Vec::new();
    let (rows, cols) = texture.dim();
    let reach = (rows.max(cols)) as f64;
    for k in 0..rays {
        let ang = k as f64 * std::f64::consts::TAU / rays as f64;
        let (dx, dy) = (ang.cos(), ang.sin());
        let mut quiet = 0usize;
        let mut first_quiet = None;
        let mut r = 0.0;
        while r < reach {
            let (x, y) = (seed[0] + r * dx, seed[1] + r * dy);
            let Some(v) = bilinear(texture.view(), x, y) else {
                break;
            };
            if v < th {
                if quiet == 0 {
                    first_quiet = Some([x, y]);
                }
                quiet += 1;
                if quiet >= run {
                    break;
                }
            } else {
                quiet = 0;
                first_quiet = None;
            }
            r += 1.0;
        }
        if let Some(p) = first_quiet {
            stops.push(p);
        }
    }
    fit_ellipse_ransac(
        &stops,
        &RobustEllipse::new(0.8 / params.pixel, 400, 1).with_axis_bounds(
            params.axis_range.0 / params.pixel,
            params.axis_range.1 / params.pixel,
        ),
    )
    .map(|(e, _)| e)
    .ok_or_else(|| Error::invalid("no initial ellipse: the texture rays found no embryo boundary"))
}

/// Fit the eggshell in one plane, starting from a `seed` inside the embryo.
pub fn fit_shell_plane<T: Copy + Into<f64>>(
    plane: ArrayView2<'_, T>,
    seed: [f64; 2],
    params: &ShellParams,
) -> Result<ShellPlaneFit> {
    let plane = plane.mapv(Into::into);
    let px = params.pixel;
    let texture = texture_rms(plane.view(), params.texture_scale / px)?;
    let e0 = initial_ellipse(&texture, seed, params)?;
    let line = &gaussian_plane(plane.view(), params.line_sigmas.0)?
        - &gaussian_plane(plane.view(), params.line_sigmas.1)?;
    let mut abs: Vec<f64> = line.iter().map(|v| v.abs()).collect();
    // P25 of |N(0, σ)| is 0.3186 σ; the medium dominates the low quantiles.
    let noise = (percentile_of(&mut abs, 25.0) / 0.3186).max(1e-12);
    let n = params.rays;

    // step 2: outermost line before background
    let r0 = e0.mean_radius();
    let dr = 0.5 / r0;
    let rho: Vec<f64> = (0..)
        .map(|k| 0.5 + k as f64 * dr)
        .take_while(|&r| r < 2.2)
        .collect();
    let mut grid = Array2::<f64>::zeros((n, rho.len()));
    let mut inside = Array2::<bool>::from_elem((n, rho.len()), false);
    for i in 0..n {
        let t = i as f64 * std::f64::consts::TAU / n as f64;
        for (j, &r) in rho.iter().enumerate() {
            let p = e0.point_scaled(t, r);
            if let Some(v) = bilinear(line.view(), p[0], p[1]) {
                grid[[i, j]] = v.abs();
                inside[[i, j]] = true;
            }
        }
    }
    let arc = std::f64::consts::TAU * r0 / n as f64;
    smooth_along_angle(&mut grid, params.along_shell / px / arc);
    // angle smoothing averages ~2.5σ samples of noise: scale the threshold down
    let level = params.line_threshold * noise;
    let run = ((params.background_run / px) / 0.5).ceil() as usize;
    let j0 = rho.iter().position(|&r| r >= 0.8).unwrap_or(0);
    let mut outer = Vec::new();
    for i in 0..n {
        let t = i as f64 * std::f64::consts::TAU / n as f64;
        let mut last_line = None;
        let mut quiet = 0usize;
        let mut found = None;
        for j in j0..rho.len() {
            if !inside[[i, j]] {
                // the plane ends: what was quiet so far counts as background
                if quiet > 0 {
                    found = last_line;
                }
                break;
            }
            if grid[[i, j]] > level {
                last_line = Some(j);
                quiet = 0;
            } else {
                quiet += 1;
                if quiet >= run {
                    found = last_line;
                    break;
                }
            }
        }
        if let Some(j) = found {
            if j > j0 {
                outer.push(e0.point_scaled(t, rho[j]));
            }
        }
    }
    let coverage = outer.len() as f64 / n as f64;
    let (e1, _) = fit_ellipse_ransac(
        &outer,
        &RobustEllipse::new(1.0 / px, 400, 2).with_axis_bounds(
            params.axis_range.0 / params.pixel,
            params.axis_range.1 / params.pixel,
        ),
    )
    .ok_or_else(|| Error::invalid("no shell: too few outermost-line points"))?;

    // step 3: ridge near e1
    let r1 = e1.mean_radius();
    let step = 0.25 / r1;
    let (win_in, win_out) = (
        params.ridge_window.0 / px / r1,
        params.ridge_window.1 / px / r1,
    );
    let rho2: Vec<f64> = (0..)
        .map(|k| 1.0 - win_in + k as f64 * step)
        .take_while(|&r| r <= 1.0 + win_out)
        .collect();
    let mut signed = Array2::<f64>::zeros((n, rho2.len()));
    for i in 0..n {
        let t = i as f64 * std::f64::consts::TAU / n as f64;
        for (j, &r) in rho2.iter().enumerate() {
            let p = e1.point_scaled(t, r);
            signed[[i, j]] = bilinear(line.view(), p[0], p[1]).unwrap_or(0.0);
        }
    }
    let arc1 = std::f64::consts::TAU * r1 / n as f64;
    smooth_along_angle(&mut signed, 0.5 / px / arc1);
    let mut ridge = Vec::with_capacity(n);
    let mut ridge_value = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 * std::f64::consts::TAU / n as f64;
        let (mut jbest, mut vbest) = (0, 0.0f64);
        for j in 0..rho2.len() {
            if signed[[i, j]].abs() > vbest.abs() {
                vbest = signed[[i, j]];
                jbest = j;
            }
        }
        ridge.push(e1.point_scaled(t, rho2[jbest]));
        ridge_value.push(vbest);
    }
    let (e2, flags) = fit_ellipse_ransac(
        &ridge,
        &RobustEllipse::new(params.ridge_tolerance / px, 400, 3).with_axis_bounds(
            params.axis_range.0 / params.pixel,
            params.axis_range.1 / params.pixel,
        ),
    )
    .ok_or_else(|| Error::invalid("no shell: the ridge points do not fit an ellipse"))?;
    let inlier_fraction = flags.iter().filter(|&&f| f).count() as f64 / n as f64;

    // step 4: contrast at the ridge points, against the final ellipse's normal
    let (gy, gx) = gaussian_gradient_plane(plane.view(), params.line_sigmas.0)?;
    let mut samples = Vec::with_capacity(n);
    for (i, p) in ridge.iter().enumerate() {
        let (c, s) = (e2.angle.cos(), e2.angle.sin());
        let (dx, dy) = (p[0] - e2.cx, p[1] - e2.cy);
        let t = ((-dx * s + dy * c) / e2.b).atan2((dx * c + dy * s) / e2.a);
        let normal = e2.normal(t);
        let gxv = bilinear(gx.view(), p[0], p[1]).unwrap_or(0.0);
        let gyv = bilinear(gy.view(), p[0], p[1]).unwrap_or(0.0);
        samples.push(EdgeSample {
            phi: normal[1].atan2(normal[0]),
            even: ridge_value[i],
            odd: -(gxv * normal[0] + gyv * normal[1]),
            inlier: flags[i],
        });
    }
    let mut snr: Vec<f64> = samples
        .iter()
        .filter(|s| s.inlier)
        .map(|s| s.even.abs() / noise)
        .collect();
    let contrast_snr = if snr.is_empty() {
        0.0
    } else {
        percentile_of(&mut snr, 50.0)
    };
    Ok(ShellPlaneFit {
        ellipse: e2,
        inlier_fraction,
        coverage,
        line_noise: noise,
        contrast_snr,
        samples,
    })
}

// ---------------------------------------------------------------------------
// the shell in 3-D
// ---------------------------------------------------------------------------

/// A shell: an ellipsoid with one axis along `z`. Physical units (µm).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ellipsoid {
    pub cx: f64,
    pub cy: f64,
    pub cz: f64,
    /// In-plane semi-axes at the equator, `a ≥ b`; `angle` is that of `a`
    /// (the anterior–posterior axis of an egg lying on its side).
    pub a: f64,
    pub b: f64,
    pub angle: f64,
    /// Semi-axis along `z`. Infinite when the planes showed no curvature.
    pub c: f64,
}

impl Ellipsoid {
    /// Elliptical radius of a physical point: < 1 inside.
    pub fn radius_of(&self, x: f64, y: f64, z: f64) -> f64 {
        let e = Ellipse::new(self.cx, self.cy, self.a, self.b, self.angle);
        let planar = e.radius_of(x, y);
        let dz = if self.c.is_finite() {
            (z - self.cz) / self.c
        } else {
            0.0
        };
        (planar * planar + dz * dz).sqrt()
    }

    /// The cross-section at height `z`, if the plane cuts the shell.
    pub fn section(&self, z: f64) -> Option<Ellipse> {
        let f = if self.c.is_finite() {
            1.0 - ((z - self.cz) / self.c).powi(2)
        } else {
            1.0
        };
        (f > 0.0).then(|| {
            Ellipse::new(
                self.cx,
                self.cy,
                self.a * f.sqrt(),
                self.b * f.sqrt(),
                self.angle,
            )
        })
    }
}

/// One plane's contribution to [`fit_ellipsoid`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlaneEllipse {
    /// Height of the plane, µm.
    pub z: f64,
    /// The plane's ellipse in µm.
    pub ellipse: Ellipse,
    /// Confidence (e.g. inlier fraction); 0 drops the plane.
    pub weight: f64,
}

/// Fit an [`Ellipsoid`] to per-plane shell ellipses. The size follows
/// `ρ(z)² = p0 + p1 z + p2 z²` (`ρ = √(ab)`), fitted with Tukey reweighting;
/// centre, aspect and angle are weighted medians over the planes within half a
/// `c` of the equator.
pub fn fit_ellipsoid(planes: &[PlaneEllipse]) -> Result<Ellipsoid> {
    let used: Vec<&PlaneEllipse> = planes
        .iter()
        .filter(|p| p.weight > 0.0 && p.ellipse.a.is_finite())
        .collect();
    if used.len() < 3 {
        return Err(Error::invalid(format!(
            "an ellipsoid needs at least three planes with a shell; got {}",
            used.len()
        )));
    }
    let rows: Vec<Vec<f64>> = used.iter().map(|p| vec![1.0, p.z, p.z * p.z]).collect();
    let y: Vec<f64> = used
        .iter()
        .map(|p| p.ellipse.mean_radius().powi(2))
        .collect();
    let mut w: Vec<f64> = used.iter().map(|p| p.weight).collect();
    let base = w.clone();
    let mut beta = vec![0.0; 3];
    for _ in 0..10 {
        beta = weighted_least_squares(&rows, &y, &w).ok_or_else(|| {
            Error::invalid("the ellipsoid size fit is singular (planes at one z?)")
        })?;
        let residuals: Vec<f64> = rows
            .iter()
            .zip(&y)
            .map(|(r, &yi)| yi - (beta[0] + beta[1] * r[1] + beta[2] * r[2]))
            .collect();
        let mut abs: Vec<f64> = residuals.iter().map(|r| r.abs()).collect();
        let scale = 1.4826 * percentile_of(&mut abs, 50.0);
        if !(scale > 0.0) {
            break;
        }
        for ((wi, r), b) in w.iter_mut().zip(&residuals).zip(&base) {
            let u = r / (4.685 * scale);
            *wi = if u.abs() < 1.0 {
                b * (1.0 - u * u).powi(2)
            } else {
                0.0
            };
        }
    }
    let (cz, r0sq, c) = if beta[2] < 0.0 {
        let cz = -beta[1] / (2.0 * beta[2]);
        let r0sq = beta[0] - beta[1] * beta[1] / (4.0 * beta[2]);
        (cz, r0sq, (-r0sq / beta[2]).sqrt())
    } else {
        // no curvature: the widest plane is the equator
        let widest = used
            .iter()
            .zip(&w)
            .filter(|(_, &wi)| wi > 0.0)
            .max_by(|a, b| {
                a.0.ellipse
                    .mean_radius()
                    .total_cmp(&b.0.ellipse.mean_radius())
            })
            .map(|(p, _)| *p)
            .unwrap_or(used[0]);
        (
            widest.z,
            widest.ellipse.mean_radius().powi(2),
            f64::INFINITY,
        )
    };
    if !(r0sq > 0.0) {
        return Err(Error::invalid(
            "the ellipsoid fit gave a non-positive equatorial size",
        ));
    }
    let near: Vec<(&PlaneEllipse, f64)> = used
        .iter()
        .zip(&w)
        .filter(|(p, &wi)| wi > 0.0 && (!c.is_finite() || (p.z - cz).abs() <= 0.5 * c))
        .map(|(p, &wi)| (*p, wi))
        .collect();
    let near = if near.is_empty() {
        used.iter()
            .zip(&w)
            .map(|(p, &wi)| (*p, wi.max(1e-9)))
            .collect()
    } else {
        near
    };
    let wmedian = |values: Vec<(f64, f64)>| -> f64 {
        let mut v = values;
        v.sort_by(|a, b| a.0.total_cmp(&b.0));
        let total: f64 = v.iter().map(|x| x.1).sum();
        let mut acc = 0.0;
        for (value, weight) in &v {
            acc += weight;
            if acc >= total / 2.0 {
                return *value;
            }
        }
        v.last().map(|x| x.0).unwrap_or(f64::NAN)
    };
    let cx = wmedian(near.iter().map(|(p, w)| (p.ellipse.cx, *w)).collect());
    let cy = wmedian(near.iter().map(|(p, w)| (p.ellipse.cy, *w)).collect());
    let aspect = wmedian(
        near.iter()
            .map(|(p, w)| (p.ellipse.a / p.ellipse.b, *w))
            .collect(),
    );
    // axial mean of the angle (period π)
    let (mut sc, mut ss) = (0.0, 0.0);
    for (p, w) in &near {
        sc += w * (2.0 * p.ellipse.angle).cos();
        ss += w * (2.0 * p.ellipse.angle).sin();
    }
    let angle = 0.5 * ss.atan2(sc);
    let r0 = r0sq.sqrt();
    Ok(Ellipsoid {
        cx,
        cy,
        cz,
        a: r0 * aspect.sqrt(),
        b: r0 / aspect.sqrt(),
        angle,
        c,
    })
}

/// The voxels of a `[z, y, x]` grid whose centres lie inside `shell` scaled by
/// `scale` (`< 1` shrinks). `spacing` is `[z, y, x]` µm per voxel, voxel `k` at
/// `k · spacing` (no offset).
pub fn ellipsoid_mask(
    shape: [usize; 3],
    spacing: [f64; 3],
    shell: &Ellipsoid,
    scale: f64,
) -> Array3<bool> {
    let scaled = Ellipsoid {
        a: shell.a * scale,
        b: shell.b * scale,
        c: shell.c * scale,
        ..*shell
    };
    Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(z, y, x)| {
        scaled.radius_of(
            x as f64 * spacing[2],
            y as f64 * spacing[1],
            z as f64 * spacing[0],
        ) < 1.0
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn the_direct_fit_recovers_an_ellipse_exactly() {
        let truth = Ellipse::new(120.0, 80.0, 60.0, 35.0, 0.6);
        let points: Vec<[f64; 2]> = (0..40).map(|k| truth.point(k as f64 * 0.157)).collect();
        let fit = fit_ellipse(&points).unwrap();
        assert!(
            (fit.cx - 120.0).abs() < 1e-6 && (fit.cy - 80.0).abs() < 1e-6,
            "{fit:?}"
        );
        assert!(
            (fit.a - 60.0).abs() < 1e-6 && (fit.b - 35.0).abs() < 1e-6,
            "{fit:?}"
        );
        assert!((fit.angle - 0.6).abs() < 1e-6, "{fit:?}");
        // the same with the axes swapped in the parametrisation
        let rotated = Ellipse::new(0.0, 0.0, 10.0, 30.0, 0.0);
        assert!((rotated.a - 30.0).abs() < 1e-12 && (rotated.angle - PI / 2.0).abs() < 1e-12);
    }

    #[test]
    fn ransac_ignores_a_neighbour() {
        let truth = Ellipse::new(200.0, 150.0, 90.0, 60.0, -0.3);
        let mut points: Vec<[f64; 2]> = (0..300)
            .map(|k| truth.point(k as f64 * PI / 150.0))
            .collect();
        for k in 0..80 {
            points.push([320.0 + k as f64, 100.0 + (k as f64 * 0.37).sin() * 40.0]);
        }
        let (fit, flags) = fit_ellipse_ransac(&points, &RobustEllipse::new(1.0, 300, 4)).unwrap();
        assert!(
            (fit.a - 90.0).abs() < 1e-6 && (fit.b - 60.0).abs() < 1e-6,
            "{fit:?}"
        );
        assert!(flags[..300].iter().all(|&f| f));
    }

    /// A synthetic DIC egg: textured embryo, a smooth gap, a thin shell line
    /// whose sign follows `cos(φ − θ)`, a touching neighbour embryo.
    fn synthetic_egg(theta: f64) -> (Array2<f64>, Ellipse) {
        let shell = Ellipse::new(150.0, 110.0, 95.0, 62.0, 0.25);
        let embryo = Ellipse::new(150.0, 110.0, 82.0, 52.0, 0.25);
        let neighbour = Ellipse::new(330.0, 110.0, 82.0, 52.0, 0.0);
        let mut state = 99u64;
        let mut noise = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        };
        let image = Array2::from_shape_fn((220, 440), |(i, j)| {
            let (x, y) = (j as f64, i as f64);
            let mut v = 100.0 + 2.0 * noise();
            if embryo.radius_of(x, y) < 1.0 || neighbour.radius_of(x, y) < 1.0 {
                v += 25.0 * ((x * 0.9).sin() * (y * 1.1).cos() + (x * 0.37 + y * 0.61).sin());
            }
            let r = shell.radius_of(x, y);
            let d = (r - 1.0) * shell.mean_radius();
            if d.abs() < 4.0 {
                // outward normal angle approx from the centre direction scaled by the axes
                let (c, s) = (shell.angle.cos(), shell.angle.sin());
                let (dx, dy) = (x - shell.cx, y - shell.cy);
                let u = (dx * c + dy * s) / (shell.a * shell.a);
                let w = (-dx * s + dy * c) / (shell.b * shell.b);
                let phi = (u * s + w * c).atan2(u * c - w * s);
                v += (4.0 + 14.0 * (phi - theta).cos()) * (-0.5 * (d / 0.9).powi(2)).exp();
            }
            v
        });
        (image, shell)
    }

    #[test]
    fn the_plane_fit_finds_the_shell_and_its_cosine() {
        let theta = -2.2;
        let (image, truth) = synthetic_egg(theta);
        let params = ShellParams::for_pixel(0.13).unwrap();
        let seed = auto_seed(image.view(), &params).unwrap();
        assert!(
            truth.radius_of(seed[0], seed[1]) < 1.0
                || Ellipse::new(330.0, 110.0, 82.0, 52.0, 0.0).radius_of(seed[0], seed[1]) < 1.0
        );
        let fit = fit_shell_plane(image.view(), [150.0, 110.0], &params).unwrap();
        let e = fit.ellipse;
        assert!(
            (e.cx - truth.cx).abs() < 1.0 && (e.cy - truth.cy).abs() < 1.0,
            "{e:?}"
        );
        assert!(
            (e.a - truth.a).abs() < 1.0 && (e.b - truth.b).abs() < 1.0,
            "{e:?}"
        );
        assert!(fit.inlier_fraction > 0.6, "{}", fit.inlier_fraction);
        let (phi, even): (Vec<f64>, Vec<f64>) = fit
            .samples
            .iter()
            .filter(|s| s.inlier)
            .map(|s| (s.phi, s.even))
            .unzip();
        let cos = super::super::dic::fit_cosine(&phi, &even, false).unwrap();
        assert!(
            super::super::dic::angle_difference(cos.theta, theta).abs() < 0.1,
            "{cos:?}"
        );
        assert!(cos.a > 0.0, "{cos:?}");
    }

    #[test]
    fn the_ellipsoid_follows_the_sections() {
        let truth = Ellipsoid {
            cx: 30.0,
            cy: 20.0,
            cz: 14.0,
            a: 25.0,
            b: 15.0,
            angle: 0.4,
            c: 15.0,
        };
        let planes: Vec<PlaneEllipse> = (0..20)
            .filter_map(|k| {
                let z = k as f64 * 1.5;
                truth.section(z).map(|e| PlaneEllipse {
                    z,
                    ellipse: e,
                    weight: 1.0,
                })
            })
            .collect();
        let fit = fit_ellipsoid(&planes).unwrap();
        for (got, want) in [
            (fit.cx, 30.0),
            (fit.cy, 20.0),
            (fit.cz, 14.0),
            (fit.a, 25.0),
            (fit.b, 15.0),
            (fit.angle, 0.4),
            (fit.c, 15.0),
        ] {
            assert!((got - want).abs() < 1e-6, "{fit:?}");
        }
        let mask = ellipsoid_mask([20, 50, 70], [1.5, 1.0, 1.0], &fit, 1.0);
        assert!(mask[[9, 20, 30]] && mask[[0, 20, 30]] && !mask[[0, 20, 50]] && !mask[[9, 0, 0]]);
    }
}
