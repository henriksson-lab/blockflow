//! Rotation-invariant features of a fitted shell, for clustering the objects.
//!
//! Every feature is measured against the object itself: shape-adapted radius
//! `s = |x - c| / R(û)` from the fitted rays, the radial direction `û`, or
//! aggregates over all directions, so none depends on how the object lies in
//! the volume. Intensities are normalised per object to its own radial
//! profile: 0 at the darkest just inside the wall (`s` 0.5–1.0), 1 at the
//! wall peak (`s` 0.8–1.2). The outside is not a reference, since it is as
//! often bright tissue as background. The groups, after
//! `imaging_nedal/docs/ROTATION_INVARIANT_FEATURES.md` §7:
//!
//! - radial profile: median intensity in 12 bins of `s`, core contrast, wall
//!   step and per-ray wall thickness;
//! - radialness: the fraction of gradient energy along `û` per zone at two
//!   scales, and how the structure tensor's flat direction aligns with `û`;
//! - object-frame variograms: radial over tangential at 2, 4 and 8 voxels;
//! - per-shell spread and skewness;
//! - intensity and shape second-moment tensors (anisotropy only, no axes);
//! - spherical-harmonic band power on three shells and the coherence of
//!   neighbouring shells per band, which tells spokes (coherent across
//!   shells) from grains (not).

use blockflow::ops::{gaussian_smooth_into_with, trilinear_at, Boundary, Gaussian};
use blockflow::sphere::{fibonacci_sphere, RaySet};
use blockflow::Result;
use ndarray::{Array3, ArrayView3};

/// Feature columns, in the order [`Extractor::measure`] returns them.
/// `wall_step_rel` is left at 0 there; it needs the whole volume's objects.
pub const COLUMNS: [&str; 63] = [
    "valid_fraction",
    "prof_00",
    "prof_01",
    "prof_02",
    "prof_03",
    "prof_04",
    "prof_05",
    "prof_06",
    "prof_07",
    "prof_08",
    "prof_09",
    "prof_10",
    "prof_11",
    "core_contrast",
    "wall_step",
    "wall_step_rel",
    "wall_thickness",
    "wall_thickness_cv",
    "frad_core_s1",
    "frad_mid_s1",
    "frad_sub_s1",
    "frad_core_s2",
    "frad_mid_s2",
    "frad_sub_s2",
    "st_radial_align",
    "st_coherence",
    "vario_ratio_2",
    "vario_ratio_4",
    "vario_ratio_8",
    "vario_tan_4",
    "shell25_std",
    "shell25_skew",
    "shell50_std",
    "shell50_skew",
    "shell75_std",
    "shell75_skew",
    "int_fa",
    "int_cl",
    "int_cp",
    "int_offset",
    "shape_fa",
    "shape_elong",
    "sh30_l1",
    "sh30_l2",
    "sh30_l3_4",
    "sh30_l5_8",
    "sh30_l9_16",
    "sh55_l1",
    "sh55_l2",
    "sh55_l3_4",
    "sh55_l5_8",
    "sh55_l9_16",
    "sh80_l1",
    "sh80_l2",
    "sh80_l3_4",
    "sh80_l5_8",
    "sh80_l9_16",
    "coh_30_55_l2_4",
    "coh_30_55_l5_8",
    "coh_30_55_l9_16",
    "coh_55_80_l2_4",
    "coh_55_80_l5_8",
    "coh_55_80_l9_16",
];

/// Highest spherical-harmonic degree of the shell spectra.
const LMAX: usize = 16;
/// Directions sampled per shell; near-uniform, so equal quadrature weights.
const DENSE: usize = 4000;
const PROFILE_BINS: usize = 12;
const PROFILE_EXTENT: f64 = 1.3;
const SH_SHELLS: [f64; 3] = [0.3, 0.55, 0.8];
const POWER_BANDS: [(usize, usize); 5] = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 16)];
const COHERENCE_BANDS: [(usize, usize); 3] = [(2, 4), (5, 8), (9, 16)];
/// Radialness zones in `s`: core, mid and outer interior. Below 0.15 the
/// radial direction is ill-defined; the outer zone stops short of the wall.
const ZONES: [(f64, f64); 3] = [(0.15, 0.35), (0.35, 0.65), (0.65, 0.9)];
/// Tangential directions per point of the variograms, evenly over half a turn,
/// so the estimate does not depend on the tangent frame.
const TANGENTS: usize = 6;

/// Shared, per-run state: the shell directions and their harmonics.
pub struct Extractor {
    dense: Vec<[f64; 3]>,
    basis: Vec<Vec<f64>>,
}

impl Default for Extractor {
    fn default() -> Self {
        let dense = fibonacci_sphere(DENSE);
        let basis = dense.iter().map(|&u| real_sh(LMAX, u)).collect();
        Self { dense, basis }
    }
}

/// The level-0 window `measure` needs around an object: start and shape.
pub fn window(centre: [f64; 3], radii: &[f32], volume: [usize; 3]) -> ([usize; 3], [usize; 3]) {
    let rmax = radii.iter().copied().fold(0.0f32, f32::max) as f64;
    let half = (1.5 * rmax + 8.0).ceil() as isize;
    let start: [usize; 3] = std::array::from_fn(|a| {
        (centre[a].round() as isize - half).clamp(0, volume[a] as isize - 1) as usize
    });
    let end: [usize; 3] = std::array::from_fn(|a| {
        ((centre[a].round() as isize + half + 1).max(1) as usize).min(volume[a])
    });
    (start, std::array::from_fn(|a| end[a] - start[a]))
}

fn smooth(raw: ArrayView3<'_, f64>, sigma: f64) -> Result<Array3<f64>> {
    let mut out = Array3::<f64>::zeros(raw.raw_dim());
    gaussian_smooth_into_with(
        raw,
        Gaussian::isotropic(sigma, 4.0)?.kernels(),
        Boundary::Clamp,
        out.view_mut(),
    )?;
    Ok(out)
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mid = v.len() / 2;
    *v.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn ratio(num: f64, den: f64) -> f64 {
    if den > 0.0 {
        num / den
    } else {
        0.0
    }
}

/// Central-difference gradient; zero on the window's faces.
fn gradient(a: &Array3<f64>, [z, y, x]: [usize; 3]) -> [f64; 3] {
    let (nz, ny, nx) = a.dim();
    if z == 0 || y == 0 || x == 0 || z + 1 >= nz || y + 1 >= ny || x + 1 >= nx {
        return [0.0; 3];
    }
    [
        0.5 * (a[[z + 1, y, x]] - a[[z - 1, y, x]]),
        0.5 * (a[[z, y + 1, x]] - a[[z, y - 1, x]]),
        0.5 * (a[[z, y, x + 1]] - a[[z, y, x - 1]]),
    ]
}

/// Two unit vectors completing `r` to an orthonormal frame.
fn tangents(r: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let a = if r[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let cross = |p: [f64; 3], q: [f64; 3]| {
        [
            p[1] * q[2] - p[2] * q[1],
            p[2] * q[0] - p[0] * q[2],
            p[0] * q[1] - p[1] * q[0],
        ]
    };
    let t = cross(r, a);
    let n = dot(t, t).sqrt();
    let t1 = [t[0] / n, t[1] / n, t[2] / n];
    (t1, cross(r, t1))
}

/// Eigenvalues (descending) and unit eigenvectors of a symmetric 3×3 matrix,
/// by cyclic Jacobi rotations.
fn eig3(mut m: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..16 {
        let off = m[0][1].powi(2) + m[0][2].powi(2) + m[1][2].powi(2);
        if off < 1e-24 * (m[0][0].powi(2) + m[1][1].powi(2) + m[2][2].powi(2)) + 1e-300 {
            break;
        }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            if m[p][q] == 0.0 {
                continue;
            }
            let theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let t = if theta == 0.0 { 1.0 } else { t };
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            for k in 0..3 {
                let (mkp, mkq) = (m[k][p], m[k][q]);
                m[k][p] = c * mkp - s * mkq;
                m[k][q] = s * mkp + c * mkq;
            }
            for k in 0..3 {
                let (mpk, mqk) = (m[p][k], m[q][k]);
                m[p][k] = c * mpk - s * mqk;
                m[q][k] = s * mpk + c * mqk;
            }
            for row in &mut v {
                let (vp, vq) = (row[p], row[q]);
                row[p] = c * vp - s * vq;
                row[q] = s * vp + c * vq;
            }
        }
    }
    let mut order = [0, 1, 2];
    order.sort_by(|&a, &b| m[b][b].total_cmp(&m[a][a]));
    let values = order.map(|i| m[i][i]);
    let vectors = order.map(|i| [v[0][i], v[1][i], v[2][i]]);
    (values, vectors)
}

/// Fractional anisotropy and the Westin linear and planar measures of
/// descending eigenvalues.
fn anisotropy(l: [f64; 3]) -> (f64, f64, f64) {
    let l = l.map(|x| x.max(0.0));
    let sum = l[0] + l[1] + l[2];
    let norm = (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]).sqrt();
    let fa = ratio(
        ((l[0] - l[1]).powi(2) + (l[1] - l[2]).powi(2) + (l[2] - l[0]).powi(2)).sqrt(),
        std::f64::consts::SQRT_2 * norm,
    );
    (
        fa,
        ratio(l[0] - l[1], sum),
        ratio(2.0 * (l[1] - l[2]), sum),
    )
}

/// Orthonormal real spherical harmonics up to degree `lmax` at the unit
/// vector `u = [z, y, x]`, indexed `l * l + l + m`.
pub fn real_sh(lmax: usize, u: [f64; 3]) -> Vec<f64> {
    let ct = u[0].clamp(-1.0, 1.0);
    let st = (1.0 - ct * ct).max(0.0).sqrt();
    let phi = u[1].atan2(u[2]);
    // fully normalised associated Legendre functions p[l][m]
    let mut p = vec![vec![0.0; lmax + 1]; lmax + 1];
    p[0][0] = (0.25 / std::f64::consts::PI).sqrt();
    for m in 1..=lmax {
        p[m][m] = -((2 * m + 1) as f64 / (2 * m) as f64).sqrt() * st * p[m - 1][m - 1];
    }
    for m in 0..lmax {
        p[m + 1][m] = ((2 * m + 3) as f64).sqrt() * ct * p[m][m];
    }
    for m in 0..=lmax {
        for l in (m + 2)..=lmax {
            let (lf, mf) = (l as f64, m as f64);
            let a = ((4.0 * lf * lf - 1.0) / (lf * lf - mf * mf)).sqrt();
            let b = (((lf - 1.0).powi(2) - mf * mf) / (4.0 * (lf - 1.0).powi(2) - 1.0)).sqrt();
            p[l][m] = a * (ct * p[l - 1][m] - b * p[l - 2][m]);
        }
    }
    let mut y = vec![0.0; (lmax + 1) * (lmax + 1)];
    for l in 0..=lmax {
        y[l * l + l] = p[l][0];
        for m in 1..=l {
            let mf = m as f64;
            y[l * l + l + m] = std::f64::consts::SQRT_2 * p[l][m] * (mf * phi).cos();
            y[l * l + l - m] = std::f64::consts::SQRT_2 * p[l][m] * (mf * phi).sin();
        }
    }
    y
}

impl Extractor {
    /// The features of one object, in [`COLUMNS`] order. `raw` is the level-0
    /// window from [`window`] at `start`; `centre` and `radii` are the fit's.
    pub fn measure(
        &self,
        rays: &RaySet,
        raw: ArrayView3<'_, f64>,
        start: [usize; 3],
        volume: [usize; 3],
        centre: [f64; 3],
        radii: &[f32],
    ) -> Result<Vec<f64>> {
        let dim = raw.dim();
        let c: [f64; 3] = std::array::from_fn(|a| centre[a] - start[a] as f64);
        let rmax = radii.iter().copied().fold(0.0f32, f32::max) as f64;
        let rbar = radii.iter().map(|&r| r as f64).sum::<f64>() / radii.len() as f64;
        let s1 = smooth(raw, 1.0)?;
        let s2 = smooth(raw, 2.0)?;
        let at = |p: [f64; 3]| -> Option<f64> { trilinear_at(s1.view(), p) };

        // shape-adapted radius of every voxel near the object (NaN elsewhere)
        let mut sv = Array3::<f32>::from_elem(dim, f32::NAN);
        for ((z, y, x), v) in sv.indexed_iter_mut() {
            let d = [z as f64 - c[0], y as f64 - c[1], x as f64 - c[2]];
            let r = dot(d, d).sqrt();
            if r <= 1.5 * rmax + 1.0 {
                *v = (r / (rays.interpolate(radii, d) as f64).max(1.0)) as f32;
            }
        }
        let voxels = || {
            sv.indexed_iter().filter_map(|((z, y, x), &s)| {
                (!s.is_nan()).then_some(([z, y, x], s as f64))
            })
        };

        // normalisation: dark band and wall peak of a fine profile (0.05 in s)
        let mut fine = vec![Vec::new(); 24];
        for (i, s) in voxels() {
            if let Some(b) = fine.get_mut((s / 0.05) as usize) {
                b.push(s1[i]);
            }
        }
        let fine: Vec<Option<f64>> = fine
            .iter_mut()
            .map(|b| (!b.is_empty()).then(|| median(b)))
            .collect();
        let extreme = |range: std::ops::Range<usize>, max: bool| {
            fine[range].iter().flatten().copied().reduce(if max { f64::max } else { f64::min })
        };
        let wall = extreme(16..24, true).unwrap_or(0.0);
        let dark = extreme(10..20, false).unwrap_or(wall).min(wall);
        let scale = (wall - dark).max(1.0);
        let norm = |v: f64| (v - dark) / scale;

        let mut out = Vec::with_capacity(COLUMNS.len());

        // validity: the fitted wall point inside the volume
        let inside = |p: [f64; 3]| (0..3).all(|a| p[a] >= 0.0 && p[a] <= (volume[a] - 1) as f64);
        let dirs = rays.directions();
        let valid = dirs
            .iter()
            .zip(radii)
            .filter(|(u, &r)| inside(std::array::from_fn(|a| centre[a] + r as f64 * u[a])))
            .count();
        out.push(valid as f64 / dirs.len() as f64);

        // radial profile
        let mut bins = vec![Vec::new(); PROFILE_BINS];
        let (mut core, mut band) = (Vec::new(), Vec::new());
        for (i, s) in voxels() {
            let v = norm(s1[i]);
            let b = (s / PROFILE_EXTENT * PROFILE_BINS as f64) as usize;
            if b < PROFILE_BINS {
                bins[b].push(v);
            }
            if s < 0.25 {
                core.push(v);
            } else if (0.4..0.7).contains(&s) {
                band.push(v);
            }
        }
        for b in &mut bins {
            out.push(median(b));
        }
        out.push(median(&mut core) - median(&mut band));
        out.push(wall - dark);
        out.push(0.0); // wall_step_rel, filled per volume

        // wall thickness along each ray: half-height crossings around the peak,
        // against the lowest value on each side
        let mut thick = Vec::new();
        for (u, &ri) in dirs.iter().zip(radii) {
            let ri = ri as f64;
            let n = ((0.8 * ri) / 0.5) as usize + 1;
            let samples: Option<Vec<(f64, f64)>> = (0..n)
                .map(|k| {
                    let r = 0.5 * ri + 0.5 * k as f64;
                    at(std::array::from_fn(|a| c[a] + r * u[a])).map(|v| (r, norm(v)))
                })
                .collect();
            let Some(samples) = samples else { continue };
            let Some((pk, &(_, peak))) = samples
                .iter()
                .enumerate()
                .filter(|(_, (r, _))| (0.8 * ri..=1.15 * ri).contains(r))
                .max_by(|a, b| a.1 .1.total_cmp(&b.1 .1))
            else {
                continue;
            };
            let low = samples[..=pk]
                .iter()
                .map(|s| s.1)
                .fold(f64::INFINITY, f64::min);
            let crossing = |level: f64, range: &mut dyn Iterator<Item = usize>| {
                let mut prev = pk;
                for k in range {
                    if samples[k].1 < level {
                        let (r0, v0) = samples[prev];
                        let (r1, v1) = samples[k];
                        return r0 + (r1 - r0) * ratio(v0 - level, v0 - v1);
                    }
                    prev = k;
                }
                samples[prev].0
            };
            let r_in = crossing(0.5 * (peak + low), &mut (0..pk).rev());
            let high = samples[pk..]
                .iter()
                .map(|s| s.1)
                .fold(f64::INFINITY, f64::min);
            let r_out = crossing(0.5 * (peak + high), &mut (pk + 1..samples.len()));
            thick.push(r_out - r_in);
        }
        let tmean = thick.iter().sum::<f64>() / thick.len().max(1) as f64;
        let tsd = (thick.iter().map(|t| (t - tmean).powi(2)).sum::<f64>()
            / thick.len().max(1) as f64)
            .sqrt();
        out.push(median(&mut thick));
        out.push(ratio(tsd, tmean));

        // radialness: gradient energy along the radial direction, per zone
        for smoothed in [&s1, &s2] {
            let mut acc = [[0.0f64; 2]; 3];
            for (i, s) in voxels() {
                let Some(z) = ZONES.iter().position(|&(lo, hi)| (lo..hi).contains(&s)) else {
                    continue;
                };
                let d = [i[0] as f64 - c[0], i[1] as f64 - c[1], i[2] as f64 - c[2]];
                let r = dot(d, d).sqrt();
                let g = gradient(smoothed, i);
                acc[z][0] += dot(g, d).powi(2) / (r * r);
                acc[z][1] += dot(g, g);
            }
            out.extend(acc.iter().map(|a| ratio(a[0], a[1])));
        }

        // structure tensor (sigma 1.5, rho 3) in the mid zone: alignment of the
        // flattest direction with the radius, weighted by coherence
        {
            let s15 = smooth(raw, 1.5)?;
            let mut products: Vec<Array3<f64>> = (0..6).map(|_| Array3::zeros(dim)).collect();
            for ((z, y, x), _) in s15.indexed_iter() {
                let g = gradient(&s15, [z, y, x]);
                for (k, (p, q)) in [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
                    .into_iter()
                    .enumerate()
                {
                    products[k][[z, y, x]] = g[p] * g[q];
                }
            }
            let j: Vec<Array3<f64>> = products
                .iter()
                .map(|p| smooth(p.view(), 3.0))
                .collect::<Result<_>>()?;
            let (mut align, mut weight, mut coherence, mut count) = (0.0, 0.0, 0.0, 0usize);
            for (i, s) in voxels() {
                if !(ZONES[1].0..ZONES[1].1).contains(&s) {
                    continue;
                }
                let e = |k: usize| j[k][i];
                let (l, v) = eig3([
                    [e(0), e(1), e(2)],
                    [e(1), e(3), e(4)],
                    [e(2), e(4), e(5)],
                ]);
                let coh = ratio(l[0] - l[2], l[0] + l[2]);
                let d = [i[0] as f64 - c[0], i[1] as f64 - c[1], i[2] as f64 - c[2]];
                let r = dot(d, d).sqrt();
                align += coh * dot(v[2], d).powi(2) / (r * r);
                weight += coh;
                coherence += coh;
                count += 1;
            }
            out.push(ratio(align, weight));
            out.push(ratio(coherence, count as f64));
        }

        // object-frame variograms over the interior
        {
            let steps = [2.0, 4.0, 8.0];
            let mut rad = [0.0f64; 3];
            let mut tan = [0.0f64; 3];
            let mut counts = [0usize; 3];
            let (mut sum, mut sum2, mut n) = (0.0, 0.0, 0usize);
            for (i, s) in voxels() {
                if !(0.2..0.85).contains(&s) {
                    continue;
                }
                let p = [i[0] as f64, i[1] as f64, i[2] as f64];
                let d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
                let r = dot(d, d).sqrt();
                let rh = [d[0] / r, d[1] / r, d[2] / r];
                let (t1, t2) = tangents(rh);
                let turn: [[f64; 3]; TANGENTS] = std::array::from_fn(|k| {
                    let a = std::f64::consts::PI * k as f64 / TANGENTS as f64;
                    std::array::from_fn(|i| a.cos() * t1[i] + a.sin() * t2[i])
                });
                let v0 = s1[i];
                sum += v0;
                sum2 += v0 * v0;
                n += 1;
                for (k, &h) in steps.iter().enumerate() {
                    // both ends in the interior: a radial pair must not reach the wall
                    if (r + h) / r * s >= 0.85 {
                        continue;
                    }
                    let shifted = |e: [f64; 3]| at(std::array::from_fn(|a| p[a] + h * e[a]));
                    let Some(a) = shifted(rh) else { continue };
                    let t: Option<f64> = turn
                        .iter()
                        .map(|&e| shifted(e).map(|b| (b - v0).powi(2)))
                        .sum();
                    if let Some(t) = t {
                        rad[k] += (a - v0).powi(2);
                        tan[k] += t / TANGENTS as f64;
                        counts[k] += 1;
                    }
                }
            }
            for k in 0..3 {
                out.push(ratio(rad[k], tan[k]));
            }
            let var = ratio(sum2, n as f64) - ratio(sum, n as f64).powi(2);
            out.push(ratio(0.5 * tan[1] / counts[1].max(1) as f64, var));
        }

        // per-shell spread and skewness of normalised intensity
        for centre_s in [0.25, 0.5, 0.75] {
            let v: Vec<f64> = voxels()
                .filter(|(_, s)| (s - centre_s).abs() < 0.05)
                .map(|(i, _)| norm(s1[i]))
                .collect();
            let n = v.len().max(1) as f64;
            let mean = v.iter().sum::<f64>() / n;
            let m2 = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
            let m3 = v.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / n;
            out.push(m2.sqrt());
            out.push(ratio(m3, m2.powf(1.5)));
        }

        // intensity second moments of the interior, and the shape's
        {
            let (mut w, mut m1) = (0.0, [0.0; 3]);
            let mut m2 = [[0.0; 3]; 3];
            for (i, s) in voxels() {
                if s >= 0.85 {
                    continue;
                }
                let wi = norm(s1[i]).max(0.0);
                let d = [i[0] as f64 - c[0], i[1] as f64 - c[1], i[2] as f64 - c[2]];
                w += wi;
                for a in 0..3 {
                    m1[a] += wi * d[a];
                    for b in 0..3 {
                        m2[a][b] += wi * d[a] * d[b];
                    }
                }
            }
            let mean: [f64; 3] = std::array::from_fn(|a| ratio(m1[a], w));
            let cov: [[f64; 3]; 3] =
                std::array::from_fn(|a| std::array::from_fn(|b| ratio(m2[a][b], w) - mean[a] * mean[b]));
            let (fa, cl, cp) = anisotropy(eig3(cov).0);
            out.extend([fa, cl, cp, dot(mean, mean).sqrt() / rbar]);
            let mut t = [[0.0; 3]; 3];
            for (u, &r) in dirs.iter().zip(radii) {
                let r5 = (r as f64).powi(5);
                for a in 0..3 {
                    for b in 0..3 {
                        t[a][b] += r5 * u[a] * u[b];
                    }
                }
            }
            let l = eig3(t).0;
            out.push(anisotropy(l).0);
            out.push(ratio(l[0], l[2]).max(0.0).sqrt());
        }

        // spherical-harmonic spectra of three shells and their coherence
        let coefficients: Vec<Vec<f64>> = SH_SHELLS
            .iter()
            .map(|&s| {
                let values: Vec<Option<f64>> = self
                    .dense
                    .iter()
                    .map(|&u| {
                        let r = s * rays.interpolate(radii, u) as f64;
                        at(std::array::from_fn(|a| c[a] + r * u[a])).map(norm)
                    })
                    .collect();
                let present: Vec<f64> = values.iter().flatten().copied().collect();
                let mean = present.iter().sum::<f64>() / present.len().max(1) as f64;
                let f: Vec<f64> = values.iter().map(|v| v.unwrap_or(mean) - mean).collect();
                let w = 4.0 * std::f64::consts::PI / DENSE as f64;
                let mut a = vec![0.0; (LMAX + 1) * (LMAX + 1)];
                for (fi, y) in f.iter().zip(&self.basis) {
                    for (ak, yk) in a.iter_mut().zip(y) {
                        *ak += w * fi * yk;
                    }
                }
                let var = f.iter().map(|x| x * x).sum::<f64>() / DENSE as f64;
                a.push(4.0 * std::f64::consts::PI * var);
                a
            })
            .collect();
        let band = |a: &[f64], b: &[f64], (lo, hi): (usize, usize)| {
            (lo * lo..(hi + 1) * (hi + 1))
                .map(|k| a[k] * b[k])
                .sum::<f64>()
        };
        for a in &coefficients {
            let total = a[a.len() - 1];
            out.extend(POWER_BANDS.iter().map(|&b| ratio(band(a, a, b), total)));
        }
        for pair in coefficients.windows(2) {
            let (a, b) = (&pair[0], &pair[1]);
            out.extend(COHERENCE_BANDS.iter().map(|&k| {
                ratio(band(a, b, k), (band(a, a, k) * band(b, b, k)).sqrt())
            }));
        }
        debug_assert_eq!(out.len(), COLUMNS.len());
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_harmonics_are_orthonormal_on_the_shell_directions() {
        let e = Extractor::default();
        let n = (LMAX + 1) * (LMAX + 1);
        let w = 4.0 * std::f64::consts::PI / DENSE as f64;
        let mut worst: f64 = 0.0;
        for i in 0..n {
            for j in i..n {
                let g: f64 = e.basis.iter().map(|y| w * y[i] * y[j]).sum();
                worst = worst.max((g - if i == j { 1.0 } else { 0.0 }).abs());
            }
        }
        assert!(worst < 0.02, "{worst}");
    }

    #[test]
    fn eig3_recovers_a_rotated_diagonal() {
        let (l, v) = eig3([[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 5.0]]);
        assert!((l[0] - 5.0).abs() < 1e-9 && (l[1] - 3.0).abs() < 1e-9 && (l[2] - 1.0).abs() < 1e-9);
        assert!((v[2][0].abs() - 0.5f64.sqrt()).abs() < 1e-9 && v[2][2].abs() < 1e-9);
    }

    /// A shell phantom: bright wall at radius 30, dark band inside, and an
    /// interior (to r = 25) of radial wedges or of isotropic grains.
    fn phantom(spokes: bool) -> (Array3<f64>, [f64; 3]) {
        let n = 96;
        let c = [47.5, 47.0, 48.25];
        let mut deltas = Array3::<f64>::zeros((n, n, n));
        for (k, u) in fibonacci_sphere(1500).iter().enumerate() {
            let r = 24.0 * ((k * 7919 % 1009) as f64 / 1009.0).cbrt();
            let p: [usize; 3] = std::array::from_fn(|a| (c[a] + r * u[a]).round() as usize);
            deltas[p] += 2400.0;
        }
        let grains = smooth(deltas.view(), 1.0).unwrap();
        let wedges = fibonacci_sphere(60);
        let img = Array3::from_shape_fn((n, n, n), |(z, y, x)| {
            let d = [z as f64 - c[0], y as f64 - c[1], x as f64 - c[2]];
            let r = dot(d, d).sqrt();
            let taper = 1.0 / (1.0 + (r - 25.0).exp());
            let texture = if spokes {
                let best = wedges
                    .iter()
                    .map(|u| dot(*u, d) / r.max(1e-9))
                    .fold(-1.0, f64::max);
                150.0 * (-(best.clamp(-1.0, 1.0).acos() / 0.1).powi(2)).exp()
            } else {
                grains[[z, y, x]]
            };
            100.0 + 400.0 * (-((r - 30.0) / 1.2).powi(2)).exp() + taper * texture
        });
        (img, c)
    }

    fn features_of(img: &Array3<f64>, c: [f64; 3]) -> Vec<f64> {
        let rays = RaySet::fibonacci(500, 8);
        let radii = vec![31.0f32; rays.len()];
        let e = Extractor::default();
        let dim = img.dim();
        e.measure(&rays, img.view(), [0; 3], [dim.0, dim.1, dim.2], c, &radii)
            .unwrap()
    }

    #[test]
    #[ignore]
    fn dump_phantom_features() {
        let (s, cs) = phantom(true);
        let (g, cg) = phantom(false);
        let n = g.dim().0;
        let rotated = Array3::from_shape_fn(g.dim(), |(a, b, d)| g[[b, d, n - 1 - a]]);
        let rc = [n as f64 - 1.0 - cg[2], cg[0], cg[1]];
        let (fs, fg, fr) = (features_of(&s, cs), features_of(&g, cg), features_of(&rotated, rc));
        for (k, name) in COLUMNS.iter().enumerate() {
            eprintln!("{name:20} spokes {:9.4} grains {:9.4} rotated {:9.4}", fs[k], fg[k], fr[k]);
        }
    }

    #[test]
    fn spokes_are_radial_and_grains_are_not() {
        let col = |name: &str| COLUMNS.iter().position(|&c| c == name).unwrap();
        let (s, cs) = phantom(true);
        let (g, cg) = phantom(false);
        let fs = features_of(&s, cs);
        let fg = features_of(&g, cg);
        for name in ["frad_mid_s1", "vario_ratio_4"] {
            assert!(fs[col(name)] < 0.7 * fg[col(name)], "{name}: {} vs {}", fs[col(name)], fg[col(name)]);
        }
        assert!(fs[col("st_radial_align")] > fg[col("st_radial_align")] + 0.15);
        assert!(fs[col("coh_30_55_l9_16")] > fg[col("coh_30_55_l9_16")] + 0.3);
    }

    #[test]
    fn features_do_not_change_under_an_axis_permutation() {
        let (img, c) = phantom(false);
        // (z, y, x) -> (x, z, y), and a flip of the new first axis
        let n = img.dim().0;
        let rotated = Array3::from_shape_fn(img.dim(), |(a, b, d)| img[[b, d, n - 1 - a]]);
        let rc = [n as f64 - 1.0 - c[2], c[0], c[1]];
        let f = features_of(&img, c);
        let fr = features_of(&rotated, rc);
        for (k, name) in COLUMNS.iter().enumerate() {
            // coherences are ill-conditioned where a band has little power
            let floor = if name.starts_with("coh_") { 0.5 } else { 0.05 };
            let scale = f[k].abs().max(fr[k].abs()).max(floor);
            assert!(
                (f[k] - fr[k]).abs() / scale < 0.08,
                "{name}: {} vs {}",
                f[k],
                fr[k]
            );
        }
    }
}
