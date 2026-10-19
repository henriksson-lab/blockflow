// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Per-synapse features: one row per image, for the cross-condition analysis.
//!
//! Intensities are not comparable between images (laser power and exposure
//! differ), so every intensity feature is a ratio within the synapse, after
//! subtracting the image's own background. Spatial features are normalised to
//! the synapse — radial position as distance-to-edge over the inradius, point
//! patterns against random points in the same mask — so that layout can be
//! separated from size downstream; the size features are reported alongside.
//!
//! | group | features |
//! |---|---|
//! | size, shape | `area_um2`, `eq_radius_um`, `inradius_um`, `eccentricity`, `solidity`, `touches_frame` |
//! | amount | `n_spots`, `spot_density_um2`, `spot_area_fraction`, `spot_area_median_px`, `spot_ecc_median` |
//! | heterogeneity | `pixel_cv`, `spot_int_cv`, `spot_int_gini`, `spot_int_log_p90_p10`, `punctate_fraction` |
//! | radial | `spot_radial_index`, `spot_radial_index_w`, `radial_bin1..4` (edge to centre), `central_enrichment` |
//! | polarity | `polarity` |
//! | point pattern | `clark_evans`, `nn_cv`, `ripley_h_*` |

use std::fmt::Write as _;

use blockflow::ops::threshold::background_mode;
use ndarray::ArrayView2;

/// One spot as the features need it: position (pixels), area, net intensity.
pub struct Spot {
    pub y: f64,
    pub x: f64,
    pub area: f64,
    pub raw_sum: f64,
    pub ecc: f64,
    /// Mean distance to the synapse edge over the spot's pixels, pixels.
    pub edge_distance: f64,
}

pub struct Inputs<'a> {
    pub raw: ArrayView2<'a, f64>,
    pub synapse: ArrayView2<'a, bool>,
    /// Distance to the synapse edge inside it, pixels; 0 outside.
    pub distance: ArrayView2<'a, f64>,
    pub spot_pixels: ArrayView2<'a, bool>,
    pub spots: &'a [Spot],
    pub eccentricity: f64,
    pub pixel_um: f64,
    pub seed: u64,
}

/// Monte Carlo draws of random spot positions per synapse.
const SIMULATIONS: usize = 99;
// not below ~0.5 um: spots are ~6 px (0.4 um) across and at least 2.5 px apart,
// so shorter scales measure the detector's exclusion zone, not the biology
const RIPLEY_UM: [f64; 3] = [0.5, 1.0, 1.5];
const RIPLEY_REL: [f64; 2] = [0.1, 0.2];
const RADIAL_BINS: usize = 4;

pub fn header() -> String {
    let mut h = String::from(
        "area_um2,eq_radius_um,inradius_um,eccentricity,solidity,touches_frame,\
         n_spots,spot_density_um2,spot_area_fraction,spot_area_median_px,spot_ecc_median,\
         background,pixel_cv,spot_int_cv,spot_int_gini,spot_int_log_p90_p10,punctate_fraction,\
         spot_radial_index,spot_radial_index_w",
    );
    for b in 1..=RADIAL_BINS {
        let _ = write!(h, ",radial_bin{b}");
    }
    h.push_str(",central_enrichment,polarity,clark_evans,nn_cv");
    for r in RIPLEY_UM {
        let _ = write!(h, ",ripley_h_{}um", fmt_scale(r));
    }
    for r in RIPLEY_REL {
        let _ = write!(h, ",ripley_h_{}R", fmt_scale(r));
    }
    h
}

fn fmt_scale(r: f64) -> String {
    format!("{r}").replace('.', "p")
}

pub fn row(inputs: &Inputs<'_>) -> String {
    let values = compute(inputs);
    values
        .iter()
        .map(|v| if v.is_finite() { format!("{v:.6}") } else { "NA".into() })
        .collect::<Vec<_>>()
        .join(",")
}

fn compute(inp: &Inputs<'_>) -> Vec<f64> {
    let (h, w) = inp.synapse.dim();
    let um = inp.pixel_um;
    let mut inside = Vec::new();
    let mut outside = Vec::new();
    let mut touches = false;
    let mut spot_px = 0usize;
    for y in 0..h {
        for x in 0..w {
            if inp.synapse[[y, x]] {
                inside.push((y, x));
                touches |= y == 0 || x == 0 || y + 1 == h || x + 1 == w;
                spot_px += usize::from(inp.spot_pixels[[y, x]]);
            } else {
                outside.push(inp.raw[[y, x]]);
            }
        }
    }
    let area_px = inside.len() as f64;
    let area = area_px * um * um;
    let eq_radius = (area / std::f64::consts::PI).sqrt();
    let inradius_px = inside
        .iter()
        .map(|&(y, x)| inp.distance[[y, x]])
        .fold(0.0, f64::max);
    let solidity = area_px / hull_pixel_area(&inside).max(area_px);

    // background: the mode of the pixels outside the synapse
    let background = background_mode(&outside, 512).map_or(f64::NAN, |(mode, _)| mode);
    let net: Vec<f64> = inside
        .iter()
        .map(|&(y, x)| inp.raw[[y, x]] - background)
        .collect();
    let net_total: f64 = net.iter().sum();

    let spot_net: Vec<f64> = inp
        .spots
        .iter()
        .map(|s| s.raw_sum - s.area * background)
        .collect();
    let n = inp.spots.len();

    // radial: distance to edge over inradius, 0 at the edge, 1 at the centre
    let rnorm = |d: f64| (d / inradius_px).clamp(0.0, 1.0);
    let uniform: f64 = inside.iter().map(|&(y, x)| rnorm(inp.distance[[y, x]])).sum::<f64>() / area_px;
    let spot_r: Vec<f64> = inp.spots.iter().map(|s| rnorm(s.edge_distance)).collect();
    let spot_radial_index = mean(&spot_r) / uniform;
    let weights: Vec<f64> = spot_net.iter().map(|v| v.max(0.0)).collect();
    let spot_radial_index_w = weighted_mean(&spot_r, &weights) / uniform;
    let mut bin_area = [0.0; RADIAL_BINS];
    let mut bin_int = [0.0; RADIAL_BINS];
    for (&(y, x), &v) in inside.iter().zip(&net) {
        let b = ((rnorm(inp.distance[[y, x]]) * RADIAL_BINS as f64) as usize).min(RADIAL_BINS - 1);
        bin_area[b] += 1.0;
        bin_int[b] += v;
    }
    // relative intensity density per bin: 1 everywhere for a flat synapse
    let radial_bins: Vec<f64> = (0..RADIAL_BINS)
        .map(|b| (bin_int[b] / net_total) / (bin_area[b] / area_px))
        .collect();
    let (mut central_int, mut central_area) = (0.0, 0.0);
    for (&(y, x), &v) in inside.iter().zip(&net) {
        if rnorm(inp.distance[[y, x]]) > 0.5 {
            central_int += v;
            central_area += 1.0;
        }
    }
    let central_enrichment = (central_int / net_total) / (central_area / area_px);

    // polarity: intensity centroid offset from the mask centroid, in radii
    let (mut cy, mut cx, mut wy, mut wx, mut wsum) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (&(y, x), &v) in inside.iter().zip(&net) {
        cy += y as f64;
        cx += x as f64;
        let v = v.max(0.0);
        wy += v * y as f64;
        wx += v * x as f64;
        wsum += v;
    }
    let polarity = ((wy / wsum - cy / area_px).hypot(wx / wsum - cx / area_px)) * um / eq_radius;

    // point pattern against random points in the same mask
    let points: Vec<[f64; 2]> = inp.spots.iter().map(|s| [s.y * um, s.x * um]).collect();
    let mut rng = SplitMix(inp.seed);
    let scales: Vec<f64> = RIPLEY_UM
        .iter()
        .copied()
        .chain(RIPLEY_REL.iter().map(|r| r * eq_radius))
        .collect();
    let (nn_obs, nn_cv) = nearest_neighbour(&points);
    let l_obs = ripley_l(&points, &scales, area);
    let mut nn_sim = Vec::with_capacity(SIMULATIONS);
    let mut l_sim = vec![Vec::with_capacity(SIMULATIONS); scales.len()];
    if n >= 2 {
        for _ in 0..SIMULATIONS {
            let sim: Vec<[f64; 2]> = (0..n)
                .map(|_| {
                    let (y, x) = inside[rng.below(inside.len())];
                    [(y as f64 + rng.unit() - 0.5) * um, (x as f64 + rng.unit() - 0.5) * um]
                })
                .collect();
            nn_sim.push(nearest_neighbour(&sim).0);
            for (k, l) in ripley_l(&sim, &scales, area).into_iter().enumerate() {
                l_sim[k].push(l);
            }
        }
    }
    let clark_evans = nn_obs / mean(&nn_sim);
    // H = (L_obs - mean L_random) / r: dimensionless, > 0 clustered at r
    let ripley: Vec<f64> = scales
        .iter()
        .enumerate()
        .map(|(k, &r)| (l_obs[k] - mean(&l_sim[k])) / r)
        .collect();

    let spot_areas: Vec<f64> = inp.spots.iter().map(|s| s.area).collect();
    let spot_eccs: Vec<f64> = inp.spots.iter().map(|s| s.ecc).filter(|e| e.is_finite()).collect();
    let mut out = vec![
        area,
        eq_radius,
        inradius_px * um,
        inp.eccentricity,
        solidity,
        f64::from(u8::from(touches)),
        n as f64,
        n as f64 / area,
        spot_px as f64 / area_px,
        median(&spot_areas),
        median(&spot_eccs),
        background,
        sd(&net) / mean(&net),
        sd(&spot_net) / mean(&spot_net),
        gini(&weights),
        (quantile(&spot_net, 0.9).max(1e-9) / quantile(&spot_net, 0.1).max(1e-9)).ln(),
        spot_net.iter().sum::<f64>() / net_total,
        spot_radial_index,
        spot_radial_index_w,
    ];
    out.extend(radial_bins);
    out.extend([central_enrichment, polarity, clark_evans, nn_cv]);
    out.extend(ripley);
    out
}

/// Area of the convex hull of the pixel squares (corners), so a filled
/// convex blob has solidity close to 1.
fn hull_pixel_area(pixels: &[(usize, usize)]) -> f64 {
    let mut pts: Vec<(i64, i64)> = Vec::with_capacity(pixels.len() * 4);
    for &(y, x) in pixels {
        let (y, x) = (y as i64, x as i64);
        pts.extend([(y, x), (y + 1, x), (y, x + 1), (y + 1, x + 1)]);
    }
    pts.sort_unstable();
    pts.dedup();
    if pts.len() < 3 {
        return 0.0;
    }
    let cross = |o: (i64, i64), a: (i64, i64), b: (i64, i64)| {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };
    let mut hull: Vec<(i64, i64)> = Vec::new();
    for pass in 0..2 {
        let start = hull.len();
        let iter: Box<dyn Iterator<Item = &(i64, i64)>> =
            if pass == 0 { Box::new(pts.iter()) } else { Box::new(pts.iter().rev()) };
        for &p in iter {
            while hull.len() >= start + 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0 {
                hull.pop();
            }
            hull.push(p);
        }
        hull.pop();
    }
    let twice: i64 = (0..hull.len())
        .map(|i| {
            let (a, b) = (hull[i], hull[(i + 1) % hull.len()]);
            a.0 * b.1 - b.0 * a.1
        })
        .sum();
    twice.abs() as f64 / 2.0
}

/// Mean nearest-neighbour distance and its coefficient of variation.
fn nearest_neighbour(points: &[[f64; 2]]) -> (f64, f64) {
    if points.len() < 2 {
        return (f64::NAN, f64::NAN);
    }
    let nn: Vec<f64> = points
        .iter()
        .enumerate()
        .map(|(i, p)| {
            points
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, q)| (p[0] - q[0]).hypot(p[1] - q[1]))
                .fold(f64::INFINITY, f64::min)
        })
        .collect();
    (mean(&nn), sd(&nn) / mean(&nn))
}

/// Besag's `L(r) = sqrt(K(r) / pi)`, `K(r) = A / (n (n - 1)) * #pairs < r`,
/// uncorrected for edges: the random reference in the same mask carries the
/// same edge effect, so the difference is what is reported.
fn ripley_l(points: &[[f64; 2]], scales: &[f64], area: f64) -> Vec<f64> {
    let n = points.len();
    if n < 2 {
        return vec![f64::NAN; scales.len()];
    }
    let mut counts = vec![0usize; scales.len()];
    for i in 0..n {
        for j in i + 1..n {
            let d = (points[i][0] - points[j][0]).hypot(points[i][1] - points[j][1]);
            for (k, &r) in scales.iter().enumerate() {
                if d < r {
                    counts[k] += 2;
                }
            }
        }
    }
    counts
        .iter()
        .map(|&c| (area * c as f64 / (n * (n - 1)) as f64 / std::f64::consts::PI).sqrt())
        .collect()
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        f64::NAN
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn weighted_mean(v: &[f64], w: &[f64]) -> f64 {
    let total: f64 = w.iter().sum();
    v.iter().zip(w).map(|(a, b)| a * b).sum::<f64>() / total
}

fn sd(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return f64::NAN;
    }
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

fn quantile(v: &[f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let rank = q * (s.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    s[lo] + (rank - lo as f64) * (s[hi] - s[lo])
}

fn median(v: &[f64]) -> f64 {
    quantile(v, 0.5)
}

/// Gini coefficient of non-negative values: 0 all equal, towards 1 when a few
/// spots hold most of the signal.
fn gini(v: &[f64]) -> f64 {
    let n = v.len();
    let total: f64 = v.iter().sum();
    if n < 2 || total <= 0.0 {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let ranked: f64 = s.iter().enumerate().map(|(i, x)| (i + 1) as f64 * x).sum();
    2.0 * ranked / (n as f64 * total) - (n as f64 + 1.0) / n as f64
}

/// A small deterministic generator, so a rerun gives the same table.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn below(&mut self, n: usize) -> usize {
        (self.unit() * n as f64) as usize % n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hull_of_a_filled_square_is_its_area() {
        let px: Vec<(usize, usize)> = (0..5).flat_map(|y| (0..5).map(move |x| (y, x))).collect();
        assert_eq!(hull_pixel_area(&px), 25.0);
    }

    #[test]
    fn gini_is_zero_for_equal_and_large_for_one_holder() {
        assert!(gini(&[1.0, 1.0, 1.0, 1.0]).abs() < 1e-12);
        assert!(gini(&[0.0, 0.0, 0.0, 10.0]) > 0.7);
    }

    #[test]
    fn ripley_l_of_a_pair_inside_r() {
        // two points 1 apart: K(2) = A / 2 * 2 = A, L = sqrt(A / pi)
        let l = ripley_l(&[[0.0, 0.0], [0.0, 1.0]], &[2.0, 0.5], 10.0);
        assert!((l[0] - (10.0 / std::f64::consts::PI).sqrt()).abs() < 1e-12);
        assert_eq!(l[1], 0.0);
    }
}
