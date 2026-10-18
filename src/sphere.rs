// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// Directions on the unit sphere and a least-squares sphere, for ops that work
// in rays from a centre (star-convex shapes) or fit round objects.
//
// `ops::label_energy` keeps its own spiral: its weights depend on the axis order
// that spiral was written in, so sharing this one would change its numbers on
// anisotropic spacing.

/// `n` near-uniform unit directions in `[z, y, x]` on a Fibonacci spiral:
/// `z = 1 - (2i + 1) / n`, azimuth `i * golden angle`, `y = r sin`, `x = r cos`.
pub fn fibonacci_sphere(n: usize) -> Vec<[f64; 3]> {
    let golden = std::f64::consts::PI * (3.0 - 5f64.sqrt());
    (0..n)
        .map(|i| {
            let z = 1.0 - (2 * i + 1) as f64 / n as f64;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let phi = i as f64 * golden;
            [z, r * phi.sin(), r * phi.cos()]
        })
        .collect()
}

/// Algebraic least-squares sphere through `points`: solves
/// `|p|^2 = 2 c.p + (R^2 - |c|^2)` for the centre `c` and radius `R`. `None`
/// for fewer than four points or a degenerate (coplanar) set.
pub fn fit_sphere(points: &[[f64; 3]]) -> Option<([f64; 3], f64)> {
    if points.len() < 4 {
        return None;
    }
    // Centre the points first so the normal equations stay well conditioned for
    // spheres far from the origin.
    let n = points.len() as f64;
    let mean: [f64; 3] = std::array::from_fn(|a| points.iter().map(|p| p[a]).sum::<f64>() / n);
    let mut m = [[0f64; 5]; 4];
    for p in points {
        let q = [p[0] - mean[0], p[1] - mean[1], p[2] - mean[2]];
        let row = [2.0 * q[0], 2.0 * q[1], 2.0 * q[2], 1.0];
        let b = q[0] * q[0] + q[1] * q[1] + q[2] * q[2];
        for i in 0..4 {
            for j in 0..4 {
                m[i][j] += row[i] * row[j];
            }
            m[i][4] += row[i] * b;
        }
    }
    let scale = m.iter().map(|r| r[..4].iter().map(|v| v.abs()).fold(0.0, f64::max)).fold(0.0, f64::max);
    for c in 0..4 {
        let pivot = (c..4).max_by(|&i, &j| m[i][c].abs().total_cmp(&m[j][c].abs()))?;
        m.swap(c, pivot);
        if m[c][c].abs() <= 1e-12 * scale.max(1.0) {
            return None;
        }
        for r in 0..4 {
            if r != c {
                let f = m[r][c] / m[c][c];
                for k in c..5 {
                    m[r][k] -= f * m[c][k];
                }
            }
        }
    }
    let x: [f64; 4] = std::array::from_fn(|i| m[i][4] / m[i][i]);
    let r2 = x[3] + x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
    (r2 > 0.0).then(|| ([x[0] + mean[0], x[1] + mean[1], x[2] + mean[2]], r2.sqrt()))
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// A fixed set of ray directions with what a star-convex shape needs from it:
/// each ray's nearest neighbours (for regularising radii along the surface) and
/// a lookup from any direction to the rays that interpolate it.
///
/// The lookup is a `theta x phi` table of the four nearest rays, weighted by
/// inverse angle, so evaluating a shape at a voxel is one `atan2`, one `acos`
/// and four multiplies rather than a search over the rays.
#[derive(Debug, Clone)]
pub struct RaySet {
    directions: Vec<[f64; 3]>,
    neighbours: Vec<Vec<usize>>,
    table: [usize; 2],
    lookup: Vec<[(u32, f32); 4]>,
}

impl RaySet {
    /// `n` Fibonacci directions, each with its `k` nearest neighbours.
    pub fn fibonacci(n: usize, k: usize) -> Self {
        let directions = fibonacci_sphere(n);
        let nearest = |v: [f64; 3], m: usize| {
            let mut d: Vec<(usize, f64)> =
                directions.iter().enumerate().map(|(j, &w)| (j, dot(v, w))).collect();
            d.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            d.truncate(m);
            d
        };
        let neighbours = directions
            .iter()
            .map(|&v| nearest(v, k + 1).into_iter().skip(1).map(|(j, _)| j).collect())
            .collect();
        let table = [64, 128];
        let mut lookup = Vec::with_capacity(table[0] * table[1]);
        for it in 0..table[0] {
            for ip in 0..table[1] {
                let theta = (it as f64 + 0.5) * std::f64::consts::PI / table[0] as f64;
                let phi = (ip as f64 + 0.5) * std::f64::consts::TAU / table[1] as f64
                    - std::f64::consts::PI;
                let v = [theta.cos(), theta.sin() * phi.sin(), theta.sin() * phi.cos()];
                let mut cell = [(0u32, 0f32); 4];
                let mut total = 0.0;
                for (slot, (j, d)) in cell.iter_mut().zip(nearest(v, 4)) {
                    let w = 1.0 / (d.clamp(-1.0, 1.0).acos() + 1e-3);
                    *slot = (j as u32, w as f32);
                    total += w as f32;
                }
                cell.iter_mut().for_each(|c| c.1 /= total);
                lookup.push(cell);
            }
        }
        Self {
            directions,
            neighbours,
            table,
            lookup,
        }
    }

    pub fn len(&self) -> usize {
        self.directions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.directions.is_empty()
    }

    pub fn directions(&self) -> &[[f64; 3]] {
        &self.directions
    }

    pub fn neighbours(&self, ray: usize) -> &[usize] {
        &self.neighbours[ray]
    }

    /// A per-ray value (a radius) interpolated in the direction of `offset`,
    /// which need not be unit length. `values` has one entry per ray.
    pub fn interpolate(&self, values: &[f32], offset: [f64; 3]) -> f32 {
        let n = dot(offset, offset).sqrt().max(1e-12);
        let theta = (offset[0] / n).clamp(-1.0, 1.0).acos();
        let phi = offset[1].atan2(offset[2]);
        let it = ((theta / std::f64::consts::PI * self.table[0] as f64) as usize)
            .min(self.table[0] - 1);
        let ip = (((phi + std::f64::consts::PI) / std::f64::consts::TAU * self.table[1] as f64)
            as usize)
            .min(self.table[1] - 1);
        self.lookup[it * self.table[1] + ip]
            .iter()
            .map(|&(j, w)| w * values[j as usize])
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fibonacci_directions_are_unit_and_balanced() {
        for n in [1, 7, 120, 500] {
            let u = fibonacci_sphere(n);
            assert_eq!(u.len(), n);
            for d in &u {
                let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                assert!((len - 1.0).abs() < 1e-12);
            }
            if n >= 120 {
                let mean: [f64; 3] = std::array::from_fn(|a| u.iter().map(|d| d[a]).sum::<f64>() / n as f64);
                assert!(mean.iter().all(|m| m.abs() < 0.02), "{n}: {mean:?}");
            }
        }
    }

    /// A sphere's constant radius interpolates to itself everywhere, and a
    /// value set on one ray dominates only around that ray.
    #[test]
    fn ray_set_interpolates() {
        let rays = RaySet::fibonacci(300, 6);
        assert_eq!(rays.len(), 300);
        assert!(rays.neighbours(0).len() == 6 && !rays.neighbours(0).contains(&0));
        let constant = vec![12.5f32; 300];
        for &u in fibonacci_sphere(97).iter() {
            assert!((rays.interpolate(&constant, [3.0 * u[0], 3.0 * u[1], 3.0 * u[2]]) - 12.5).abs() < 1e-4);
        }
        let mut spike = vec![0f32; 300];
        spike[150] = 1.0;
        let along = rays.directions()[150];
        assert!(rays.interpolate(&spike, along) > 0.5);
        assert_eq!(rays.interpolate(&spike, [-along[0], -along[1], -along[2]]), 0.0);
    }

    #[test]
    fn sphere_fit_recovers_centre_and_radius() {
        let centre = [300.5, -12.0, 4000.25];
        let radius = 27.5;
        let mut seed = 11u64;
        let mut jitter = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.4
        };
        // a partial sphere (one hemisphere and a bit), as at a volume face
        let points: Vec<[f64; 3]> = fibonacci_sphere(400)
            .into_iter()
            .filter(|u| u[0] > -0.3)
            .map(|u| std::array::from_fn(|a| centre[a] + (radius + jitter()) * u[a]))
            .collect();
        let (c, r) = fit_sphere(&points).unwrap();
        assert!((0..3).all(|a| (c[a] - centre[a]).abs() < 0.1), "{c:?}");
        assert!((r - radius).abs() < 0.1, "{r}");
        assert!(fit_sphere(&points[..3]).is_none());
        let plane: Vec<[f64; 3]> = (0..20).map(|i| [0.0, i as f64, (i * i % 7) as f64]).collect();
        assert!(fit_sphere(&plane).is_none());
    }
}
