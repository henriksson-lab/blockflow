// SPDX-License-Identifier: MIT
// Original work for this crate.

//! How alike two images on one lattice are: the numbers `register` records
//! after each stage and the comparison with the reference reports.
//!
//! Both are over every voxel of the lattice, background included, which is
//! what a registration of a whole sample is judged on. `compare.py` in
//! `reference-clearmap/` computes the same two numbers the same way.

use ndarray::Array3;

/// Pearson correlation and mutual information of two images.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Similarity {
    pub correlation: f64,
    /// In nats, from a 64 × 64 joint histogram with equal-width bins over
    /// each image's own range.
    pub mutual_information: f64,
}

/// Bins per image of [`Similarity::mutual_information`].
pub const SIMILARITY_BINS: usize = 64;

pub fn similarity(a: &Array3<f64>, b: &Array3<f64>) -> Similarity {
    assert_eq!(
        a.shape(),
        b.shape(),
        "similarity of images of different shapes"
    );
    let n = a.len() as f64;
    let (mut sa, mut sb, mut saa, mut sbb, mut sab) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b.iter()) {
        sa += x;
        sb += y;
        saa += x * x;
        sbb += y * y;
        sab += x * y;
    }
    let cov = sab - sa * sb / n;
    let va = saa - sa * sa / n;
    let vb = sbb - sb * sb / n;
    let correlation = if va > 0.0 && vb > 0.0 {
        cov / (va * vb).sqrt()
    } else {
        0.0
    };
    let range = |image: &Array3<f64>| {
        image
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
                (lo.min(*v), hi.max(*v))
            })
    };
    let (alo, ahi) = range(a);
    let (blo, bhi) = range(b);
    let bin = |v: f64, lo: f64, hi: f64| {
        if hi > lo {
            (((v - lo) / (hi - lo) * SIMILARITY_BINS as f64) as usize).min(SIMILARITY_BINS - 1)
        } else {
            0
        }
    };
    let mut joint = vec![0.0f64; SIMILARITY_BINS * SIMILARITY_BINS];
    for (x, y) in a.iter().zip(b.iter()) {
        joint[bin(*x, alo, ahi) * SIMILARITY_BINS + bin(*y, blo, bhi)] += 1.0;
    }
    let mut pa = vec![0.0; SIMILARITY_BINS];
    let mut pb = vec![0.0; SIMILARITY_BINS];
    for i in 0..SIMILARITY_BINS {
        for j in 0..SIMILARITY_BINS {
            let p = joint[i * SIMILARITY_BINS + j] / n;
            pa[i] += p;
            pb[j] += p;
        }
    }
    let mut mutual_information = 0.0;
    for i in 0..SIMILARITY_BINS {
        for j in 0..SIMILARITY_BINS {
            let p = joint[i * SIMILARITY_BINS + j] / n;
            if p > 0.0 {
                mutual_information += p * (p / (pa[i] * pb[j])).ln();
            }
        }
    }
    Similarity {
        correlation,
        mutual_information,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_image_against_itself_correlates_fully_and_shares_its_entropy() {
        let image = Array3::from_shape_fn((6, 7, 8), |(i, j, k)| (i * 7 + j * 3 + k) as f64);
        let s = similarity(&image, &image);
        assert!((s.correlation - 1.0).abs() < 1e-12);
        let flat = Array3::from_elem((6, 7, 8), 2.0);
        let t = similarity(&image, &flat);
        assert_eq!(t.correlation, 0.0);
        assert!(t.mutual_information.abs() < 1e-12);
        assert!(s.mutual_information > 1.0);
    }
}
