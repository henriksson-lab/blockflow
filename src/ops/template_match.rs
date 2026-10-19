// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **Normalised cross-correlation with radial templates, as a composition.**
//
// No new arithmetic: for a kernel `t` (zero mean, unit norm over its support
// `M` of `n` taps) the NCC at a voxel is
//
//     sum_M t u  /  sqrt( sum_M u^2 - (sum_M u)^2 / n )
//
// and each of the three sums is a correlation — with `t`, with the indicator of
// `M`, and with the indicator over the squared input. One
// `transform_bank::TransformBankOp` computes them: the image and its square are
// transformed once per tile, each radius is a group of three terms combined to
// its NCC, and the radii are folded by a maximum.
// The denominator is floored at `n * variance_floor` so flat regions (outside a
// field of view, saturated background) do not divide noise by noise.
//
// The templates here are **radial**: an intensity profile over `rho = r / R`,
// scaled to each radius, with an optional band `[rho0, rho1]` so that only a
// shell of the profile is matched (an object's interior may vary while its
// wall does not). [`radial_profile`] samples one from an image around a known
// object, and [`RadialTemplate::average`] builds one from many.

use ndarray::ArrayView3;

use crate::error::{Error, Result};
use std::sync::Arc;

use crate::op::Chain;
use crate::ops::convolve::{Kernel, Sense};
use crate::ops::edges::trilinear_at;
use crate::ops::element::StructuringElement;
use crate::ops::transform_bank::{BankGroup, InputMap, TransformBankOp};
use crate::ops::Boundary;

/// An intensity profile over `rho = r / R`, sampled every `step`.
#[derive(Debug, Clone, PartialEq)]
pub struct RadialTemplate {
    pub step: f64,
    pub values: Vec<f64>,
}

impl RadialTemplate {
    pub fn new(step: f64, values: Vec<f64>) -> Result<Self> {
        if !(step.is_finite() && step > 0.0)
            || values.len() < 2
            || values.iter().any(|v| !v.is_finite())
        {
            return Err(Error::invalid(
                "a radial template needs a positive step and at least two finite values",
            ));
        }
        Ok(Self { step, values })
    }

    /// Largest `rho` the profile covers.
    pub fn extent(&self) -> f64 {
        (self.values.len() - 1) as f64 * self.step
    }

    /// Linear interpolation, clamped to the ends.
    pub fn at(&self, rho: f64) -> f64 {
        let x = (rho / self.step).max(0.0);
        let i = (x.floor() as usize).min(self.values.len() - 2);
        let t = (x - i as f64).clamp(0.0, 1.0);
        self.values[i] * (1.0 - t) + self.values[i + 1] * t
    }

    /// The mean of `profiles` after each is z-scored over `band`, so differently
    /// exposed objects weigh equally. All must share `step` and length.
    pub fn average(profiles: &[RadialTemplate], band: [f64; 2]) -> Result<Self> {
        let first = profiles
            .first()
            .ok_or_else(|| Error::invalid("averaging templates needs at least one"))?;
        if profiles
            .iter()
            .any(|p| p.step != first.step || p.values.len() != first.values.len())
        {
            return Err(Error::invalid(
                "templates to average must share step and length",
            ));
        }
        let n = first.values.len();
        let lo = ((band[0] / first.step) as usize).min(n - 1);
        let hi = ((band[1] / first.step) as usize).min(n - 1);
        let mut out = vec![0.0; n];
        for p in profiles {
            let segment = &p.values[lo..=hi];
            let mean = segment.iter().sum::<f64>() / segment.len() as f64;
            let sd = (segment.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / segment.len() as f64)
                .sqrt()
                .max(1e-12);
            for (o, v) in out.iter_mut().zip(&p.values) {
                *o += (v - mean) / sd / profiles.len() as f64;
            }
        }
        Self::new(first.step, out)
    }
}

/// The mean of `values` on the sphere of each radius `rho * radius` around
/// `centre` (array index coordinates, clamped trilinear sampling), for
/// `rho = 0, step, ..` up to `extent`, averaged over `directions`.
pub fn radial_profile(
    values: ArrayView3<'_, f64>,
    centre: [f64; 3],
    radius: f64,
    directions: &[[f64; 3]],
    step: f64,
    extent: f64,
) -> Result<RadialTemplate> {
    if directions.is_empty() || !(radius > 0.0) {
        return Err(Error::invalid(
            "a radial profile needs directions and a positive radius",
        ));
    }
    let shape = values.shape();
    let count = (extent / step).round() as usize + 1;
    let profile = (0..count)
        .map(|k| {
            let r = k as f64 * step * radius;
            directions
                .iter()
                .map(|u| {
                    let p: [f64; 3] = std::array::from_fn(|a| {
                        (centre[a] + r * u[a]).clamp(0.0, (shape[a] - 1) as f64)
                    });
                    trilinear_at(values, p).unwrap_or(0.0)
                })
                .sum::<f64>()
                / directions.len() as f64
        })
        .collect();
    RadialTemplate::new(step, profile)
}

/// The two kernels of one radius: the zero-mean, unit-norm template over the
/// band, and the indicator of the band.
#[derive(Debug, Clone)]
pub struct ShellKernels {
    pub template: Kernel,
    pub mask: Kernel,
    pub taps: usize,
}

/// Kernels of `template` scaled to `radius` (grid voxels), over the shell
/// `band[0] <= rho <= band[1]`.
pub fn shell_kernels(
    template: &RadialTemplate,
    radius: f64,
    band: [f64; 2],
) -> Result<ShellKernels> {
    if !(radius > 0.0) || !(band[0] >= 0.0 && band[1] > band[0]) {
        return Err(Error::invalid(format!(
            "a shell kernel needs a positive radius and a band lo < hi, got {radius} and {band:?}"
        )));
    }
    let extent = (radius * band[1]).ceil() as isize;
    let mut taps: Vec<([isize; 3], f64)> = Vec::new();
    for z in -extent..=extent {
        for y in -extent..=extent {
            for x in -extent..=extent {
                let rho = ((z * z + y * y + x * x) as f64).sqrt() / radius;
                if rho >= band[0] && rho <= band[1] {
                    taps.push(([z, y, x], template.at(rho)));
                }
            }
        }
    }
    if taps.len() < 2 {
        return Err(Error::invalid(format!(
            "the shell at radius {radius} over {band:?} has fewer than two voxels"
        )));
    }
    // The element orders its offsets lexicographically; so do these loops.
    let n = taps.len();
    let mean = taps.iter().map(|t| t.1).sum::<f64>() / n as f64;
    let norm = taps
        .iter()
        .map(|t| (t.1 - mean).powi(2))
        .sum::<f64>()
        .sqrt();
    if norm == 0.0 {
        return Err(Error::invalid(
            "a constant template has no correlation to measure",
        ));
    }
    let offsets: Vec<[isize; 3]> = taps.iter().map(|t| t.0).collect();
    let template_kernel = Kernel::new(
        StructuringElement::from_offsets(offsets.clone())?,
        taps.iter().map(|t| (t.1 - mean) / norm).collect(),
    )?;
    let mask = Kernel::new(StructuringElement::from_offsets(offsets)?, vec![1.0; n])?;
    Ok(ShellKernels {
        template: template_kernel,
        mask,
        taps: n,
    })
}

/// The NCC of one kernel pair as a group of a transform bank whose inputs are
/// the image and its square. `variance_floor` is per voxel, in the input's
/// units squared.
fn ncc_group(kernels: &ShellKernels, variance_floor: f64) -> BankGroup {
    let n = kernels.taps as f64;
    let floor = variance_floor * n;
    BankGroup {
        terms: vec![
            (0, kernels.template.clone()),
            (0, kernels.mask.clone()),
            (1, kernels.mask.clone()),
        ],
        combine: Arc::new(move |x| x[0] / (x[2] - x[1] * x[1] / n).max(floor).sqrt()),
    }
}

/// The NCC of one kernel pair as a chain over the phase input; output `f64`.
/// `variance_floor` is per voxel, in the input's units squared.
pub fn ncc_chain(kernels: &ShellKernels, tile: [usize; 3], variance_floor: f64) -> Result<Chain> {
    ncc_bank(vec![kernels.clone()], tile, variance_floor)
}

/// The maximum over `radii` (grid voxels) of [`ncc_chain`] for `template`
/// scaled to each.
pub fn multi_radius_ncc_chain(
    template: &RadialTemplate,
    radii: &[f64],
    band: [f64; 2],
    tile: [usize; 3],
    variance_floor: f64,
) -> Result<Chain> {
    if radii.is_empty() {
        return Err(Error::invalid("a template scan needs at least one radius"));
    }
    let kernels = radii
        .iter()
        .map(|&r| shell_kernels(template, r, band))
        .collect::<Result<Vec<_>>>()?;
    ncc_bank(kernels, tile, variance_floor)
}

fn ncc_bank(kernels: Vec<ShellKernels>, tile: [usize; 3], variance_floor: f64) -> Result<Chain> {
    if !(variance_floor.is_finite() && variance_floor >= 0.0) {
        return Err(Error::invalid(
            "the NCC variance floor must be finite and non-negative",
        ));
    }
    let identity: InputMap = Arc::new(|v| v);
    let square: InputMap = Arc::new(|v| v * v);
    Ok(Chain::op(TransformBankOp::new(
        "ncc",
        vec![identity, square],
        kernels
            .iter()
            .map(|k| ncc_group(k, variance_floor))
            .collect(),
        Arc::new(f64::max),
        Sense::Correlate,
        Boundary::Clamp,
        tile,
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assemble::PlanBuilder;
    use crate::dtype::Dtype;
    use crate::env::ArrayEnvironment;
    use crate::geometry::BlockGrid;
    use crate::sphere::fibonacci_sphere;
    use crate::strategy::{execute_phases, Hints};
    use crate::voxels::Voxels;
    use ndarray::Array3;

    fn shell_image(volume: [usize; 3], centre: [f64; 3], radius: f64) -> Array3<f64> {
        Array3::from_shape_fn((volume[0], volume[1], volume[2]), |(z, y, x)| {
            let r = ((z as f64 - centre[0]).powi(2)
                + (y as f64 - centre[1]).powi(2)
                + (x as f64 - centre[2]).powi(2))
            .sqrt()
                / radius;
            let wall = if (0.85..1.0).contains(&r) {
                3.0
            } else if (0.7..0.85).contains(&r) {
                -1.0
            } else {
                0.0
            };
            wall + 0.3 * ((z * 5 + y * 3 + x) % 7) as f64
        })
    }

    /// Resident NCC at one voxel, straight from the definition.
    fn ncc_at(image: &Array3<f64>, k: &ShellKernels, at: [usize; 3], floor: f64) -> f64 {
        let shape = image.shape();
        let (mut num, mut s1, mut s2) = (0.0, 0.0, 0.0);
        for (o, w) in k
            .template
            .element()
            .offsets()
            .iter()
            .zip(k.template.weights())
        {
            let p: [usize; 3] = std::array::from_fn(|a| {
                (at[a] as isize + o[a]).clamp(0, shape[a] as isize - 1) as usize
            });
            let v = image[p];
            num += w * v;
            s1 += v;
            s2 += v * v;
        }
        let n = k.taps as f64;
        num / (s2 - s1 * s1 / n).max(floor * n).sqrt()
    }

    #[test]
    fn planned_ncc_matches_definition_and_peaks_at_the_shell() {
        let volume = [24, 30, 28];
        let centre = [11.0, 14.0, 15.0];
        let image = shell_image(volume, centre, 7.0);
        let template =
            radial_profile(image.view(), centre, 7.0, &fibonacci_sphere(200), 0.05, 1.6).unwrap();
        let radii = [5.0, 7.0, 9.0];
        let band = [0.6, 1.4];
        let floor = 0.01;
        let mut answers = Vec::new();
        for block in [[24, 30, 28], [8, 10, 7]] {
            let mut builder =
                PlanBuilder::new(volume, Dtype::F64, BlockGrid::new(volume, block).unwrap());
            builder
                .pixels(multi_radius_ncc_chain(&template, &radii, band, [8, 10, 7], floor).unwrap())
                .unwrap();
            let assembly = builder.finish().unwrap();
            let env = ArrayEnvironment::for_decomposition(
                Voxels::from(image.clone()),
                &assembly.decomposition,
                block,
            )
            .unwrap();
            execute_phases(
                "ncc",
                &assembly.workflow,
                &assembly.decomposition,
                &Hints::default(),
                &env,
                &[],
                &assembly.work(),
            )
            .unwrap();
            answers.push(env.output().view::<f64>().unwrap().to_owned());
        }
        let kernels: Vec<ShellKernels> = radii
            .iter()
            .map(|&r| shell_kernels(&template, r, band).unwrap())
            .collect();
        for at in [[11, 14, 15], [3, 4, 5], [20, 25, 2], [12, 13, 16]] {
            let expected = kernels
                .iter()
                .map(|k| ncc_at(&image, k, at, floor))
                .fold(f64::NEG_INFINITY, f64::max);
            for got in &answers {
                assert!(
                    (got[at] - expected).abs() < 1e-6,
                    "{at:?}: {} vs {expected}",
                    got[at]
                );
            }
        }
        let (peak, _) = answers[0]
            .indexed_iter()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        assert_eq!([peak.0, peak.1, peak.2], [11, 14, 15]);
    }

    #[test]
    fn averaged_template_is_scale_free() {
        let a = RadialTemplate::new(0.5, vec![1.0, 2.0, 3.0]).unwrap();
        let b = RadialTemplate::new(0.5, vec![10.0, 30.0, 50.0]).unwrap();
        let m = RadialTemplate::average(&[a.clone(), b], [0.0, 1.0]).unwrap();
        let z = RadialTemplate::average(&[a], [0.0, 1.0]).unwrap();
        for (x, y) in m.values.iter().zip(&z.values) {
            assert!((x - y).abs() < 1e-12);
        }
    }
}
