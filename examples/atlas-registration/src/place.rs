// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Initial placement: where the fixed image lies in the moving one, found by
//! a coarse-to-fine search over translations, before the fit starts.
//!
//! A partial mosaic (some tiles of a sample) whose autofluorescence carries
//! no stage position cannot be started from its frames or from the two
//! centres; this finds the translation instead. Both images are on lattices
//! of one voxel size (the registration's), each over its own box.
//!
//! 1. **Pyramid.** 2x block means of both lattices, down to the coarsest
//!    level at which the fixed image still spans [`MIN_COARSE_EXTENT`] voxels
//!    on every axis (at most [`MAX_COARSE_FACTOR`] times the lattice), and
//!    coarser still if the search's transforms would not fit their memory
//!    bound. Intensities are `ln(1 + v)` by default, so the sparse bright
//!    structures of a signal channel do not decide the match alone.
//! 2. **Exhaustive search at the coarsest level**: the masked normalised
//!    cross-correlation of the two images at *every* integer translation
//!    with at least `min_overlap` of the smaller image overlapping, from six
//!    correlations through the correlation theorem (`ops::fft`). The NCC is
//!    taken over the overlap only, both means and variances included, so a
//!    partial overlap is scored on what overlaps.
//! 3. **Candidates.** The local maxima (within two voxels), best first, up to
//!    `candidates` of them.
//! 4. **Refinement.** At each finer level every surviving candidate climbs
//!    the NCC by integer steps from where the level above put it; candidates
//!    more than `prune` below the best are dropped and ones that meet are
//!    merged. At the finest level a parabola through the peak and its two
//!    neighbours, per axis, gives the sub-voxel position.
//! 5. **Verdict.** The best candidate is the placement. It is refused when its
//!    NCC is below `min_peak`, or when a distinct runner-up comes within
//!    `min_margin` of it: then the images do not say where the fixed one is,
//!    and an explicit `--initial-translation` is needed.
//!
//! Translation only: rotations and scales are left to the fit that follows,
//! which is why the NCC of even a right placement is well below 1 when the
//! two frames disagree by a few per cent in scale.

use std::collections::HashMap;

use blockflow::error::{Error, Result};
use blockflow::ops::fft::{next_smooth_length, RealTransform3, Spectrum3};
use ndarray::{s, Array3, ArrayView3, Axis};
use rayon::prelude::*;
use serde_json::{json, Value};

use crate::rescale::Frame;

/// The fixed image spans at least this many voxels on every axis at the
/// coarsest level the automatic choice picks.
pub const MIN_COARSE_EXTENT: usize = 16;
/// The coarsest level the automatic choice goes to, as a multiple of the
/// lattice spacing.
pub const MAX_COARSE_FACTOR: usize = 8;
/// Upper bound on the exhaustive search's working set (six half-spectra and
/// six correlation volumes), bytes. The coarsest level is made coarser until
/// the search fits.
pub const MAX_SEARCH_BYTES: u64 = 1 << 30;

/// How the search runs and when it refuses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchParams {
    /// Pyramid levels, the lattice included; `None`: the automatic choice
    /// above.
    pub levels: Option<usize>,
    /// Smallest overlap a translation may have, as a fraction of the smaller
    /// image's voxels.
    pub min_overlap: f64,
    /// Coarse candidates carried into refinement.
    pub candidates: usize,
    /// Refuse a best NCC below this.
    pub min_peak: f64,
    /// Refuse when a distinct runner-up is within this of the best NCC.
    pub min_margin: f64,
    /// Drop a candidate more than this below the best at a level.
    pub prune: f64,
    /// Match `ln(1 + v)` rather than `v` (non-negative images only).
    pub log_intensity: bool,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            levels: None,
            min_overlap: 0.5,
            candidates: 8,
            min_peak: 0.2,
            min_margin: 0.05,
            prune: 0.15,
            log_intensity: true,
        }
    }
}

/// One translation and how well it matches.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    /// `[z, y, x]`, physical: the fixed image's point `p` lies at the moving
    /// image's point `p + translation`.
    pub translation: [f64; 3],
    pub ncc: f64,
    /// Overlapping voxels as a fraction of the fixed image's.
    pub overlap: f64,
}

impl Candidate {
    fn to_json(self) -> Value {
        json!({ "translation": self.translation, "ncc": self.ncc, "overlap": self.overlap })
    }
}

/// The alternative a placement was decided against.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Runner {
    pub candidate: Candidate,
    /// The level's spacing.
    pub spacing: [f64; 3],
    /// The best NCC minus this one's, at that level.
    pub margin: f64,
}

/// One pyramid level of the search.
#[derive(Debug, Clone, PartialEq)]
pub struct LevelReport {
    pub spacing: [f64; 3],
    pub fixed_shape: [usize; 3],
    pub moving_shape: [usize; 3],
    /// Translations whose NCC was evaluated at this level.
    pub evaluated: usize,
    /// The candidates after this level, best first.
    pub candidates: Vec<Candidate>,
}

/// The search's answer.
#[derive(Debug, Clone, PartialEq)]
pub struct Placement {
    pub best: Candidate,
    /// The strongest distinct alternative, at the finest level that still
    /// compared one with the best (later levels refine the survivors only).
    pub runner_up: Option<Runner>,
    /// Coarsest first.
    pub levels: Vec<LevelReport>,
    pub seconds: f64,
}

impl Placement {
    pub fn translation(&self) -> [f64; 3] {
        self.best.translation
    }

    /// Best NCC minus the runner-up's, where they were compared (the best's
    /// own NCC without one).
    pub fn margin(&self) -> f64 {
        self.runner_up.map_or(self.best.ncc, |r| r.margin)
    }

    pub fn to_json(&self, params: &SearchParams) -> Value {
        json!({
            "translation": self.best.translation,
            "ncc": self.best.ncc,
            "overlap": self.best.overlap,
            "runner_up": self.runner_up.map(|r| json!({
                "translation": r.candidate.translation,
                "ncc": r.candidate.ncc,
                "overlap": r.candidate.overlap,
                "spacing": r.spacing,
                "margin": r.margin,
            })),
            "margin": self.margin(),
            "seconds": self.seconds,
            "params": {
                "levels": params.levels,
                "min_overlap": params.min_overlap,
                "candidates": params.candidates,
                "min_peak": params.min_peak,
                "min_margin": params.min_margin,
                "prune": params.prune,
                "log_intensity": params.log_intensity,
            },
            "levels": self.levels.iter().map(|level| json!({
                "spacing": level.spacing,
                "fixed_shape": level.fixed_shape,
                "moving_shape": level.moving_shape,
                "evaluated": level.evaluated,
                "candidates": level.candidates.iter().map(|c| c.to_json()).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
        })
    }
}

/// One image at one pyramid level.
struct Level {
    image: Array3<f64>,
    frame: Frame,
}

impl Level {
    fn shape(&self) -> [usize; 3] {
        let d = self.image.dim();
        [d.0, d.1, d.2]
    }

    /// 2x block means; a trailing odd voxel is dropped. The coarse voxel's
    /// centre is the middle of the two it averages.
    fn halve(&self) -> Result<Self> {
        let [nz, ny, nx] = self.shape();
        let (hz, hy, hx) = ((nz / 2).max(1), (ny / 2).max(1), (nx / 2).max(1));
        let factor = [nz >= 2, ny >= 2, nx >= 2];
        let mut data = vec![0.0f64; hz * hy * hx];
        data.par_chunks_mut(hy * hx)
            .enumerate()
            .for_each(|(z, plane)| {
                let zs: &[usize] = if factor[0] { &[2 * z, 2 * z + 1] } else { &[z] };
                for y in 0..hy {
                    let ys: &[usize] = if factor[1] { &[2 * y, 2 * y + 1] } else { &[y] };
                    for x in 0..hx {
                        let xs: &[usize] = if factor[2] { &[2 * x, 2 * x + 1] } else { &[x] };
                        let mut sum = 0.0;
                        for &a in zs {
                            for &b in ys {
                                for &c in xs {
                                    sum += self.image[[a, b, c]];
                                }
                            }
                        }
                        plane[y * hx + x] = sum / (zs.len() * ys.len() * xs.len()) as f64;
                    }
                }
            });
        let image = Array3::from_shape_vec((hz, hy, hx), data).map_err(Error::backend)?;
        let scale: [f64; 3] =
            std::array::from_fn(|a| self.frame.scale[a] * if factor[a] { 2.0 } else { 1.0 });
        let translation = std::array::from_fn(|a| {
            self.frame.translation[a] + (scale[a] - self.frame.scale[a]) / 2.0
        });
        Ok(Self {
            image,
            frame: Frame::new(scale, translation)?,
        })
    }
}

fn volume(shape: [usize; 3]) -> usize {
    shape.iter().product()
}

/// The exhaustive search's working set at these shapes, bytes.
fn search_bytes(fixed: [usize; 3], moving: [usize; 3]) -> u64 {
    let padded: [usize; 3] = std::array::from_fn(|a| next_smooth_length(fixed[a] + moving[a] - 1));
    let voxels = volume(padded) as u64;
    let spectrum = (padded[0] * padded[1] * (padded[2] / 2 + 1)) as u64;
    // six half-spectra of complex f64, six correlations of f64
    6 * spectrum * 16 + 6 * voxels * 8
}

/// Find the translation that places `fixed` in `moving`. Both are lattices of
/// one voxel size on their own frames.
pub fn search_translation(
    fixed: ArrayView3<f64>,
    fixed_frame: &Frame,
    moving: ArrayView3<f64>,
    moving_frame: &Frame,
    params: &SearchParams,
) -> Result<Placement> {
    let started = std::time::Instant::now();
    if (0..3)
        .any(|a| (fixed_frame.scale[a] - moving_frame.scale[a]).abs() > 1e-9 * fixed_frame.scale[a])
    {
        return Err(Error::invalid(format!(
            "placement search: both lattices need one voxel size, got {:?} and {:?}",
            fixed_frame.scale, moving_frame.scale
        )));
    }
    if !(params.min_overlap > 0.0 && params.min_overlap <= 1.0) || params.candidates == 0 {
        return Err(Error::invalid(
            "placement search: the minimum overlap must be in (0, 1] and at least one candidate kept",
        ));
    }
    // The two pyramids, in float64: at most 8/7 of each lattice.
    let pyramid_voxels = [fixed.shape(), moving.shape()]
        .iter()
        .map(|shape| shape.iter().map(|&n| n as u64).product::<u64>())
        .sum::<u64>();
    let _pyramids = crate::memory::lease(
        8 * pyramid_voxels * 8 / 7,
        "the placement search's pyramids",
    )?;
    let prepare = |image: ArrayView3<f64>| -> Array3<f64> {
        let log = params.log_intensity && image.iter().all(|v| *v >= 0.0);
        image.mapv(|v| {
            if !v.is_finite() {
                0.0
            } else if log {
                v.ln_1p()
            } else {
                v
            }
        })
    };
    let mut fixed_levels = vec![Level {
        image: prepare(fixed),
        frame: *fixed_frame,
    }];
    let mut moving_levels = vec![Level {
        image: prepare(moving),
        frame: *moving_frame,
    }];
    // Pyramid depth: as asked, else the coarsest level that keeps the fixed
    // image MIN_COARSE_EXTENT voxels wide; then coarser while the search
    // does not fit its bound.
    let wanted = match params.levels {
        Some(0) => return Err(Error::invalid("placement search: at least one level")),
        Some(levels) => levels,
        None => {
            let narrowest = fixed_levels[0].shape().into_iter().min().unwrap_or(1);
            let mut levels = 1;
            while (1 << levels) <= MAX_COARSE_FACTOR && narrowest >> levels >= MIN_COARSE_EXTENT {
                levels += 1;
            }
            levels
        }
    };
    // Within what the budget leaves beside the volumes already held (the
    // caches give theirs back when the search is leased).
    let budget = blockflow::budget::process_budget();
    let cap = MAX_SEARCH_BYTES
        .min(blockflow::budget::process_cap().usable_bytes() / 4)
        .min(budget.total().saturating_sub(budget.reserved_in_use()));
    loop {
        let (f, m) = (fixed_levels.last().unwrap(), moving_levels.last().unwrap());
        let fits = search_bytes(f.shape(), m.shape()) <= cap;
        if fixed_levels.len() >= wanted && fits {
            break;
        }
        if f.shape().iter().chain(m.shape().iter()).all(|&n| n < 2) {
            return Err(Error::invalid(
                "placement search: the images are too large to search within its memory bound",
            ));
        }
        let (f, m) = (f.halve()?, m.halve()?);
        fixed_levels.push(f);
        moving_levels.push(m);
    }
    let floor = |f: &Level, m: &Level| {
        (params.min_overlap * volume(f.shape()).min(volume(m.shape())) as f64).max(1.0)
    };

    // The exhaustive search at the coarsest level.
    let coarsest = fixed_levels.len() - 1;
    let (f, m) = (&fixed_levels[coarsest], &moving_levels[coarsest]);
    let lease = crate::memory::lease(
        search_bytes(f.shape(), m.shape()),
        "the placement search's correlations",
    )?;
    let field = ncc_field(&f.image, &m.image, floor(f, m))?;
    drop(lease);
    let peaks = local_maxima(&field.ncc, params.candidates);
    let offset = [
        f.shape()[0] as isize - 1,
        f.shape()[1] as isize - 1,
        f.shape()[2] as isize - 1,
    ];
    let mut candidates: Vec<([isize; 3], f64, f64)> = peaks
        .into_iter()
        .map(|(index, ncc)| {
            let shift = std::array::from_fn(|a| index[a] as isize - offset[a]);
            (shift, ncc, field.overlap[index] / volume(f.shape()) as f64)
        })
        .collect();
    if candidates.is_empty() {
        return Err(Error::invalid(format!(
            "placement search: no translation overlaps {:.0}% of the smaller image",
            100.0 * params.min_overlap
        )));
    }
    let to_candidate = |f: &Level, m: &Level, shift: [f64; 3], ncc: f64, overlap: f64| Candidate {
        translation: std::array::from_fn(|a| {
            m.frame.translation[a] + shift[a] * m.frame.scale[a] - f.frame.translation[a]
        }),
        ncc,
        overlap,
    };
    let mut reports = vec![LevelReport {
        spacing: f.frame.scale,
        fixed_shape: f.shape(),
        moving_shape: m.shape(),
        evaluated: field.ncc.len(),
        candidates: candidates
            .iter()
            .map(|(s, ncc, o)| to_candidate(f, m, s.map(|v| v as f64), *ncc, *o))
            .collect(),
    }];

    // Refinement, level by level, down to the lattice.
    let mut best: Vec<Candidate> = reports[0].candidates.clone();
    let mut runner_up = (best.len() > 1).then(|| Runner {
        candidate: best[1],
        spacing: f.frame.scale,
        margin: best[0].ncc - best[1].ncc,
    });
    // A single level is searched and refined at the lattice itself.
    let refined: Vec<usize> = if coarsest == 0 {
        vec![0]
    } else {
        (0..coarsest).rev().collect()
    };
    for level in refined {
        let (f, m) = (&fixed_levels[level], &moving_levels[level]);
        let floor = floor(f, m);
        let mut memo: HashMap<[isize; 3], Option<(f64, f64)>> = HashMap::new();
        let mut climbed: Vec<([isize; 3], f64, f64)> = Vec::new();
        for candidate in &best {
            let start: [isize; 3] = std::array::from_fn(|a| {
                ((candidate.translation[a] + f.frame.translation[a] - m.frame.translation[a])
                    / m.frame.scale[a])
                    .round() as isize
            });
            if let Some(found) = climb(&f.image, &m.image, start, floor, &mut memo) {
                climbed.push(found);
            }
        }
        climbed.sort_by(|a, b| b.1.total_cmp(&a.1));
        // merge candidates that met, then drop the clearly worse
        let mut distinct: Vec<([isize; 3], f64, f64)> = Vec::new();
        for c in climbed {
            let near = distinct
                .iter()
                .any(|k| (0..3).all(|a| (k.0[a] - c.0[a]).abs() <= 1));
            if !near {
                distinct.push(c);
            }
        }
        if distinct.len() > 1 {
            let (shift, ncc, overlap) = distinct[1];
            runner_up = Some(Runner {
                candidate: to_candidate(
                    f,
                    m,
                    shift.map(|v| v as f64),
                    ncc,
                    overlap / volume(f.shape()) as f64,
                ),
                spacing: f.frame.scale,
                margin: distinct[0].1 - ncc,
            });
        }
        let top = distinct.first().map_or(f64::NEG_INFINITY, |c| c.1);
        candidates = distinct
            .into_iter()
            .filter(|c| c.1 >= top - params.prune)
            .collect();
        if candidates.is_empty() {
            return Err(Error::invalid(format!(
                "placement search: no candidate keeps {:.0}% overlap at {:?}",
                100.0 * params.min_overlap,
                f.frame.scale
            )));
        }
        best = candidates
            .iter()
            .map(|(shift, ncc, overlap)| {
                let refined = if level == 0 {
                    subvoxel(*shift, &memo)
                } else {
                    shift.map(|v| v as f64)
                };
                to_candidate(f, m, refined, *ncc, overlap / volume(f.shape()) as f64)
            })
            .collect();
        reports.push(LevelReport {
            spacing: f.frame.scale,
            fixed_shape: f.shape(),
            moving_shape: m.shape(),
            evaluated: memo.len(),
            candidates: best.clone(),
        });
    }
    let placement = Placement {
        best: best[0],
        runner_up,
        levels: reports,
        seconds: started.elapsed().as_secs_f64(),
    };
    let listed = || {
        best.iter()
            .take(1)
            .chain(placement.runner_up.as_ref().map(|r| &r.candidate))
            .map(|c| {
                format!(
                    "({:.0}, {:.0}, {:.0}) NCC {:.3}",
                    c.translation[0], c.translation[1], c.translation[2], c.ncc
                )
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    if placement.best.ncc < params.min_peak {
        return Err(Error::invalid(format!(
            "placement search: the best translation matches with NCC {:.3}, below {:.3}; the \
             images may not overlap, or be too different to place by correlation. Candidates: \
             {}. Give the translation with --initial-translation",
            placement.best.ncc,
            params.min_peak,
            listed()
        )));
    }
    if placement.runner_up.is_some() && placement.margin() < params.min_margin {
        return Err(Error::invalid(format!(
            "placement search: ambiguous, two translations match within {:.3} NCC (at {:?}): \
             {}. Give the translation with --initial-translation",
            params.min_margin,
            placement.runner_up.map(|r| r.spacing).unwrap_or_default(),
            listed()
        )));
    }
    Ok(placement)
}

struct Field {
    /// NCC per translation, `-inf` where the overlap is too small or flat;
    /// index `i` is the shift `i - (fixed shape - 1)`.
    ncc: Array3<f64>,
    overlap: Array3<f64>,
}

/// Masked NCC of `fixed` over `moving` at every integer shift `s` (the fixed
/// voxel `x` against the moving voxel `x + s`), from the six correlations
/// `sum_x a(x) b(x + s)` of {box, f, f^2} with {box, m, m^2} it needs.
fn ncc_field(fixed: &Array3<f64>, moving: &Array3<f64>, floor: f64) -> Result<Field> {
    let fd = fixed.dim();
    let md = moving.dim();
    let fs = [fd.0, fd.1, fd.2];
    let ms = [md.0, md.1, md.2];
    let padded: [usize; 3] = std::array::from_fn(|a| next_smooth_length(fs[a] + ms[a] - 1));
    let template = RealTransform3::new(padded)?;
    let forward = |image: Array3<f64>| -> Result<Spectrum3> {
        let mut transform = template.clone();
        let mut out = transform.spectrum();
        transform.forward_zero_padded(image.view(), &mut out)?;
        Ok(out)
    };
    let fixed_inputs = [
        Array3::<f64>::ones(fd),
        fixed.clone(),
        fixed.mapv(|v| v * v),
    ];
    let moving_inputs = [
        Array3::<f64>::ones(md),
        moving.clone(),
        moving.mapv(|v| v * v),
    ];
    let spectra: Vec<Spectrum3> = fixed_inputs
        .into_par_iter()
        .chain(moving_inputs.into_par_iter())
        .map(forward)
        .collect::<Result<_>>()?;
    let (fixed_spectra, moving_spectra) = spectra.split_at(3);
    // (fixed term, moving term): n, Sf, Sff, Sm, Smm, Sfm
    let pairs = [(0, 0), (1, 0), (2, 0), (0, 1), (0, 2), (1, 1)];
    let shape = (fs[0] + ms[0] - 1, fs[1] + ms[1] - 1, fs[2] + ms[2] - 1);
    let terms: Vec<Array3<f64>> = pairs
        .into_par_iter()
        .map(|(a, b)| -> Result<Array3<f64>> {
            let mut product = fixed_spectra[a].clone();
            product
                .iter_mut()
                .zip(moving_spectra[b].iter())
                .for_each(|(p, q)| *p = p.conj() * q);
            let mut transform = template.clone();
            let mut full = Array3::<f64>::zeros((padded[0], padded[1], padded[2]));
            transform.inverse(&mut product, &mut full)?;
            // shift s at index s + (fs - 1); the correlation holds it at s mod P
            Ok(Array3::from_shape_fn(shape, |(i, j, k)| {
                let index = [i, j, k];
                let at: [usize; 3] = std::array::from_fn(|axis| {
                    let s = index[axis] as isize - (fs[axis] as isize - 1);
                    s.rem_euclid(padded[axis] as isize) as usize
                });
                full[at]
            }))
        })
        .collect::<Result<_>>()?;
    let [n, sf, sff, sm, smm, sfm] = <[Array3<f64>; 6]>::try_from(terms)
        .map_err(|_| Error::invalid("placement search: six correlations"))?;
    let overlap = n.mapv(f64::round);
    let ncc = Array3::from_shape_fn(shape, |index| {
        let count = overlap[index];
        if count < floor {
            return f64::NEG_INFINITY;
        }
        masked_ncc(
            count, sf[index], sff[index], sm[index], smm[index], sfm[index],
        )
        .unwrap_or(f64::NEG_INFINITY)
    });
    Ok(Field { ncc, overlap })
}

/// NCC over an overlap of `n` voxels from its five sums; `None` if either side
/// is flat there.
fn masked_ncc(n: f64, sf: f64, sff: f64, sm: f64, smm: f64, sfm: f64) -> Option<f64> {
    let vf = sff - sf * sf / n;
    let vm = smm - sm * sm / n;
    // relative floors: a constant overlap leaves rounding noise only
    if !(vf > 1e-9 * sff.abs().max(1e-300)) || !(vm > 1e-9 * smm.abs().max(1e-300)) {
        return None;
    }
    Some((sfm - sf * sm / n) / (vf * vm).sqrt())
}

/// The `limit` best local maxima (over a 5^3 neighbourhood) of a field.
fn local_maxima(field: &Array3<f64>, limit: usize) -> Vec<([usize; 3], f64)> {
    let (nz, ny, nx) = field.dim();
    let mut peaks: Vec<([usize; 3], f64)> = (0..nz)
        .into_par_iter()
        .flat_map_iter(|z| {
            let mut found = Vec::new();
            for y in 0..ny {
                for x in 0..nx {
                    let value = field[[z, y, x]];
                    if !value.is_finite() {
                        continue;
                    }
                    let window = field.slice(s![
                        z.saturating_sub(2)..(z + 3).min(nz),
                        y.saturating_sub(2)..(y + 3).min(ny),
                        x.saturating_sub(2)..(x + 3).min(nx)
                    ]);
                    if window.iter().all(|&v| v <= value) {
                        found.push(([z, y, x], value));
                    }
                }
            }
            found
        })
        .collect();
    peaks.sort_by(|a, b| b.1.total_cmp(&a.1));
    // plateaus: keep one per neighbourhood
    let mut kept: Vec<([usize; 3], f64)> = Vec::new();
    for peak in peaks {
        if kept
            .iter()
            .any(|k| (0..3).all(|a| k.0[a].abs_diff(peak.0[a]) <= 2))
        {
            continue;
        }
        kept.push(peak);
        if kept.len() == limit {
            break;
        }
    }
    kept
}

/// Masked NCC and overlap at one integer shift, directly.
fn ncc_at(fixed: &Array3<f64>, moving: &Array3<f64>, shift: [isize; 3]) -> (Option<f64>, f64) {
    let fd = fixed.dim();
    let md = moving.dim();
    let fs = [fd.0 as isize, fd.1 as isize, fd.2 as isize];
    let ms = [md.0 as isize, md.1 as isize, md.2 as isize];
    let lo: [isize; 3] = std::array::from_fn(|a| 0.max(-shift[a]));
    let hi: [isize; 3] = std::array::from_fn(|a| fs[a].min(ms[a] - shift[a]));
    if (0..3).any(|a| hi[a] <= lo[a]) {
        return (None, 0.0);
    }
    let f = fixed.slice(s![lo[0]..hi[0], lo[1]..hi[1], lo[2]..hi[2]]);
    let m = moving.slice(s![
        lo[0] + shift[0]..hi[0] + shift[0],
        lo[1] + shift[1]..hi[1] + shift[1],
        lo[2] + shift[2]..hi[2] + shift[2]
    ]);
    let sums = (0..f.len_of(Axis(0)))
        .into_par_iter()
        .map(|z| {
            let (fp, mp) = (f.index_axis(Axis(0), z), m.index_axis(Axis(0), z));
            let mut t = [0.0f64; 5];
            for (fr, mr) in fp.outer_iter().zip(mp.outer_iter()) {
                for (&a, &b) in fr.iter().zip(mr.iter()) {
                    t[0] += a;
                    t[1] += a * a;
                    t[2] += b;
                    t[3] += b * b;
                    t[4] += a * b;
                }
            }
            t
        })
        .reduce(|| [0.0; 5], |a, b| std::array::from_fn(|i| a[i] + b[i]));
    let n = (0..3).map(|a| (hi[a] - lo[a]) as f64).product::<f64>();
    (
        masked_ncc(n, sums[0], sums[1], sums[2], sums[3], sums[4]),
        n,
    )
}

/// Hill-climb the NCC from `start` over the 26 neighbours until no neighbour
/// is better; `(shift, ncc, overlap voxels)`, or `None` if the start does not
/// overlap enough. Evaluations are kept in `memo` (`None`: too little overlap
/// or flat).
fn climb(
    fixed: &Array3<f64>,
    moving: &Array3<f64>,
    start: [isize; 3],
    floor: f64,
    memo: &mut HashMap<[isize; 3], Option<(f64, f64)>>,
) -> Option<([isize; 3], f64, f64)> {
    let eval = |shift: [isize; 3], memo: &mut HashMap<[isize; 3], Option<(f64, f64)>>| {
        *memo.entry(shift).or_insert_with(|| {
            let (ncc, n) = ncc_at(fixed, moving, shift);
            ncc.filter(|_| n >= floor).map(|v| (v, n))
        })
    };
    let mut at = start;
    let (mut value, mut overlap) = eval(at, memo)?;
    // a few hundred steps at most: the levels above put the start near the peak
    for _ in 0..256 {
        let mut next = None;
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dz == 0 && dy == 0 && dx == 0 {
                        continue;
                    }
                    let shift = [at[0] + dz, at[1] + dy, at[2] + dx];
                    if let Some((v, n)) = eval(shift, memo) {
                        if v > value && next.is_none_or(|(_, best, _)| v > best) {
                            next = Some((shift, v, n));
                        }
                    }
                }
            }
        }
        match next {
            Some((shift, v, n)) => {
                at = shift;
                value = v;
                overlap = n;
            }
            None => break,
        }
    }
    Some((at, value, overlap))
}

/// The peak at `shift` refined per axis by a parabola through it and its two
/// neighbours (all evaluated by the climb that stopped there).
fn subvoxel(shift: [isize; 3], memo: &HashMap<[isize; 3], Option<(f64, f64)>>) -> [f64; 3] {
    let value = |s: [isize; 3]| memo.get(&s).copied().flatten().map(|(v, _)| v);
    std::array::from_fn(|axis| {
        let mut below = shift;
        below[axis] -= 1;
        let mut above = shift;
        above[axis] += 1;
        match (value(below), value(shift), value(above)) {
            (Some(a), Some(b), Some(c)) => {
                let curvature = a - 2.0 * b + c;
                let delta = if curvature < 0.0 {
                    (0.5 * (a - c) / curvature).clamp(-0.5, 0.5)
                } else {
                    0.0
                };
                shift[axis] as f64 + delta
            }
            _ => shift[axis] as f64,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A field of blobs on a background, deterministic.
    fn scene(shape: [usize; 3], seed: u64) -> Array3<f64> {
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let blobs: Vec<([f64; 3], f64, f64)> = (0..60)
            .map(|_| {
                (
                    std::array::from_fn(|a| shape[a] as f64 * next()),
                    2.0 + 4.0 * next(),
                    100.0 + 900.0 * next(),
                )
            })
            .collect();
        Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(z, y, x)| {
            let p = [z as f64, y as f64, x as f64];
            20.0 + blobs
                .iter()
                .map(|(c, w, h)| {
                    let d: f64 = (0..3).map(|a| ((p[a] - c[a]) / w).powi(2)).sum();
                    h * (-0.5 * d).exp()
                })
                .sum::<f64>()
        })
    }

    #[test]
    fn the_fft_field_matches_the_direct_ncc() {
        let moving = scene([12, 14, 10], 3);
        let fixed = moving.slice(s![2..9, 3..11, 1..8]).to_owned();
        let field = ncc_field(&fixed, &moving, 1.0).unwrap();
        for shift in [[2, 3, 1], [0, 0, 0], [-3, 2, 4], [4, -5, 2]] {
            let (direct, n) = ncc_at(&fixed, &moving, shift);
            let fs = fixed.dim();
            let at = [
                (shift[0] + fs.0 as isize - 1) as usize,
                (shift[1] + fs.1 as isize - 1) as usize,
                (shift[2] + fs.2 as isize - 1) as usize,
            ];
            assert!((field.overlap[at] - n).abs() < 1e-6, "{shift:?}");
            let (a, b) = (field.ncc[at], direct.unwrap());
            assert!((a - b).abs() < 1e-6, "{shift:?}: {a} vs {b}");
        }
        assert!((field.ncc[[2 + 6, 3 + 7, 1 + 6]] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn a_crop_is_placed_where_it_was_cut() {
        let spacing = 2.0;
        let moving = scene([40, 72, 64], 11);
        let origin = [5usize, 37, 9];
        let fixed = moving
            .slice(s![
                origin[0]..origin[0] + 32,
                origin[1]..origin[1] + 30,
                origin[2]..origin[2] + 34
            ])
            .to_owned();
        let moving_frame = Frame::new([spacing; 3], [1.0, -4.0, 7.0]).unwrap();
        // the crop's frame knows nothing of where it came from
        let fixed_frame = Frame::new([spacing; 3], [100.0, 200.0, -300.0]).unwrap();
        let placement = search_translation(
            fixed.view(),
            &fixed_frame,
            moving.view(),
            &moving_frame,
            &SearchParams::default(),
        )
        .unwrap();
        let truth: [f64; 3] = std::array::from_fn(|a| {
            moving_frame.translation[a] + origin[a] as f64 * spacing - fixed_frame.translation[a]
        });
        for a in 0..3 {
            assert!(
                (placement.translation()[a] - truth[a]).abs() < 0.25 * spacing,
                "{:?} vs {truth:?}",
                placement.translation()
            );
        }
        assert!(placement.best.ncc > 0.99, "{}", placement.best.ncc);
        assert!(placement.levels.len() >= 2, "{:?}", placement.levels.len());
    }

    #[test]
    fn a_periodic_scene_is_refused_as_ambiguous() {
        // two identical copies side by side: the crop fits both
        let half = scene([24, 24, 24], 5);
        let mut moving = Array3::<f64>::zeros((24, 48, 24));
        moving.slice_mut(s![.., 0..24, ..]).assign(&half);
        moving.slice_mut(s![.., 24..48, ..]).assign(&half);
        let fixed = half.slice(s![4..20, 4..20, 4..20]).to_owned();
        let frame = Frame::new([1.0; 3], [0.0; 3]).unwrap();
        let params = SearchParams {
            levels: Some(1),
            ..SearchParams::default()
        };
        let error = search_translation(fixed.view(), &frame, moving.view(), &frame, &params)
            .unwrap_err()
            .to_string();
        assert!(error.contains("ambiguous"), "{error}");
    }

    #[test]
    fn unrelated_images_are_refused() {
        let frame = Frame::new([1.0; 3], [0.0; 3]).unwrap();
        let fixed = scene([20, 20, 20], 1);
        let moving = Array3::from_shape_fn((30, 30, 30), |(z, y, x)| {
            ((z * 7 + y * 13 + x * 3) % 5) as f64
        });
        let params = SearchParams {
            min_peak: 0.5,
            ..SearchParams::default()
        };
        let error = search_translation(fixed.view(), &frame, moving.view(), &frame, &params)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("below") || error.contains("ambiguous"),
            "{error}"
        );
    }
}
