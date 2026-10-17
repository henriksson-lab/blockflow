// SPDX-License-Identifier: MIT
//
// Original work for this crate.

//! Voxel-level Potts refinement of a dense label field, with a hard topology
//! constraint.
//!
//! The input is a labelling from an earlier segmentation (a `u32` per voxel,
//! dense: there is no background label, every label is a cell), a boundary
//! evidence field `p` in `[0, 1]` (membrane probability) and optionally an
//! appearance field `f`. The labelling is moved, one voxel at a time, down
//!
//! ```text
//! E(L) = Σ_v  λ_data · (f_v − μ_{L_v})² / σ²                       (data, optional)
//!      + Σ_{(u,v) ∈ N26}  w_uv · [L_u ≠ L_v]                          (boundary)
//!
//! w_uv = β · c_uv · ((1 − p̄_uv)^α + ε),   p̄_uv = (p_u + p_v) / 2
//! ```
//!
//! What each part is for
//! ---------------------
//! * **`c_uv`, the geometric weights**, are Boykov & Kolmogorov (ICCV 2003,
//!   "Computing geodesics and minimal surfaces via graph cuts"): by the
//!   Cauchy–Crofton formula, the area of a surface is `1/π` times the measure of
//!   the lines crossing it. Discretised over the 13 undirected neighbour
//!   directions `e_k` of the 26-neighbourhood, with `ΔΦ_k` the solid angle of the
//!   Voronoi cell of `±e_k` on the unit hemisphere (so `Σ ΔΦ_k = 2π`) and the
//!   lattice lines along `e_k` occupying one per `V / |e_k|` of cross-section
//!   (`V` the voxel volume),
//!
//!   ```text
//!   c_k = ΔΦ_k · V / (π · |e_k|)
//!   ```
//!
//!   in physical units, so a cut through the lattice costs its Euclidean area to
//!   within the angular resolution of 26 directions rather than the up-to-`√3`
//!   metrication error of a uniform 6- or 26-neighbour count.
//!   [`geometric_weights`] computes `ΔΦ_k` numerically (a Fibonacci lattice of
//!   sphere points, each assigned to its nearest direction), because with
//!   anisotropic spacing the Voronoi cells have no convenient closed form.
//! * **The `(1 − p̄)^α + ε` factor** makes a boundary nearly free where the
//!   evidence says there is a membrane and expensive in cytoplasm. That — not a
//!   generic smoothness term — is what lets a thin process survive: its surface
//!   lies on membrane, so keeping it costs about `ε`, while moving its boundary
//!   inward into cytoplasm costs about `β` per unit area.
//! * **The data term** is piecewise-constant appearance per label (Chan–Vese /
//!   Mumford–Shah). The means `μ` are a table, refreshed between sweeps by a
//!   global reduction ([`label_means`]); within a sweep they are constants.
//!
//! The smoothing bias, and why it is not a term of `E`
//! -------------------------------------------------------
//! The boundary term is an area, and the gradient flow of an area is mean
//! curvature flow: a tube of radius `r` shrinks at speed `1/r` under it. A plain
//! unweighted area term for smoothing would therefore shrink exactly the thin
//! processes this refinement is meant to keep. What is used instead is a
//! **curvature-thresholded majority vote**: with `n_x` the number of
//! 26-neighbours of `v` labelled `x`, a flip `a → b` is biased by
//!
//! ```text
//! λ_smooth · (h(n_b) − h(n_a)),   h(n) = max(0, τ − n).
//! ```
//!
//! `h` is zero for any voxel with at least `τ` like neighbours. A voxel on a flat
//! interface has 17 like neighbours and 9 unlike; a single-voxel bump on that
//! interface has 9; the corner of a 3×3 square tube has 11. With `τ = 10` the
//! bias removes bumps, dents and single-voxel jaggies (curvature of order one
//! voxel) and exerts **no force at all** on flat interfaces or on tubes three or
//! more voxels across — the low-curvature regime where area flow would still be
//! shrinking them. Thinner tubes than that are protected by the topology
//! constraint instead. A 3×3×3 majority vote is a small-kernel
//! Merriman–Bence–Osher threshold step, which is the sense in which this
//! approximates curvature flow; the threshold is what restricts it to the
//! highest curvatures.
//!
//! As an energy, `Σ_v h(n_same(v))` would be a 27-voxel clique, and two voxels
//! of one parity class (two apart) share such cliques, so the eight-class
//! parallel update below would no longer be exact. It is therefore applied as a
//! **proposal bias**: it shifts which flips are taken but is not part of `E`.
//! [`SweepStats::delta_energy`] and [`label_energy`] report `E` alone, and ICM is
//! guaranteed monotone in `E` only with `λ_smooth = 0`.
//!
//! Topology: which connectivity, and the simple-point test
//! --------------------------------------------------------
//! A flip `a → b` at `v` changes two sets, `{L = a}` (loses `v`) and `{L = b}`
//! (gains `v`); no other label's set changes. It preserves the topology of both
//! — component count, tunnels and cavities of the set and of its complement —
//! exactly when `v` is a simple point of `{L = a}` and of `{L = b} ∪ {v}`. Each
//! label set is taken against its complement (the union of all other labels and
//! the outside of the volume).
//!
//! **Labels are 6-connected and complements 26-connected**, the `(6, 26)` pair,
//! by default ([`is_simple_label_flip`]). In a dense labelling a 26-connected
//! label would count two voxels touching only at an edge or a corner as one
//! cell, which lets a process "survive" as a chain of corner contacts that a
//! membrane passes straight through. Under `(6, 26)` a process is preserved as a
//! face-connected tube. `(26, 6)` is available through
//! [`is_simple_label_flip_with`].
//!
//! The test is [`super::skeleton::is_simple_point`], which is the `(26, 6)`
//! characterisation, through the standard duality: `v` is `(6, 26)`-simple in `X`
//! exactly when it is `(26, 6)`-simple in `X̄ ∪ {v}`. It is answered from a
//! 2^26-bit table of that predicate ([`simple_point_table`], 8 MiB, built once
//! per process and checked against the flood fill over every configuration in
//! the tests), which makes the flip test about ten times cheaper than the flood
//! fill and is the same table the GPU kernel reads. Because every voxel lies in
//! the 3×3×3 window of no other voxel of its own parity class, the simplicity of
//! one flip cannot be changed by a simultaneous flip in the same class, so the
//! parallel update preserves topology exactly as a sequential one would.
//!
//! **Known limit: ends at the volume face.** Outside the volume counts as
//! complement of every label, so a process cut by the volume face has an end
//! there, and removing an end voxel is simple. Where the energy favours it, such
//! a process can retract from the face. Block seams do not cause this — outside
//! the buffer at a seam lies inside the halo — only the real volume edge does.
//!
//! Proposals are only ever the labels of a voxel's own 26 neighbours (cellular
//! Potts style), so there is no global label count and no `K` in the cost.
//!
//! The optimiser
//! -------------
//! A sweep visits the eight parity classes of the **global** coordinates in a
//! fixed order ([`SWEEP_ORDER`]). No two voxels of one class are 26-neighbours,
//! and every term of `E` is a pair inside N26, so the voxels of a class are
//! updated in parallel with exactly the energy change each computes. Each voxel
//! compares its current label with every distinct neighbour label: at `T = 0`
//! (ICM) it takes the lowest-scoring flip if that is negative, at `T > 0`
//! (Metropolis) it draws one candidate uniformly and accepts with
//! `min(1, exp(−ΔE / T))`, the acceptance draw taken on 2^16 levels
//! ([`Acceptance`]). A flip failing the simple-point test is rejected.
//!
//! The randomness is a counter-based hash of `(seed, sweep, class, global voxel
//! index)`, so the answer does not depend on the block decomposition, on the
//! thread count or on the order voxels are visited in.
//!
//! The per-voxel decision is one pure function, [`decide`], of the 3×3×3 label
//! and `p` windows, the centre's `f`, the means table, the parameters and the
//! two counter words. Everything else here is gathering and bookkeeping;
//! `gpu::label_energy` reimplements `decide` and the gather as a kernel and keeps
//! this as the reference.
//!
//! Fixed point: why every decision is an integer
//! ---------------------------------------------
//! So that a GPU takes **the same decisions to the bit**, the energy is defined
//! in integer units of a [`LabelEnergy::quantum`] chosen from the parameters
//! alone (the largest boundary weight is 2^16 units, or `λ_smooth / 256` is when
//! that is larger), never from the data, so it is the same in every block:
//!
//! * `p` is quantised to [`P_LEVELS`] levels and `w_uv` read from a table by
//!   direction and level sum, rounded to units on the host;
//! * the data cost of one voxel is `min((f − μ)² · λ/(σ² quantum), 2^29)` in
//!   `f32`, truncated: a subtraction and two products, which every device
//!   rounds alike and no compiler can fuse. **It saturates** at `2^29` units
//!   (`2^13` largest boundary weights), past which a voxel's appearance no longer
//!   ranks two labels whose means are both that far away;
//! * the smoothing bias is an integer multiple of `λ_smooth` in units;
//! * Metropolis compares the integer score with a per-sweep integer threshold
//!   table, and the candidate pick is an integer product of a hash word.
//!
//! Every score then fits an `i32` and sums exactly, `E` ([`label_energy`]) is an
//! exact integer sum times the quantum, and ICM is exactly monotone. The
//! quantisation is far below anything the evidence resolves (`p` to 1/4096, a
//! boundary weight to 1/65536 of the largest).
//!
//! As a `BlockOp`
//! --------------
//! [`LabelEnergySweepOp`] is one whole sweep (all eight classes) over a `u32`
//! label image, with `p` and `f` read as stored source images
//! ([`BlockOp::source_inputs`]). Its reach is derived from the class order
//! ([`sweep_reach`]): a voxel's final value depends on a chain of earlier class
//! updates, each one step away, and a step can move along an axis only when the
//! two classes differ in that axis's parity bit. The reach along an axis is
//! therefore the number of runs of that parity bit in the order — `[3, 4, 3]`
//! for [`SWEEP_ORDER`], a Hamiltonian path of the parity cube chosen to keep the
//! sum low and balanced, against `[2, 4, 8]` for the natural order `0..8`.
//! One op per sweep, rather than one per class, because a `Chain::sequence` of
//! eight ops cannot give each op its own operand window; eight phases per sweep
//! would materialise eight intermediates. The means table is part of the op, so
//! refreshing it means building the next sweep's op after a reduction.

use std::collections::HashMap;
use std::sync::Arc;

use ndarray::{Array3, ArrayView3, ArrayViewMut3};
use rayon::prelude::*;

use crate::assemble::{ImageId, Phase, PlanBuilder};
use crate::dtype::Dtype;
use crate::error::{bail, ensure, Error, Result};
use crate::op::{Anchor, BlockOp, Chain, SourceInput, SourceInputs};
use crate::reach::Reach;
use crate::voxels::Voxels;

#[cfg(test)]
use super::skeleton::is_simple_point;
use super::skeleton::{neighbour_offset, Adjacency, Subfield, CENTRE, NEIGHBOURHOOD};

/// The label a gathered neighbourhood holds where it runs past the buffer.
///
/// Reserved: a label field containing it is refused. It never equals a real
/// label, so it is never proposed, contributes no boundary, and counts as
/// complement in the simple-point test.
pub const OUTSIDE: u32 = u32::MAX;

// ------------------------------------------------------------ topology --

/// The 26 neighbours of a window as bits: bit `k` for neighbour `k < CENTRE`,
/// bit `k - 1` above it. The index into [`simple_point_table`].
#[inline]
pub fn neighbour_bits(mut member: impl FnMut(usize) -> bool) -> u32 {
    let mut bits = 0u32;
    for k in 0..NEIGHBOURHOOD {
        if k != CENTRE && member(k) {
            bits |= 1 << (k - usize::from(k > CENTRE));
        }
    }
    bits
}

/// Bit `c` (word `c >> 5`, bit `c & 31`) says whether the centre is a simple
/// point ([`super::skeleton::is_simple_point`], the `(26, 6)`
/// characterisation) of the object whose 26 neighbours are the bits of `c`
/// ([`neighbour_bits`]).
///
/// 2^26 bits, 8 MiB, built once per process on first use (a fraction of a
/// second on all cores) by a bitmask flood fill that is checked against
/// `is_simple_point` over every configuration in this module's tests. It
/// turns the simple-point test of a flip into two lookups, on the CPU and the
/// GPU alike.
pub fn simple_point_table() -> &'static [u32] {
    static TABLE: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();
    // Built on threads of its own, not rayon's: a rayon worker that reaches
    // this while another thread initialises it would block inside the pool
    // the initialiser needs.
    TABLE.get_or_init(|| {
        let masks = BitNeighbourhood::new();
        let mut table = vec![0u32; 1 << 21];
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let per = table.len().div_ceil(threads);
        std::thread::scope(|scope| {
            for (chunk, words) in table.chunks_mut(per).enumerate() {
                let masks = &masks;
                scope.spawn(move || {
                    for (offset, slot) in words.iter_mut().enumerate() {
                        let word = (chunk * per + offset) as u32;
                        for bit in 0..32 {
                            if masks.is_simple(word << 5 | bit) {
                                *slot |= 1 << bit;
                            }
                        }
                    }
                });
            }
        });
        table
    })
}

/// Adjacency within the 3×3×3 window as 27-bit masks, for the flood fills that
/// build [`simple_point_table`].
struct BitNeighbourhood {
    /// Window positions 26-adjacent to each position, the centre excluded.
    adjacent26: [u32; NEIGHBOURHOOD],
    /// Window positions 6-adjacent to each position, the centre excluded.
    adjacent6: [u32; NEIGHBOURHOOD],
    /// The 18-neighbourhood (no corners), the centre excluded.
    n18: u32,
    faces: u32,
}

impl BitNeighbourhood {
    fn new() -> Self {
        let mut adjacent26 = [0u32; NEIGHBOURHOOD];
        let mut adjacent6 = [0u32; NEIGHBOURHOOD];
        let mut n18 = 0;
        let mut faces = 0;
        for i in 0..NEIGHBOURHOOD {
            let a = neighbour_offset(i);
            let l1: isize = a.iter().map(|v| v.abs()).sum();
            if i != CENTRE && l1 <= 2 {
                n18 |= 1 << i;
            }
            if l1 == 1 {
                faces |= 1 << i;
            }
            for j in 0..NEIGHBOURHOOD {
                if i == j || j == CENTRE {
                    continue;
                }
                let b = neighbour_offset(j);
                let d: [isize; 3] = std::array::from_fn(|axis| (a[axis] - b[axis]).abs());
                if d.iter().all(|&v| v <= 1) {
                    adjacent26[i] |= 1 << j;
                }
                if d.iter().sum::<isize>() == 1 {
                    adjacent6[i] |= 1 << j;
                }
            }
        }
        Self {
            adjacent26,
            adjacent6,
            n18,
            faces,
        }
    }

    /// The component of `within` containing `seed` under `adjacent`.
    fn flood(seed: u32, within: u32, adjacent: &[u32; NEIGHBOURHOOD]) -> u32 {
        let mut component = seed;
        let mut frontier = seed;
        while frontier != 0 {
            let mut next = 0;
            let mut rest = frontier;
            while rest != 0 {
                next |= adjacent[rest.trailing_zeros() as usize];
                rest &= rest - 1;
            }
            frontier = next & within & !component;
            component |= frontier;
        }
        component
    }

    /// `C* = 1` and `Cbar = 1` for the 26-neighbour configuration `config`.
    fn is_simple(&self, config: u32) -> bool {
        let low = (1u32 << CENTRE) - 1;
        let object = (config & low) | ((config & !low) << 1);
        if object == 0 {
            return false;
        }
        let first = object.isolate_lowest_one();
        if Self::flood(first, object, &self.adjacent26) != object {
            return false;
        }
        let background = !object & self.n18;
        let faces = background & self.faces;
        if faces == 0 {
            return false;
        }
        let first = faces.isolate_lowest_one();
        faces & !Self::flood(first, background, &self.adjacent6) == 0
    }
}

#[inline]
fn table_says(table: &[u32], config: u32) -> bool {
    (table[(config >> 5) as usize] >> (config & 31)) & 1 == 1
}

/// The [`simple_point_table`] index for `v` leaving `{label == x}`: the set
/// itself under `(26, 6)`, its complement under `(6, 26)` (the duality in the
/// module header).
#[inline]
pub fn removal_config(n: &[u32; NEIGHBOURHOOD], x: u32, adjacency: Adjacency) -> u32 {
    match adjacency {
        Adjacency::TwentySix => neighbour_bits(|k| n[k] == x),
        Adjacency::Six => neighbour_bits(|k| n[k] != x),
    }
}

/// Can `v` leave the set `{label == x}` (with `v` counted in it) without
/// changing its topology, `x` being `adjacency`-connected and its complement
/// the other one?
fn removable(n: &[u32; NEIGHBOURHOOD], x: u32, adjacency: Adjacency) -> bool {
    table_says(simple_point_table(), removal_config(n, x, adjacency))
}

/// [`removable`] by the flood fill of [`is_simple_point`]: the definition the
/// table is tested against.
#[cfg(test)]
fn removable_by_flood(n: &[u32; NEIGHBOURHOOD], x: u32, adjacency: Adjacency) -> bool {
    let mut mask = [false; NEIGHBOURHOOD];
    for (index, slot) in mask.iter_mut().enumerate() {
        *slot = match adjacency {
            Adjacency::TwentySix => n[index] == x,
            Adjacency::Six => n[index] != x,
        };
    }
    mask[CENTRE] = true;
    is_simple_point(&mask)
}

/// Is changing the centre of `n` from `a` to `b` topology-preserving for both
/// label sets, labels being `adjacency`-connected and their complements the
/// other connectivity? `n[CENTRE]` is taken to be `a`; [`OUTSIDE`] counts as
/// complement of every label.
pub fn is_simple_label_flip_with(
    n: &[u32; NEIGHBOURHOOD],
    a: u32,
    b: u32,
    adjacency: Adjacency,
) -> bool {
    a != b && removable(n, a, adjacency) && removable(n, b, adjacency)
}

/// [`is_simple_label_flip_with`] under this module's convention: labels
/// 6-connected, complements 26-connected.
pub fn is_simple_label_flip(n: &[u32; NEIGHBOURHOOD], a: u32, b: u32) -> bool {
    is_simple_label_flip_with(n, a, b, Adjacency::Six)
}

// -------------------------------------------------------------- weights --

/// Sphere points used to measure the Voronoi solid angles in
/// [`geometric_weights`]; the solid-angle error is about `1/sqrt(N)` per cell.
const SPHERE_SAMPLES: usize = 200_000;

/// Boykov–Kolmogorov 2003 geometric edge weights for the 26-neighbourhood with
/// physical voxel `spacing`, indexed like a gathered neighbourhood
/// (`skeleton::neighbour_index`); the centre is zero.
///
/// `c_k = ΔΦ_k · V / (π · |e_k|)` with `ΔΦ_k` the hemisphere solid angle of the
/// Voronoi cell of `±e_k` among the 13 physical directions. Summing `c` over
/// the edges a surface cuts approximates its area in physical units.
pub fn geometric_weights(spacing: [f64; 3]) -> Result<[f64; NEIGHBOURHOOD]> {
    ensure!(
        spacing.iter().all(|s| s.is_finite() && *s > 0.0),
        "label energy: spacing must be finite and positive, got {spacing:?}"
    );
    // the 13 undirected directions: the neighbourhood indices after the centre
    let directions: Vec<(usize, [f64; 3], f64)> = (CENTRE + 1..NEIGHBOURHOOD)
        .map(|index| {
            let o = neighbour_offset(index);
            let e = [
                o[0] as f64 * spacing[0],
                o[1] as f64 * spacing[1],
                o[2] as f64 * spacing[2],
            ];
            let length = (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt();
            (index, [e[0] / length, e[1] / length, e[2] / length], length)
        })
        .collect();
    let mut counts = [0usize; NEIGHBOURHOOD];
    let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    for sample in 0..SPHERE_SAMPLES {
        let z = 1.0 - (2.0 * sample as f64 + 1.0) / SPHERE_SAMPLES as f64;
        let r = (1.0 - z * z).sqrt();
        let phi = golden * sample as f64;
        let u = [r * phi.cos(), r * phi.sin(), z];
        let mut best = 0;
        let mut best_dot = -1.0;
        for (slot, (_, d, _)) in directions.iter().enumerate() {
            let dot = (u[0] * d[0] + u[1] * d[1] + u[2] * d[2]).abs();
            if dot > best_dot {
                best_dot = dot;
                best = slot;
            }
        }
        counts[directions[best].0] += 1;
    }
    let volume = spacing[0] * spacing[1] * spacing[2];
    let mut weights = [0.0; NEIGHBOURHOOD];
    for &(index, _, length) in &directions {
        // fraction of the sphere, times 2π: the hemisphere solid angle of ±e_k
        let solid = 2.0 * std::f64::consts::PI * counts[index] as f64 / SPHERE_SAMPLES as f64;
        let weight = solid * volume / (std::f64::consts::PI * length);
        weights[index] = weight;
        weights[NEIGHBOURHOOD - 1 - index] = weight; // the antipode
    }
    Ok(weights)
}

// ------------------------------------------------------------ parameters --

/// The energy's parameters. Construct with [`LabelEnergyParams::new`] (boundary
/// term only) and add the optional parts with the `with_*` methods.
#[derive(Debug, Clone, PartialEq)]
pub struct LabelEnergyParams {
    /// Physical voxel spacing per axis, for the geometric weights.
    pub spacing: [f64; 3],
    /// Boundary weight `β`.
    pub beta: f64,
    /// Exponent `α` on `1 − p̄`.
    pub alpha: f64,
    /// Floor `ε` on the boundary factor, so a boundary on membrane is cheap, not
    /// free.
    pub epsilon: f64,
    /// Data weight `λ_data`; zero disables the data term.
    pub lambda_data: f64,
    /// Appearance scale `σ` of the data term.
    pub sigma: f64,
    /// Smoothing-bias weight `λ_smooth`; zero disables it.
    pub lambda_smooth: f64,
    /// Like-neighbour count `τ` below which the smoothing bias acts.
    pub smooth_threshold: u32,
    /// Label connectivity for the hard topology constraint, or `None` for no
    /// constraint.
    pub topology: Option<Adjacency>,
    /// Seed of the counter-based random draws.
    pub seed: u64,
}

impl LabelEnergyParams {
    /// The boundary term alone, with the `(6, 26)` topology constraint on.
    pub fn new(spacing: [f64; 3], beta: f64, alpha: f64, epsilon: f64) -> Self {
        Self {
            spacing,
            beta,
            alpha,
            epsilon,
            lambda_data: 0.0,
            sigma: 1.0,
            lambda_smooth: 0.0,
            smooth_threshold: 0,
            topology: Some(Adjacency::Six),
            seed: 0,
        }
    }

    pub fn with_data(mut self, lambda_data: f64, sigma: f64) -> Self {
        self.lambda_data = lambda_data;
        self.sigma = sigma;
        self
    }

    pub fn with_smoothing(mut self, lambda_smooth: f64, threshold: u32) -> Self {
        self.lambda_smooth = lambda_smooth;
        self.smooth_threshold = threshold;
        self
    }

    pub fn with_topology(mut self, topology: Option<Adjacency>) -> Self {
        self.topology = topology;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    fn validate(&self) -> Result<()> {
        let finite_non_negative = |value: f64| value.is_finite() && value >= 0.0;
        ensure!(
            finite_non_negative(self.beta)
                && finite_non_negative(self.alpha)
                && finite_non_negative(self.epsilon),
            "label energy: beta, alpha and epsilon must be finite and non-negative, got {}, {}, {}",
            self.beta,
            self.alpha,
            self.epsilon
        );
        ensure!(
            finite_non_negative(self.lambda_data) && finite_non_negative(self.lambda_smooth),
            "label energy: lambda_data and lambda_smooth must be finite and non-negative, got {} \
             and {}",
            self.lambda_data,
            self.lambda_smooth
        );
        ensure!(
            self.sigma.is_finite() && self.sigma > 0.0,
            "label energy: sigma must be finite and positive, got {}",
            self.sigma
        );
        ensure!(
            self.smooth_threshold <= 26,
            "label energy: the smoothing threshold counts 26-neighbours, got {}",
            self.smooth_threshold
        );
        Ok(())
    }
}

// ----------------------------------------------------------- fixed point --

/// Levels `p` is quantised to before it weights a boundary: `p` stands for
/// `⌊clamp(p, 0, 1) · P_LEVELS + ½⌋ / P_LEVELS` ([`quantise_p`]).
pub const P_LEVELS: u32 = 4096;

/// Entries per direction of the edge table: every sum of two quantised `p`.
pub const PAIR_LEVELS: usize = 2 * P_LEVELS as usize + 1;

/// Rows of the edge table: one per undirected direction, `k` and its antipode
/// `26 − k` sharing row `min(k, 26 − k)`.
pub const EDGE_ROWS: usize = CENTRE;

/// The largest boundary weight, or `λ_smooth / SMOOTH_HEADROOM` when that is
/// larger, is this many energy units.
const BASE_UNITS: f64 = 65536.0;

/// How many largest boundary weights `λ_smooth` may be before it, rather than
/// the boundary, sets the unit.
const SMOOTH_HEADROOM: f64 = 256.0;

/// Ceiling on one voxel's data cost, in energy units.
pub const DATA_CAP: i32 = 1 << 29;

/// Accept-draw levels of Metropolis: the acceptance draw is `(k + ½) / ACCEPT_LEVELS`.
pub const ACCEPT_LEVELS: usize = 1 << 16;

/// `p` as the integer level the energy uses. `NaN` is level 0.
#[inline]
pub fn quantise_p(p: f64) -> u32 {
    (p.clamp(0.0, 1.0) * f64::from(P_LEVELS) + 0.5) as u32
}

/// The edge-table row of neighbour `k`.
#[inline]
pub const fn edge_row(k: usize) -> usize {
    if k < CENTRE {
        k
    } else {
        NEIGHBOURHOOD - 1 - k
    }
}

/// The parameters with the geometric weights computed from them, once, and the
/// fixed-point tables every decision is taken in (see the module header).
#[derive(Debug, Clone, PartialEq)]
pub struct LabelEnergy {
    params: LabelEnergyParams,
    weights: [f64; NEIGHBOURHOOD],
    quantum: f64,
    edges: EdgeTable,
    smooth_units: i32,
    data_scale: f32,
}

/// `w` in energy units by `(edge_row(k), q_u + q_v)`; behind an `Arc` so the
/// energy stays cheap to clone, and with a `Debug` that does not print it.
#[derive(Clone, PartialEq)]
struct EdgeTable(Arc<[i32]>);

impl std::fmt::Debug for EdgeTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EdgeTable({} entries)", self.0.len())
    }
}

impl LabelEnergy {
    pub fn new(params: LabelEnergyParams) -> Result<Self> {
        params.validate()?;
        let weights = geometric_weights(params.spacing)?;
        let largest = weights.iter().copied().fold(0.0, f64::max);
        let mut base = (params.beta * largest * (1.0 + params.epsilon))
            .max(params.lambda_smooth / SMOOTH_HEADROOM);
        if base <= 0.0 {
            base = params.lambda_data; // data alone: one unit (f − μ)² / σ²
        }
        if base <= 0.0 {
            base = 1.0; // nothing to weigh
        }
        let quantum = base / BASE_UNITS;
        let mut edges = vec![0i32; EDGE_ROWS * PAIR_LEVELS];
        for row in 0..EDGE_ROWS {
            for sum in 0..PAIR_LEVELS {
                let open = (1.0 - sum as f64 / f64::from(2 * P_LEVELS)).max(0.0);
                let w = params.beta * weights[row] * (open.powf(params.alpha) + params.epsilon);
                edges[row * PAIR_LEVELS + sum] = (w / quantum).round() as i32;
            }
        }
        let smooth_units = (params.lambda_smooth / quantum).round() as i32;
        let data_scale = ((params.lambda_data / (params.sigma * params.sigma)) / quantum)
            .min(f64::from(f32::MAX)) as f32;
        Ok(Self {
            params,
            weights,
            quantum,
            edges: EdgeTable(edges.into()),
            smooth_units,
            data_scale,
        })
    }

    pub fn params(&self) -> &LabelEnergyParams {
        &self.params
    }

    /// The geometric weights `c_k`, indexed like a gathered neighbourhood.
    pub fn weights(&self) -> &[f64; NEIGHBOURHOOD] {
        &self.weights
    }

    /// The energy of one unit: every term of `E` is an integer multiple of it.
    pub fn quantum(&self) -> f64 {
        self.quantum
    }

    /// `w_uv` in units, by `edge_row(k) * PAIR_LEVELS + q_u + q_v`.
    pub fn edge_table(&self) -> &[i32] {
        &self.edges.0
    }

    /// `λ_smooth` in units.
    pub fn smooth_units(&self) -> i32 {
        self.smooth_units
    }

    /// `λ_data / (σ² · quantum)` as the `f32` the data cost is computed with;
    /// zero when the data term is off.
    pub fn data_scale(&self) -> f32 {
        self.data_scale
    }

    /// `w_uv` in units for the neighbour at index `k`, given quantised `p` at
    /// both ends.
    #[inline]
    fn edge(&self, k: usize, q_u: u32, q_v: u32) -> i64 {
        i64::from(self.edges.0[edge_row(k) * PAIR_LEVELS + (q_u + q_v) as usize])
    }

    /// One voxel's data cost in units: `min((f − μ)² · data_scale, DATA_CAP)`,
    /// truncated, every step an `f32` operation a GPU rounds alike.
    #[inline]
    pub fn data_units(&self, f: f32, mu: f32) -> i32 {
        let d = f - mu;
        ((d * d) * self.data_scale).min(DATA_CAP as f32) as i32
    }

    fn uses_data(&self) -> bool {
        self.params.lambda_data > 0.0
    }

    /// Whether the data term can contribute at all.
    pub fn data_on(&self) -> bool {
        self.uses_data() && self.data_scale > 0.0
    }
}

/// The mean of `label` as the data term uses it: `f32`, and absent when it is
/// not finite.
#[inline]
pub fn mean_f32(means: &HashMap<u32, f64>, label: u32) -> Option<f32> {
    means
        .get(&label)
        .map(|&mu| mu as f32)
        .filter(|mu| mu.is_finite())
}

/// The temperature of one sweep, as the integer thresholds Metropolis accepts
/// against. Built once per sweep; ICM carries none.
///
/// A positive score `s` (in units) is accepted with draw level `k` iff
/// `s < thresholds[k]`, `thresholds[k] = ⌈−T · ln((k + ½) / ACCEPT_LEVELS) / quantum⌉`:
/// `u < exp(−s · quantum / T)` for `u` the bucket's midpoint, decided in
/// integers so that every device decides it alike.
#[derive(Debug, Clone, PartialEq)]
pub struct Acceptance {
    temperature: f64,
    thresholds: Option<Arc<[i32]>>,
}

impl Acceptance {
    pub fn new(energy: &LabelEnergy, temperature: f64) -> Self {
        let thresholds = (temperature > 0.0).then(|| {
            (0..ACCEPT_LEVELS)
                .map(|k| {
                    let u = (k as f64 + 0.5) / ACCEPT_LEVELS as f64;
                    let x = -temperature * u.ln() / energy.quantum;
                    x.ceil().min(f64::from(i32::MAX)) as i32
                })
                .collect()
        });
        Self {
            temperature,
            thresholds,
        }
    }

    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// The Metropolis thresholds, `None` for ICM (`T <= 0`).
    pub fn thresholds(&self) -> Option<&[i32]> {
        self.thresholds.as_deref()
    }
}

// --------------------------------------------------------------- kernel --

/// What one voxel sees: its 3×3×3 label window, its quantised `p` window
/// ([`quantise_p`]) and its own `f` as `f32` (`NaN` when there is no
/// appearance field).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighbourhood {
    pub labels: [u32; NEIGHBOURHOOD],
    pub p: [u32; NEIGHBOURHOOD],
    pub f: f32,
}

/// The outcome of [`decide`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Decision {
    /// The voxel's label after the update.
    pub label: u32,
    /// The exact change in `E` (smoothing bias excluded) in energy units
    /// ([`LabelEnergy::quantum`]); zero when kept.
    pub delta_units: i64,
    /// Whether a flip that would otherwise have been taken was refused by the
    /// topology constraint.
    pub blocked: bool,
}

/// The candidate index and acceptance level of a Metropolis step, from the two
/// counter words ([`counter_words`]): `pick = ⌊(h >> 32) · count / 2^32⌋`,
/// `level = g >> 48`. Integer, so every device draws alike.
#[inline]
pub fn metropolis_draw(words: [u64; 2], count: usize) -> (usize, usize) {
    let pick = (((words[0] >> 32) * count as u64) >> 32) as usize;
    (pick, (words[1] >> 48) as usize)
}

/// The per-voxel update: a pure function of the neighbourhood, the energy, the
/// means table, the sweep's [`Acceptance`] and two counter words.
///
/// ICM (no thresholds): the candidate with the lowest score (energy change plus
/// smoothing bias) is taken if the score is negative, falling back to the next
/// if the topology refuses it; ties go to the lower label. Metropolis: the
/// words pick one candidate and an acceptance level ([`metropolis_draw`]).
/// Every quantity is an integer number of energy units, so the decision is
/// reproducible to the bit on any device.
pub fn decide(
    energy: &LabelEnergy,
    n: &Neighbourhood,
    means: &HashMap<u32, f64>,
    acceptance: &Acceptance,
    words: [u64; 2],
) -> Decision {
    let a = n.labels[CENTRE];
    let keep = Decision {
        label: a,
        delta_units: 0,
        blocked: false,
    };
    let mut candidates = [0u32; NEIGHBOURHOOD];
    let mut count = 0;
    let mut outside = 0u32;
    for (k, &label) in n.labels.iter().enumerate() {
        if k == CENTRE || label == a {
            continue;
        }
        if label == OUTSIDE {
            outside += 1;
            continue;
        }
        if !candidates[..count].contains(&label) {
            candidates[count] = label;
            count += 1;
        }
    }
    if count == 0 {
        return keep;
    }

    let params = &energy.params;
    let mut weight = [0i64; NEIGHBOURHOOD];
    let mut keep_cost = 0i64; // boundary the centre has to label `a` now: Σ_{L_u = a} w
    let mut like_a = 0u32;
    for k in 0..NEIGHBOURHOOD {
        let label = n.labels[k];
        if k == CENTRE || label == OUTSIDE {
            continue;
        }
        weight[k] = energy.edge(k, n.p[CENTRE], n.p[k]);
        if label == a {
            keep_cost += weight[k];
            like_a += 1;
        }
    }
    let data_a = if energy.data_on() && n.f.is_finite() {
        mean_f32(means, a).map(|mu| energy.data_units(n.f, mu))
    } else {
        None
    };
    let smooth = i64::from(energy.smooth_units);
    let hinge = |like: u32| i64::from(params.smooth_threshold.saturating_sub(like + outside));

    // (score, delta) for flipping to `b`
    let evaluate = |b: u32| -> (i64, i64) {
        let mut gain = 0i64;
        let mut like_b = 0u32;
        for k in 0..NEIGHBOURHOOD {
            if k != CENTRE && n.labels[k] == b {
                gain += weight[k];
                like_b += 1;
            }
        }
        let mut delta = keep_cost - gain;
        if let (Some(data_a), Some(mu_b)) = (data_a, mean_f32(means, b)) {
            delta += i64::from(energy.data_units(n.f, mu_b)) - i64::from(data_a);
        }
        let bias = if smooth > 0 {
            smooth * (hinge(like_b) - hinge(like_a))
        } else {
            0
        };
        (delta + bias, delta)
    };
    let allowed = |b: u32| match params.topology {
        None => true,
        Some(adjacency) => is_simple_label_flip_with(&n.labels, a, b, adjacency),
    };

    match acceptance.thresholds() {
        None => {
            let mut scored = [(0i64, 0u32, 0i64); NEIGHBOURHOOD];
            for (slot, &b) in candidates[..count].iter().enumerate() {
                let (score, delta) = evaluate(b);
                scored[slot] = (score, b, delta);
            }
            let scored = &mut scored[..count];
            scored.sort_unstable_by_key(|&(score, b, _)| (score, b));
            let mut blocked = false;
            for &(score, b, delta) in scored.iter() {
                if score >= 0 {
                    break;
                }
                if allowed(b) {
                    return Decision {
                        label: b,
                        delta_units: delta,
                        blocked,
                    };
                }
                blocked = true;
            }
            Decision { blocked, ..keep }
        }
        Some(thresholds) => {
            let (pick, level) = metropolis_draw(words, count);
            let b = candidates[pick];
            let (score, delta) = evaluate(b);
            let accept = score <= 0 || score < i64::from(thresholds[level]);
            if !accept {
                return keep;
            }
            if allowed(b) {
                Decision {
                    label: b,
                    delta_units: delta,
                    blocked: false,
                }
            } else {
                Decision {
                    blocked: true,
                    ..keep
                }
            }
        }
    }
}

// ------------------------------------------------------------ randomness --

/// The splitmix64 finaliser.
#[inline]
pub fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The part of the counter hash shared by every voxel of one class of one
/// sweep: `mix(mix(mix(seed) ^ sweep) ^ class)`.
#[inline]
pub fn class_prefix(seed: u64, sweep: u64, class: u64) -> u64 {
    splitmix(splitmix(splitmix(seed) ^ sweep) ^ class)
}

/// The second word is derived from the first with this salt.
pub const SECOND_WORD_SALT: u64 = 0xd1b5_4a32_d192_ed03;

/// Two 64-bit words that are a function of the counter alone.
///
/// A splitmix64 finaliser folded over the four words: no state, so any thread
/// visiting any voxel in any block draws the same numbers for it.
pub fn counter_words(seed: u64, sweep: u64, class: u64, index: u64) -> [u64; 2] {
    let h = splitmix(class_prefix(seed, sweep, class) ^ index);
    [h, splitmix(h ^ SECOND_WORD_SALT)]
}

/// [`counter_words`] as two uniform draws in `[0, 1)` (53 bits each).
pub fn counter_draws(seed: u64, sweep: u64, class: u64, index: u64) -> [f64; 2] {
    let [h, g] = counter_words(seed, sweep, class, index);
    const SCALE: f64 = 1.0 / (1u64 << 53) as f64;
    [(h >> 11) as f64 * SCALE, (g >> 11) as f64 * SCALE]
}

// ---------------------------------------------------------------- sweeps --

/// The order a sweep visits the eight parity classes in (`Subfield` index:
/// bit 2 is the parity of axis 0, bit 0 of axis 2).
///
/// A Hamiltonian path of the parity cube, `000 001 011 111 101 100 110 010`,
/// chosen for its [`sweep_reach`] of `[3, 4, 3]`; see the module header.
pub const SWEEP_ORDER: [usize; Subfield::COUNT] = [0, 1, 3, 7, 5, 4, 6, 2];

/// Voxels one sweep in `order` reads beyond the voxel it writes, per axis: the
/// number of runs of that axis's parity bit along the order.
///
/// A voxel's final value depends on a neighbour updated in an earlier class,
/// which depends on one updated earlier still, and so on to the input; each link
/// is one step, and it can move along an axis only if the two classes differ in
/// that axis's bit. The longest such chain alternates the bit as often as the
/// order does (runs − 1 times), and the last link reads the input one step
/// further.
pub fn sweep_reach(order: &[usize]) -> [usize; 3] {
    let mut reach = [0; 3];
    for (axis, slot) in reach.iter_mut().enumerate() {
        let bit = 2 - axis;
        let mut runs = 0;
        let mut last = None;
        for &class in order {
            let value = (class >> bit) & 1;
            if last != Some(value) {
                runs += 1;
                last = Some(value);
            }
        }
        *slot = runs;
    }
    reach
}

/// What one sweep did.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SweepStats {
    pub sweep: usize,
    pub temperature: f64,
    /// Voxels whose label changed.
    pub flips: u64,
    /// Flips that would have been taken and were refused by the topology
    /// constraint.
    pub topology_rejections: u64,
    /// Sum of the exact `E` changes of the flips taken.
    pub delta_energy: f64,
    /// `E` before and after the sweep, when the caller computed them (the
    /// in-memory driver does; [`label_energy_sweep_into`] leaves them `NaN`).
    pub energy_before: f64,
    pub energy_after: f64,
}

fn shape3<T>(view: &ArrayView3<'_, T>) -> [usize; 3] {
    [view.shape()[0], view.shape()[1], view.shape()[2]]
}

#[inline]
fn gather(
    labels: &ArrayView3<'_, u32>,
    p: &ArrayView3<'_, f64>,
    f: Option<&ArrayView3<'_, f64>>,
    at: [usize; 3],
    shape: [usize; 3],
) -> Option<Neighbourhood> {
    let centre = labels[at];
    let mut n = Neighbourhood {
        labels: [OUTSIDE; NEIGHBOURHOOD],
        p: [0; NEIGHBOURHOOD],
        f: f32::NAN,
    };
    let mut differs = false;
    for k in 0..NEIGHBOURHOOD {
        let o = neighbour_offset(k);
        let q = [
            at[0] as isize + o[0],
            at[1] as isize + o[1],
            at[2] as isize + o[2],
        ];
        if (0..3).any(|axis| q[axis] < 0 || q[axis] >= shape[axis] as isize) {
            continue;
        }
        let q = [q[0] as usize, q[1] as usize, q[2] as usize];
        let label = labels[q];
        differs |= label != centre;
        n.labels[k] = label;
    }
    if !differs {
        return None; // interior voxel: no candidate, nothing to gather further
    }
    for k in 0..NEIGHBOURHOOD {
        if n.labels[k] == OUTSIDE {
            continue;
        }
        let o = neighbour_offset(k);
        let q = [
            (at[0] as isize + o[0]) as usize,
            (at[1] as isize + o[1]) as usize,
            (at[2] as isize + o[2]) as usize,
        ];
        n.p[k] = quantise_p(p[q]);
    }
    if let Some(f) = f {
        n.f = f[at] as f32;
    }
    Some(n)
}

/// One parity class's updates, read from `labels` and returned as
/// `(position, new label)` in a fixed order, with the class's statistics.
#[allow(clippy::too_many_arguments)]
fn class_updates(
    energy: &LabelEnergy,
    labels: ArrayView3<'_, u32>,
    p: ArrayView3<'_, f64>,
    f: Option<ArrayView3<'_, f64>>,
    means: &HashMap<u32, f64>,
    sweep: usize,
    acceptance: &Acceptance,
    class: usize,
    at: &Anchor,
) -> (Vec<([usize; 3], u32)>, SweepStats) {
    let shape = shape3(&labels);
    let parity = [(class >> 2) & 1, (class >> 1) & 1, class & 1];
    // first local index along each axis whose global coordinate has the parity
    let first: [usize; 3] = std::array::from_fn(|axis| (parity[axis] + (at.offset[axis] & 1)) & 1);
    let volume = at.volume;
    let prefix = class_prefix(energy.params.seed, sweep as u64, class as u64);
    let metropolis = acceptance.thresholds().is_some();
    let planes: Vec<(Vec<([usize; 3], u32)>, SweepStats, i64)> = (first[0]..shape[0])
        .step_by(2)
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|i| {
            let mut updates = Vec::new();
            let mut stats = SweepStats::default();
            let mut delta_units = 0i64;
            for j in (first[1]..shape[1]).step_by(2) {
                for k in (first[2]..shape[2]).step_by(2) {
                    let Some(n) = gather(&labels, &p, f.as_ref(), [i, j, k], shape) else {
                        continue;
                    };
                    let g = [at.offset[0] + i, at.offset[1] + j, at.offset[2] + k];
                    let index = ((g[0] * volume[1] + g[1]) * volume[2] + g[2]) as u64;
                    let words = if metropolis {
                        let h = splitmix(prefix ^ index);
                        [h, splitmix(h ^ SECOND_WORD_SALT)]
                    } else {
                        [0, 0]
                    };
                    let decision = decide(energy, &n, means, acceptance, words);
                    stats.topology_rejections += u64::from(decision.blocked);
                    if decision.label != n.labels[CENTRE] {
                        stats.flips += 1;
                        delta_units += decision.delta_units;
                        updates.push(([i, j, k], decision.label));
                    }
                }
            }
            (updates, stats, delta_units)
        })
        .collect();
    let mut all = Vec::new();
    let mut stats = SweepStats::default();
    let mut delta_units = 0i64;
    for (updates, plane, units) in planes {
        all.extend(updates);
        stats.flips += plane.flips;
        stats.topology_rejections += plane.topology_rejections;
        delta_units += units;
    }
    stats.delta_energy = delta_units as f64 * energy.quantum;
    (all, stats)
}

pub(crate) fn check_inputs(
    labels: &ArrayView3<'_, u32>,
    p: &ArrayView3<'_, f64>,
    f: Option<&ArrayView3<'_, f64>>,
    energy: &LabelEnergy,
) -> Result<()> {
    let shape = shape3(labels);
    ensure!(
        shape3(p) == shape,
        "label energy: p has shape {:?}, labels {:?}",
        p.shape(),
        shape
    );
    if let Some(f) = f {
        ensure!(
            shape3(f) == shape,
            "label energy: f has shape {:?}, labels {:?}",
            f.shape(),
            shape
        );
    } else if energy.uses_data() {
        bail!("label energy: the data term is on (lambda_data > 0) and no appearance field f was given");
    }
    ensure!(
        !labels.iter().any(|&label| label == OUTSIDE),
        "label energy: label {OUTSIDE} is reserved for the outside of the buffer"
    );
    Ok(())
}

/// One sweep (all eight parity classes in [`SWEEP_ORDER`]) over a buffer at
/// `at` in the volume, written into `out`.
///
/// Parity and random draws come from global coordinates, so a block of a
/// larger volume, read with [`sweep_reach`] voxels of halo, gives the same core
/// as the whole volume. Outside the buffer is [`OUTSIDE`]: at a real volume
/// edge that is the definition, at a block seam it is inside the halo.
#[allow(clippy::too_many_arguments)]
pub fn label_energy_sweep_into(
    energy: &LabelEnergy,
    labels: ArrayView3<'_, u32>,
    p: ArrayView3<'_, f64>,
    f: Option<ArrayView3<'_, f64>>,
    means: &HashMap<u32, f64>,
    sweep: usize,
    temperature: f64,
    at: &Anchor,
    mut out: ArrayViewMut3<'_, u32>,
) -> Result<SweepStats> {
    check_inputs(&labels, &p, f.as_ref(), energy)?;
    ensure!(
        out.shape() == labels.shape(),
        "label energy: out has shape {:?}, labels {:?}",
        out.shape(),
        labels.shape()
    );
    ensure!(
        temperature.is_finite(),
        "label energy: temperature must be finite, got {temperature}"
    );
    out.assign(&labels);
    let acceptance = Acceptance::new(energy, temperature);
    let mut stats = SweepStats {
        sweep,
        temperature,
        energy_before: f64::NAN,
        energy_after: f64::NAN,
        ..SweepStats::default()
    };
    for &class in &SWEEP_ORDER {
        let (updates, class_stats) = class_updates(
            energy,
            out.view(),
            p,
            f,
            means,
            sweep,
            &acceptance,
            class,
            at,
        );
        for (position, label) in updates {
            out[position] = label;
        }
        stats.flips += class_stats.flips;
        stats.topology_rejections += class_stats.topology_rejections;
        stats.delta_energy += class_stats.delta_energy;
    }
    Ok(stats)
}

// ---------------------------------------------------------------- energy --

/// `E` split into its two terms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnergyTerms {
    pub boundary: f64,
    pub data: f64,
}

impl EnergyTerms {
    pub fn total(&self) -> f64 {
        self.boundary + self.data
    }
}

/// `E(labels)` over the whole array (the smoothing bias is not part of it).
///
/// Summed per plane along axis 0 and the planes in order, so the answer does
/// not depend on the thread count.
pub fn label_energy(
    energy: &LabelEnergy,
    labels: ArrayView3<'_, u32>,
    p: ArrayView3<'_, f64>,
    f: Option<ArrayView3<'_, f64>>,
    means: &HashMap<u32, f64>,
) -> Result<EnergyTerms> {
    check_inputs(&labels, &p, f.as_ref(), energy)?;
    let shape = shape3(&labels);
    let data_on = energy.data_on();
    let planes: Vec<(i64, i64)> = (0..shape[0])
        .into_par_iter()
        .map(|i| {
            let mut boundary = 0i64;
            let mut data = 0i64;
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    let label = labels[[i, j, k]];
                    let pv = quantise_p(p[[i, j, k]]);
                    // each undirected pair once: the offsets after the centre
                    for index in CENTRE + 1..NEIGHBOURHOOD {
                        let o = neighbour_offset(index);
                        let q = [i as isize + o[0], j as isize + o[1], k as isize + o[2]];
                        if (0..3).any(|axis| q[axis] < 0 || q[axis] >= shape[axis] as isize) {
                            continue;
                        }
                        let q = [q[0] as usize, q[1] as usize, q[2] as usize];
                        if labels[q] != label {
                            boundary += energy.edge(index, pv, quantise_p(p[q]));
                        }
                    }
                    if data_on {
                        if let (Some(f), Some(mu)) = (f.as_ref(), mean_f32(means, label)) {
                            let fv = f[[i, j, k]] as f32;
                            if fv.is_finite() {
                                data += i64::from(energy.data_units(fv, mu));
                            }
                        }
                    }
                }
            }
            (boundary, data)
        })
        .collect();
    let (boundary, data) = planes
        .into_iter()
        .fold((0i64, 0i64), |acc, (b, d)| (acc.0 + b, acc.1 + d));
    Ok(EnergyTerms {
        boundary: boundary as f64 * energy.quantum,
        data: data as f64 * energy.quantum,
    })
}

/// Mean of `f` per label: the global reduction that refreshes the data term's
/// table between sweeps. Deterministic regardless of thread count.
pub fn label_means(
    labels: ArrayView3<'_, u32>,
    f: ArrayView3<'_, f64>,
) -> Result<HashMap<u32, f64>> {
    ensure!(
        labels.shape() == f.shape(),
        "label means: labels have shape {:?}, f {:?}",
        labels.shape(),
        f.shape()
    );
    let shape = shape3(&labels);
    let planes: Vec<HashMap<u32, (f64, u64)>> = (0..shape[0])
        .into_par_iter()
        .map(|i| {
            let mut sums: HashMap<u32, (f64, u64)> = HashMap::new();
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    let entry = sums.entry(labels[[i, j, k]]).or_insert((0.0, 0));
                    entry.0 += f[[i, j, k]];
                    entry.1 += 1;
                }
            }
            sums
        })
        .collect();
    let mut total: HashMap<u32, (f64, u64)> = HashMap::new();
    for plane in planes {
        // plane order is fixed; within a plane each label has one entry
        for (label, (sum, count)) in plane {
            let entry = total.entry(label).or_insert((0.0, 0));
            entry.0 += sum;
            entry.1 += count;
        }
    }
    Ok(total
        .into_iter()
        .map(|(label, (sum, count))| (label, sum / count as f64))
        .collect())
}

// ---------------------------------------------------------------- driver --

/// Temperatures per sweep: `icm_before` sweeps at `T = 0`, then `annealing`
/// sweeps at `t0 · decay^s`, then `icm_after` at `T = 0` (ICM → annealing → ICM).
#[derive(Debug, Clone, PartialEq)]
pub struct AnnealingSchedule {
    pub icm_before: usize,
    pub t0: f64,
    pub decay: f64,
    pub annealing: usize,
    pub icm_after: usize,
}

impl AnnealingSchedule {
    pub fn new(
        icm_before: usize,
        t0: f64,
        decay: f64,
        annealing: usize,
        icm_after: usize,
    ) -> Result<Self> {
        ensure!(
            t0.is_finite() && t0 >= 0.0,
            "annealing: t0 must be finite and non-negative, got {t0}"
        );
        ensure!(
            decay.is_finite() && decay > 0.0 && decay <= 1.0,
            "annealing: decay must be in (0, 1], got {decay}"
        );
        Ok(Self {
            icm_before,
            t0,
            decay,
            annealing,
            icm_after,
        })
    }

    /// `sweeps` of ICM.
    pub fn icm(sweeps: usize) -> Self {
        Self {
            icm_before: sweeps,
            t0: 0.0,
            decay: 1.0,
            annealing: 0,
            icm_after: 0,
        }
    }

    pub fn temperatures(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.icm_before];
        let mut t = self.t0;
        for _ in 0..self.annealing {
            out.push(t);
            t *= self.decay;
        }
        out.extend(std::iter::repeat_n(0.0, self.icm_after));
        out
    }
}

/// Refine `labels` in memory over the whole `schedule`, refreshing the means
/// table before every sweep when the data term is on.
///
/// Returns the final labels and one [`SweepStats`] per sweep, with `E` before
/// and after (under that sweep's means).
pub fn refine_labels(
    energy: &LabelEnergy,
    labels: ArrayView3<'_, u32>,
    p: ArrayView3<'_, f64>,
    f: Option<ArrayView3<'_, f64>>,
    schedule: &AnnealingSchedule,
) -> Result<(Array3<u32>, Vec<SweepStats>)> {
    check_inputs(&labels, &p, f.as_ref(), energy)?;
    ensure!(
        p.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)),
        "label energy: p must lie in [0, 1]"
    );
    let shape = shape3(&labels);
    let at = Anchor::whole(shape);
    let mut current = labels.to_owned();
    let mut next = Array3::zeros(labels.raw_dim());
    let mut stats = Vec::new();
    for (sweep, temperature) in schedule.temperatures().into_iter().enumerate() {
        let means = match (energy.uses_data(), f) {
            (true, Some(f)) => label_means(current.view(), f)?,
            _ => HashMap::new(),
        };
        let before = label_energy(energy, current.view(), p, f, &means)?.total();
        let mut sweep_stats = label_energy_sweep_into(
            energy,
            current.view(),
            p,
            f,
            &means,
            sweep,
            temperature,
            &at,
            next.view_mut(),
        )?;
        std::mem::swap(&mut current, &mut next);
        sweep_stats.energy_before = before;
        sweep_stats.energy_after = label_energy(energy, current.view(), p, f, &means)?.total();
        stats.push(sweep_stats);
    }
    Ok((current, stats))
}

// --------------------------------------------------------------- BlockOp --

/// One sweep of the refinement as a block op over a `u32` label image, with
/// `p` (and `f`, when the data term is on) read as stored `f64` source images.
///
/// Reach is [`sweep_reach`] of [`SWEEP_ORDER`]. The source images are declared
/// at the same reach as the labels — `f` is only read at the centre and needs
/// one voxel less — so that the three buffers cover one region and the kernel
/// indexes them alike.
#[derive(Debug, Clone)]
pub struct LabelEnergySweepOp {
    energy: LabelEnergy,
    p: ImageId,
    f: Option<ImageId>,
    means: Arc<HashMap<u32, f64>>,
    sweep: usize,
    temperature: f64,
    cost: f64,
}

/// Placeholder per-voxel cost, in voxelwise-map units; most voxels are
/// interior and exit after the label gather, boundary voxels cost far more.
/// To be retaken through the cost harness before planner tuning.
pub const LABEL_ENERGY_SWEEP_COST: f64 = 60.0;

impl LabelEnergySweepOp {
    pub fn new(
        energy: LabelEnergy,
        p: impl Into<ImageId>,
        f: Option<ImageId>,
        means: Arc<HashMap<u32, f64>>,
        sweep: usize,
        temperature: f64,
    ) -> Result<Self> {
        ensure!(
            !(energy.uses_data() && f.is_none()),
            "label energy sweep: the data term is on and no appearance image was named"
        );
        ensure!(
            temperature.is_finite(),
            "label energy sweep: temperature must be finite, got {temperature}"
        );
        Ok(Self {
            energy,
            p: p.into(),
            f,
            means,
            sweep,
            temperature,
            cost: LABEL_ENERGY_SWEEP_COST,
        })
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }

    pub fn energy(&self) -> &LabelEnergy {
        &self.energy
    }

    /// The boundary-evidence image and the appearance image, if any.
    pub fn images(&self) -> (ImageId, Option<ImageId>) {
        (self.p, self.f)
    }

    pub fn means(&self) -> &Arc<HashMap<u32, f64>> {
        &self.means
    }

    pub fn sweep(&self) -> usize {
        self.sweep
    }

    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    fn reach3(&self) -> [usize; 3] {
        sweep_reach(&SWEEP_ORDER)
    }
}

impl BlockOp for LabelEnergySweepOp {
    fn name(&self) -> &'static str {
        "label-energy-sweep"
    }

    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        self.reach3()[axis]
    }

    fn reach_spec(&self, _volume: [usize; 3]) -> Reach {
        Reach::symmetric(self.reach3())
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        let reach = Reach::symmetric(self.reach3());
        let mut inputs = vec![SourceInput::new(self.p, reach.clone()).holding(Dtype::F64)];
        if let Some(f) = self.f {
            inputs.push(SourceInput::new(f, reach).holding(Dtype::F64));
        }
        inputs
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        dtype == Dtype::U32
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::U32
    }

    fn apply(&self, _input: &Voxels, _out: &mut Voxels, _at: &Anchor) -> Result<()> {
        Err(Error::InvalidArgument(format!(
            "{}: the boundary evidence comes from image {}, so this op has no answer from its \
             input alone. It is applied through `apply_with`.",
            self.name(),
            self.p.index()
        )))
    }

    fn apply_with(
        &self,
        input: &Voxels,
        sources: SourceInputs<'_>,
        out: &mut Voxels,
        at: &Anchor,
    ) -> Result<()> {
        let labels = input.view::<u32>()?;
        let p = sources.get(self.p)?.view::<f64>()?;
        let f = match self.f {
            Some(image) => Some(sources.get(image)?.view::<f64>()?),
            None => None,
        };
        let out = out.view_mut::<u32>()?;
        label_energy_sweep_into(
            &self.energy,
            labels,
            p,
            f,
            &self.means,
            self.sweep,
            self.temperature,
            at,
            out,
        )?;
        Ok(())
    }

    /// A constant labelling has no candidate anywhere, so it is a fixed point.
    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        Some(value)
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

/// Append one pixel phase per sweep, at the given temperatures, numbering the
/// sweeps from `first_sweep` (which seeds the random draws). The means table is
/// fixed across these phases; refreshing it is a reduction between runs.
pub fn append_label_energy_phases(
    builder: &mut PlanBuilder,
    energy: &LabelEnergy,
    p: impl Into<ImageId> + Copy,
    f: Option<ImageId>,
    means: Arc<HashMap<u32, f64>>,
    temperatures: &[f64],
    first_sweep: usize,
) -> Result<Vec<Phase>> {
    append_label_energy_phases_on(
        builder,
        energy,
        p,
        f,
        means,
        temperatures,
        first_sweep,
        crate::gpu::DevicePolicy::Cpu,
    )
}

/// [`append_label_energy_phases`] with each sweep run where `device` says:
/// wrapped in `gpu::label_energy::accelerated` in a build with a GPU runtime
/// (bit-identical to the CPU sweep), and on the CPU otherwise. In a build
/// without one, [`crate::gpu::DevicePolicy::Gpu`] is refused here rather than
/// at the first block.
#[allow(clippy::too_many_arguments)]
pub fn append_label_energy_phases_on(
    builder: &mut PlanBuilder,
    energy: &LabelEnergy,
    p: impl Into<ImageId> + Copy,
    f: Option<ImageId>,
    means: Arc<HashMap<u32, f64>>,
    temperatures: &[f64],
    first_sweep: usize,
    device: crate::gpu::DevicePolicy,
) -> Result<Vec<Phase>> {
    #[cfg(not(feature = "gpu"))]
    ensure!(
        device != crate::gpu::DevicePolicy::Gpu,
        "label energy: DevicePolicy::Gpu needs a build with a GPU runtime (feature gpu-cuda or \
         gpu-wgpu); use DevicePolicy::Auto to fall back to the CPU"
    );
    let mut phases = Vec::with_capacity(temperatures.len());
    for (offset, &temperature) in temperatures.iter().enumerate() {
        let op = LabelEnergySweepOp::new(
            energy.clone(),
            p,
            f,
            Arc::clone(&means),
            first_sweep + offset,
            temperature,
        )?;
        #[cfg(feature = "gpu")]
        if device != crate::gpu::DevicePolicy::Cpu {
            let fast = crate::gpu::label_energy::accelerated(op, device);
            phases.push(builder.pixels(Chain::op(fast))?);
            continue;
        }
        phases.push(builder.pixels(Chain::op(op))?);
    }
    Ok(phases)
}

// ----------------------------------------------------------------- tests --

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::skeleton::{connected_components, neighbour_index};
    use ndarray::{s, Array3};

    fn unit() -> LabelEnergy {
        LabelEnergy::new(LabelEnergyParams::new([1.0; 3], 1.0, 1.0, 0.01)).unwrap()
    }

    fn hash01(i: usize, j: usize, k: usize, salt: u64) -> f64 {
        counter_draws(salt, i as u64, j as u64, k as u64)[0]
    }

    fn components(labels: &Array3<u32>, label: u32, how: Adjacency) -> usize {
        let mask = labels.mapv(|l| l == label);
        connected_components(mask.view(), true, how)
    }

    fn complement_components(labels: &Array3<u32>, label: u32, how: Adjacency) -> usize {
        let mask = labels.mapv(|l| l == label);
        connected_components(mask.view(), false, how)
    }

    fn window(f: impl Fn([isize; 3]) -> u32) -> [u32; NEIGHBOURHOOD] {
        std::array::from_fn(|index| f(neighbour_offset(index)))
    }

    // ------------------------------------------------- (e) by hand --

    #[test]
    fn simple_label_flips_by_hand() {
        // flat interface: a below (axis 0 <= 0), b above
        let flat = window(|o| if o[0] <= 0 { 1 } else { 2 });
        assert!(
            is_simple_label_flip(&flat, 1, 2),
            "a voxel on a flat interface"
        );
        // a one-voxel line of a along axis 2 through b: removing the centre cuts it
        let line = window(|o| if o[0] == 0 && o[1] == 0 { 1 } else { 2 });
        assert!(
            !is_simple_label_flip(&line, 1, 2),
            "cutting a one-voxel line"
        );
        // an isolated voxel: its label would vanish
        let isolated = window(|o| if o == [0, 0, 0] { 1 } else { 2 });
        assert!(
            !is_simple_label_flip(&isolated, 1, 2),
            "deleting a whole label"
        );
        // b in two separate pieces of the window: adding the centre joins them
        let bridge = window(|o| {
            if o[2] != 0 && o[0] == 0 && o[1] == 0 {
                2
            } else {
                1
            }
        });
        assert!(
            !is_simple_label_flip(&bridge, 1, 2),
            "bridging two pieces of b"
        );
        // the centre is 1's only voxel even though its faces are 2 and the rest 3
        let cavity = window(|o| {
            if o == [0, 0, 0] {
                1
            } else if o.iter().map(|v| v.abs()).sum::<isize>() == 1 {
                2
            } else {
                3
            }
        });
        assert!(!is_simple_label_flip(&cavity, 1, 2), "label 1 would vanish");
        // three labels on a flat interface, a third label in a far corner
        let three = window(|o| {
            if o == [1, 1, 1] {
                3
            } else if o[0] <= 0 {
                1
            } else {
                2
            }
        });
        assert!(is_simple_label_flip(&three, 1, 2));
        assert!(
            !is_simple_label_flip(&three, 1, 3),
            "3 only touches at a corner"
        );
        // (6, 26) against (26, 6): b is two face neighbours of the centre that
        // share an edge, so they are 26- but not 6-adjacent to each other
        let diagonal = window(|o| {
            if o == [0, 1, 0] || o == [1, 0, 0] {
                2
            } else {
                1
            }
        });
        assert!(!is_simple_label_flip_with(&diagonal, 1, 2, Adjacency::Six));
        // under (26, 6) adding the centre to b is fine (the two are already one
        // 26-component), but removing it from 1 joins two 6-pieces of 1's
        // complement, so the flip is refused for the other set
        assert!(removable(&diagonal, 2, Adjacency::TwentySix));
        assert!(!removable(&diagonal, 1, Adjacency::TwentySix));
        assert!(is_simple_label_flip_with(&flat, 1, 2, Adjacency::TwentySix));
        assert!(!is_simple_label_flip(&flat, 1, 1));
    }

    #[test]
    fn the_simple_point_table_is_the_flood_fill_definition() {
        let table = simple_point_table();
        let step = if cfg!(debug_assertions) { 997 } else { 1 };
        let mismatches = (0..1u32 << 26)
            .into_par_iter()
            .step_by(step)
            .filter(|&config| {
                let mut mask = [false; NEIGHBOURHOOD];
                for (k, slot) in mask.iter_mut().enumerate() {
                    *slot = k == CENTRE || (config >> (k - usize::from(k > CENTRE))) & 1 == 1;
                }
                table_says(table, config) != is_simple_point(&mask)
            })
            .count();
        assert_eq!(mismatches, 0);
        // and through the label windows, both connectivities
        for salt in 0..20_000u64 {
            let window: [u32; NEIGHBOURHOOD] = std::array::from_fn(|k| {
                let r = counter_words(salt, 1, 2, k as u64)[0];
                if r % 11 == 0 {
                    OUTSIDE
                } else {
                    (r % 3) as u32
                }
            });
            for x in 0..3 {
                for how in [Adjacency::Six, Adjacency::TwentySix] {
                    assert_eq!(
                        removable(&window, x, how),
                        removable_by_flood(&window, x, how),
                        "{window:?} {x} {how:?}"
                    );
                }
            }
        }
    }

    // ---------------------------------------------------- weights --

    #[test]
    fn geometric_weights_measure_area_better_than_a_face_count() {
        for spacing in [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]] {
            let energy = LabelEnergy::new(LabelEnergyParams::new(spacing, 1.0, 1.0, 0.0)).unwrap();
            let r = 9.0;
            let n = [(2.0 * r / spacing[0]) as usize + 6, 24, 24];
            let centre = [n[0] as f64 / 2.0, 12.0, 12.0];
            let labels = Array3::from_shape_fn(n, |(i, j, k)| {
                let d = [
                    (i as f64 + 0.5 - centre[0]) * spacing[0],
                    (j as f64 + 0.5 - centre[1]) * spacing[1],
                    (k as f64 + 0.5 - centre[2]) * spacing[2],
                ];
                u32::from(d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= r * r) + 1
            });
            let p = Array3::zeros(n);
            let area = label_energy(&energy, labels.view(), p.view(), None, &HashMap::new())
                .unwrap()
                .boundary;
            let exact = 4.0 * std::f64::consts::PI * r * r;
            let error = (area - exact).abs() / exact;
            assert!(error < 0.06, "spacing {spacing:?}: {area} against {exact}");
        }
        // the weights are symmetric and sum, over one face's crossings, to ~1
        let w = unit().weights;
        for k in 0..NEIGHBOURHOOD {
            assert_eq!(w[k], w[NEIGHBOURHOOD - 1 - k]);
        }
        let face = w[neighbour_index(1, 0, 0)];
        let edge = w[neighbour_index(1, 1, 0)];
        let corner = w[neighbour_index(1, 1, 1)];
        let plane = face + 4.0 * edge + 4.0 * corner;
        // 13 directions are a coarse quadrature of Cauchy-Crofton: an axis plane
        // comes out about 7% light, against 0% for a face count (which is
        // then 41% heavy on a diagonal plane)
        assert!(
            (plane - 1.0).abs() < 0.1,
            "an axis plane costs {plane} per unit area"
        );
    }

    // ------------------------------------------------------ (a) ICM --

    fn noisy_cube(n: usize) -> Array3<u32> {
        let (lo, hi) = (n / 4, 3 * n / 4);
        Array3::from_shape_fn((n, n, n), |(i, j, k)| {
            let inside = [i, j, k].iter().all(|&c| (lo..hi).contains(&c));
            let near = [i, j, k].iter().all(|&c| (lo - 1..=hi).contains(&c))
                && [i, j, k].iter().any(|&c| c <= lo || c >= hi - 1);
            let flip = near && hash01(i, j, k, 7) < 0.35;
            if inside != flip {
                2
            } else {
                1
            }
        })
    }

    fn rough_pairs(labels: &Array3<u32>) -> usize {
        let sh = labels.shape();
        let mut count = 0;
        for ((i, j, k), &l) in labels.indexed_iter() {
            if i + 1 < sh[0] && labels[[i + 1, j, k]] != l {
                count += 1;
            }
            if j + 1 < sh[1] && labels[[i, j + 1, k]] != l {
                count += 1;
            }
            if k + 1 < sh[2] && labels[[i, j, k + 1]] != l {
                count += 1;
            }
        }
        count
    }

    #[test]
    fn icm_smooths_a_noisy_cube_and_never_raises_the_energy() {
        let labels = noisy_cube(24);
        let p = Array3::zeros(labels.raw_dim());
        let energy =
            LabelEnergy::new(LabelEnergyParams::new([1.0; 3], 1.0, 1.0, 0.0).with_topology(None))
                .unwrap();
        let (out, stats) = refine_labels(
            &energy,
            labels.view(),
            p.view(),
            None,
            &AnnealingSchedule::icm(8),
        )
        .unwrap();
        assert!(stats[0].flips > 0);
        for s in &stats {
            assert!(s.energy_after <= s.energy_before + 1e-9, "{s:?}");
            assert!(s.delta_energy <= 0.0);
            assert!(
                (s.energy_after - s.energy_before - s.delta_energy).abs() < 1e-6,
                "the kernel's ΔE agrees with the global energy: {s:?}"
            );
        }
        for pair in stats.windows(2) {
            assert!(pair[1].energy_before <= pair[0].energy_after + 1e-9);
        }
        assert!(stats.last().unwrap().energy_after < 0.8 * stats[0].energy_before);
        assert!(rough_pairs(&out) < rough_pairs(&labels) * 3 / 4);
    }

    #[test]
    fn the_kernel_delta_matches_the_energy_with_every_term_and_metropolis() {
        let n = [12, 13, 14];
        let labels = Array3::from_shape_fn(n, |(i, j, k)| 1 + ((i / 4 + j / 5 + k / 6) % 3) as u32);
        let p = Array3::from_shape_fn(n, |(i, j, k)| hash01(i, j, k, 1));
        let f = Array3::from_shape_fn(n, |(i, j, k)| {
            f64::from(labels[[i, j, k]]) + hash01(i, j, k, 2)
        });
        let energy = LabelEnergy::new(
            LabelEnergyParams::new([1.5, 1.0, 1.0], 2.0, 2.0, 0.05).with_data(0.5, 1.0),
        )
        .unwrap();
        let means = label_means(labels.view(), f.view()).unwrap();
        let mut out = Array3::zeros(n);
        let at = Anchor::whole(n);
        let before = label_energy(&energy, labels.view(), p.view(), Some(f.view()), &means)
            .unwrap()
            .total();
        let stats = label_energy_sweep_into(
            &energy,
            labels.view(),
            p.view(),
            Some(f.view()),
            &means,
            3,
            1.0,
            &at,
            out.view_mut(),
        )
        .unwrap();
        let after = label_energy(&energy, out.view(), p.view(), Some(f.view()), &means)
            .unwrap()
            .total();
        assert!(stats.flips > 10, "{stats:?}");
        assert!(
            (after - before - stats.delta_energy).abs() < 1e-6,
            "{before} {after} {stats:?}"
        );
    }

    // --------------------------------------------- smoothing bias --

    #[test]
    fn the_smoothing_bias_removes_bumps_and_leaves_flat_faces_and_wide_tubes() {
        let energy = LabelEnergy::new(
            LabelEnergyParams::new([1.0; 3], 0.0, 1.0, 0.0)
                .with_smoothing(1.0, 10)
                .with_topology(None),
        )
        .unwrap();
        let none = HashMap::new();
        let decide_on = |labels: [u32; NEIGHBOURHOOD]| {
            let n = Neighbourhood {
                labels,
                p: [0; NEIGHBOURHOOD],
                f: f32::NAN,
            };
            decide(&energy, &n, &none, &Acceptance::new(&energy, 0.0), [0; 2]).label
        };
        // a bump of 1 on a flat face of 1 against 2: the centre has 9 like neighbours
        let bump = window(|o| if o[0] < 0 || o == [0, 0, 0] { 1 } else { 2 });
        assert_eq!(decide_on(bump), 2);
        // the flat face itself is left alone
        let flat = window(|o| if o[0] <= 0 { 1 } else { 2 });
        assert_eq!(decide_on(flat), 1);
        // the corner of a 3x3 square tube along axis 2 (cells 0..3 in axes 0, 1,
        // centre at (0, 0)): 11 like neighbours
        let corner = window(|o| if o[0] >= 0 && o[1] >= 0 { 1 } else { 2 });
        assert_eq!(decide_on(corner), 1);
    }

    // ------------------------------------------------- (b) tubes --

    /// Square rings in planes of constant axis 0: a one-voxel line (label 2)
    /// and a tube with a 3x3 cross-section (label 3), in label 1. Rings rather
    /// than straight tubes because the end of a tube at the volume face is an
    /// end, and an end may retract (see the module header).
    fn rings(n: [usize; 3]) -> Array3<u32> {
        Array3::from_shape_fn(n, |(i, j, k)| {
            let on_square = |lo: usize, hi: usize| {
                ((j == lo || j == hi) && (lo..=hi).contains(&k))
                    || ((k == lo || k == hi) && (lo..=hi).contains(&j))
            };
            let band = |lo: usize, hi: usize, width: usize| {
                let inside = |a: usize, b: usize| (a..=b).contains(&j) && (a..=b).contains(&k);
                inside(lo, hi) && !inside(lo + width, hi - width)
            };
            if i == 2 && on_square(3, 26) {
                2
            } else if (6..9).contains(&i) && band(5, 24, 3) {
                3
            } else {
                1
            }
        })
    }

    /// 50 sweeps (ICM, low-temperature annealing, ICM) over [`rings`] with
    /// membrane evidence on every voxel next to an interface; the per-sweep
    /// component counts of each label and of its complement, and the result.
    fn run_rings(topology: Option<Adjacency>) -> (Array3<u32>, Vec<[(usize, usize); 3]>, u64) {
        let n = [12, 30, 30];
        let labels = rings(n);
        let p = Array3::from_shape_fn(n, |(i, j, k)| {
            let l = labels[[i, j, k]];
            let on = (0..NEIGHBOURHOOD).any(|index| {
                let o = neighbour_offset(index);
                let q = [i as isize + o[0], j as isize + o[1], k as isize + o[2]];
                (0..3).all(|a| q[a] >= 0 && (q[a] as usize) < n[a])
                    && labels[[q[0] as usize, q[1] as usize, q[2] as usize]] != l
            });
            if on {
                0.95
            } else {
                0.05 * hash01(i, j, k, 3)
            }
        });
        let energy = LabelEnergy::new(
            LabelEnergyParams::new([1.0; 3], 1.0, 1.0, 0.02)
                .with_smoothing(0.3, 10)
                .with_topology(topology)
                .with_seed(11),
        )
        .unwrap();
        let at = Anchor::whole(n);
        let mut current = labels.clone();
        let mut next = Array3::zeros(n);
        let schedule = AnnealingSchedule::new(5, 0.02, 0.93, 40, 5).unwrap();
        assert_eq!(schedule.temperatures().len(), 50);
        let mut counts = Vec::new();
        let mut flips = 0;
        for (sweep, t) in schedule.temperatures().into_iter().enumerate() {
            let stats = label_energy_sweep_into(
                &energy,
                current.view(),
                p.view(),
                None,
                &HashMap::new(),
                sweep,
                t,
                &at,
                next.view_mut(),
            )
            .unwrap();
            flips += stats.flips;
            std::mem::swap(&mut current, &mut next);
            counts.push(std::array::from_fn(|index| {
                let label = index as u32 + 1;
                (
                    components(&current, label, Adjacency::Six),
                    complement_components(&current, label, Adjacency::TwentySix),
                )
            }));
        }
        (current, counts, flips)
    }

    #[test]
    fn thin_tubes_on_membrane_survive_fifty_sweeps() {
        let labels = rings([12, 30, 30]);
        let initial: [(usize, usize); 3] = std::array::from_fn(|index| {
            let label = index as u32 + 1;
            (
                components(&labels, label, Adjacency::Six),
                complement_components(&labels, label, Adjacency::TwentySix),
            )
        });
        let (out, counts, flips) = run_rings(Some(Adjacency::Six));
        assert!(flips > 0, "the refinement moved something");
        for (sweep, count) in counts.iter().enumerate() {
            assert_eq!(
                *count, initial,
                "sweep {sweep}: a label broke, vanished or changed holes"
            );
        }
        let wide = labels.iter().filter(|&&l| l == 3).count();
        let kept = out.iter().filter(|&&l| l == 3).count();
        assert!(
            kept * 10 >= wide * 9,
            "the 3x3 tube kept its volume: {kept} of {wide}"
        );

        // the constraint is what keeps the one-voxel ring: without it, the same
        // run erases or breaks it
        let (_, counts, _) = run_rings(None);
        assert_ne!(
            counts.last().unwrap()[1],
            initial[1],
            "the line survived unprotected"
        );
    }

    // ----------------------------------------- (c) topology holds --

    #[test]
    fn with_the_constraint_no_label_vanishes_splits_or_changes_its_holes() {
        let n = [14, 16, 18];
        let seeds = [[3, 4, 4], [10, 3, 12], [7, 12, 6], [4, 12, 15], [11, 9, 3]];
        let labels = Array3::from_shape_fn(n, |(i, j, k)| {
            let mut best = (usize::MAX, 0u32);
            for (index, s) in seeds.iter().enumerate() {
                let d = (i as isize - s[0]).pow(2)
                    + (j as isize - s[1]).pow(2)
                    + (k as isize - s[2]).pow(2);
                let jitter = (hash01(i, j, k, 5) * 6.0) as isize;
                best = best.min(((d + jitter) as usize, index as u32 + 1));
            }
            best.1
        });
        let p = Array3::from_shape_fn(n, |(i, j, k)| hash01(i, j, k, 9));
        let count = |labels: &Array3<u32>| -> Vec<(usize, usize)> {
            (1..=seeds.len() as u32)
                .map(|l| {
                    (
                        components(labels, l, Adjacency::Six),
                        complement_components(labels, l, Adjacency::TwentySix),
                    )
                })
                .collect()
        };
        let initial = count(&labels);
        let energy =
            LabelEnergy::new(LabelEnergyParams::new([1.0; 3], 1.0, 1.0, 0.05).with_seed(3))
                .unwrap();
        let at = Anchor::whole(n);
        let mut current = labels.clone();
        let mut next = Array3::zeros(n);
        let mut rejected = 0;
        for sweep in 0..12 {
            // hot: most proposals are accepted and only the constraint holds
            let stats = label_energy_sweep_into(
                &energy,
                current.view(),
                p.view(),
                None,
                &HashMap::new(),
                sweep,
                5.0,
                &at,
                next.view_mut(),
            )
            .unwrap();
            rejected += stats.topology_rejections;
            std::mem::swap(&mut current, &mut next);
            assert_eq!(count(&current), initial, "sweep {sweep}");
        }
        assert!(rejected > 0, "the constraint was exercised");
        assert_ne!(current, labels);
    }

    // -------------------------------------- (d) decomposition-free --

    fn fixture() -> (Array3<u32>, Array3<f64>, Array3<f64>, LabelEnergy) {
        let n = [17, 20, 23];
        let labels = Array3::from_shape_fn(n, |(i, j, k)| {
            1 + ((i / 5 + 2 * (j / 6) + (k / 7)) % 4) as u32
        });
        let p = Array3::from_shape_fn(n, |(i, j, k)| hash01(i, j, k, 21));
        let f = Array3::from_shape_fn(n, |(i, j, k)| {
            f64::from(labels[[i, j, k]]) * 0.3 + hash01(i, j, k, 22)
        });
        let energy = LabelEnergy::new(
            LabelEnergyParams::new([2.0, 1.0, 1.0], 1.0, 2.0, 0.05)
                .with_data(0.3, 0.5)
                .with_smoothing(0.2, 10)
                .with_seed(99),
        )
        .unwrap();
        (labels, p, f, energy)
    }

    fn whole_sweep(
        energy: &LabelEnergy,
        labels: &Array3<u32>,
        p: &Array3<f64>,
        f: &Array3<f64>,
        means: &HashMap<u32, f64>,
        sweep: usize,
        t: f64,
    ) -> Array3<u32> {
        let shape = [labels.shape()[0], labels.shape()[1], labels.shape()[2]];
        let mut out = Array3::zeros(shape);
        label_energy_sweep_into(
            energy,
            labels.view(),
            p.view(),
            Some(f.view()),
            means,
            sweep,
            t,
            &Anchor::whole(shape),
            out.view_mut(),
        )
        .unwrap();
        out
    }

    #[test]
    fn blockwise_and_any_thread_count_give_the_whole_volume_answer() {
        let (labels, p, f, energy) = fixture();
        let shape = [17, 20, 23];
        let means = label_means(labels.view(), f.view()).unwrap();
        for (sweep, t) in [(0usize, 0.0), (1, 0.8), (2, 3.0)] {
            let reference = whole_sweep(&energy, &labels, &p, &f, &means, sweep, t);
            assert_ne!(reference, labels, "sweep {sweep} moved something");
            for threads in [1, 3, 8] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let got = pool.install(|| whole_sweep(&energy, &labels, &p, &f, &means, sweep, t));
                assert_eq!(got, reference, "{threads} threads");
            }
            // blockwise through the BlockOp, each block read with exactly the
            // declared reach of halo
            let op = LabelEnergySweepOp::new(
                energy.clone(),
                1usize,
                Some(ImageId::from(2usize)),
                Arc::new(means.clone()),
                sweep,
                t,
            )
            .unwrap();
            let reach: [usize; 3] = std::array::from_fn(|axis| op.reach(axis, shape[axis]));
            for core in [[5, 6, 7], [4, 9, 4], [17, 3, 23]] {
                let mut stitched = Array3::<u32>::zeros(shape);
                for i0 in (0..shape[0]).step_by(core[0]) {
                    for j0 in (0..shape[1]).step_by(core[1]) {
                        for k0 in (0..shape[2]).step_by(core[2]) {
                            let start = [i0, j0, k0];
                            let lo: [usize; 3] =
                                std::array::from_fn(|a| start[a].saturating_sub(reach[a]));
                            let end: [usize; 3] =
                                std::array::from_fn(|a| (start[a] + core[a]).min(shape[a]));
                            let hi: [usize; 3] =
                                std::array::from_fn(|a| (end[a] + reach[a]).min(shape[a]));
                            let cut = s![lo[0]..hi[0], lo[1]..hi[1], lo[2]..hi[2]];
                            let input: Voxels = labels.slice(cut).to_owned().into();
                            let pv: Voxels = p.slice(cut).to_owned().into();
                            let fv: Voxels = f.slice(cut).to_owned().into();
                            let entries =
                                [(ImageId::from(1usize), &pv), (ImageId::from(2usize), &fv)];
                            let mut out = Voxels::zeros(Dtype::U32, input.shape()).unwrap();
                            op.apply_with(
                                &input,
                                SourceInputs::new(&entries),
                                &mut out,
                                &Anchor::new(lo, shape),
                            )
                            .unwrap();
                            let out = out.view::<u32>().unwrap();
                            stitched
                                .slice_mut(s![start[0]..end[0], start[1]..end[1], start[2]..end[2]])
                                .assign(&out.slice(s![
                                    start[0] - lo[0]..end[0] - lo[0],
                                    start[1] - lo[1]..end[1] - lo[1],
                                    start[2] - lo[2]..end[2] - lo[2]
                                ]));
                        }
                    }
                }
                assert_eq!(stitched, reference, "sweep {sweep}, core {core:?}");
            }
        }
    }

    /// The derived reach is the exact influence radius: perturb one input voxel
    /// and propagate "may have changed" through the class order.
    #[test]
    fn the_sweep_reach_is_the_influence_radius_of_the_class_order() {
        for order in [SWEEP_ORDER, [0, 1, 2, 3, 4, 5, 6, 7]] {
            let size = 41usize;
            let c = size / 2;
            // perturbations of each parity: the reach is the max over them
            let mut extent = [0usize; 3];
            for origin_class in 0..8usize {
                let o = [
                    c + ((origin_class >> 2) & 1),
                    c + ((origin_class >> 1) & 1),
                    c + (origin_class & 1),
                ];
                let mut dirty = Array3::from_elem((size, size, size), false);
                dirty[o] = true;
                for &class in &order {
                    let before = dirty.clone();
                    for ((i, j, k), d) in dirty.indexed_iter_mut() {
                        if Subfield::of([i, j, k]).index() != class
                            || i == 0
                            || j == 0
                            || k == 0
                            || i + 1 == size
                            || j + 1 == size
                            || k + 1 == size
                        {
                            continue;
                        }
                        let near = before
                            .slice(s![i - 1..=i + 1, j - 1..=j + 1, k - 1..=k + 1])
                            .iter()
                            .any(|&v| v);
                        *d |= near;
                    }
                }
                for ((i, j, k), &d) in dirty.indexed_iter() {
                    if d {
                        let p = [i, j, k];
                        for a in 0..3 {
                            extent[a] = extent[a].max(p[a].abs_diff(o[a]));
                        }
                    }
                }
            }
            assert_eq!(extent, sweep_reach(&order), "order {order:?}");
        }
        assert_eq!(sweep_reach(&SWEEP_ORDER), [3, 4, 3]);
    }

    #[test]
    fn inputs_are_checked() {
        let energy = unit();
        let labels = Array3::from_elem((3, 3, 3), OUTSIDE);
        let p = Array3::zeros((3, 3, 3));
        assert!(refine_labels(
            &energy,
            labels.view(),
            p.view(),
            None,
            &AnnealingSchedule::icm(1)
        )
        .is_err());
        let with_data =
            LabelEnergy::new(LabelEnergyParams::new([1.0; 3], 1.0, 1.0, 0.0).with_data(1.0, 1.0))
                .unwrap();
        let labels = Array3::from_elem((3, 3, 3), 1u32);
        assert!(refine_labels(
            &with_data,
            labels.view(),
            p.view(),
            None,
            &AnnealingSchedule::icm(1)
        )
        .is_err());
        assert!(AnnealingSchedule::new(0, 1.0, 1.5, 3, 0).is_err());
        assert_eq!(
            AnnealingSchedule::new(1, 2.0, 0.5, 2, 1)
                .unwrap()
                .temperatures(),
            vec![0.0, 2.0, 1.0, 0.0]
        );
    }
}
