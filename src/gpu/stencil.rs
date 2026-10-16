// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// A 3x3x3 stencil on labels: one colour-class update of a weighted
// 26-neighbourhood majority vote. The template for a checkerboard label-energy
// sweep on the GPU.
//
// The shape it fixes
// ------------------
// `u32` labels and an `f32` field in, `u32` labels out, reach one voxel on
// every axis, and only the voxels of one **colour class** rewritten — the rest
// are copied. A class is a set of voxels no two of which are neighbours, so
// every voxel of it can be updated at once from the previous labels and the
// result equals a sequential sweep over the class in any order. That is the
// property that makes a Gibbs / ICM sweep parallel, on a GPU or on a block
// lattice. Two colourings:
//
// * [`Colouring::Octant`] — `(z&1, y&1, x&1)`, eight classes. Independent for
//   the 26-neighbourhood, which is what this vote reads.
// * [`Colouring::Checkerboard`] — `(z+y+x)&1`, two classes. Independent only
//   for the 6-neighbourhood; offered because a 6-connected energy uses it, and
//   with a 26-neighbour vote it is a (deterministic, well-defined) Jacobi-style
//   update inside the class, not a Gibbs sweep.
//
// The colour and the random numbers are functions of **global** coordinates
// (`Anchor::offset` + local), so the update is block-cut invariant: a
// decomposed run equals a whole-volume one. The tie-breaking "RNG" is a
// counter-based hash — `lowbias32` chained over (seed, sweep, z, y, x, label) —
// so it needs no state, is the same on CPU and GPU, and differs per sweep.
//
// The update (the part a real energy replaces)
// --------------------------------------------
// For a voxel `v` of the class, the candidates are the labels in its 3x3x3
// neighbourhood (itself included; positions outside the array are absent). A
// candidate label `L` scores `sum field[u]` over the neighbourhood positions
// `u` holding `L`, so `field` is a non-negative per-voxel vote weight (all ones
// is a plain majority). The highest score wins; a tie goes to the smaller hash.
// Both loops run in the fixed order `dz, dy, dx` ascending.
//
// Parity: **exact**. The score is a sum of `f32` field values with no
// multiplication, accumulated in the same order on both sides, so there is no
// contraction for the GPU compiler to make and the labels agree bit for bit
// (CUDA builds with fast-math, which flushes subnormal weights to zero; a field
// of subnormals is outside the contract).

use ndarray::{ArrayView3, ArrayViewMut3};

use crate::dtype::Dtype;
use crate::error::{ensure, Error, Result};
use crate::op::{Anchor, Combine};
use crate::voxels::Voxels;

/// Which voxels form a colour class. See the module header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Colouring {
    /// `(z + y + x) & 1`: two classes.
    Checkerboard,
    /// `(z & 1) << 2 | (y & 1) << 1 | (x & 1)`: eight classes.
    Octant,
}

impl Colouring {
    pub fn classes(self) -> u32 {
        match self {
            Colouring::Checkerboard => 2,
            Colouring::Octant => 8,
        }
    }

    /// The class of the voxel at global position `[z, y, x]`.
    pub fn class_of(self, [z, y, x]: [u32; 3]) -> u32 {
        match self {
            Colouring::Checkerboard => (z.wrapping_add(y).wrapping_add(x)) & 1,
            Colouring::Octant => ((z & 1) << 2) | ((y & 1) << 1) | (x & 1),
        }
    }
}

/// One sweep step: which class is rewritten, and the RNG counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VoteSweep {
    pub colouring: Colouring,
    pub class: u32,
    pub seed: u32,
    pub sweep: u32,
}

impl VoteSweep {
    pub fn new(colouring: Colouring, class: u32, seed: u32, sweep: u32) -> Result<Self> {
        ensure!(
            class < colouring.classes(),
            "class {class} does not exist in a {colouring:?} colouring of {} classes",
            colouring.classes()
        );
        Ok(Self {
            colouring,
            class,
            seed,
            sweep,
        })
    }
}

/// Chris Wellons' `lowbias32`: a stateless 32-bit mixer.
pub fn mix(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

/// The tie-break key of `label` at global `[z, y, x]` in this sweep.
pub fn tie_hash(seed: u32, sweep: u32, [z, y, x]: [u32; 3], label: u32) -> u32 {
    let mut h = mix(seed);
    h = mix(h ^ sweep);
    h = mix(h ^ z);
    h = mix(h ^ y);
    h = mix(h ^ x);
    mix(h ^ label)
}

/// The CPU reference: one class update of `labels` into `out`.
///
/// `origin` is the global position of `labels[0, 0, 0]` (`Anchor::offset`).
/// Positions outside the arrays are absent from every neighbourhood, so under
/// decomposition a block needs a halo of one voxel and its core is exact.
pub fn vote_sweep_into(
    labels: ArrayView3<'_, u32>,
    field: ArrayView3<'_, f32>,
    sweep: &VoteSweep,
    origin: [usize; 3],
    mut out: ArrayViewMut3<'_, u32>,
) -> Result<()> {
    ensure!(
        labels.shape() == field.shape() && labels.shape() == out.shape(),
        "vote sweep: labels {:?}, field {:?} and output {:?} must have one shape",
        labels.shape(),
        field.shape(),
        out.shape()
    );
    let n = [labels.shape()[0], labels.shape()[1], labels.shape()[2]];
    for i0 in 0..n[0] {
        for i1 in 0..n[1] {
            for i2 in 0..n[2] {
                let global = [
                    (origin[0] + i0) as u32,
                    (origin[1] + i1) as u32,
                    (origin[2] + i2) as u32,
                ];
                let current = labels[[i0, i1, i2]];
                if sweep.colouring.class_of(global) != sweep.class {
                    out[[i0, i1, i2]] = current;
                    continue;
                }
                let neighbour = |k: usize| -> Option<[usize; 3]> {
                    let d = [k / 9, (k / 3) % 3, k % 3];
                    let at = [i0 + d[0], i1 + d[1], i2 + d[2]];
                    (at.iter().all(|&a| a >= 1) && (0..3).all(|a| at[a] - 1 < n[a]))
                        .then(|| [at[0] - 1, at[1] - 1, at[2] - 1])
                };
                let mut best: Option<(f32, u32, u32)> = None;
                for i in 0..27 {
                    let Some(u) = neighbour(i) else { continue };
                    let candidate = labels[u];
                    let mut score = 0.0f32;
                    for j in 0..27 {
                        if let Some(w) = neighbour(j) {
                            if labels[w] == candidate {
                                score += field[w];
                            }
                        }
                    }
                    let hash = tie_hash(sweep.seed, sweep.sweep, global, candidate);
                    let better = match best {
                        None => true,
                        Some((s, h, _)) => score > s || (score == s && hash < h),
                    };
                    if better {
                        best = Some((score, hash, candidate));
                    }
                }
                out[[i0, i1, i2]] = best.map_or(current, |(_, _, label)| label);
            }
        }
    }
    Ok(())
}

/// [`vote_sweep_into`] as a fan-in: branch 0 is the labels (`u32`), branch 1
/// the field (`f32`). A [`Combine`] because the op reads two arrays.
#[derive(Debug, Clone)]
pub struct VoteSweepOp {
    name: &'static str,
    sweep: VoteSweep,
}

impl VoteSweepOp {
    pub fn new(name: &'static str, sweep: VoteSweep) -> Self {
        Self { name, sweep }
    }

    pub fn sweep(&self) -> &VoteSweep {
        &self.sweep
    }
}

impl Combine for VoteSweepOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// One voxel: the 3x3x3 neighbourhood. Tight — the corner neighbour's label
    /// is a candidate.
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        1
    }

    fn accepts(&self, inputs: &[Dtype]) -> bool {
        inputs == [Dtype::U32, Dtype::F32]
    }

    fn produces(&self, _inputs: &[Dtype]) -> Dtype {
        Dtype::U32
    }

    fn output_shape(&self, inputs: &[[usize; 3]]) -> Result<[usize; 3]> {
        match inputs {
            [labels, field] if labels == field => Ok(*labels),
            _ => Err(Error::InvalidArgument(format!(
                "{}: wants a label block and a field block of one shape, got {inputs:?}",
                self.name
            ))),
        }
    }

    fn apply(&self, inputs: &[&Voxels], out: &mut Voxels, at: &Anchor) -> Result<()> {
        let [labels, field] = inputs else {
            return Err(Error::InvalidArgument(format!(
                "{}: takes two inputs",
                self.name
            )));
        };
        vote_sweep_into(
            labels.view::<u32>()?,
            field.view::<f32>()?,
            &self.sweep,
            at.offset,
            out.view_mut::<u32>()?,
        )
    }

    /// Constant labels stay that label: it is the only candidate.
    fn constant_maps_to(&self, values: &[f64]) -> Option<f64> {
        values.first().copied()
    }

    /// Measured with `benchmark_vote_sweep` in `src/gpu/tests.rs` (6400 and
    /// 10400 in two runs on a loaded machine): the CPU reference is a
    /// straightforward `27 x 27` loop per voxel of the class — it is the
    /// definition, not a tuned kernel — and this is the average over a sweep of
    /// one checkerboard class.
    fn cost_per_voxel(&self, _branches: usize) -> f64 {
        VOTE_SWEEP_CPU_COST
    }
}

pub(crate) const VOTE_SWEEP_CPU_COST: f64 = 8000.0;

// ------------------------------------------------------------------- GPU --

#[cfg(feature = "gpu")]
pub use gpu::{accelerated, GpuVoteSweep};

#[cfg(feature = "gpu")]
mod gpu {
    use std::sync::Arc;

    use cubecl::prelude::*;

    use super::{Colouring, VoteSweep, VoteSweepOp};
    use crate::dtype::Dtype;
    use crate::error::{Error, Result};
    use crate::gpu::device::{check_indexable, device, launch_shape, Rt};
    use crate::gpu::{admit, Accelerated, DevicePolicy, GpuKernel};
    use crate::op::Anchor;
    use crate::voxels::Voxels;

    #[cube]
    fn mix(x: u32) -> u32 {
        let mut x = x;
        x ^= x >> 16u32;
        x *= 0x7feb_352du32;
        x ^= x >> 15u32;
        x *= 0x846c_a68bu32;
        x ^= x >> 16u32;
        x
    }

    /// Is neighbour `k` (0..27) of local `i` inside an axis of `n`, and where?
    /// Returned as `(inside, index)`; `index` is meaningless when outside.
    #[cube]
    fn along(i: u32, d: u32, n: u32) -> (bool, u32) {
        let shifted = i + d;
        let inside = shifted >= 1u32 && shifted - 1u32 < n;
        let mut index = 0u32;
        if inside {
            index = shifted - 1u32;
        }
        (inside, index)
    }

    #[cube(launch_unchecked)]
    #[allow(clippy::too_many_arguments)]
    fn vote_sweep_kernel(
        labels: &[u32],
        field: &[f32],
        out: &mut [u32],
        n0: u32,
        n1: u32,
        n2: u32,
        o0: u32,
        o1: u32,
        o2: u32,
        seed: u32,
        sweep: u32,
        class: u32,
        #[comptime] octant: bool,
    ) {
        let total = (n0 * n1 * n2) as usize;
        if ABSOLUTE_POS < total {
            let p = ABSOLUTE_POS as u32;
            let i2 = p % n2;
            let i1 = (p / n2) % n1;
            let i0 = p / (n2 * n1);
            let g0 = o0 + i0;
            let g1 = o1 + i1;
            let g2 = o2 + i2;
            let mut mine = (g0 + g1 + g2) & 1u32;
            if octant {
                mine = ((g0 & 1u32) << 2u32) | ((g1 & 1u32) << 1u32) | (g2 & 1u32);
            }
            let current = labels[ABSOLUTE_POS];
            let mut best_label = current;
            if mine == class {
                let prefix = mix(mix(mix(mix(mix(seed) ^ sweep) ^ g0) ^ g1) ^ g2);
                let mut found = false;
                let mut best_score = 0.0f32;
                let mut best_hash = 0u32;
                for i in 0..27u32 {
                    let (in0, a0) = along(i0, i / 9u32, n0);
                    let (in1, a1) = along(i1, (i / 3u32) % 3u32, n1);
                    let (in2, a2) = along(i2, i % 3u32, n2);
                    if in0 && in1 && in2 {
                        let candidate = labels[((a0 * n1 + a1) * n2 + a2) as usize];
                        let mut score = 0.0f32;
                        for j in 0..27u32 {
                            let (jn0, b0) = along(i0, j / 9u32, n0);
                            let (jn1, b1) = along(i1, (j / 3u32) % 3u32, n1);
                            let (jn2, b2) = along(i2, j % 3u32, n2);
                            if jn0 && jn1 && jn2 {
                                let w = ((b0 * n1 + b1) * n2 + b2) as usize;
                                if labels[w] == candidate {
                                    score += field[w];
                                }
                            }
                        }
                        let hash = mix(prefix ^ candidate);
                        let better = !found
                            || score > best_score
                            || (score == best_score && hash < best_hash);
                        if better {
                            found = true;
                            best_score = score;
                            best_hash = hash;
                            best_label = candidate;
                        }
                    }
                }
            }
            out[ABSOLUTE_POS] = best_label;
        }
    }

    /// The GPU implementation of a [`VoteSweep`].
    pub struct GpuVoteSweep {
        sweep: VoteSweep,
    }

    impl GpuVoteSweep {
        pub fn new(sweep: VoteSweep) -> Self {
            Self { sweep }
        }
    }

    impl GpuKernel for GpuVoteSweep {
        /// Labels, field and output: twelve bytes a voxel, plus pool slack.
        fn device_bytes(&self, block: [usize; 3]) -> u64 {
            12 * block.iter().product::<usize>() as u64 + (1 << 20)
        }

        fn cost_per_voxel(&self) -> f64 {
            VOTE_SWEEP_GPU_COST
        }

        fn apply(&self, inputs: &[&Voxels], out: &mut Voxels, at: &Anchor) -> Result<()> {
            let [labels, field] = inputs else {
                return Err(Error::InvalidArgument(
                    "gpu vote sweep takes two inputs".into(),
                ));
            };
            if field.dtype() != Dtype::F32 {
                return Err(Error::InvalidArgument(format!(
                    "gpu vote sweep: the field must be float32, got {}",
                    field.dtype().numpy_name()
                )));
            }
            let gpu = device().ok_or_else(|| Error::InvalidArgument("no GPU device".into()))?;
            let shape = labels.shape();
            check_indexable(shape, "gpu vote sweep")?;
            let _admission = admit(self.device_bytes(shape));
            if shape.contains(&0) {
                return Ok(());
            }
            let labels = gpu.upload_u32(labels)?;
            let field = gpu.upload_f32(field)?;
            let result = gpu.empty(shape, Dtype::U32);
            let (count, dim) = launch_shape(labels.len());
            let [n0, n1, n2] = shape.map(|n| n as u32);
            let [o0, o1, o2] = at.offset.map(|o| o as u32);
            // SAFETY: the three buffers hold `n0 * n1 * n2` elements, and the
            // kernel indexes them only below that.
            unsafe {
                vote_sweep_kernel::launch_unchecked::<Rt>(
                    gpu.client(),
                    count,
                    dim,
                    labels.arg(),
                    field.arg(),
                    result.arg(),
                    n0,
                    n1,
                    n2,
                    o0,
                    o1,
                    o2,
                    self.sweep.seed,
                    self.sweep.sweep,
                    self.sweep.class,
                    self.sweep.colouring == Colouring::Octant,
                );
            }
            gpu.download_into(result, out)
        }
    }

    /// Measured with `benchmark_vote_sweep`, transfer included (12 bytes a
    /// voxel across PCIe dominate): 32 to 54 units over runs.
    pub(crate) const VOTE_SWEEP_GPU_COST: f64 = 40.0;

    /// [`VoteSweepOp`] with its GPU implementation attached, under `policy`.
    pub fn accelerated(op: VoteSweepOp, policy: DevicePolicy) -> Accelerated<VoteSweepOp> {
        let kernel = Arc::new(GpuVoteSweep::new(*op.sweep()));
        Accelerated::with_kernel(op, kernel, policy)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{s, Array3};

    use super::*;

    fn fixture(shape: (usize, usize, usize)) -> (Array3<u32>, Array3<f32>) {
        let labels =
            Array3::from_shape_fn(shape, |(z, y, x)| mix((z * 131 + y * 17 + x) as u32) % 4);
        let field = Array3::from_shape_fn(shape, |(z, y, x)| {
            (mix((z * 7 + y * 3 + x * 11) as u32) % 1000) as f32 / 1000.0
        });
        (labels, field)
    }

    fn run(
        labels: &Array3<u32>,
        field: &Array3<f32>,
        sweep: &VoteSweep,
        origin: [usize; 3],
    ) -> Array3<u32> {
        let mut out = Array3::<u32>::zeros(labels.dim());
        vote_sweep_into(labels.view(), field.view(), sweep, origin, out.view_mut()).unwrap();
        out
    }

    #[test]
    fn a_lone_dissenter_joins_its_unanimous_neighbourhood() {
        let mut labels = Array3::<u32>::from_elem((3, 3, 3), 7);
        labels[[1, 1, 1]] = 2;
        let field = Array3::<f32>::ones((3, 3, 3));
        // [1,1,1] is class 1 of the checkerboard and class 7 of the octants.
        for sweep in [
            VoteSweep::new(Colouring::Checkerboard, 1, 0, 0).unwrap(),
            VoteSweep::new(Colouring::Octant, 7, 0, 0).unwrap(),
        ] {
            let out = run(&labels, &field, &sweep, [0, 0, 0]);
            assert_eq!(out, Array3::from_elem((3, 3, 3), 7));
        }
    }

    #[test]
    fn only_the_named_class_changes() {
        let (labels, field) = fixture((6, 7, 8));
        let sweep = VoteSweep::new(Colouring::Octant, 3, 9, 1).unwrap();
        let out = run(&labels, &field, &sweep, [0, 0, 0]);
        for ((z, y, x), &value) in out.indexed_iter() {
            if Colouring::Octant.class_of([z as u32, y as u32, x as u32]) != 3 {
                assert_eq!(value, labels[[z, y, x]]);
            }
        }
        assert_ne!(out, labels, "the fixture is not already a fixed point");
    }

    #[test]
    fn the_field_weights_the_vote() {
        // Labels 1 on the z = 0 plane, 2 on z = 2; the centre is a tie at a
        // uniform field and goes to whichever plane has the heavier field.
        let mut labels = Array3::<u32>::from_elem((3, 3, 3), 1);
        labels.slice_mut(s![2, .., ..]).fill(2);
        labels.slice_mut(s![1, .., ..]).fill(0);
        let mut field = Array3::<f32>::ones((3, 3, 3));
        field.slice_mut(s![1, .., ..]).fill(0.0);
        field.slice_mut(s![2, .., ..]).fill(2.0);
        let sweep = VoteSweep::new(Colouring::Octant, 7, 0, 0).unwrap();
        assert_eq!(run(&labels, &field, &sweep, [0, 0, 0])[[1, 1, 1]], 2);
    }

    /// Block-cut invariance: each block, cut with a one-voxel halo clipped at
    /// the volume and anchored at its global offset, reproduces the
    /// whole-volume answer on its core.
    #[test]
    fn decomposed_equals_whole() {
        let shape = (9, 10, 11);
        let (labels, field) = fixture(shape);
        for colouring in [Colouring::Checkerboard, Colouring::Octant] {
            let sweep = VoteSweep::new(colouring, 1, 42, 5).unwrap();
            let whole = run(&labels, &field, &sweep, [0, 0, 0]);
            let edge = 4;
            for z in (0..shape.0).step_by(edge) {
                for y in (0..shape.1).step_by(edge) {
                    for x in (0..shape.2).step_by(edge) {
                        let core = [z, y, x];
                        let end = [
                            (z + edge).min(shape.0),
                            (y + edge).min(shape.1),
                            (x + edge).min(shape.2),
                        ];
                        let lo = core.map(|c| c.saturating_sub(1));
                        let hi = [
                            (end[0] + 1).min(shape.0),
                            (end[1] + 1).min(shape.1),
                            (end[2] + 1).min(shape.2),
                        ];
                        let cut = s![lo[0]..hi[0], lo[1]..hi[1], lo[2]..hi[2]];
                        let block = run(
                            &labels.slice(cut).to_owned(),
                            &field.slice(cut).to_owned(),
                            &sweep,
                            lo,
                        );
                        let inner = s![
                            core[0] - lo[0]..end[0] - lo[0],
                            core[1] - lo[1]..end[1] - lo[1],
                            core[2] - lo[2]..end[2] - lo[2]
                        ];
                        let want =
                            whole.slice(s![core[0]..end[0], core[1]..end[1], core[2]..end[2]]);
                        assert_eq!(block.slice(inner), want, "{colouring:?} block at {core:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn the_combine_reads_labels_then_field() {
        let (labels, field) = fixture((4, 5, 6));
        let sweep = VoteSweep::new(Colouring::Checkerboard, 0, 1, 2).unwrap();
        let op = VoteSweepOp::new("vote", sweep);
        assert!(op.accepts(&[Dtype::U32, Dtype::F32]));
        assert!(!op.accepts(&[Dtype::F32, Dtype::U32]));
        let inputs = [Voxels::U32(labels.clone()), Voxels::F32(field.clone())];
        let mut out = Voxels::zeros(Dtype::U32, [4, 5, 6]).unwrap();
        op.apply(
            &[&inputs[0], &inputs[1]],
            &mut out,
            &Anchor::new([2, 0, 1], [10, 10, 10]),
        )
        .unwrap();
        assert_eq!(
            out.view::<u32>().unwrap(),
            run(&labels, &field, &sweep, [2, 0, 1])
        );
        assert!(VoteSweep::new(Colouring::Checkerboard, 2, 0, 0).is_err());
    }
}
