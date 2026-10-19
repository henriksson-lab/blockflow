// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **A bank of linear filters over one input, transformed once.**
//
// Many detectors are a handful of correlations of the same image, combined
// nonlinearly per voxel: a normalised cross-correlation is three (template,
// support, support over the squared input), a multi-scale response is one per
// scale with a maximum across them, a quadrature-pair energy is two per
// orientation summed. Written as a `Chain::parallel` of
// [`TransformConvolveOp`](super::convolve::TransformConvolveOp)s each branch
// transforms the same window again, so `k` kernels cost `2k` transforms per
// tile. Here each **input** — the image, or a pointwise map of it such as its
// square — is transformed once per tile, and each kernel costs one multiply and
// one inverse: `m + k` transforms for `m` inputs.
//
// The shape of the combination is fixed and small, so that a tile never holds
// more than one group's results: terms form **groups**, each group is combined
// to one value per voxel by its own function, and the group values are folded by
// one `reduce` (a maximum, a sum). One group is the plain "combine these
// correlations" case.
//
// Everything else is `TransformConvolveOp`'s: tiles anchored to the volume, the
// same boundary convention, the same alignment-aware reach (over the widest
// kernel on each side), tiles computed in parallel. It agrees with the
// equivalent chain of `TransformConvolveOp`s to a rounding, not to a bit — the
// transform length is the bank's, not each kernel's.

use std::sync::Arc;

use ndarray::{Array3, ArrayView3, ArrayViewMut3};

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::op::{Anchor, BlockOp};
use crate::reach::{AxisReach, Reach};
use crate::voxels::Voxels;

use super::convolve::{
    fill_window, landing_slice, run_tiles, Kernel, Sense, TRANSFORM_CONVOLVE_COST,
};
use super::fft::{next_smooth_length, RealTransform3, Spectrum3};
use super::ridge::Boundary;
use super::shapes_agree;

/// A pointwise map applied to the input before it is transformed.
pub type InputMap = Arc<dyn Fn(f64) -> f64 + Send + Sync>;
/// One group's combination of its terms' correlations at a voxel.
pub type GroupCombine = Arc<dyn Fn(&[f64]) -> f64 + Send + Sync>;
/// The fold over group values.
pub type GroupReduce = Arc<dyn Fn(f64, f64) -> f64 + Send + Sync>;

/// Correlations of one input with several kernels, combined per voxel.
#[derive(Clone)]
pub struct BankGroup {
    /// `(input, kernel)`: the input is an index into the bank's inputs.
    pub terms: Vec<(usize, Kernel)>,
    pub combine: GroupCombine,
}

/// See this module's header.
#[derive(Clone)]
pub struct TransformBankOp {
    name: &'static str,
    inputs: Vec<InputMap>,
    groups: Vec<BankGroup>,
    reduce: GroupReduce,
    boundary: Boundary,
    tile: [usize; 3],
    lo: [usize; 3],
    hi: [usize; 3],
    length: [usize; 3],
    transform: RealTransform3,
    /// Conjugated kernel spectra, one per term, group by group.
    spectra: Vec<Vec<Spectrum3>>,
}

impl std::fmt::Debug for TransformBankOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransformBankOp")
            .field("name", &self.name)
            .field("inputs", &self.inputs.len())
            .field(
                "groups",
                &self
                    .groups
                    .iter()
                    .map(|g| g.terms.len())
                    .collect::<Vec<_>>(),
            )
            .field("tile", &self.tile)
            .field("length", &self.length)
            .finish()
    }
}

impl TransformBankOp {
    /// `inputs` pointwise maps of the phase input (the identity is
    /// `Arc::new(|v| v)`), `groups` of correlations against them, folded by
    /// `reduce`. Every kernel is applied with `sense` and `boundary` over
    /// volume-anchored tiles of `tile`.
    pub fn new(
        name: &'static str,
        inputs: Vec<InputMap>,
        groups: Vec<BankGroup>,
        reduce: GroupReduce,
        sense: Sense,
        boundary: Boundary,
        tile: [usize; 3],
    ) -> Result<Self> {
        if tile.contains(&0) {
            return Err(Error::invalid(format!(
                "a transform bank's tile must be non-empty, got {tile:?}"
            )));
        }
        if inputs.is_empty() || groups.is_empty() || groups.iter().any(|g| g.terms.is_empty()) {
            return Err(Error::invalid(
                "a transform bank needs an input and non-empty groups",
            ));
        }
        if let Some((input, _)) = groups
            .iter()
            .flat_map(|g| &g.terms)
            .find(|(i, _)| *i >= inputs.len())
        {
            return Err(Error::invalid(format!(
                "a transform bank term reads input {input} of {}",
                inputs.len()
            )));
        }
        let side = |kernel: &Kernel, axis: usize| {
            let (low, high) = kernel.element().sides(axis);
            match sense {
                Sense::Correlate => (low, high),
                Sense::Convolve => (high, low),
            }
        };
        let kernels = || groups.iter().flat_map(|g| g.terms.iter().map(|t| &t.1));
        let lo: [usize; 3] =
            std::array::from_fn(|a| kernels().map(|k| side(k, a).0).max().unwrap_or(0));
        let hi: [usize; 3] =
            std::array::from_fn(|a| kernels().map(|k| side(k, a).1).max().unwrap_or(0));
        let length: [usize; 3] =
            std::array::from_fn(|a| next_smooth_length(tile[a] + lo[a] + hi[a]));
        let mut transform = RealTransform3::new(length)?;
        let mut spectra = Vec::with_capacity(groups.len());
        for group in &groups {
            let mut group_spectra = Vec::with_capacity(group.terms.len());
            for (_, kernel) in &group.terms {
                let mut dense =
                    Array3::<f64>::zeros((lo[0] + hi[0] + 1, lo[1] + hi[1] + 1, lo[2] + hi[2] + 1));
                for (offset, &weight) in kernel
                    .element()
                    .offsets()
                    .iter()
                    .zip(kernel.weights().iter())
                {
                    let placed = sense.displace(*offset);
                    let index: [usize; 3] =
                        std::array::from_fn(|a| (placed[a] + lo[a] as isize) as usize);
                    dense[index] = weight;
                }
                let mut spectrum = transform.spectrum();
                transform.forward_zero_padded(dense.view(), &mut spectrum)?;
                for value in spectrum.iter_mut() {
                    value.im = -value.im;
                }
                group_spectra.push(spectrum);
            }
            spectra.push(group_spectra);
        }
        Ok(Self {
            name,
            inputs,
            groups,
            reduce,
            boundary,
            tile,
            lo,
            hi,
            length,
            transform,
            spectra,
        })
    }

    pub fn tile(&self) -> [usize; 3] {
        self.tile
    }

    /// The smooth length one transform runs at.
    pub fn transform_shape(&self) -> [usize; 3] {
        self.length
    }

    /// Transforms per tile: one forward per input, one inverse per term.
    pub fn transforms_per_tile(&self) -> usize {
        self.inputs.len() + self.groups.iter().map(|g| g.terms.len()).sum::<usize>()
    }

    fn run<T>(
        &self,
        input: ArrayView3<'_, T>,
        mut out: ArrayViewMut3<'_, f64>,
        at: &Anchor,
    ) -> Result<()>
    where
        T: Copy + Into<f64> + Sync,
    {
        shapes_agree(input.shape(), out.shape(), "transform bank")?;
        let window_shape = (
            self.tile[0] + self.lo[0] + self.hi[0],
            self.tile[1] + self.lo[1] + self.hi[1],
            self.tile[2] + self.lo[2] + self.hi[2],
        );
        let widest = self.groups.iter().map(|g| g.terms.len()).max().unwrap_or(1);
        let length = (self.length[0], self.length[1], self.length[2]);
        run_tiles(
            at,
            self.tile,
            (self.lo, self.hi),
            &mut out,
            || {
                let transform = self.transform.clone();
                let inputs: Vec<Spectrum3> =
                    self.inputs.iter().map(|_| transform.spectrum()).collect();
                let scratch = transform.spectrum();
                (
                    transform,
                    Array3::<f64>::zeros(window_shape),
                    Array3::<f64>::zeros(window_shape),
                    inputs,
                    scratch,
                    (0..widest)
                        .map(|_| Array3::<f64>::zeros(length))
                        .collect::<Vec<_>>(),
                )
            },
            |(transform, window, mapped, inputs, scratch, results), origin, landing, mut piece| {
                fill_window(
                    &input,
                    at,
                    origin,
                    self.tile,
                    self.lo,
                    self.hi,
                    self.boundary,
                    window,
                );
                for (map, spectrum) in self.inputs.iter().zip(inputs.iter_mut()) {
                    mapped.zip_mut_with(window, |m, &w| *m = map(w));
                    transform.forward_zero_padded(mapped.view(), spectrum)?;
                }
                let mut values = vec![0.0; widest];
                for (g, (group, spectra)) in self.groups.iter().zip(&self.spectra).enumerate() {
                    for (t, ((input, _), kernel)) in group.terms.iter().zip(spectra).enumerate() {
                        scratch.assign(&inputs[*input]);
                        scratch.zip_mut_with(kernel, |v, k| *v *= *k);
                        transform.inverse(scratch, &mut results[t])?;
                    }
                    let n = group.terms.len();
                    let views: Vec<_> = results[..n]
                        .iter()
                        .map(|r| r.slice(landing_slice(landing)))
                        .collect();
                    for (index, cell) in piece.indexed_iter_mut() {
                        for (value, view) in values.iter_mut().zip(&views) {
                            *value = view[index];
                        }
                        let combined = (group.combine)(&values[..n]);
                        *cell = if g == 0 {
                            combined
                        } else {
                            (self.reduce)(*cell, combined)
                        };
                    }
                }
                Ok(())
            },
        )
    }

    fn axis_reach(&self, axis: usize) -> AxisReach {
        AxisReach::aligned(self.tile[axis], self.lo[axis], self.hi[axis])
    }
}

impl BlockOp for TransformBankOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// `TransformConvolveOp::reach`'s statement over the widest kernel.
    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        (self.tile[axis] - 1 + self.lo[axis]).max(self.tile[axis] - 1 + self.hi[axis])
    }

    fn reach_spec(&self, _volume: [usize; 3]) -> Reach {
        Reach::per_axis([self.axis_reach(0), self.axis_reach(1), self.axis_reach(2)])
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        matches!(
            dtype,
            Dtype::Bool
                | Dtype::U8
                | Dtype::U16
                | Dtype::U32
                | Dtype::U64
                | Dtype::I8
                | Dtype::I16
                | Dtype::I32
                | Dtype::I64
                | Dtype::F32
                | Dtype::F64
        )
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::F64
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, at: &Anchor) -> Result<()> {
        let out = out.view_mut::<f64>()?;
        match input.dtype() {
            Dtype::U8 => self.run(input.view::<u8>()?, out, at),
            Dtype::U16 => self.run(input.view::<u16>()?, out, at),
            Dtype::U32 => self.run(input.view::<u32>()?, out, at),
            Dtype::I8 => self.run(input.view::<i8>()?, out, at),
            Dtype::I16 => self.run(input.view::<i16>()?, out, at),
            Dtype::I32 => self.run(input.view::<i32>()?, out, at),
            Dtype::F32 => self.run(input.view::<f32>()?, out, at),
            Dtype::F64 => self.run(input.view::<f64>()?, out, at),
            Dtype::Bool | Dtype::U64 | Dtype::I64 => {
                let widened = input.widened();
                self.run(widened.view(), out, at)
            }
            Dtype::F16 => Err(Error::invalid(format!(
                "{}: no buffer holds half-precision",
                self.name
            ))),
        }
    }

    /// `None`, for `TransformConvolveOp::constant_maps_to`'s reason.
    fn constant_maps_to(&self, _value: f64) -> Option<f64> {
        None
    }

    /// `TransformConvolveOp`'s price per transform, times the transforms a tile
    /// runs.
    fn cost_per_voxel(&self) -> f64 {
        let padded = (self.length[0] * self.length[1] * self.length[2]) as f64;
        let tile = (self.tile[0] * self.tile[1] * self.tile[2]) as f64;
        TRANSFORM_CONVOLVE_COST * padded * padded.log2() / tile * self.transforms_per_tile() as f64
            / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::convolve::convolve_into;
    use crate::ops::element::{ElementShape, StructuringElement};

    fn field(shape: [usize; 3]) -> Array3<f64> {
        Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
            (i * 37 + j * 11 + k * 3) as f64 % 13.0 - 4.0 + (i * j) as f64 * 0.25
        })
    }

    fn lopsided() -> Kernel {
        Kernel::new(
            StructuringElement::from_offsets([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 2]]).unwrap(),
            vec![1.0, -3.0, 0.5, 2.0],
        )
        .unwrap()
    }

    fn boxed(r: usize) -> Kernel {
        let element = StructuringElement::from_radius(ElementShape::Box, [r, r, r]);
        let n = element.len();
        Kernel::new(element, (0..n).map(|i| 1.0 + (i % 5) as f64).collect()).unwrap()
    }

    /// Two groups over two inputs, against direct correlations combined by
    /// hand, on blocks cut two ways (one block; blocks smaller than a tile).
    #[test]
    fn a_bank_matches_direct_correlations() {
        let shape = [13, 11, 12];
        let image = field(shape);
        let square: InputMap = Arc::new(|v| v * v);
        let identity: InputMap = Arc::new(|v| v);
        let groups = vec![
            BankGroup {
                terms: vec![(0, lopsided()), (1, boxed(1))],
                combine: Arc::new(|x| x[0] - 0.1 * x[1]),
            },
            BankGroup {
                terms: vec![(0, boxed(2))],
                combine: Arc::new(|x| 0.5 * x[0]),
            },
        ];
        let bank = TransformBankOp::new(
            "bank",
            vec![identity, square],
            groups,
            Arc::new(f64::max),
            Sense::Correlate,
            Boundary::Clamp,
            [5, 4, 6],
        )
        .unwrap();
        assert_eq!(bank.transforms_per_tile(), 5);
        let direct = |input: &Array3<f64>, kernel: &Kernel| {
            let mut out = Array3::zeros(input.raw_dim());
            convolve_into(
                input.view(),
                kernel,
                Sense::Correlate,
                Boundary::Clamp,
                out.view_mut(),
            )
            .unwrap();
            out
        };
        let squared = image.mapv(|v| v * v);
        let (a, b, c) = (
            direct(&image, &lopsided()),
            direct(&squared, &boxed(1)),
            direct(&image, &boxed(2)),
        );
        let expected =
            Array3::from_shape_fn(image.raw_dim(), |i| (a[i] - 0.1 * b[i]).max(0.5 * c[i]));

        let mut whole = Voxels::from(Array3::<f64>::zeros(image.raw_dim()));
        bank.apply(
            &Voxels::from(image.clone()),
            &mut whole,
            &Anchor::whole(shape),
        )
        .unwrap();
        let whole = whole.view::<f64>().unwrap().to_owned();
        for (got, want) in whole.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-9, "{got} vs {want}");
        }
        // a block in the middle, read with the declared halo
        let core = [[4usize, 9], [3, 8], [6, 12]];
        let read: [[usize; 2]; 3] = std::array::from_fn(|a| {
            let r = bank.reach(a, shape[a]);
            [core[a][0].saturating_sub(r), (core[a][1] + r).min(shape[a])]
        });
        let buffer = image
            .slice(ndarray::s![
                read[0][0]..read[0][1],
                read[1][0]..read[1][1],
                read[2][0]..read[2][1]
            ])
            .to_owned();
        let mut out = Voxels::from(Array3::<f64>::zeros(buffer.raw_dim()));
        bank.apply(
            &Voxels::from(buffer),
            &mut out,
            &Anchor::new([read[0][0], read[1][0], read[2][0]], shape),
        )
        .unwrap();
        let out = out.view::<f64>().unwrap();
        for z in core[0][0]..core[0][1] {
            for y in core[1][0]..core[1][1] {
                for x in core[2][0]..core[2][1] {
                    let got = out[[z - read[0][0], y - read[1][0], x - read[2][0]]];
                    assert_eq!(got.to_bits(), whole[[z, y, x]].to_bits(), "{z} {y} {x}");
                }
            }
        }
    }

    /// Planned over lattices whose block edge is a whole number of tiles, is
    /// not, and is smaller than a tile: both transform ops give the bits of one
    /// block holding the volume, so skipping halo tiles drops nothing kept.
    #[test]
    fn planned_transforms_are_lattice_invariant() {
        use crate::assemble::PlanBuilder;
        use crate::env::ArrayEnvironment;
        use crate::geometry::BlockGrid;
        use crate::op::Chain;
        use crate::ops::convolve::TransformConvolveOp;
        use crate::strategy::{execute_phases, Hints};

        let volume = [23, 19, 21];
        let image = field(volume);
        let tile = [4, 4, 4];
        let chains = || -> Vec<Chain> {
            let identity: InputMap = Arc::new(|v| v);
            let square: InputMap = Arc::new(|v| v * v);
            vec![
                Chain::op(
                    TransformConvolveOp::new(
                        "conv",
                        lopsided(),
                        Sense::Correlate,
                        Boundary::Clamp,
                        tile,
                    )
                    .unwrap(),
                ),
                Chain::op(
                    TransformBankOp::new(
                        "bank",
                        vec![identity, square],
                        vec![
                            BankGroup {
                                terms: vec![(0, lopsided()), (1, boxed(1))],
                                combine: Arc::new(|x| x[0] - 0.1 * x[1]),
                            },
                            BankGroup {
                                terms: vec![(0, boxed(2))],
                                combine: Arc::new(|x| 0.5 * x[0]),
                            },
                        ],
                        Arc::new(f64::max),
                        Sense::Correlate,
                        Boundary::Clamp,
                        tile,
                    )
                    .unwrap(),
                ),
            ]
        };
        for which in 0..2 {
            let mut answers = Vec::new();
            for block in [volume, [8, 8, 8], [7, 9, 6], [3, 3, 3]] {
                let mut builder =
                    PlanBuilder::new(volume, Dtype::F64, BlockGrid::new(volume, block).unwrap());
                builder.pixels(chains().swap_remove(which)).unwrap();
                let assembly = builder.finish().unwrap();
                let env = ArrayEnvironment::for_decomposition(
                    Voxels::from(image.clone()),
                    &assembly.decomposition,
                    block,
                )
                .unwrap();
                execute_phases(
                    "transform",
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
            assert!(answers[0].iter().any(|v| *v != 0.0));
            for (n, other) in answers.iter().enumerate().skip(1) {
                let differing = answers[0]
                    .iter()
                    .zip(other.iter())
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                assert_eq!(
                    differing, 0,
                    "op {which}, lattice {n}: {differing} voxels differ"
                );
            }
        }
    }
}
