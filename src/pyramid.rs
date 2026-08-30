// SPDX-License-Identifier: MIT
//
// Derived image levels.
//
// This module is about applying ordinary image operations while deriving a
// smaller lattice. It is not tied to any file format and it is not a registration
// schedule. A storage pyramid is an externally useful set of arrays; a derived
// pyramid is a working set a caller may build from those arrays before running
// its own algorithm.

use ndarray::Array3;

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::op::{Anchor, Chain};
use crate::voxels::{VoxelElement, Voxels};

/// One derived level: first apply `ops`, then decimate by `factor`.
pub struct LevelRecipe {
    pub factor: [usize; 3],
    pub ops: Chain,
}

impl LevelRecipe {
    pub fn new(factor: [usize; 3], ops: Chain) -> Result<Self> {
        if factor.contains(&0) || factor == [1, 1, 1] {
            return Err(Error::InvalidArgument(format!(
                "pyramid level factor {factor:?} must be positive and must shrink at least one \
                 axis"
            )));
        }
        Ok(Self { factor, ops })
    }

    pub fn decimate(factor: [usize; 3]) -> Result<Self> {
        Self::new(factor, Chain::sequence(Vec::new()))
    }
}

/// A recipe for derived levels. Level 0 is the input unchanged; each entry
/// derives the next level from the previous one.
pub struct PyramidRecipe {
    levels: Vec<LevelRecipe>,
}

impl PyramidRecipe {
    pub fn new(levels: Vec<LevelRecipe>) -> Self {
        Self { levels }
    }

    pub fn decimation(factors: Vec<[usize; 3]>) -> Result<Self> {
        let levels = factors
            .into_iter()
            .map(LevelRecipe::decimate)
            .collect::<Result<Vec<_>>>()?;
        Ok(Self::new(levels))
    }

    pub fn len(&self) -> usize {
        self.levels.len() + 1
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    pub fn levels(&self) -> &[LevelRecipe] {
        &self.levels
    }

    pub fn build_resident(&self, input: &Voxels) -> Result<Vec<Voxels>> {
        let mut out = Vec::with_capacity(self.len());
        out.push(input.clone());
        for (index, level) in self.levels.iter().enumerate() {
            let prior = out.last().expect("level 0 exists");
            out.push(derive_level(prior, level, index + 1)?);
        }
        Ok(out)
    }
}

fn derive_level(input: &Voxels, recipe: &LevelRecipe, level: usize) -> Result<Voxels> {
    let filtered = apply_level_ops(input, &recipe.ops)?;
    decimate(&filtered, recipe.factor, level)
}

fn apply_level_ops(input: &Voxels, ops: &Chain) -> Result<Voxels> {
    let shape = ops.output_shape(input.shape())?;
    let dtype = ops.produces(input.dtype())?;
    let mut out = Voxels::zeros(dtype, shape)?;
    ops.apply(input, &mut out, &Anchor::whole(input.shape()))?;
    Ok(out)
}

fn decimate(input: &Voxels, factor: [usize; 3], level: usize) -> Result<Voxels> {
    let shape = input.shape();
    let mut out = [0usize; 3];
    for axis in 0..3 {
        out[axis] = shape[axis] / factor[axis];
        if out[axis] == 0 {
            return Err(Error::InvalidArgument(format!(
                "derived pyramid level {level} would shrink axis {axis} from {} by factor {} to \
                 zero voxels",
                shape[axis], factor[axis]
            )));
        }
    }
    by_dtype(input.dtype(), input, factor, out)
}

fn by_dtype(dtype: Dtype, input: &Voxels, factor: [usize; 3], out: [usize; 3]) -> Result<Voxels> {
    Ok(match dtype {
        Dtype::Bool => downsample::<bool>(input, factor, out)?.into(),
        Dtype::U8 => downsample::<u8>(input, factor, out)?.into(),
        Dtype::U16 => downsample::<u16>(input, factor, out)?.into(),
        Dtype::U32 => downsample::<u32>(input, factor, out)?.into(),
        Dtype::U64 => downsample::<u64>(input, factor, out)?.into(),
        Dtype::I8 => downsample::<i8>(input, factor, out)?.into(),
        Dtype::I16 => downsample::<i16>(input, factor, out)?.into(),
        Dtype::I32 => downsample::<i32>(input, factor, out)?.into(),
        Dtype::I64 => downsample::<i64>(input, factor, out)?.into(),
        Dtype::F32 => downsample::<f32>(input, factor, out)?.into(),
        Dtype::F64 => downsample::<f64>(input, factor, out)?.into(),
        Dtype::F16 => {
            return Err(Error::InvalidArgument(
                "derived pyramids cannot hold float16 because this crate has no float16 voxel \
                 buffer"
                    .to_string(),
            ));
        }
    })
}

fn downsample<T>(input: &Voxels, factor: [usize; 3], out: [usize; 3]) -> Result<Array3<T>>
where
    T: VoxelElement,
{
    let values = T::peek(input).ok_or_else(|| {
        Error::InvalidArgument(format!(
            "derived pyramid expected {:?}, got {:?}",
            T::DTYPE,
            input.dtype()
        ))
    })?;
    Ok(Array3::from_shape_fn(
        (out[0], out[1], out[2]),
        |(x, y, z)| {
            if T::DTYPE == Dtype::Bool {
                return values[[x * factor[0], y * factor[1], z * factor[2]]];
            }
            let mut sum = 0.0;
            let mut count = 0usize;
            for dz in 0..factor[2] {
                for dy in 0..factor[1] {
                    for dx in 0..factor[0] {
                        sum += values[[x * factor[0] + dx, y * factor[1] + dy, z * factor[2] + dz]]
                            .into_f64();
                        count += 1;
                    }
                }
            }
            T::from_f64(sum / count as f64)
        },
    ))
}
