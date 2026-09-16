// SPDX-License-Identifier: MIT
//
// Original work for this crate.

//! Geometric image transforms.
//!
//! The maps here are inverse maps: for each output voxel, they say which source
//! coordinate is sampled. That is the convention that makes interpolation local
//! and keeps holes out of the output.

use ndarray::{ArrayView3, ArrayViewMut3};

use crate::assemble::{Phase, PlanBuilder};
use crate::decomposition::PhaseDecomposition;
use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::geometry::BlockGrid;
use crate::op::{Anchor, BlockOp, Geometry, InputMap, Placement, Slicing, SourceInputs};
use crate::reach::Reach;
use crate::region::Region;
use crate::voxels::{VoxelElement, Voxels};
use crate::Chain;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformInterpolation {
    Nearest,
    Linear,
}

impl TransformInterpolation {
    fn accepts(self, dtype: Dtype) -> bool {
        match self {
            Self::Nearest => dtype != Dtype::F16,
            Self::Linear => !matches!(dtype, Dtype::Bool | Dtype::F16),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransformBoundary {
    Constant(f64),
    Clamp,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CoordinateMap {
    Affine {
        matrix: [[f64; 3]; 3],
        offset: [f64; 3],
    },
    Projective {
        matrix: [[f64; 4]; 4],
    },
    Polar {
        center: [f64; 2],
        radius_scale: f64,
        angle_scale: f64,
        z_scale: f64,
        log_radius: bool,
    },
}

impl CoordinateMap {
    pub fn affine(matrix: [[f64; 3]; 3], offset: [f64; 3]) -> Result<Self> {
        validate_finite_matrix3(matrix, offset)?;
        Ok(Self::Affine { matrix, offset })
    }

    pub fn projective(matrix: [[f64; 4]; 4]) -> Result<Self> {
        validate_finite_matrix4(matrix)?;
        Ok(Self::Projective { matrix })
    }

    pub fn rotation_2d(center: [f64; 2], angle_radians: f64, z_scale: f64) -> Result<Self> {
        if !angle_radians.is_finite() {
            return Err(Error::InvalidArgument(format!(
                "rotation angle must be finite, got {angle_radians}"
            )));
        }
        validate_finite_vector([center[0], center[1], z_scale], "rotation")?;
        if z_scale == 0.0 {
            return Err(Error::InvalidArgument(
                "rotation z scale must be non-zero".to_string(),
            ));
        }
        let (sin, cos) = angle_radians.sin_cos();
        let matrix = [[cos, sin, 0.0], [-sin, cos, 0.0], [0.0, 0.0, 1.0 / z_scale]];
        let offset = [
            center[0] - cos * center[0] - sin * center[1],
            center[1] + sin * center[0] - cos * center[1],
            0.0,
        ];
        Self::affine(matrix, offset)
    }

    pub fn polar(
        center: [f64; 2],
        radius_scale: f64,
        angle_scale: f64,
        z_scale: f64,
    ) -> Result<Self> {
        validate_polar(center, radius_scale, angle_scale, z_scale)?;
        Ok(Self::Polar {
            center,
            radius_scale,
            angle_scale,
            z_scale,
            log_radius: false,
        })
    }

    pub fn log_polar(
        center: [f64; 2],
        radius_scale: f64,
        angle_scale: f64,
        z_scale: f64,
    ) -> Result<Self> {
        validate_polar(center, radius_scale, angle_scale, z_scale)?;
        Ok(Self::Polar {
            center,
            radius_scale,
            angle_scale,
            z_scale,
            log_radius: true,
        })
    }

    fn at(&self, out: [usize; 3]) -> Option<[f64; 3]> {
        let x = out[0] as f64;
        let y = out[1] as f64;
        let z = out[2] as f64;
        match *self {
            Self::Affine { matrix, offset } => Some([
                matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + offset[0],
                matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + offset[1],
                matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + offset[2],
            ]),
            Self::Projective { matrix } => {
                let w = matrix[3][0] * x + matrix[3][1] * y + matrix[3][2] * z + matrix[3][3];
                if w == 0.0 || !w.is_finite() {
                    return None;
                }
                Some([
                    (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3]) / w,
                    (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3]) / w,
                    (matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3]) / w,
                ])
            }
            Self::Polar {
                center,
                radius_scale,
                angle_scale,
                z_scale,
                log_radius,
            } => {
                let radius = if log_radius {
                    (x / radius_scale).exp() - 1.0
                } else {
                    x / radius_scale
                };
                let angle = y / angle_scale;
                Some([
                    center[0] + radius * angle.cos(),
                    center[1] + radius * angle.sin(),
                    z / z_scale,
                ])
            }
        }
    }
}

pub fn warp_into<T>(
    input: ArrayView3<'_, T>,
    map: &CoordinateMap,
    output_shape: [usize; 3],
    interpolation: TransformInterpolation,
    boundary: TransformBoundary,
    mut out: ArrayViewMut3<'_, T>,
) -> Result<()>
where
    T: VoxelElement,
{
    if !interpolation.accepts(T::DTYPE) {
        return Err(Error::InvalidArgument(format!(
            "{interpolation:?} transform interpolation does not support {:?}",
            T::DTYPE
        )));
    }
    shapes_match(output_shape, out.shape(), "warp_into output")?;
    for i in 0..output_shape[0] {
        for j in 0..output_shape[1] {
            for k in 0..output_shape[2] {
                out[[i, j, k]] = sample(input, map.at([i, j, k]), interpolation, boundary);
            }
        }
    }
    Ok(())
}

pub fn affine_transform_into<T>(
    input: ArrayView3<'_, T>,
    matrix: [[f64; 3]; 3],
    offset: [f64; 3],
    interpolation: TransformInterpolation,
    boundary: TransformBoundary,
    out: ArrayViewMut3<'_, T>,
) -> Result<()>
where
    T: VoxelElement,
{
    if !interpolation.accepts(T::DTYPE) {
        return Err(Error::InvalidArgument(format!(
            "{interpolation:?} transform interpolation does not support {:?}",
            T::DTYPE
        )));
    }
    let output_shape = [out.shape()[0], out.shape()[1], out.shape()[2]];
    warp_into(
        input,
        &CoordinateMap::affine(matrix, offset)?,
        output_shape,
        interpolation,
        boundary,
        out,
    )
}

pub fn projective_transform_into<T>(
    input: ArrayView3<'_, T>,
    matrix: [[f64; 4]; 4],
    interpolation: TransformInterpolation,
    boundary: TransformBoundary,
    out: ArrayViewMut3<'_, T>,
) -> Result<()>
where
    T: VoxelElement,
{
    let output_shape = [out.shape()[0], out.shape()[1], out.shape()[2]];
    warp_into(
        input,
        &CoordinateMap::projective(matrix)?,
        output_shape,
        interpolation,
        boundary,
        out,
    )
}

pub fn rotate_into<T>(
    input: ArrayView3<'_, T>,
    center: [f64; 2],
    angle_radians: f64,
    interpolation: TransformInterpolation,
    boundary: TransformBoundary,
    out: ArrayViewMut3<'_, T>,
) -> Result<()>
where
    T: VoxelElement,
{
    let output_shape = [out.shape()[0], out.shape()[1], out.shape()[2]];
    warp_into(
        input,
        &CoordinateMap::rotation_2d(center, angle_radians, 1.0)?,
        output_shape,
        interpolation,
        boundary,
        out,
    )
}

pub fn polar_transform_into<T>(
    input: ArrayView3<'_, T>,
    center: [f64; 2],
    radius_scale: f64,
    angle_scale: f64,
    interpolation: TransformInterpolation,
    boundary: TransformBoundary,
    out: ArrayViewMut3<'_, T>,
) -> Result<()>
where
    T: VoxelElement,
{
    let output_shape = [out.shape()[0], out.shape()[1], out.shape()[2]];
    warp_into(
        input,
        &CoordinateMap::polar(center, radius_scale, angle_scale, 1.0)?,
        output_shape,
        interpolation,
        boundary,
        out,
    )
}

pub fn log_polar_transform_into<T>(
    input: ArrayView3<'_, T>,
    center: [f64; 2],
    radius_scale: f64,
    angle_scale: f64,
    interpolation: TransformInterpolation,
    boundary: TransformBoundary,
    out: ArrayViewMut3<'_, T>,
) -> Result<()>
where
    T: VoxelElement,
{
    let output_shape = [out.shape()[0], out.shape()[1], out.shape()[2]];
    warp_into(
        input,
        &CoordinateMap::log_polar(center, radius_scale, angle_scale, 1.0)?,
        output_shape,
        interpolation,
        boundary,
        out,
    )
}

pub fn remap_into<T>(
    input: ArrayView3<'_, T>,
    coordinates: [ArrayView3<'_, f64>; 3],
    interpolation: TransformInterpolation,
    boundary: TransformBoundary,
    mut out: ArrayViewMut3<'_, T>,
) -> Result<()>
where
    T: VoxelElement,
{
    let output_shape = [out.shape()[0], out.shape()[1], out.shape()[2]];
    for coordinate in &coordinates {
        shapes_match(output_shape, coordinate.shape(), "remap coordinate")?;
    }
    for i in 0..output_shape[0] {
        for j in 0..output_shape[1] {
            for k in 0..output_shape[2] {
                out[[i, j, k]] = sample(
                    input,
                    Some([
                        coordinates[0][[i, j, k]],
                        coordinates[1][[i, j, k]],
                        coordinates[2][[i, j, k]],
                    ]),
                    interpolation,
                    boundary,
                );
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
/// Planner-visible analytic warp.
///
/// Use [`append_warp_phase`] for shape-changing warps; it cuts the phase on the
/// output grid and records source fetch regions in the input image's coordinate
/// space. The generic `PlanBuilder::pixels` path is still fine for same-volume
/// warps.
pub struct WarpOp {
    name: &'static str,
    map: CoordinateMap,
    output_shape: [usize; 3],
    interpolation: TransformInterpolation,
    boundary: TransformBoundary,
    cost: f64,
}

impl WarpOp {
    pub fn new(
        name: &'static str,
        map: CoordinateMap,
        output_shape: [usize; 3],
        interpolation: TransformInterpolation,
        boundary: TransformBoundary,
    ) -> Result<Self> {
        if output_shape.contains(&0) {
            return Err(Error::InvalidArgument(format!(
                "{name} needs a non-empty output shape, got {output_shape:?}"
            )));
        }
        Ok(Self {
            name,
            map,
            output_shape,
            interpolation,
            boundary,
            cost: transform_cost(interpolation),
        })
    }

    pub fn affine(
        name: &'static str,
        matrix: [[f64; 3]; 3],
        offset: [f64; 3],
        output_shape: [usize; 3],
        interpolation: TransformInterpolation,
        boundary: TransformBoundary,
    ) -> Result<Self> {
        Self::new(
            name,
            CoordinateMap::affine(matrix, offset)?,
            output_shape,
            interpolation,
            boundary,
        )
    }

    pub fn projective(
        name: &'static str,
        matrix: [[f64; 4]; 4],
        output_shape: [usize; 3],
        interpolation: TransformInterpolation,
        boundary: TransformBoundary,
    ) -> Result<Self> {
        Self::new(
            name,
            CoordinateMap::projective(matrix)?,
            output_shape,
            interpolation,
            boundary,
        )
    }

    pub fn rotate_2d(
        name: &'static str,
        center: [f64; 2],
        angle_radians: f64,
        output_shape: [usize; 3],
        interpolation: TransformInterpolation,
        boundary: TransformBoundary,
    ) -> Result<Self> {
        Self::new(
            name,
            CoordinateMap::rotation_2d(center, angle_radians, 1.0)?,
            output_shape,
            interpolation,
            boundary,
        )
    }

    pub fn polar(
        name: &'static str,
        center: [f64; 2],
        radius_scale: f64,
        angle_scale: f64,
        output_shape: [usize; 3],
        interpolation: TransformInterpolation,
        boundary: TransformBoundary,
    ) -> Result<Self> {
        Self::new(
            name,
            CoordinateMap::polar(center, radius_scale, angle_scale, 1.0)?,
            output_shape,
            interpolation,
            boundary,
        )
    }

    pub fn log_polar(
        name: &'static str,
        center: [f64; 2],
        radius_scale: f64,
        angle_scale: f64,
        output_shape: [usize; 3],
        interpolation: TransformInterpolation,
        boundary: TransformBoundary,
    ) -> Result<Self> {
        Self::new(
            name,
            CoordinateMap::log_polar(center, radius_scale, angle_scale, 1.0)?,
            output_shape,
            interpolation,
            boundary,
        )
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }

    fn source_region(&self, output: &Region, input_volume: [usize; 3]) -> Region {
        source_region_for(
            &self.map,
            output,
            input_volume,
            self.interpolation,
            self.boundary,
        )
    }
}

/// The phase shape an analytic warp needs: blocks are cut over the output
/// volume, while every block fetches the input region its transformed output
/// footprint can sample.
///
/// The bound is still axis-aligned in input space, which is intentionally the
/// same rectangular contract the rest of the executor moves, but it is derived
/// from the actual inverse-map samples rather than conservatively fetching the
/// whole source.
pub fn warp_phase(
    slots: Vec<usize>,
    names: Vec<String>,
    op: &WarpOp,
    input_volume: [usize; 3],
    output_grid: BlockGrid,
) -> Result<PhaseDecomposition> {
    let output_volume = output_grid.volume();
    if output_volume != op.output_shape {
        return Err(Error::InvalidArgument(format!(
            "{} writes {:?}, but the supplied warp grid is over {output_volume:?}",
            op.name, op.output_shape
        )));
    }
    if input_volume.contains(&0) || output_volume.contains(&0) {
        return Err(Error::InvalidArgument(format!(
            "warp phase needs non-empty input and output volumes, got input {input_volume:?}, \
             output {output_volume:?}"
        )));
    }
    Ok(
        PhaseDecomposition::derive(slots, names, Reach::all(), Reach::all(), output_grid)
            .with_sources(|block| op.source_region(&block.valid, input_volume)),
    )
}

/// Append a planner-visible analytic warp phase.
///
/// This is the shape-changing counterpart to `PlanBuilder::pixels`: the caller
/// supplies a grid over `op`'s output volume, and the phase records each block's
/// source fetch region in the input volume.
pub fn append_warp_phase(
    builder: &mut PlanBuilder,
    op: WarpOp,
    input_volume: [usize; 3],
    output_grid: BlockGrid,
) -> Result<Phase> {
    if output_grid.volume() != op.output_shape {
        return Err(Error::InvalidArgument(format!(
            "{} writes {:?}, but the supplied warp grid is over {:?}",
            op.name,
            op.output_shape,
            output_grid.volume()
        )));
    }
    let phase = warp_phase(
        vec![0],
        vec![op.name().to_string()],
        &op,
        input_volume,
        output_grid,
    )?;
    builder.pixels_decomposed(Chain::op(op), phase)
}

impl BlockOp for WarpOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, volume_len: usize) -> usize {
        volume_len
    }

    fn reach_spec(&self, _volume: [usize; 3]) -> Reach {
        Reach::all()
    }

    fn geometry(&self, _input_volume: [usize; 3]) -> Geometry {
        Geometry::new(self.output_shape, vec![InputMap::Stencil(Reach::all())])
    }

    fn output_shape(&self, _input: [usize; 3]) -> [usize; 3] {
        self.output_shape
    }

    fn takes_extent_from_placement(&self) -> bool {
        true
    }

    fn placed_output_shape(&self, _input: [usize; 3], at: &Placement) -> [usize; 3] {
        at.writes().unwrap_or(self.output_shape)
    }

    fn slicing(&self) -> Slicing {
        Slicing::Stencil
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        self.interpolation.accepts(dtype)
    }

    fn produces(&self, input: Dtype) -> Dtype {
        input
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, at: &Anchor) -> Result<()> {
        let placement = Placement::same(at.clone()).writing(out.shape());
        self.apply_placed(input, SourceInputs::none(), out, &placement)
    }

    fn apply_placed(
        &self,
        input: &Voxels,
        _sources: SourceInputs<'_>,
        out: &mut Voxels,
        at: &Placement,
    ) -> Result<()> {
        let held_shape = input.shape();
        for axis in 0..3 {
            if at.input.offset[axis] + held_shape[axis] > at.input.volume[axis] {
                return Err(Error::InvalidArgument(format!(
                    "{} source buffer spans axis {axis} {}..{} of {:?}",
                    self.name,
                    at.input.offset[axis],
                    at.input.offset[axis] + held_shape[axis],
                    at.input.volume
                )));
            }
        }
        if at.input.volume.contains(&0) {
            return Err(Error::InvalidArgument(format!(
                "{} needs a non-empty source volume",
                self.name
            )));
        }
        self.apply_typed_dispatch(
            input,
            out,
            at.output.offset,
            at.input.offset,
            at.input.volume,
        )
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

impl WarpOp {
    fn apply_typed<T>(
        &self,
        input: &Voxels,
        out: &mut Voxels,
        output_offset: [usize; 3],
        source_offset: [usize; 3],
        full_shape: [usize; 3],
    ) -> Result<()>
    where
        T: VoxelElement,
    {
        let input = input.view::<T>()?;
        let mut out = out.view_mut::<T>()?;
        let shape = [out.shape()[0], out.shape()[1], out.shape()[2]];
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    let global = [
                        output_offset[0] + i,
                        output_offset[1] + j,
                        output_offset[2] + k,
                    ];
                    out[[i, j, k]] = sample_from_region(
                        input,
                        self.map.at(global),
                        self.interpolation,
                        self.boundary,
                        source_offset,
                        full_shape,
                    );
                }
            }
        }
        Ok(())
    }

    fn apply_typed_dispatch(
        &self,
        input: &Voxels,
        out: &mut Voxels,
        output_offset: [usize; 3],
        source_offset: [usize; 3],
        full_shape: [usize; 3],
    ) -> Result<()> {
        if !self.accepts(input.dtype()) {
            return Err(Error::InvalidArgument(format!(
                "{} does not support {:?} with {:?} interpolation",
                self.name,
                input.dtype(),
                self.interpolation
            )));
        }
        match input.dtype() {
            Dtype::Bool => {
                self.apply_typed::<bool>(input, out, output_offset, source_offset, full_shape)
            }
            Dtype::U8 => {
                self.apply_typed::<u8>(input, out, output_offset, source_offset, full_shape)
            }
            Dtype::U16 => {
                self.apply_typed::<u16>(input, out, output_offset, source_offset, full_shape)
            }
            Dtype::U32 => {
                self.apply_typed::<u32>(input, out, output_offset, source_offset, full_shape)
            }
            Dtype::U64 => {
                self.apply_typed::<u64>(input, out, output_offset, source_offset, full_shape)
            }
            Dtype::I8 => {
                self.apply_typed::<i8>(input, out, output_offset, source_offset, full_shape)
            }
            Dtype::I16 => {
                self.apply_typed::<i16>(input, out, output_offset, source_offset, full_shape)
            }
            Dtype::I32 => {
                self.apply_typed::<i32>(input, out, output_offset, source_offset, full_shape)
            }
            Dtype::I64 => {
                self.apply_typed::<i64>(input, out, output_offset, source_offset, full_shape)
            }
            Dtype::F32 => {
                self.apply_typed::<f32>(input, out, output_offset, source_offset, full_shape)
            }
            Dtype::F64 => {
                self.apply_typed::<f64>(input, out, output_offset, source_offset, full_shape)
            }
            Dtype::F16 => Err(Error::InvalidArgument(
                "no buffer holds half-precision".to_string(),
            )),
        }
    }
}

fn sample<T>(
    input: ArrayView3<'_, T>,
    point: Option<[f64; 3]>,
    interpolation: TransformInterpolation,
    boundary: TransformBoundary,
) -> T
where
    T: VoxelElement,
{
    let full_shape = [input.shape()[0], input.shape()[1], input.shape()[2]];
    sample_from_region(input, point, interpolation, boundary, [0, 0, 0], full_shape)
}

fn sample_from_region<T>(
    input: ArrayView3<'_, T>,
    point: Option<[f64; 3]>,
    interpolation: TransformInterpolation,
    boundary: TransformBoundary,
    source_offset: [usize; 3],
    full_shape: [usize; 3],
) -> T
where
    T: VoxelElement,
{
    match interpolation {
        TransformInterpolation::Nearest => {
            sample_nearest(input, point, boundary, source_offset, full_shape)
        }
        TransformInterpolation::Linear => {
            sample_linear(input, point, boundary, source_offset, full_shape)
        }
    }
}

fn sample_nearest<T>(
    input: ArrayView3<'_, T>,
    point: Option<[f64; 3]>,
    boundary: TransformBoundary,
    source_offset: [usize; 3],
    full_shape: [usize; 3],
) -> T
where
    T: VoxelElement,
{
    let Some(point) = point else {
        return T::from_f64(boundary_value(boundary));
    };
    let held_shape = [input.shape()[0], input.shape()[1], input.shape()[2]];
    let mut at = [0usize; 3];
    for axis in 0..3 {
        let rounded = point[axis].round();
        if !rounded.is_finite() {
            return T::from_f64(boundary_value(boundary));
        }
        match resolve_axis(rounded, full_shape[axis], boundary) {
            Some(value) => match local_axis(value, source_offset[axis], held_shape[axis]) {
                Some(local) => at[axis] = local,
                None => return T::from_f64(boundary_value(boundary)),
            },
            None => return T::from_f64(boundary_value(boundary)),
        }
    }
    input[at]
}

fn sample_linear<T>(
    input: ArrayView3<'_, T>,
    point: Option<[f64; 3]>,
    boundary: TransformBoundary,
    source_offset: [usize; 3],
    full_shape: [usize; 3],
) -> T
where
    T: VoxelElement,
{
    let Some(point) = point else {
        return T::from_f64(boundary_value(boundary));
    };
    let held_shape = [input.shape()[0], input.shape()[1], input.shape()[2]];
    let mut low = [0isize; 3];
    let mut fraction = [0.0f64; 3];
    for axis in 0..3 {
        if !point[axis].is_finite() {
            return T::from_f64(boundary_value(boundary));
        }
        low[axis] = point[axis].floor() as isize;
        fraction[axis] = point[axis] - low[axis] as f64;
    }
    let mut out = 0.0;
    for da in 0..=1 {
        for db in 0..=1 {
            for dc in 0..=1 {
                let choice = [da, db, dc];
                let mut at = [0usize; 3];
                let mut weight = 1.0;
                let mut outside = false;
                for axis in 0..3 {
                    let source = low[axis] + choice[axis] as isize;
                    weight *= if choice[axis] == 0 {
                        1.0 - fraction[axis]
                    } else {
                        fraction[axis]
                    };
                    match resolve_axis(source as f64, full_shape[axis], boundary) {
                        Some(value) => {
                            match local_axis(value, source_offset[axis], held_shape[axis]) {
                                Some(local) => at[axis] = local,
                                None => {
                                    outside = true;
                                    break;
                                }
                            }
                        }
                        None => {
                            outside = true;
                            break;
                        }
                    }
                }
                let value = if outside {
                    boundary_value(boundary)
                } else {
                    input[at].into_f64()
                };
                out += weight * value;
            }
        }
    }
    T::from_f64(out)
}

fn local_axis(global: usize, source_offset: usize, held_len: usize) -> Option<usize> {
    let local = global.checked_sub(source_offset)?;
    (local < held_len).then_some(local)
}

#[derive(Debug, Clone)]
struct SourceBounds {
    low: [usize; 3],
    high: [usize; 3],
    any: bool,
}

impl SourceBounds {
    fn empty() -> Self {
        Self {
            low: [usize::MAX; 3],
            high: [0; 3],
            any: false,
        }
    }

    fn include(&mut self, point: [usize; 3]) {
        self.any = true;
        for (axis, &value) in point.iter().enumerate() {
            self.low[axis] = self.low[axis].min(value);
            self.high[axis] = self.high[axis].max(value + 1);
        }
    }

    fn region(self) -> Region {
        if !self.any {
            return Region::new(&[0, 0, 0], &[1, 1, 1]);
        }
        Region::from_ranges(&[
            (self.low[0], self.high[0]),
            (self.low[1], self.high[1]),
            (self.low[2], self.high[2]),
        ])
    }
}

fn source_region_for(
    map: &CoordinateMap,
    output: &Region,
    input_volume: [usize; 3],
    interpolation: TransformInterpolation,
    boundary: TransformBoundary,
) -> Region {
    let start = [output.start[0], output.start[1], output.start[2]];
    let shape = [output.shape[0], output.shape[1], output.shape[2]];
    let mut bounds = SourceBounds::empty();
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                include_sample_footprint(
                    map.at([start[0] + i, start[1] + j, start[2] + k]),
                    input_volume,
                    interpolation,
                    boundary,
                    &mut bounds,
                );
            }
        }
    }
    bounds.region()
}

fn include_sample_footprint(
    point: Option<[f64; 3]>,
    input_volume: [usize; 3],
    interpolation: TransformInterpolation,
    boundary: TransformBoundary,
    bounds: &mut SourceBounds,
) {
    let Some(point) = point else {
        return;
    };
    match interpolation {
        TransformInterpolation::Nearest => {
            include_nearest_footprint(point, input_volume, boundary, bounds)
        }
        TransformInterpolation::Linear => {
            include_linear_footprint(point, input_volume, boundary, bounds)
        }
    }
}

fn include_nearest_footprint(
    point: [f64; 3],
    input_volume: [usize; 3],
    boundary: TransformBoundary,
    bounds: &mut SourceBounds,
) {
    let mut at = [0usize; 3];
    for axis in 0..3 {
        let rounded = point[axis].round();
        if !rounded.is_finite() {
            return;
        }
        let Some(value) = resolve_axis(rounded, input_volume[axis], boundary) else {
            return;
        };
        at[axis] = value;
    }
    bounds.include(at);
}

fn include_linear_footprint(
    point: [f64; 3],
    input_volume: [usize; 3],
    boundary: TransformBoundary,
    bounds: &mut SourceBounds,
) {
    let mut low = [0isize; 3];
    let mut fraction = [0.0f64; 3];
    for axis in 0..3 {
        if !point[axis].is_finite() {
            return;
        }
        low[axis] = point[axis].floor() as isize;
        fraction[axis] = point[axis] - low[axis] as f64;
    }
    for da in 0..=1 {
        for db in 0..=1 {
            for dc in 0..=1 {
                let choice = [da, db, dc];
                let mut at = [0usize; 3];
                let mut weight = 1.0;
                for axis in 0..3 {
                    weight *= if choice[axis] == 0 {
                        1.0 - fraction[axis]
                    } else {
                        fraction[axis]
                    };
                    let source = low[axis] + choice[axis] as isize;
                    let Some(value) = resolve_axis(source as f64, input_volume[axis], boundary)
                    else {
                        weight = 0.0;
                        break;
                    };
                    at[axis] = value;
                }
                if weight != 0.0 {
                    bounds.include(at);
                }
            }
        }
    }
}

fn resolve_axis(value: f64, len: usize, boundary: TransformBoundary) -> Option<usize> {
    if value >= 0.0 && value < len as f64 {
        return Some(value as usize);
    }
    match boundary {
        TransformBoundary::Constant(_) => None,
        TransformBoundary::Clamp => Some((value as isize).clamp(0, len as isize - 1) as usize),
    }
}

fn boundary_value(boundary: TransformBoundary) -> f64 {
    match boundary {
        TransformBoundary::Constant(value) => value,
        TransformBoundary::Clamp => 0.0,
    }
}

fn shapes_match(expected: [usize; 3], found: &[usize], context: &str) -> Result<()> {
    if found != expected.as_slice() {
        return Err(Error::InvalidArgument(format!(
            "{context}: expected shape {expected:?}, got {found:?}"
        )));
    }
    Ok(())
}

fn validate_finite_matrix3(matrix: [[f64; 3]; 3], offset: [f64; 3]) -> Result<()> {
    for row in matrix {
        validate_finite_vector(row, "affine transform")?;
    }
    validate_finite_vector(offset, "affine transform")
}

fn validate_finite_matrix4(matrix: [[f64; 4]; 4]) -> Result<()> {
    for row in matrix {
        for value in row {
            if !value.is_finite() {
                return Err(Error::InvalidArgument(format!(
                    "projective transform matrix entries must be finite, got {value}"
                )));
            }
        }
    }
    Ok(())
}

fn validate_finite_vector(values: [f64; 3], context: &str) -> Result<()> {
    for value in values {
        if !value.is_finite() {
            return Err(Error::InvalidArgument(format!(
                "{context} parameters must be finite, got {value}"
            )));
        }
    }
    Ok(())
}

fn validate_polar(
    center: [f64; 2],
    radius_scale: f64,
    angle_scale: f64,
    z_scale: f64,
) -> Result<()> {
    validate_finite_vector([center[0], center[1], z_scale], "polar transform")?;
    if !(radius_scale.is_finite() && radius_scale > 0.0) {
        return Err(Error::InvalidArgument(format!(
            "polar radius scale must be positive and finite, got {radius_scale}"
        )));
    }
    if !(angle_scale.is_finite() && angle_scale > 0.0) {
        return Err(Error::InvalidArgument(format!(
            "polar angle scale must be positive and finite, got {angle_scale}"
        )));
    }
    if z_scale == 0.0 {
        return Err(Error::InvalidArgument(
            "polar z scale must be non-zero".to_string(),
        ));
    }
    Ok(())
}

fn transform_cost(interpolation: TransformInterpolation) -> f64 {
    match interpolation {
        TransformInterpolation::Nearest => 7.0,
        TransformInterpolation::Linear => 36.0,
    }
}
