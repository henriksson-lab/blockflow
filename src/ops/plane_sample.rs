// SPDX-License-Identifier: MIT
//! Physical planar sampling, separate from registration or acceptance policy.
//!
//! Physical vectors and input arrays use z,y,x order. Integer voxel indices
//! denote centers: physical = origin + index * spacing. A plane is centered at
//! `center`; row/column tangents and its normal are unit orthogonal vectors.
//! Positive depth follows the supplied normal. All distances use the same
//! physical unit; the caller owns the meaning of that unit and normal direction.
//!
//! Output has shape [1, rows, columns]. Each depth is a separate plane/run, so
//! arbitrary depth lists do not pretend to be a regular volume. Sample an image
//! and a numeric/bool support image with the same op to obtain separate outputs.
//! No support division, depth averaging or acceptance threshold is applied here.
use crate::{
    assemble::{Phase, PlanBuilder},
    decomposition::PhaseDecomposition,
    dtype::Dtype,
    error::{Error, Result},
    geometry::BlockGrid,
    op::{Anchor, BlockOp, Geometry, Placement, Slicing, SourceInputs},
    ops::transform::{
        warp_phase, CoordinateMap, TransformBoundary, TransformInterpolation, WarpOp,
    },
    reach::Reach,
    voxels::{VoxelElement, Voxels},
    Chain,
};

#[derive(Clone, Debug, PartialEq)]
pub struct PhysicalGrid {
    origin: [f64; 3],
    spacing: [f64; 3],
}
impl PhysicalGrid {
    pub fn new(origin: [f64; 3], spacing: [f64; 3]) -> Result<Self> {
        if origin.iter().any(|x| !x.is_finite())
            || spacing.iter().any(|x| !x.is_finite() || *x <= 0.)
        {
            return Err(Error::invalid(
                "physical grid needs finite origin and positive finite spacing",
            ));
        }
        Ok(Self { origin, spacing })
    }
    pub fn origin(&self) -> [f64; 3] {
        self.origin
    }
    pub fn spacing(&self) -> [f64; 3] {
        self.spacing
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlaneGrid {
    center: [f64; 3],
    row: [f64; 3],
    column: [f64; 3],
    normal: [f64; 3],
    shape: [usize; 2],
    spacing: [f64; 2],
}
impl PlaneGrid {
    /// Tangents and normal must already be unit and orthogonal (tolerance 1e-8).
    /// Either handedness is allowed; orientation is never silently changed.
    pub fn new(
        center: [f64; 3],
        row: [f64; 3],
        column: [f64; 3],
        normal: [f64; 3],
        shape: [usize; 2],
        spacing: [f64; 2],
    ) -> Result<Self> {
        let dot = |a: [f64; 3], b: [f64; 3]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        let axes = [row, column, normal];
        if center
            .iter()
            .chain(axes.iter().flatten())
            .any(|x| !x.is_finite())
            || shape.contains(&0)
            || shape[0].checked_mul(shape[1]).is_none()
            || spacing.iter().any(|x| !x.is_finite() || *x <= 0.)
            || axes.iter().any(|a| (dot(*a, *a) - 1.).abs() > 1e-8)
            || dot(row, column).abs() > 1e-8
            || dot(row, normal).abs() > 1e-8
            || dot(column, normal).abs() > 1e-8
        {
            return Err(Error::invalid(
                "plane needs finite center, orthonormal axes, nonempty shape and positive spacing",
            ));
        }
        Ok(Self {
            center,
            row,
            column,
            normal,
            shape,
            spacing,
        })
    }
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
    /// Continuous input voxel coordinates at the output pixel center.
    pub fn coordinate(
        &self,
        grid: &PhysicalGrid,
        row: usize,
        column: usize,
        depth: f64,
    ) -> [f64; 3] {
        let r = (row as f64 - (self.shape[0] - 1) as f64 / 2.) * self.spacing[0];
        let c = (column as f64 - (self.shape[1] - 1) as f64 / 2.) * self.spacing[1];
        std::array::from_fn(|a| {
            let base = (self.center[a] + self.row[a] * r) + self.column[a] * c;
            (base + self.normal[a] * depth - grid.origin[a]) / grid.spacing[a]
        })
    }
    fn affine(&self, grid: &PhysicalGrid, depth: f64) -> Result<CoordinateMap> {
        let offset = self.coordinate(grid, 0, 0, depth);
        let matrix = std::array::from_fn(|a| {
            [
                0.,
                self.row[a] * self.spacing[0] / grid.spacing[a],
                self.column[a] * self.spacing[1] / grid.spacing[a],
            ]
        });
        CoordinateMap::affine(matrix, offset)
    }
}

/// Strict trilinear interpolation, with f64 accumulation in z,y,x bit order.
/// Returns None for empty axes, coordinates outside [0,n-1], or a nonfinite
/// contributing value. Zero-weight corners are not read. Upper neighbors clamp
/// at the final center. The getter provides resident or region-local storage;
/// no ndarray version or whole-image allocation is imposed on the caller.
pub fn trilinear(
    shape: [usize; 3],
    point: [f64; 3],
    mut get: impl FnMut([usize; 3]) -> f64,
) -> Option<f64> {
    if (0..3).any(|a| {
        shape[a] == 0 || !point[a].is_finite() || point[a] < 0. || point[a] > (shape[a] - 1) as f64
    }) {
        return None;
    }
    let lo: [usize; 3] = std::array::from_fn(|a| point[a].floor() as usize);
    let hi: [usize; 3] = std::array::from_fn(|a| (lo[a] + 1).min(shape[a] - 1));
    let f: [f64; 3] = std::array::from_fn(|a| point[a] - lo[a] as f64);
    let mut value = 0.;
    for bits in 0..8 {
        let mut at = [0; 3];
        let mut weight = 1.;
        for a in 0..3 {
            let up = bits & (1 << a) != 0;
            at[a] = if up { hi[a] } else { lo[a] };
            weight *= if up { f[a] } else { 1. - f[a] };
        }
        if weight == 0. {
            continue;
        }
        let v = get(at);
        if !v.is_finite() {
            return None;
        }
        value += v * weight;
    }
    value.is_finite().then_some(value)
}

/// Resident plane samples. Invalid samples are NaN, independent of support.
/// Use the same function with a 0/1 mask getter to sample support separately.
pub fn sample_plane(
    grid: &PhysicalGrid,
    plane: &PlaneGrid,
    depth: f64,
    shape: [usize; 3],
    mut get: impl FnMut([usize; 3]) -> f64,
) -> Result<Vec<f64>> {
    if !depth.is_finite() {
        return Err(Error::invalid("plane depth must be finite"));
    }
    let mut out = Vec::with_capacity(plane.shape[0] * plane.shape[1]);
    for row in 0..plane.shape[0] {
        for col in 0..plane.shape[1] {
            out.push(
                trilinear(shape, plane.coordinate(grid, row, col, depth), &mut get)
                    .unwrap_or(f64::NAN),
            );
        }
    }
    Ok(out)
}

/// Strict physical plane sampler. Always produces f64, including for bool mask
/// input. NaN denotes an invalid sample; geometric validity is not mask support.
#[derive(Clone, Debug)]
pub struct PlaneSampleOp {
    grid: PhysicalGrid,
    plane: PlaneGrid,
    depth: f64,
    footprint: WarpOp,
}
impl PlaneSampleOp {
    pub fn new(grid: PhysicalGrid, plane: PlaneGrid, depth: f64) -> Result<Self> {
        if !depth.is_finite() {
            return Err(Error::invalid("plane depth must be finite"));
        }
        let footprint = WarpOp::new(
            "physical-plane",
            plane.affine(&grid, depth)?,
            [1, plane.shape[0], plane.shape[1]],
            TransformInterpolation::Linear,
            TransformBoundary::Clamp,
        )?;
        Ok(Self {
            grid,
            plane,
            depth,
            footprint,
        })
    }
    /// The affine warp provides the source footprint. Expand by one source
    /// voxel for floating-point differences between physical-coordinate and
    /// affine evaluation at integer boundaries. Execution checks actual corners
    /// and refuses an understated fetch instead of substituting background.
    pub fn phase(
        &self,
        input_shape: [usize; 3],
        output_grid: BlockGrid,
    ) -> Result<PhaseDecomposition> {
        let footprints = warp_phase(
            vec![0],
            vec!["physical-plane".into()],
            &self.footprint,
            input_shape,
            output_grid.clone(),
        )?;
        // A planar sample reads no neighboring OUTPUT pixels. The transformed
        // input neighborhood is represented by `source`, not an output halo.
        // Retaining WarpOp's Reach::all output halo would compute the whole
        // plane for each tile from only that tile's fetched source region.
        let mut phase = PhaseDecomposition::derive(
            vec![0],
            vec!["physical-plane".into()],
            [0; 3],
            [0; 3],
            output_grid,
        );
        for (block, footprint) in phase.blocks.iter_mut().zip(footprints.blocks) {
            block.source = footprint.source;
            let source = &mut block.source;
            for a in 0..3 {
                let end = (source.start[a] + source.shape[a])
                    .saturating_add(1)
                    .min(input_shape[a]);
                source.start[a] = source.start[a].saturating_sub(1);
                source.shape[a] = end - source.start[a];
            }
        }
        Ok(phase)
    }
    fn apply_typed<T: VoxelElement>(
        &self,
        input: &Voxels,
        out: &mut Voxels,
        at: &Placement,
    ) -> Result<()> {
        let input = input.view::<T>()?;
        let mut output = out.view_mut::<f64>()?;
        let held = input.shape();
        for r in 0..output.shape()[1] {
            for c in 0..output.shape()[2] {
                let point = self.plane.coordinate(
                    &self.grid,
                    at.output.offset[1] + r,
                    at.output.offset[2] + c,
                    self.depth,
                );
                let mut missing = false;
                let value = trilinear(at.input.volume, point, |global| {
                    let mut local = [0; 3];
                    for a in 0..3 {
                        match global[a].checked_sub(at.input.offset[a]) {
                            Some(i) if i < held[a] => local[a] = i,
                            _ => {
                                missing = true;
                                return f64::NAN;
                            }
                        }
                    }
                    input[local].into_f64()
                });
                if missing {
                    return Err(Error::invalid(format!(
                        "physical-plane source fetch does not cover interpolation corners: point {point:?}, input {:?}, held {held:?}, output {:?}", at.input, at.output
                    )));
                }
                output[[0, r, c]] = value.unwrap_or(f64::NAN);
            }
        }
        Ok(())
    }
}
/// Append a physical plane on an output grid `[1, rows, columns]`.
pub fn append_plane_sample_phase(
    builder: &mut PlanBuilder,
    op: PlaneSampleOp,
    input_shape: [usize; 3],
    output_grid: BlockGrid,
) -> Result<Phase> {
    let phase = op.phase(input_shape, output_grid)?;
    builder.pixels_decomposed(Chain::op(op), phase)
}
impl BlockOp for PlaneSampleOp {
    fn name(&self) -> &'static str {
        "physical-plane"
    }
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }
    fn reach_spec(&self, _volume: [usize; 3]) -> Reach {
        [0; 3].into()
    }
    fn geometry(&self, volume: [usize; 3]) -> Geometry {
        self.footprint.geometry(volume)
    }
    fn output_shape(&self, _input: [usize; 3]) -> [usize; 3] {
        [1, self.plane.shape[0], self.plane.shape[1]]
    }
    fn takes_extent_from_placement(&self) -> bool {
        true
    }
    fn placed_output_shape(&self, _input: [usize; 3], at: &Placement) -> [usize; 3] {
        at.writes().unwrap_or(self.output_shape([1; 3]))
    }
    fn slicing(&self) -> Slicing {
        Slicing::Stencil
    }
    fn accepts(&self, dtype: Dtype) -> bool {
        dtype != Dtype::F16
    }
    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::F64
    }
    fn cost_per_voxel(&self) -> f64 {
        100.
    }
    fn apply(&self, input: &Voxels, out: &mut Voxels, at: &Anchor) -> Result<()> {
        self.apply_placed(
            input,
            SourceInputs::none(),
            out,
            &Placement::new(at.clone(), Anchor::whole(self.output_shape(at.volume)))
                .writing(out.shape()),
        )
    }
    fn apply_placed(
        &self,
        input: &Voxels,
        _sources: SourceInputs<'_>,
        out: &mut Voxels,
        at: &Placement,
    ) -> Result<()> {
        if out.shape()[0] != 1
            || at.output.offset[0] != 0
            || at.output.volume != self.output_shape(at.input.volume)
            || (0..3).any(|a| {
                at.input.offset[a]
                    .checked_add(input.shape()[a])
                    .is_none_or(|end| end > at.input.volume[a])
                    || at.output.offset[a]
                        .checked_add(out.shape()[a])
                        .is_none_or(|end| end > at.output.volume[a])
            })
        {
            return Err(Error::invalid(
                "physical-plane placement is outside input or plane",
            ));
        }
        match input {
            Voxels::Bool(_) => self.apply_typed::<bool>(input, out, at),
            Voxels::U8(_) => self.apply_typed::<u8>(input, out, at),
            Voxels::U16(_) => self.apply_typed::<u16>(input, out, at),
            Voxels::U32(_) => self.apply_typed::<u32>(input, out, at),
            Voxels::U64(_) => self.apply_typed::<u64>(input, out, at),
            Voxels::I8(_) => self.apply_typed::<i8>(input, out, at),
            Voxels::I16(_) => self.apply_typed::<i16>(input, out, at),
            Voxels::I32(_) => self.apply_typed::<i32>(input, out, at),
            Voxels::I64(_) => self.apply_typed::<i64>(input, out, at),
            Voxels::F32(_) => self.apply_typed::<f32>(input, out, at),
            Voxels::F64(_) => self.apply_typed::<f64>(input, out, at),
        }
    }
}
