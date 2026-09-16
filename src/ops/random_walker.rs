// SPDX-License-Identifier: MIT
//
// Original work for this crate.

//! Seeded random-walker segmentation.
//!
//! The data-layer implementation here solves the reduced graph Laplacian over
//! unseeded voxels:
//!
//! ```text
//! L_UU x = -L_UL y
//! ```
//!
//! Seeds are fixed Dirichlet values. Only unseeded voxels become unknown rows,
//! and seeded neighbours contribute to the right-hand side.

use std::collections::VecDeque;
use std::sync::Arc;

use ndarray::{Array3, Array4, ArrayView3, ArrayViewMut3};

use crate::assemble::{Phase, PlanBuilder};
use crate::decomposition::PhaseDecomposition;
use crate::dtype::Dtype;
use crate::env::BlockBuf;
use crate::error::{Error, Result};
use crate::fragment::{
    pack_u64, unpack_u64, BlockOutput, BlockView, Coverage, FragmentInput, FragmentOp,
    FragmentOutput, PhaseView, SeamFold, SidecarSize, SourceBlocks,
};
use crate::geometry::BlockGrid;
use crate::op::{
    Anchor, BlockOp, Geometry, InputMap, Placement, Slicing, SourceInput, SourceInputs,
};
use crate::reach::Reach;
use crate::region::Region;
use crate::sidecar::Lifecycle;
use crate::table::{Column, RowBuilder, Schema, Table, Value};
use crate::voxels::VoxelElement;
use crate::{Chain, Voxels};

const NO_ROW: usize = usize::MAX;
const NO_ROW_U64: u64 = u64::MAX;
const RANDOM_WALKER_SOLUTION_MAGIC: u64 = 0x4257_5241_4e44_534f;
const RANDOM_WALKER_SOLUTION_VERSION: u64 = 1;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RandomWalkerConfig {
    pub max_residual_norm: f64,
    pub max_iterations: usize,
}

impl Default for RandomWalkerConfig {
    fn default() -> Self {
        Self {
            max_residual_norm: 1.0e-8,
            max_iterations: 10_000,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradyWeights {
    pub beta: f64,
    pub min_edge_weight: f64,
}

impl GradyWeights {
    pub fn new(beta: f64, min_edge_weight: f64) -> Result<Self> {
        if !(beta.is_finite() && beta >= 0.0) {
            return Err(Error::InvalidArgument(format!(
                "random-walker beta must be finite and non-negative, got {beta}"
            )));
        }
        if !(min_edge_weight.is_finite() && min_edge_weight >= 0.0) {
            return Err(Error::InvalidArgument(format!(
                "random-walker minimum edge weight must be finite and non-negative, got \
                 {min_edge_weight}"
            )));
        }
        Ok(Self {
            beta,
            min_edge_weight,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RandomWalkerWeights {
    edges: Array4<f64>,
}

impl RandomWalkerWeights {
    pub fn grady<T>(input: ArrayView3<'_, T>, params: GradyWeights) -> Result<Self>
    where
        T: VoxelElement,
    {
        let shape = shape_of(input);
        let mut edges = Array4::<f64>::zeros((shape[0], shape[1], shape[2], 3));
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    let here = input[[i, j, k]].into_f64();
                    if !here.is_finite() {
                        return Err(Error::InvalidArgument(format!(
                            "random-walker input contains non-finite value at [{i}, {j}, {k}]"
                        )));
                    }
                    let at = [i, j, k];
                    for axis in 0..3 {
                        let mut next = at;
                        next[axis] += 1;
                        if next[axis] >= shape[axis] {
                            continue;
                        }
                        let there = input[next].into_f64();
                        if !there.is_finite() {
                            return Err(Error::InvalidArgument(format!(
                                "random-walker input contains non-finite value at {:?}",
                                next
                            )));
                        }
                        let diff = here - there;
                        edges[[i, j, k, axis]] = (-params.beta * diff * diff)
                            .exp()
                            .max(params.min_edge_weight);
                    }
                }
            }
        }
        Ok(Self { edges })
    }

    pub fn shape(&self) -> [usize; 3] {
        [
            self.edges.shape()[0],
            self.edges.shape()[1],
            self.edges.shape()[2],
        ]
    }

    pub fn edge(&self, lower: [usize; 3], axis: usize) -> f64 {
        self.edges[[lower[0], lower[1], lower[2], axis]]
    }

    pub fn from_packed(packed: ArrayView3<'_, f64>, input_shape: [usize; 3]) -> Result<Self> {
        shapes_match(
            packed_weight_shape(input_shape),
            packed.shape(),
            "packed random-walker weights",
        )?;
        let mut edges = Array4::<f64>::zeros((input_shape[0], input_shape[1], input_shape[2], 3));
        for i in 0..input_shape[0] {
            for j in 0..input_shape[1] {
                for k in 0..input_shape[2] {
                    for axis in 0..3 {
                        let value = packed[[i, j, packed_axis(k, axis)]];
                        if !(value.is_finite() && value >= 0.0) {
                            return Err(Error::InvalidArgument(format!(
                                "packed random-walker weight at [{i}, {j}, {}, axis {axis}] must \
                                 be finite and non-negative, got {value}",
                                packed_axis(k, axis)
                            )));
                        }
                        edges[[i, j, k, axis]] = value;
                    }
                }
            }
        }
        Ok(Self { edges })
    }

    pub fn write_packed_into(&self, mut out: ArrayViewMut3<'_, f64>) -> Result<()> {
        shapes_match(
            packed_weight_shape(self.shape()),
            out.shape(),
            "packed random-walker weights",
        )?;
        for i in 0..self.shape()[0] {
            for j in 0..self.shape()[1] {
                for k in 0..self.shape()[2] {
                    for axis in 0..3 {
                        out[[i, j, packed_axis(k, axis)]] = self.edge([i, j, k], axis);
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RandomWalkerSolve {
    pub iterations: usize,
    pub residual_norm: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RandomWalkerSystem {
    rows: Vec<Vec<(usize, f64)>>,
    rhs: Vec<f64>,
    row_ids: Array3<usize>,
}

impl RandomWalkerSystem {
    pub fn rows(&self) -> usize {
        self.rows.len()
    }

    pub fn row_entries(&self, row: usize) -> &[(usize, f64)] {
        &self.rows[row]
    }

    pub fn rhs(&self) -> &[f64] {
        &self.rhs
    }

    pub fn row_id_at(&self, at: [usize; 3]) -> Option<usize> {
        let row = self.row_ids[[at[0], at[1], at[2]]];
        (row != NO_ROW).then_some(row)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RandomWalkerImages {
    pub seed_image: usize,
    pub intensity: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RandomWalkerSolveImages {
    pub seed_image: usize,
    pub row_ids: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RandomWalkerSparseColumns {
    pub row: usize,
    pub col: usize,
    pub value: usize,
    pub rhs: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RandomWalkerRowsOp {
    name: &'static str,
    stream: String,
    lifecycle: Lifecycle,
    input_shape: [usize; 3],
    images: RandomWalkerImages,
    params: GradyWeights,
    cost: f64,
}

impl RandomWalkerRowsOp {
    pub fn new(
        name: &'static str,
        stream: impl Into<String>,
        lifecycle: Lifecycle,
        input_shape: [usize; 3],
        images: RandomWalkerImages,
        params: GradyWeights,
    ) -> Result<Self> {
        if input_shape.contains(&0) {
            return Err(Error::InvalidArgument(format!(
                "{name} needs a non-empty random-walker volume, got {input_shape:?}"
            )));
        }
        Ok(Self {
            name,
            stream: stream.into(),
            lifecycle,
            input_shape,
            images,
            params,
            cost: 12.0,
        })
    }

    pub fn stream(&self) -> &str {
        &self.stream
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }
}

impl FragmentOp for RandomWalkerRowsOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        1
    }

    fn reads_pixels(&self) -> bool {
        true
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![
            FragmentOutput::new(self.stream.clone(), self.lifecycle, Coverage::EveryBlock)
                .sized(SidecarSize::row_table(&random_walker_sparse_schema(), 7)),
        ]
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        vec![
            SourceInput::new(self.images.seed_image, Reach::symmetric([1, 1, 1]))
                .holding(Dtype::F64),
            SourceInput::new(self.images.intensity, Reach::symmetric([1, 1, 1]))
                .holding(Dtype::F64),
        ]
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::PerBlock)
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        self.apply_with(at, SourceBlocks::none())
    }

    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let BlockBuf::Array(row_ids) = at.pixels()? else {
            return Ok(BlockOutput::fragment(
                self.stream.clone(),
                RowBuilder::new(Arc::new(random_walker_sparse_schema())).encode(),
            ));
        };
        let BlockBuf::Array(seed_image) = sources.get(self.images.seed_image)? else {
            return Err(Error::InvalidArgument(format!(
                "{} needs seed image {} as an array source",
                self.name, self.images.seed_image
            )));
        };
        let BlockBuf::Array(intensity) = sources.get(self.images.intensity)? else {
            return Err(Error::InvalidArgument(format!(
                "{} needs intensity image {} as an array source",
                self.name, self.images.intensity
            )));
        };
        Ok(BlockOutput::fragment(
            self.stream.clone(),
            encode_random_walker_rows(
                row_ids.view::<u64>()?,
                seed_image.view::<f64>()?,
                intensity.view::<f64>()?,
                self.params,
                self.input_shape,
                at.read,
                at.core,
            )?,
        ))
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RandomWalkerSolveOp {
    name: &'static str,
    input: String,
    input_phase: usize,
    lattice: [usize; 3],
    images: RandomWalkerSolveImages,
    config: RandomWalkerConfig,
    cost: f64,
}

impl RandomWalkerSolveOp {
    pub fn new(
        name: &'static str,
        input: impl Into<String>,
        input_phase: usize,
        lattice: [usize; 3],
        images: RandomWalkerSolveImages,
        config: RandomWalkerConfig,
    ) -> Result<Self> {
        if lattice.contains(&0) {
            return Err(Error::InvalidArgument(format!(
                "{name} needs a non-empty sparse-row lattice, got {lattice:?}"
            )));
        }
        Ok(Self {
            name,
            input: input.into(),
            input_phase,
            lattice,
            images,
            config,
            cost: 1.0,
        })
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }
}

impl FragmentOp for RandomWalkerSolveOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn writes_pixels(&self) -> bool {
        true
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::F64
    }

    fn barrier(&self) -> bool {
        true
    }

    fn gathers(&self) -> bool {
        false
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own(self.input.clone(), self.input_phase).with_reach(self.lattice)]
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        vec![
            SourceInput::voxelwise(self.images.seed_image).holding(Dtype::F64),
            SourceInput::voxelwise(self.images.row_ids).holding(Dtype::U64),
        ]
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::Unordered)
    }

    fn reduce(&self, at: &PhaseView<'_>) -> Result<Vec<u8>> {
        let fragments = at
            .fragments(&self.input)?
            .into_iter()
            .map(|(key, bytes)| (key.block, bytes));
        let table = collect_random_walker_rows(at.volume(), fragments)?;
        let (solution, report) = solve_random_walker_sparse_table(&table, self.config)?;
        encode_random_walker_solution(&solution, &report)
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        self.apply_with(at, SourceBlocks::none())
    }

    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let (solution, _) = decode_random_walker_solution(at.reduced)?;
        let BlockBuf::Array(seed_image) = sources.get(self.images.seed_image)? else {
            return Err(Error::InvalidArgument(format!(
                "{} needs seed image {} as an array source",
                self.name, self.images.seed_image
            )));
        };
        let BlockBuf::Array(row_ids) = sources.get(self.images.row_ids)? else {
            return Err(Error::InvalidArgument(format!(
                "{} needs row-id image {} as an array source",
                self.name, self.images.row_ids
            )));
        };
        let mut out = at.output_buffer(0.0)?;
        if let Some(array) = out.as_array_mut() {
            reconstruct_random_walker_block_into(
                seed_image.view::<f64>()?,
                row_ids.view::<u64>()?,
                &solution,
                at.read,
                array.view_mut::<f64>()?,
            )?;
        }
        Ok(BlockOutput::nothing().with_pixels(out))
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RandomWalkerRowIdOp {
    name: &'static str,
    input_shape: [usize; 3],
    cost: f64,
}

impl RandomWalkerRowIdOp {
    pub fn new(name: &'static str, input_shape: [usize; 3]) -> Result<Self> {
        if input_shape.contains(&0) {
            return Err(Error::InvalidArgument(format!(
                "{name} needs a non-empty seed image shape, got {input_shape:?}"
            )));
        }
        Ok(Self {
            name,
            input_shape,
            cost: 1.0,
        })
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }
}

impl BlockOp for RandomWalkerRowIdOp {
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
        Geometry::new(self.input_shape, vec![InputMap::Stencil(Reach::all())])
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        dtype == Dtype::F64
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::U64
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, at: &Anchor) -> Result<()> {
        self.apply_placed(
            input,
            SourceInputs::none(),
            out,
            &Placement::same(at.clone()),
        )
    }

    fn apply_placed(
        &self,
        input: &Voxels,
        _sources: SourceInputs<'_>,
        out: &mut Voxels,
        at: &Placement,
    ) -> Result<()> {
        if at.input.offset != [0, 0, 0] || input.shape() != self.input_shape {
            return Err(Error::InvalidArgument(format!(
                "{} needs the whole seed image {:?}, got offset {:?} and held shape {:?}",
                self.name,
                self.input_shape,
                at.input.offset,
                input.shape()
            )));
        }
        random_walker_row_ids_block_into(
            input.view::<f64>()?,
            at.output.offset,
            at.output.volume,
            out.view_mut::<u64>()?,
        )
        .map(|_| ())
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

pub fn random_walker_binary_into<T>(
    input: ArrayView3<'_, T>,
    seeds: ArrayView3<'_, Option<f64>>,
    weights: GradyWeights,
    config: RandomWalkerConfig,
    out: ArrayViewMut3<'_, f64>,
) -> Result<RandomWalkerSolve>
where
    T: VoxelElement,
{
    let weights = RandomWalkerWeights::grady(input, weights)?;
    let system = assemble_random_walker_system(&weights, seeds)?;
    solve_random_walker_system_into(&system, seeds, config, out)
}

pub fn random_walker_binary_seed_image_into<T>(
    input: ArrayView3<'_, T>,
    seed_image: ArrayView3<'_, f64>,
    weights: GradyWeights,
    config: RandomWalkerConfig,
    out: ArrayViewMut3<'_, f64>,
) -> Result<RandomWalkerSolve>
where
    T: VoxelElement,
{
    let seeds = seed_probabilities_from_image(seed_image)?;
    random_walker_binary_into(input, seeds.view(), weights, config, out)
}

pub fn random_walker_row_ids_into(
    seed_image: ArrayView3<'_, f64>,
    out: ArrayViewMut3<'_, u64>,
) -> Result<usize> {
    random_walker_row_ids_block_into(seed_image, [0, 0, 0], shape_of(seed_image), out)
}

fn random_walker_row_ids_block_into(
    seed_image: ArrayView3<'_, f64>,
    output_offset: [usize; 3],
    output_extent: [usize; 3],
    mut out: ArrayViewMut3<'_, u64>,
) -> Result<usize> {
    shapes_match(output_extent, out.shape(), "random-walker row-id output")?;
    let seeds = seed_probabilities_from_image(seed_image)?;
    let (row_ids, rows) = row_ids_for_seeds(seeds.view());
    let shape = shape_of(seeds.view());
    if output_offset
        .iter()
        .zip(output_extent)
        .zip(shape)
        .any(|((&offset, extent), axis_len)| offset + extent > axis_len)
    {
        return Err(Error::InvalidArgument(format!(
            "random-walker row-id block offset {output_offset:?} with extent {output_extent:?} is \
             outside seed image {shape:?}"
        )));
    }
    for i in 0..output_extent[0] {
        for j in 0..output_extent[1] {
            for k in 0..output_extent[2] {
                let at = [
                    output_offset[0] + i,
                    output_offset[1] + j,
                    output_offset[2] + k,
                ];
                out[[i, j, k]] = match row_ids[[at[0], at[1], at[2]]] {
                    NO_ROW => NO_ROW_U64,
                    row => u64::try_from(row).map_err(|_| {
                        Error::InvalidArgument(
                            "random-walker row id does not fit in u64".to_string(),
                        )
                    })?,
                };
            }
        }
    }
    Ok(rows)
}

pub fn random_walker_sparse_schema() -> Schema {
    Schema::new(vec![
        Column::u64("row"),
        Column::u64("col"),
        Column::f64("value"),
        Column::f64("rhs"),
    ])
    .expect("random-walker sparse schema names each column once")
}

pub fn random_walker_sparse_columns() -> RandomWalkerSparseColumns {
    RandomWalkerSparseColumns {
        row: 0,
        col: 1,
        value: 2,
        rhs: 3,
    }
}

pub fn encode_random_walker_rows(
    row_ids: ArrayView3<'_, u64>,
    seed_image: ArrayView3<'_, f64>,
    intensity: ArrayView3<'_, f64>,
    params: GradyWeights,
    volume: [usize; 3],
    read: &Region,
    core: &Region,
) -> Result<Vec<u8>> {
    shapes_match(
        shape_of(row_ids),
        seed_image.shape(),
        "random-walker row seed block",
    )?;
    shapes_match(
        shape_of(row_ids),
        intensity.shape(),
        "random-walker row intensity block",
    )?;
    let mut rows = RowBuilder::new(Arc::new(random_walker_sparse_schema()));
    for i in core.start[0]..core.start[0] + core.shape[0] {
        for j in core.start[1]..core.start[1] + core.shape[1] {
            for k in core.start[2]..core.start[2] + core.shape[2] {
                let at = [i, j, k];
                let local = local_in_read(read, at)?;
                let row = row_ids[local];
                if row == NO_ROW_U64 {
                    continue;
                }
                let mut diagonal = 0.0;
                let mut rhs = 0.0;
                let mut off_diagonal = Vec::with_capacity(6);
                for neighbour in face_neighbours(at, volume) {
                    let neighbour_local = local_in_read(read, neighbour)?;
                    let weight = grady_weight_between(intensity, read, at, neighbour, params)?;
                    if weight == 0.0 {
                        continue;
                    }
                    diagonal += weight;
                    match seed_image[neighbour_local] {
                        seed if seed.is_nan() => {
                            let col = row_ids[neighbour_local];
                            if col == NO_ROW_U64 {
                                return Err(Error::InvalidArgument(format!(
                                    "random-walker neighbour {neighbour:?} is unseeded but has \
                                     no row id"
                                )));
                            }
                            off_diagonal.push((col, -weight));
                        }
                        seed if (0.0..=1.0).contains(&seed) => {
                            rhs += weight * seed;
                        }
                        seed => {
                            return Err(Error::InvalidArgument(format!(
                                "random-walker seed image at {neighbour:?} must be NaN for \
                                 unknown or a probability in 0..=1, got {seed}"
                            )));
                        }
                    }
                }
                rows.push(
                    at,
                    &[
                        Value::U64(row),
                        Value::U64(row),
                        Value::F64(diagonal),
                        Value::F64(rhs),
                    ],
                )?;
                for (col, value) in off_diagonal {
                    rows.push(
                        at,
                        &[
                            Value::U64(row),
                            Value::U64(col),
                            Value::F64(value),
                            Value::F64(0.0),
                        ],
                    )?;
                }
            }
        }
    }
    Ok(rows.encode())
}

pub fn collect_random_walker_rows(
    volume: [usize; 3],
    fragments: impl IntoIterator<Item = ([usize; 3], Vec<u8>)>,
) -> Result<Table> {
    let mut table = Table::new(volume, random_walker_sparse_schema())?;
    for (block, bytes) in fragments {
        table.write(block, &bytes)?;
    }
    table.seal()?;
    Ok(table)
}

pub fn assemble_random_walker_system_from_sparse_table(
    table: &Table,
    row_ids: ArrayView3<'_, u64>,
) -> Result<RandomWalkerSystem> {
    let shape = shape_of(row_ids);
    if table.schema() != &random_walker_sparse_schema() {
        return Err(Error::InvalidArgument(
            "random-walker sparse table has the wrong schema".to_string(),
        ));
    }
    if table.volume() != shape {
        return Err(Error::InvalidArgument(format!(
            "random-walker sparse table is over {:?}, but row IDs have shape {shape:?}",
            table.volume()
        )));
    }
    let (row_id_array, rows) = decode_row_id_image(row_ids)?;
    let mut matrix = vec![Vec::new(); rows];
    let mut rhs = vec![0.0; rows];
    let columns = random_walker_sparse_columns();
    for entry in table.scan(&Region::new(&[0, 0, 0], &shape))? {
        let at = entry.at();
        let row = usize::try_from(entry.u64(columns.row)?).map_err(|_| {
            Error::InvalidArgument(format!(
                "random-walker sparse row at {at:?} does not fit usize"
            ))
        })?;
        let col = usize::try_from(entry.u64(columns.col)?).map_err(|_| {
            Error::InvalidArgument(format!(
                "random-walker sparse column at {at:?} does not fit usize"
            ))
        })?;
        if row >= rows || col >= rows {
            return Err(Error::InvalidArgument(format!(
                "random-walker sparse entry at {at:?} names row {row} column {col}, but there \
                 are {rows} unknown row(s)"
            )));
        }
        if row_id_array[[at[0], at[1], at[2]]] != row {
            return Err(Error::InvalidArgument(format!(
                "random-walker sparse entry at {at:?} names row {row}, but the row-id image \
                 says {:?}",
                row_id_array[[at[0], at[1], at[2]]]
            )));
        }
        let value = entry.f64(columns.value)?;
        let row_rhs = entry.f64(columns.rhs)?;
        if !(value.is_finite() && row_rhs.is_finite()) {
            return Err(Error::InvalidArgument(format!(
                "random-walker sparse entry at {at:?} contains non-finite value {value} or RHS \
                 {row_rhs}"
            )));
        }
        matrix[row].push((col, value));
        rhs[row] += row_rhs;
    }
    Ok(RandomWalkerSystem {
        rows: matrix,
        rhs,
        row_ids: row_id_array,
    })
}

pub fn grady_weights_packed_into<T>(
    input: ArrayView3<'_, T>,
    params: GradyWeights,
    out: ArrayViewMut3<'_, f64>,
) -> Result<()>
where
    T: VoxelElement,
{
    let shape = shape_of(input);
    grady_weights_packed_block_into(input, params, [0, 0, 0], packed_weight_shape(shape), out)
}

fn grady_weights_packed_block_into<T>(
    input: ArrayView3<'_, T>,
    params: GradyWeights,
    output_offset: [usize; 3],
    output_shape: [usize; 3],
    mut out: ArrayViewMut3<'_, f64>,
) -> Result<()>
where
    T: VoxelElement,
{
    let input_shape = shape_of(input);
    if output_shape != packed_weight_shape(input_shape) {
        return Err(Error::InvalidArgument(format!(
            "Grady random-walker weights for {input_shape:?} write {:?}, got declared output \
             shape {output_shape:?}",
            packed_weight_shape(input_shape)
        )));
    }
    for i in 0..out.shape()[0] {
        for j in 0..out.shape()[1] {
            for packed in 0..out.shape()[2] {
                let global = [
                    output_offset[0] + i,
                    output_offset[1] + j,
                    output_offset[2] + packed,
                ];
                out[[i, j, packed]] = grady_weight_at(input, params, global)?;
            }
        }
    }
    Ok(())
}

fn grady_weight_at<T>(
    input: ArrayView3<'_, T>,
    params: GradyWeights,
    output: [usize; 3],
) -> Result<f64>
where
    T: VoxelElement,
{
    let input_shape = shape_of(input);
    let axis = output[2] % 3;
    let k = output[2] / 3;
    let at = [output[0], output[1], k];
    if at[0] >= input_shape[0] || at[1] >= input_shape[1] || at[2] >= input_shape[2] {
        return Err(Error::InvalidArgument(format!(
            "packed random-walker weight coordinate {output:?} is outside {:?}",
            packed_weight_shape(input_shape)
        )));
    }
    let mut next = at;
    next[axis] += 1;
    if next[axis] >= input_shape[axis] {
        return Ok(0.0);
    }
    let here = input[at].into_f64();
    let there = input[next].into_f64();
    if !(here.is_finite() && there.is_finite()) {
        return Err(Error::InvalidArgument(format!(
            "random-walker input contains non-finite value at edge {:?} axis {axis}",
            at
        )));
    }
    let diff = here - there;
    Ok((-params.beta * diff * diff)
        .exp()
        .max(params.min_edge_weight))
}

#[derive(Debug, Clone, PartialEq)]
pub struct GradyWeightOp {
    name: &'static str,
    input_shape: [usize; 3],
    params: GradyWeights,
    cost: f64,
}

impl GradyWeightOp {
    pub fn new(name: &'static str, input_shape: [usize; 3], params: GradyWeights) -> Result<Self> {
        if input_shape.contains(&0) {
            return Err(Error::InvalidArgument(format!(
                "{name} needs a non-empty input shape, got {input_shape:?}"
            )));
        }
        Ok(Self {
            name,
            input_shape,
            params,
            cost: 10.0,
        })
    }

    pub fn output_shape(&self) -> [usize; 3] {
        packed_weight_shape(self.input_shape)
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }

    fn apply_typed<T>(
        &self,
        input: &Voxels,
        out: &mut Voxels,
        output_offset: [usize; 3],
        output_extent: [usize; 3],
    ) -> Result<()>
    where
        T: VoxelElement,
    {
        grady_weights_packed_block_into(
            input.view::<T>()?,
            self.params,
            output_offset,
            output_extent,
            out.view_mut::<f64>()?,
        )
    }

    fn dispatch(
        &self,
        input: &Voxels,
        out: &mut Voxels,
        output_offset: [usize; 3],
        output_extent: [usize; 3],
    ) -> Result<()> {
        match input.dtype() {
            Dtype::Bool | Dtype::F16 => Err(Error::InvalidArgument(format!(
                "{} cannot compute Grady random-walker weights from {:?}",
                self.name,
                input.dtype()
            ))),
            Dtype::U8 => self.apply_typed::<u8>(input, out, output_offset, output_extent),
            Dtype::U16 => self.apply_typed::<u16>(input, out, output_offset, output_extent),
            Dtype::U32 => self.apply_typed::<u32>(input, out, output_offset, output_extent),
            Dtype::U64 => self.apply_typed::<u64>(input, out, output_offset, output_extent),
            Dtype::I8 => self.apply_typed::<i8>(input, out, output_offset, output_extent),
            Dtype::I16 => self.apply_typed::<i16>(input, out, output_offset, output_extent),
            Dtype::I32 => self.apply_typed::<i32>(input, out, output_offset, output_extent),
            Dtype::I64 => self.apply_typed::<i64>(input, out, output_offset, output_extent),
            Dtype::F32 => self.apply_typed::<f32>(input, out, output_offset, output_extent),
            Dtype::F64 => self.apply_typed::<f64>(input, out, output_offset, output_extent),
        }
    }
}

impl BlockOp for GradyWeightOp {
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
        Geometry::new(self.output_shape(), vec![InputMap::Stencil(Reach::all())])
    }

    fn output_shape(&self, _input: [usize; 3]) -> [usize; 3] {
        self.output_shape()
    }

    fn takes_extent_from_placement(&self) -> bool {
        true
    }

    fn placed_output_shape(&self, _input: [usize; 3], at: &Placement) -> [usize; 3] {
        at.writes().unwrap_or(self.output_shape())
    }

    fn slicing(&self) -> Slicing {
        Slicing::Stencil
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        !matches!(dtype, Dtype::Bool | Dtype::F16)
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::F64
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, at: &Anchor) -> Result<()> {
        let out_shape = out.shape();
        self.apply_placed(
            input,
            SourceInputs::none(),
            out,
            &Placement::same(at.clone()).writing(out_shape),
        )
    }

    fn apply_placed(
        &self,
        input: &Voxels,
        _sources: SourceInputs<'_>,
        out: &mut Voxels,
        at: &Placement,
    ) -> Result<()> {
        if at.input.offset != [0, 0, 0] || input.shape() != self.input_shape {
            return Err(Error::InvalidArgument(format!(
                "{} needs the whole input image {:?}, got offset {:?} and held shape {:?}",
                self.name,
                self.input_shape,
                at.input.offset,
                input.shape()
            )));
        }
        self.dispatch(input, out, at.output.offset, at.output.volume)
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

pub fn grady_weight_phase(
    slots: Vec<usize>,
    names: Vec<String>,
    input_volume: [usize; 3],
    output_grid: BlockGrid,
) -> Result<PhaseDecomposition> {
    let output_volume = packed_weight_shape(input_volume);
    if output_grid.volume() != output_volume {
        return Err(Error::InvalidArgument(format!(
            "Grady random-walker weights for {input_volume:?} write {output_volume:?}, but the \
             supplied grid is over {:?}",
            output_grid.volume()
        )));
    }
    Ok(
        PhaseDecomposition::derive(slots, names, Reach::all(), Reach::all(), output_grid)
            .with_sources(|_| Region::new(&[0, 0, 0], &input_volume)),
    )
}

pub fn append_grady_weight_phase(
    builder: &mut PlanBuilder,
    op: GradyWeightOp,
    output_grid: BlockGrid,
) -> Result<Phase> {
    let phase = grady_weight_phase(
        vec![0],
        vec![op.name().to_string()],
        op.input_shape,
        output_grid,
    )?;
    builder.pixels_decomposed(Chain::op(op), phase)
}

pub fn random_walker_row_id_phase(
    slots: Vec<usize>,
    names: Vec<String>,
    input_volume: [usize; 3],
    output_grid: BlockGrid,
) -> Result<PhaseDecomposition> {
    if output_grid.volume() != input_volume {
        return Err(Error::InvalidArgument(format!(
            "random-walker row IDs for {input_volume:?} write the same shape, but the supplied \
             grid is over {:?}",
            output_grid.volume()
        )));
    }
    Ok(
        PhaseDecomposition::derive(slots, names, Reach::all(), Reach::all(), output_grid)
            .with_sources(|_| Region::new(&[0, 0, 0], &input_volume)),
    )
}

pub fn append_random_walker_row_id_phase(
    builder: &mut PlanBuilder,
    op: RandomWalkerRowIdOp,
    output_grid: BlockGrid,
) -> Result<Phase> {
    let phase = random_walker_row_id_phase(
        vec![0],
        vec![op.name().to_string()],
        op.input_shape,
        output_grid,
    )?;
    builder.pixels_decomposed(Chain::op(op), phase)
}

pub fn append_random_walker_binary_seed_image_phases(
    builder: &mut PlanBuilder,
    stream: impl Into<String>,
    lifecycle: Lifecycle,
    images: RandomWalkerImages,
    params: GradyWeights,
    config: RandomWalkerConfig,
) -> Result<Phase> {
    let stream = stream.into();
    let grid = builder.grid().clone();
    let volume = grid.volume();
    let row_ids = append_random_walker_row_id_phase(
        builder,
        RandomWalkerRowIdOp::new("random-walker row IDs", volume)?,
        grid.clone(),
    )?;
    let row_id_image = row_ids.writes().ok_or_else(|| {
        Error::InvalidArgument("random-walker row-id phase did not write an image".to_string())
    })?;
    let rows = builder.fragments(RandomWalkerRowsOp::new(
        "random-walker sparse rows",
        stream.clone(),
        lifecycle,
        volume,
        images,
        params,
    )?)?;
    builder.fragments(RandomWalkerSolveOp::new(
        "random-walker solve",
        stream,
        rows.index(),
        grid.blocks_per_axis(),
        RandomWalkerSolveImages {
            seed_image: images.seed_image,
            row_ids: row_id_image.index(),
        },
        config,
    )?)
}

pub fn assemble_random_walker_system(
    weights: &RandomWalkerWeights,
    seeds: ArrayView3<'_, Option<f64>>,
) -> Result<RandomWalkerSystem> {
    let shape = weights.shape();
    shapes_match(shape, seeds.shape(), "random-walker seeds")?;
    validate_seeds(seeds)?;
    validate_seed_reachability(weights, seeds)?;

    let (row_ids, rows) = row_ids_for_seeds(seeds);

    let mut matrix = vec![Vec::new(); rows];
    let mut rhs = vec![0.0; rows];
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let row = row_ids[[i, j, k]];
                if row == NO_ROW {
                    continue;
                }
                let at = [i, j, k];
                let mut diagonal = 0.0;
                for (neighbour, weight) in neighbours(weights, at) {
                    if weight == 0.0 {
                        continue;
                    }
                    diagonal += weight;
                    if let Some(seed) = seeds[neighbour] {
                        rhs[row] += weight * seed;
                    } else {
                        matrix[row].push((row_ids[neighbour], -weight));
                    }
                }
                matrix[row].push((row, diagonal));
            }
        }
    }

    Ok(RandomWalkerSystem {
        rows: matrix,
        rhs,
        row_ids,
    })
}

pub fn assemble_random_walker_system_from_seed_image(
    weights: &RandomWalkerWeights,
    seed_image: ArrayView3<'_, f64>,
) -> Result<RandomWalkerSystem> {
    let seeds = seed_probabilities_from_image(seed_image)?;
    assemble_random_walker_system(weights, seeds.view())
}

pub fn assemble_random_walker_system_from_packed(
    packed_weights: ArrayView3<'_, f64>,
    input_shape: [usize; 3],
    seeds: ArrayView3<'_, Option<f64>>,
) -> Result<RandomWalkerSystem> {
    let weights = RandomWalkerWeights::from_packed(packed_weights, input_shape)?;
    assemble_random_walker_system(&weights, seeds)
}

pub fn assemble_random_walker_system_from_packed_seed_image(
    packed_weights: ArrayView3<'_, f64>,
    input_shape: [usize; 3],
    seed_image: ArrayView3<'_, f64>,
) -> Result<RandomWalkerSystem> {
    let seeds = seed_probabilities_from_image(seed_image)?;
    assemble_random_walker_system_from_packed(packed_weights, input_shape, seeds.view())
}

pub fn solve_random_walker_system_into(
    system: &RandomWalkerSystem,
    seeds: ArrayView3<'_, Option<f64>>,
    config: RandomWalkerConfig,
    mut out: ArrayViewMut3<'_, f64>,
) -> Result<RandomWalkerSolve> {
    shapes_match(shape_of(seeds), out.shape(), "random-walker output")?;
    shapes_match(
        shape_of(seeds),
        system.row_ids.shape(),
        "random-walker system row ids",
    )?;
    if !(config.max_residual_norm.is_finite() && config.max_residual_norm >= 0.0) {
        return Err(Error::InvalidArgument(format!(
            "random-walker residual tolerance must be finite and non-negative, got {}",
            config.max_residual_norm
        )));
    }
    if config.max_iterations == 0 && system.rows() > 0 {
        return Err(Error::InvalidArgument(
            "random-walker max_iterations must be non-zero when unknown voxels exist".to_string(),
        ));
    }

    let (x, solved) = solve_pcg(system, config)?;
    for ((i, j, k), value) in out.indexed_iter_mut() {
        *value = match seeds[[i, j, k]] {
            Some(seed) => seed,
            None => x[system.row_ids[[i, j, k]]],
        };
    }
    Ok(solved)
}

pub fn solve_random_walker_seed_image_system_into(
    system: &RandomWalkerSystem,
    seed_image: ArrayView3<'_, f64>,
    config: RandomWalkerConfig,
    out: ArrayViewMut3<'_, f64>,
) -> Result<RandomWalkerSolve> {
    let seeds = seed_probabilities_from_image(seed_image)?;
    solve_random_walker_system_into(system, seeds.view(), config, out)
}

pub fn solve_random_walker_sparse_table_seed_image_into(
    table: &Table,
    row_ids: ArrayView3<'_, u64>,
    seed_image: ArrayView3<'_, f64>,
    config: RandomWalkerConfig,
    out: ArrayViewMut3<'_, f64>,
) -> Result<RandomWalkerSolve> {
    let system = assemble_random_walker_system_from_sparse_table(table, row_ids)?;
    solve_random_walker_seed_image_system_into(&system, seed_image, config, out)
}

fn solve_random_walker_sparse_table(
    table: &Table,
    config: RandomWalkerConfig,
) -> Result<(Vec<f64>, RandomWalkerSolve)> {
    if table.schema() != &random_walker_sparse_schema() {
        return Err(Error::InvalidArgument(
            "random-walker sparse table has the wrong schema".to_string(),
        ));
    }
    let rows = sparse_table_row_count(table)?;
    let mut matrix = vec![Vec::new(); rows];
    let mut rhs = vec![0.0; rows];
    let columns = random_walker_sparse_columns();
    let volume = table.volume();
    for entry in table.scan(&Region::new(&[0, 0, 0], &volume))? {
        let row = usize::try_from(entry.u64(columns.row)?).map_err(|_| {
            Error::InvalidArgument("random-walker sparse row does not fit usize".to_string())
        })?;
        let col = usize::try_from(entry.u64(columns.col)?).map_err(|_| {
            Error::InvalidArgument("random-walker sparse column does not fit usize".to_string())
        })?;
        let value = entry.f64(columns.value)?;
        let row_rhs = entry.f64(columns.rhs)?;
        if row >= rows || col >= rows {
            return Err(Error::InvalidArgument(format!(
                "random-walker sparse entry names row {row} column {col}, but there are {rows} \
                 row(s)"
            )));
        }
        if !(value.is_finite() && row_rhs.is_finite()) {
            return Err(Error::InvalidArgument(format!(
                "random-walker sparse entry contains non-finite value {value} or RHS {row_rhs}"
            )));
        }
        matrix[row].push((col, value));
        rhs[row] += row_rhs;
    }
    let dummy_row_ids = Array3::<usize>::from_elem((volume[0], volume[1], volume[2]), NO_ROW);
    solve_pcg(
        &RandomWalkerSystem {
            rows: matrix,
            rhs,
            row_ids: dummy_row_ids,
        },
        config,
    )
}

fn sparse_table_row_count(table: &Table) -> Result<usize> {
    let columns = random_walker_sparse_columns();
    let volume = table.volume();
    let mut max_row = None;
    for entry in table.scan(&Region::new(&[0, 0, 0], &volume))? {
        let row = entry.u64(columns.row)?;
        let col = entry.u64(columns.col)?;
        max_row = Some(max_row.map_or(row.max(col), |max: u64| max.max(row).max(col)));
    }
    let rows = max_row.map_or(0, |max| max + 1);
    usize::try_from(rows).map_err(|_| {
        Error::InvalidArgument(format!(
            "random-walker sparse table has {rows} row(s), which does not fit usize"
        ))
    })
}

fn encode_random_walker_solution(solution: &[f64], report: &RandomWalkerSolve) -> Result<Vec<u8>> {
    if !report.residual_norm.is_finite() {
        return Err(Error::InvalidArgument(format!(
            "random-walker solution residual is not finite: {}",
            report.residual_norm
        )));
    }
    if let Some((index, value)) = solution
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(Error::InvalidArgument(format!(
            "random-walker solution value {index} is not finite: {value}"
        )));
    }
    let mut words = Vec::with_capacity(5 + solution.len());
    words.push(RANDOM_WALKER_SOLUTION_MAGIC);
    words.push(RANDOM_WALKER_SOLUTION_VERSION);
    words.push(solution.len() as u64);
    words.push(report.iterations as u64);
    words.push(report.residual_norm.to_bits());
    words.extend(solution.iter().map(|value| value.to_bits()));
    Ok(pack_u64(&words))
}

fn decode_random_walker_solution(bytes: &[u8]) -> Result<(Vec<f64>, RandomWalkerSolve)> {
    let words = unpack_u64(bytes)?;
    if words.len() < 5 {
        return Err(Error::InvalidArgument(format!(
            "random-walker solution blob has {} word(s), expected at least 5",
            words.len()
        )));
    }
    if words[0] != RANDOM_WALKER_SOLUTION_MAGIC {
        return Err(Error::InvalidArgument(
            "random-walker solution blob has the wrong magic".to_string(),
        ));
    }
    if words[1] != RANDOM_WALKER_SOLUTION_VERSION {
        return Err(Error::InvalidArgument(format!(
            "random-walker solution blob is version {}, expected {}",
            words[1], RANDOM_WALKER_SOLUTION_VERSION
        )));
    }
    let rows = usize::try_from(words[2]).map_err(|_| {
        Error::InvalidArgument("random-walker solution row count does not fit usize".to_string())
    })?;
    if words.len() != 5 + rows {
        return Err(Error::InvalidArgument(format!(
            "random-walker solution blob declares {rows} row(s) but has {} value word(s)",
            words.len().saturating_sub(5)
        )));
    }
    let iterations = usize::try_from(words[3]).map_err(|_| {
        Error::InvalidArgument(
            "random-walker solution iteration count does not fit usize".to_string(),
        )
    })?;
    let residual_norm = f64::from_bits(words[4]);
    if !residual_norm.is_finite() {
        return Err(Error::InvalidArgument(format!(
            "random-walker solution residual is not finite: {residual_norm}"
        )));
    }
    let mut solution = Vec::with_capacity(rows);
    for (index, word) in words[5..].iter().copied().enumerate() {
        let value = f64::from_bits(word);
        if !value.is_finite() {
            return Err(Error::InvalidArgument(format!(
                "random-walker solution value {index} is not finite: {value}"
            )));
        }
        solution.push(value);
    }
    Ok((
        solution,
        RandomWalkerSolve {
            iterations,
            residual_norm,
        },
    ))
}

fn reconstruct_random_walker_block_into(
    seed_image: ArrayView3<'_, f64>,
    row_ids: ArrayView3<'_, u64>,
    solution: &[f64],
    read: &Region,
    mut out: ArrayViewMut3<'_, f64>,
) -> Result<()> {
    shapes_match(
        shape_of(seed_image),
        row_ids.shape(),
        "random-walker reconstruction row IDs",
    )?;
    shapes_match(
        shape_of(seed_image),
        out.shape(),
        "random-walker reconstruction output",
    )?;
    for ((i, j, k), value) in out.indexed_iter_mut() {
        let seed = seed_image[[i, j, k]];
        *value = if seed.is_nan() {
            let row = row_ids[[i, j, k]];
            if row == NO_ROW_U64 {
                let at = [read.start[0] + i, read.start[1] + j, read.start[2] + k];
                return Err(Error::InvalidArgument(format!(
                    "random-walker output voxel {at:?} is unknown but has no row id"
                )));
            }
            let row = usize::try_from(row).map_err(|_| {
                Error::InvalidArgument("random-walker row id does not fit usize".to_string())
            })?;
            *solution.get(row).ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "random-walker row id {row} has no solved value; solution has {} row(s)",
                    solution.len()
                ))
            })?
        } else if (0.0..=1.0).contains(&seed) {
            seed
        } else {
            let at = [read.start[0] + i, read.start[1] + j, read.start[2] + k];
            return Err(Error::InvalidArgument(format!(
                "random-walker seed image at {at:?} must be NaN for unknown or a probability in \
                 0..=1, got {seed}"
            )));
        };
    }
    Ok(())
}

fn solve_pcg(
    system: &RandomWalkerSystem,
    config: RandomWalkerConfig,
) -> Result<(Vec<f64>, RandomWalkerSolve)> {
    let n = system.rows();
    if n == 0 {
        return Ok((
            Vec::new(),
            RandomWalkerSolve {
                iterations: 0,
                residual_norm: 0.0,
            },
        ));
    }
    let mut inv_diag = vec![0.0; n];
    for (row, entries) in system.rows.iter().enumerate() {
        let Some((_, diagonal)) = entries.iter().find(|(col, _)| *col == row) else {
            return Err(Error::InvalidArgument(format!(
                "random-walker row {row} has no diagonal"
            )));
        };
        if !(*diagonal > 0.0 && diagonal.is_finite()) {
            return Err(Error::InvalidArgument(format!(
                "random-walker row {row} has invalid diagonal {diagonal}"
            )));
        }
        inv_diag[row] = 1.0 / diagonal;
    }

    let mut x = vec![0.5; n];
    let mut ax = matvec(&system.rows, &x);
    let mut r: Vec<f64> = system
        .rhs
        .iter()
        .zip(ax.iter())
        .map(|(b, ax)| b - ax)
        .collect();
    let mut z: Vec<f64> = r
        .iter()
        .zip(inv_diag.iter())
        .map(|(r, inv)| r * inv)
        .collect();
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let mut residual = norm(&r);
    if residual <= config.max_residual_norm {
        return Ok((
            x,
            RandomWalkerSolve {
                iterations: 0,
                residual_norm: residual,
            },
        ));
    }

    for iteration in 0..config.max_iterations {
        ax = matvec(&system.rows, &p);
        let denom = dot(&p, &ax);
        if !(denom.is_finite() && denom > 0.0) {
            return Err(Error::InvalidArgument(format!(
                "random-walker PCG encountered invalid direction product {denom}"
            )));
        }
        let alpha = rz / denom;
        for row in 0..n {
            x[row] += alpha * p[row];
            r[row] -= alpha * ax[row];
        }
        residual = norm(&r);
        if residual <= config.max_residual_norm {
            return Ok((
                x,
                RandomWalkerSolve {
                    iterations: iteration + 1,
                    residual_norm: residual,
                },
            ));
        }
        z = r
            .iter()
            .zip(inv_diag.iter())
            .map(|(r, inv)| r * inv)
            .collect();
        let next_rz = dot(&r, &z);
        let beta = next_rz / rz;
        for row in 0..n {
            p[row] = z[row] + beta * p[row];
        }
        rz = next_rz;
    }

    Err(Error::InvalidArgument(format!(
        "random-walker PCG did not converge within {} iterations; residual norm is {}",
        config.max_iterations, residual
    )))
}

fn validate_seeds(seeds: ArrayView3<'_, Option<f64>>) -> Result<()> {
    for ((i, j, k), seed) in seeds.indexed_iter() {
        if let Some(seed) = seed {
            if !(seed.is_finite() && (0.0..=1.0).contains(seed)) {
                return Err(Error::InvalidArgument(format!(
                    "random-walker seed at [{i}, {j}, {k}] must be a finite probability in \
                     0..=1, got {seed}"
                )));
            }
        }
    }
    Ok(())
}

fn row_ids_for_seeds(seeds: ArrayView3<'_, Option<f64>>) -> (Array3<usize>, usize) {
    let shape = shape_of(seeds);
    let mut row_ids = Array3::<usize>::from_elem((shape[0], shape[1], shape[2]), NO_ROW);
    let mut rows = 0usize;
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                if seeds[[i, j, k]].is_none() {
                    row_ids[[i, j, k]] = rows;
                    rows += 1;
                }
            }
        }
    }
    (row_ids, rows)
}

fn decode_row_id_image(row_ids: ArrayView3<'_, u64>) -> Result<(Array3<usize>, usize)> {
    let shape = shape_of(row_ids);
    let mut decoded = Array3::<usize>::from_elem((shape[0], shape[1], shape[2]), NO_ROW);
    let mut max_row = None;
    for ((i, j, k), row) in row_ids.indexed_iter() {
        if *row == NO_ROW_U64 {
            continue;
        }
        let row = usize::try_from(*row).map_err(|_| {
            Error::InvalidArgument(format!(
                "random-walker row id at [{i}, {j}, {k}] does not fit usize"
            ))
        })?;
        decoded[[i, j, k]] = row;
        max_row = Some(max_row.map_or(row, |max: usize| max.max(row)));
    }
    let rows = max_row.map_or(0, |max| max + 1);
    let mut present = vec![false; rows];
    for row in decoded.iter().copied().filter(|row| *row != NO_ROW) {
        present[row] = true;
    }
    if let Some(missing) = present.iter().position(|present| !*present) {
        return Err(Error::InvalidArgument(format!(
            "random-walker row-id image skips row {missing}; row ids must be contiguous"
        )));
    }
    Ok((decoded, rows))
}

fn seed_probabilities_from_image(seed_image: ArrayView3<'_, f64>) -> Result<Array3<Option<f64>>> {
    let shape = shape_of(seed_image);
    let mut seeds = Array3::<Option<f64>>::from_elem((shape[0], shape[1], shape[2]), None);
    for ((i, j, k), value) in seed_image.indexed_iter() {
        if value.is_nan() {
            continue;
        }
        if !((0.0..=1.0).contains(value)) {
            return Err(Error::InvalidArgument(format!(
                "random-walker seed image at [{i}, {j}, {k}] must be NaN for unknown or a \
                 probability in 0..=1, got {value}"
            )));
        }
        seeds[[i, j, k]] = Some(*value);
    }
    Ok(seeds)
}

fn validate_seed_reachability(
    weights: &RandomWalkerWeights,
    seeds: ArrayView3<'_, Option<f64>>,
) -> Result<()> {
    let shape = weights.shape();
    let mut seen = Array3::<bool>::from_elem((shape[0], shape[1], shape[2]), false);
    let mut queue = VecDeque::new();
    let mut unknowns = 0usize;
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                if seeds[[i, j, k]].is_some() {
                    seen[[i, j, k]] = true;
                    queue.push_back([i, j, k]);
                } else {
                    unknowns += 1;
                }
            }
        }
    }
    if unknowns == 0 {
        return Ok(());
    }
    if queue.is_empty() {
        return Err(Error::InvalidArgument(
            "random-walker needs at least one seed when unknown voxels exist".to_string(),
        ));
    }
    while let Some(at) = queue.pop_front() {
        for (neighbour, weight) in neighbours(weights, at) {
            if weight > 0.0 && !seen[neighbour] {
                seen[neighbour] = true;
                queue.push_back(neighbour);
            }
        }
    }
    for ((i, j, k), seed) in seeds.indexed_iter() {
        if seed.is_none() && !seen[[i, j, k]] {
            return Err(Error::InvalidArgument(format!(
                "random-walker unknown voxel [{i}, {j}, {k}] is not connected to any seed by \
                 positive-weight edges"
            )));
        }
    }
    Ok(())
}

fn neighbours(weights: &RandomWalkerWeights, at: [usize; 3]) -> Vec<([usize; 3], f64)> {
    let shape = weights.shape();
    let mut out = Vec::with_capacity(6);
    for axis in 0..3 {
        if at[axis] > 0 {
            let mut lower = at;
            lower[axis] -= 1;
            out.push((lower, weights.edge(lower, axis)));
        }
        if at[axis] + 1 < shape[axis] {
            let mut upper = at;
            upper[axis] += 1;
            out.push((upper, weights.edge(at, axis)));
        }
    }
    out
}

fn face_neighbours(at: [usize; 3], shape: [usize; 3]) -> Vec<[usize; 3]> {
    let mut out = Vec::with_capacity(6);
    for axis in 0..3 {
        if at[axis] > 0 {
            let mut lower = at;
            lower[axis] -= 1;
            out.push(lower);
        }
        if at[axis] + 1 < shape[axis] {
            let mut upper = at;
            upper[axis] += 1;
            out.push(upper);
        }
    }
    out
}

fn grady_weight_between(
    intensity: ArrayView3<'_, f64>,
    read: &Region,
    left: [usize; 3],
    right: [usize; 3],
    params: GradyWeights,
) -> Result<f64> {
    let left_value = intensity[local_in_read(read, left)?];
    let right_value = intensity[local_in_read(read, right)?];
    if !(left_value.is_finite() && right_value.is_finite()) {
        return Err(Error::InvalidArgument(format!(
            "random-walker intensity contains non-finite value on edge {left:?} to {right:?}"
        )));
    }
    let diff = left_value - right_value;
    Ok((-params.beta * diff * diff)
        .exp()
        .max(params.min_edge_weight))
}

fn local_in_read(read: &Region, at: [usize; 3]) -> Result<[usize; 3]> {
    let mut local = [0usize; 3];
    for axis in 0..3 {
        if at[axis] < read.start[axis] || at[axis] - read.start[axis] >= read.shape[axis] {
            return Err(Error::InvalidArgument(format!(
                "random-walker coordinate {at:?} is outside read region start {:?} shape {:?}",
                read.start, read.shape
            )));
        }
        local[axis] = at[axis] - read.start[axis];
    }
    Ok(local)
}

pub fn packed_weight_shape(input_shape: [usize; 3]) -> [usize; 3] {
    [input_shape[0], input_shape[1], input_shape[2] * 3]
}

fn packed_axis(k: usize, axis: usize) -> usize {
    k * 3 + axis
}

fn matvec(rows: &[Vec<(usize, f64)>], x: &[f64]) -> Vec<f64> {
    rows.iter()
        .map(|row| row.iter().map(|(col, value)| value * x[*col]).sum())
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn shape_of<T>(array: ArrayView3<'_, T>) -> [usize; 3] {
    [array.shape()[0], array.shape()[1], array.shape()[2]]
}

fn shapes_match(expected: [usize; 3], found: &[usize], context: &str) -> Result<()> {
    if found != expected.as_slice() {
        return Err(Error::InvalidArgument(format!(
            "{context}: expected shape {expected:?}, got {found:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assemble::ImageId;
    use crate::env::{ArrayEnvironment, Environment};
    use crate::strategy::execute_phases;
    use crate::strategy::{execute, Hints};
    use ndarray::Array3;

    fn seeds(values: &[Option<f64>]) -> Array3<Option<f64>> {
        Array3::from_shape_vec((values.len(), 1, 1), values.to_vec()).unwrap()
    }

    fn seed_image(values: &[f64]) -> Array3<f64> {
        Array3::from_shape_vec((values.len(), 1, 1), values.to_vec()).unwrap()
    }

    fn sparse_records(table: &Table, volume: [usize; 3]) -> Vec<([usize; 3], u64, u64, f64, f64)> {
        let columns = random_walker_sparse_columns();
        table
            .query(&Region::new(&[0, 0, 0], &volume))
            .unwrap()
            .into_iter()
            .map(|entry| {
                (
                    entry.at(),
                    entry.u64(columns.row).unwrap(),
                    entry.u64(columns.col).unwrap(),
                    entry.f64(columns.value).unwrap(),
                    entry.f64(columns.rhs).unwrap(),
                )
            })
            .collect()
    }

    fn expected_sparse_records(
        system: &RandomWalkerSystem,
        volume: [usize; 3],
    ) -> Vec<([usize; 3], u64, u64, f64, f64)> {
        let mut records = Vec::new();
        for i in 0..volume[0] {
            for j in 0..volume[1] {
                for k in 0..volume[2] {
                    let at = [i, j, k];
                    let Some(row) = system.row_id_at(at) else {
                        continue;
                    };
                    for &(col, value) in system.row_entries(row) {
                        records.push((
                            at,
                            row as u64,
                            col as u64,
                            value,
                            if col == row { system.rhs()[row] } else { 0.0 },
                        ));
                    }
                }
            }
        }
        records.sort_by_key(|(at, row, col, value, rhs)| {
            (*at, *row, *col, value.to_bits(), rhs.to_bits())
        });
        records
    }

    #[test]
    fn grady_weights_store_positive_axis_edges() {
        let input = Array3::from_shape_vec((3, 1, 1), vec![0.0, 1.0, 1.0]).unwrap();
        let weights =
            RandomWalkerWeights::grady(input.view(), GradyWeights::new(2.0, 0.1).unwrap()).unwrap();

        assert!((weights.edge([0, 0, 0], 0) - (-2.0f64).exp()).abs() < 1.0e-12);
        assert!((weights.edge([1, 0, 0], 0) - 1.0).abs() < 1.0e-12);
        assert_eq!(weights.edge([2, 0, 0], 0), 0.0);
    }

    #[test]
    fn binary_random_walker_solves_linear_chain() {
        let input = Array3::from_elem((5, 1, 1), 0.0);
        let seeds = seeds(&[Some(0.0), None, None, None, Some(1.0)]);
        let mut out = Array3::<f64>::zeros((5, 1, 1));

        let report = random_walker_binary_into(
            input.view(),
            seeds.view(),
            GradyWeights::new(0.0, 0.0).unwrap(),
            RandomWalkerConfig::default(),
            out.view_mut(),
        )
        .unwrap();

        assert!(report.iterations > 0);
        assert!(report.residual_norm <= RandomWalkerConfig::default().max_residual_norm);
        for (i, want) in [0.0, 0.25, 0.5, 0.75, 1.0].into_iter().enumerate() {
            assert!((out[[i, 0, 0]] - want).abs() < 1.0e-9, "{i}");
        }
    }

    #[test]
    fn all_seeded_volume_needs_no_unknown_rows() {
        let input = Array3::from_elem((3, 1, 1), 0.0);
        let seeds = seeds(&[Some(0.0), Some(1.0), Some(0.5)]);
        let weights =
            RandomWalkerWeights::grady(input.view(), GradyWeights::new(0.0, 0.0).unwrap()).unwrap();
        let system = assemble_random_walker_system(&weights, seeds.view()).unwrap();
        let mut out = Array3::<f64>::zeros((3, 1, 1));

        let report = solve_random_walker_system_into(
            &system,
            seeds.view(),
            RandomWalkerConfig::default(),
            out.view_mut(),
        )
        .unwrap();

        assert_eq!(system.rows(), 0);
        assert_eq!(report.iterations, 0);
        assert_eq!(out[[0, 0, 0]], 0.0);
        assert_eq!(out[[1, 0, 0]], 1.0);
        assert_eq!(out[[2, 0, 0]], 0.5);
    }

    #[test]
    fn unknowns_without_any_seed_are_refused() {
        let input = Array3::from_elem((3, 1, 1), 0.0);
        let seeds = seeds(&[None, None, None]);
        let weights =
            RandomWalkerWeights::grady(input.view(), GradyWeights::new(0.0, 0.0).unwrap()).unwrap();

        let error = assemble_random_walker_system(&weights, seeds.view()).unwrap_err();
        assert!(error.to_string().contains("at least one seed"));
    }

    #[test]
    fn seed_probabilities_are_validated() {
        let input = Array3::from_elem((2, 1, 1), 0.0);
        let seeds = seeds(&[Some(2.0), None]);
        let weights =
            RandomWalkerWeights::grady(input.view(), GradyWeights::new(0.0, 0.0).unwrap()).unwrap();

        let error = assemble_random_walker_system(&weights, seeds.view()).unwrap_err();
        assert!(error.to_string().contains("finite probability"));
    }

    #[test]
    fn disconnected_unknown_region_is_refused() {
        let input = Array3::from_elem((3, 1, 1), 0.0);
        let mut weights =
            RandomWalkerWeights::grady(input.view(), GradyWeights::new(0.0, 0.0).unwrap()).unwrap();
        weights.edges[[1, 0, 0, 0]] = 0.0;
        let seeds = seeds(&[Some(0.0), None, None]);

        let error = assemble_random_walker_system(&weights, seeds.view()).unwrap_err();
        assert!(error.to_string().contains("not connected"));
    }

    #[test]
    fn pcg_matches_independent_dense_solve_on_small_2d_system() {
        let input = Array3::from_shape_fn((3, 3, 1), |(i, j, _)| (i + j) as f64);
        let mut seeds = Array3::<Option<f64>>::from_elem((3, 3, 1), None);
        seeds[[0, 0, 0]] = Some(0.0);
        seeds[[2, 2, 0]] = Some(1.0);
        seeds[[0, 2, 0]] = Some(0.25);
        let weights =
            RandomWalkerWeights::grady(input.view(), GradyWeights::new(0.2, 0.05).unwrap())
                .unwrap();
        let system = assemble_random_walker_system(&weights, seeds.view()).unwrap();
        let expected = dense_solve(&system.rows, &system.rhs);
        let mut out = Array3::<f64>::zeros((3, 3, 1));

        solve_random_walker_system_into(
            &system,
            seeds.view(),
            RandomWalkerConfig::default(),
            out.view_mut(),
        )
        .unwrap();

        for ((i, j, k), seed) in seeds.indexed_iter() {
            if seed.is_none() {
                let row = system.row_ids[[i, j, k]];
                assert!((out[[i, j, k]] - expected[row]).abs() < 1.0e-8);
            }
        }
    }

    #[test]
    fn pcg_residual_decreases_on_representative_fixture() {
        let volume = [5, 4, 1];
        let input = Array3::from_shape_fn((volume[0], volume[1], volume[2]), |(i, j, _)| {
            ((i * i + 3 * j) as f64).sin()
        });
        let mut seeds = Array3::<Option<f64>>::from_elem((volume[0], volume[1], volume[2]), None);
        seeds[[0, 0, 0]] = Some(0.0);
        seeds[[4, 3, 0]] = Some(1.0);
        seeds[[2, 0, 0]] = Some(0.35);
        let weights =
            RandomWalkerWeights::grady(input.view(), GradyWeights::new(0.4, 0.02).unwrap())
                .unwrap();
        let system = assemble_random_walker_system(&weights, seeds.view()).unwrap();
        let initial = vec![0.5; system.rows()];
        let initial_residual = norm(
            &system
                .rhs()
                .iter()
                .zip(matvec(&system.rows, &initial).iter())
                .map(|(b, ax)| b - ax)
                .collect::<Vec<_>>(),
        );

        let (_, report) = solve_pcg(&system, RandomWalkerConfig::default()).unwrap();

        assert!(report.iterations > 0);
        assert!(
            report.residual_norm < initial_residual,
            "final residual {} should be below initial residual {}",
            report.residual_norm,
            initial_residual
        );
    }

    #[test]
    fn packed_weights_assemble_the_same_reduced_system() {
        let input = Array3::from_shape_fn((3, 2, 2), |(i, j, k)| (i * 2 + j * 3 + k) as f64);
        let mut seeds = Array3::<Option<f64>>::from_elem((3, 2, 2), None);
        seeds[[0, 0, 0]] = Some(0.0);
        seeds[[2, 1, 1]] = Some(1.0);
        let params = GradyWeights::new(0.3, 0.01).unwrap();
        let weights = RandomWalkerWeights::grady(input.view(), params).unwrap();
        let output = packed_weight_shape(weights.shape());
        let mut packed = Array3::<f64>::zeros((output[0], output[1], output[2]));
        weights.write_packed_into(packed.view_mut()).unwrap();

        let direct = assemble_random_walker_system(&weights, seeds.view()).unwrap();
        let from_packed =
            assemble_random_walker_system_from_packed(packed.view(), weights.shape(), seeds.view())
                .unwrap();

        assert_eq!(from_packed, direct);
    }

    #[test]
    fn packed_weights_refuse_invalid_values() {
        let shape = [2, 1, 1];
        let mut packed = Array3::<f64>::zeros((shape[0], shape[1], shape[2] * 3));
        packed[[0, 0, 0]] = -1.0;

        let error = RandomWalkerWeights::from_packed(packed.view(), shape).unwrap_err();
        assert!(error.to_string().contains("finite and non-negative"));
    }

    #[test]
    fn seed_image_path_matches_option_seed_path() {
        let input = Array3::from_elem((5, 1, 1), 0.0);
        let option_seeds = seeds(&[Some(0.0), None, None, None, Some(1.0)]);
        let image_seeds = seed_image(&[0.0, f64::NAN, f64::NAN, f64::NAN, 1.0]);
        let mut expected = Array3::<f64>::zeros((5, 1, 1));
        let mut actual = Array3::<f64>::zeros((5, 1, 1));

        random_walker_binary_into(
            input.view(),
            option_seeds.view(),
            GradyWeights::new(0.0, 0.0).unwrap(),
            RandomWalkerConfig::default(),
            expected.view_mut(),
        )
        .unwrap();
        random_walker_binary_seed_image_into(
            input.view(),
            image_seeds.view(),
            GradyWeights::new(0.0, 0.0).unwrap(),
            RandomWalkerConfig::default(),
            actual.view_mut(),
        )
        .unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn packed_weights_with_seed_image_assemble_same_system() {
        let input = Array3::from_shape_fn((3, 1, 1), |(i, _, _)| i as f64);
        let image_seeds = seed_image(&[0.0, f64::NAN, 1.0]);
        let option_seeds = seeds(&[Some(0.0), None, Some(1.0)]);
        let params = GradyWeights::new(0.25, 0.01).unwrap();
        let weights = RandomWalkerWeights::grady(input.view(), params).unwrap();
        let output = packed_weight_shape(weights.shape());
        let mut packed = Array3::<f64>::zeros((output[0], output[1], output[2]));
        weights.write_packed_into(packed.view_mut()).unwrap();

        let expected = assemble_random_walker_system(&weights, option_seeds.view()).unwrap();
        let actual = assemble_random_walker_system_from_packed_seed_image(
            packed.view(),
            weights.shape(),
            image_seeds.view(),
        )
        .unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn seed_image_rejects_infinite_and_out_of_range_values() {
        let input = Array3::from_elem((2, 1, 1), 0.0);
        let weights =
            RandomWalkerWeights::grady(input.view(), GradyWeights::new(0.0, 0.0).unwrap()).unwrap();
        let seeds = seed_image(&[f64::INFINITY, f64::NAN]);

        let error = assemble_random_walker_system_from_seed_image(&weights, seeds.view())
            .unwrap_err()
            .to_string();

        assert!(error.contains("NaN for unknown or a probability"));
    }

    #[test]
    fn row_id_image_matches_system_rows() {
        let input = Array3::from_elem((4, 1, 1), 0.0);
        let option_seeds = seeds(&[Some(0.0), None, None, Some(1.0)]);
        let image_seeds = seed_image(&[0.0, f64::NAN, f64::NAN, 1.0]);
        let weights =
            RandomWalkerWeights::grady(input.view(), GradyWeights::new(0.0, 0.0).unwrap()).unwrap();
        let system = assemble_random_walker_system(&weights, option_seeds.view()).unwrap();
        let mut row_ids = Array3::<u64>::zeros((4, 1, 1));

        let rows = random_walker_row_ids_into(image_seeds.view(), row_ids.view_mut()).unwrap();

        assert_eq!(rows, system.rows());
        assert_eq!(row_ids[[0, 0, 0]], NO_ROW_U64);
        assert_eq!(
            row_ids[[1, 0, 0]],
            system.row_id_at([1, 0, 0]).unwrap() as u64
        );
        assert_eq!(
            row_ids[[2, 0, 0]],
            system.row_id_at([2, 0, 0]).unwrap() as u64
        );
        assert_eq!(row_ids[[3, 0, 0]], NO_ROW_U64);
    }

    #[test]
    fn row_id_phase_matches_data_layer_when_decomposed() {
        let volume = [4, 3, 2];
        let seed_image = Array3::from_shape_fn((volume[0], volume[1], volume[2]), |(i, j, k)| {
            if (i + j + k) % 3 == 0 {
                1.0
            } else {
                f64::NAN
            }
        });
        let mut expected = Array3::<u64>::zeros((volume[0], volume[1], volume[2]));
        random_walker_row_ids_into(seed_image.view(), expected.view_mut()).unwrap();

        let mut builder =
            PlanBuilder::new(volume, Dtype::F64, BlockGrid::new(volume, volume).unwrap());
        append_random_walker_row_id_phase(
            &mut builder,
            RandomWalkerRowIdOp::new("rw-row-ids", volume).unwrap(),
            BlockGrid::new(volume, [2, 2, 1]).unwrap(),
        )
        .unwrap();
        let assembly = builder.finish().unwrap();
        let env = ArrayEnvironment::for_decomposition(
            Voxels::from(seed_image),
            &assembly.decomposition,
            [2, 2, 1],
        )
        .unwrap();
        execute(
            "rw row ids",
            &assembly.workflow,
            &assembly.decomposition,
            &Hints::default(),
            &env,
        )
        .unwrap();

        assert_eq!(env.output().view::<u64>().unwrap(), &expected);
    }

    #[test]
    fn sparse_rows_match_reduced_system() {
        let volume = [3, 2, 1];
        let input = Array3::from_shape_fn((volume[0], volume[1], volume[2]), |(i, j, _)| {
            (i * 3 + j) as f64
        });
        let mut seed_image = Array3::<f64>::from_elem((volume[0], volume[1], volume[2]), f64::NAN);
        seed_image[[0, 0, 0]] = 0.0;
        seed_image[[2, 1, 0]] = 1.0;
        let seeds = seed_probabilities_from_image(seed_image.view()).unwrap();
        let params = GradyWeights::new(0.2, 0.05).unwrap();
        let weights = RandomWalkerWeights::grady(input.view(), params).unwrap();
        let system = assemble_random_walker_system(&weights, seeds.view()).unwrap();
        let mut row_ids = Array3::<u64>::zeros((volume[0], volume[1], volume[2]));
        random_walker_row_ids_into(seed_image.view(), row_ids.view_mut()).unwrap();
        let read = Region::new(&[0, 0, 0], &volume);
        let bytes = encode_random_walker_rows(
            row_ids.view(),
            seed_image.view(),
            input.view(),
            params,
            volume,
            &read,
            &read,
        )
        .unwrap();
        let table = collect_random_walker_rows(volume, [([0, 0, 0], bytes)]).unwrap();

        assert_eq!(
            sparse_records(&table, volume),
            expected_sparse_records(&system, volume)
        );
    }

    #[test]
    fn sparse_table_system_solves_like_direct_system() {
        let volume = [4, 2, 1];
        let input = Array3::from_shape_fn((volume[0], volume[1], volume[2]), |(i, j, _)| {
            (i * 2 + j) as f64
        });
        let mut seed_image = Array3::<f64>::from_elem((volume[0], volume[1], volume[2]), f64::NAN);
        seed_image[[0, 0, 0]] = 0.0;
        seed_image[[3, 1, 0]] = 1.0;
        let seeds = seed_probabilities_from_image(seed_image.view()).unwrap();
        let params = GradyWeights::new(0.1, 0.01).unwrap();
        let weights = RandomWalkerWeights::grady(input.view(), params).unwrap();
        let direct = assemble_random_walker_system(&weights, seeds.view()).unwrap();
        let mut row_ids = Array3::<u64>::zeros((volume[0], volume[1], volume[2]));
        random_walker_row_ids_into(seed_image.view(), row_ids.view_mut()).unwrap();
        let whole = Region::new(&[0, 0, 0], &volume);
        let table = collect_random_walker_rows(
            volume,
            [(
                [0, 0, 0],
                encode_random_walker_rows(
                    row_ids.view(),
                    seed_image.view(),
                    input.view(),
                    params,
                    volume,
                    &whole,
                    &whole,
                )
                .unwrap(),
            )],
        )
        .unwrap();
        let mut expected = Array3::<f64>::zeros((volume[0], volume[1], volume[2]));
        let mut actual = Array3::<f64>::zeros((volume[0], volume[1], volume[2]));

        solve_random_walker_system_into(
            &direct,
            seeds.view(),
            RandomWalkerConfig::default(),
            expected.view_mut(),
        )
        .unwrap();
        solve_random_walker_sparse_table_seed_image_into(
            &table,
            row_ids.view(),
            seed_image.view(),
            RandomWalkerConfig::default(),
            actual.view_mut(),
        )
        .unwrap();

        for (got, want) in actual.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1.0e-9, "got {got}, want {want}");
        }
    }

    #[test]
    fn sparse_table_refuses_non_contiguous_row_ids() {
        let volume = [3, 1, 1];
        let mut row_ids = Array3::<u64>::from_elem((3, 1, 1), NO_ROW_U64);
        row_ids[[1, 0, 0]] = 2;
        let table = collect_random_walker_rows(
            volume,
            [([0, 0, 0], {
                let mut rows = RowBuilder::new(Arc::new(random_walker_sparse_schema()));
                rows.push(
                    [1, 0, 0],
                    &[
                        Value::U64(2),
                        Value::U64(2),
                        Value::F64(1.0),
                        Value::F64(0.0),
                    ],
                )
                .unwrap();
                rows.encode()
            })],
        )
        .unwrap();

        let error = assemble_random_walker_system_from_sparse_table(&table, row_ids.view())
            .unwrap_err()
            .to_string();

        assert!(error.contains("skips row"));
    }

    #[test]
    fn solve_phase_reconstructs_probability_image() {
        let volume = [4, 3, 2];
        let input = Array3::from_shape_fn((volume[0], volume[1], volume[2]), |(i, j, k)| {
            (i * 7 + j * 3 + k) as f64
        });
        let mut seed_image = Array3::<f64>::from_elem((volume[0], volume[1], volume[2]), f64::NAN);
        seed_image[[0, 0, 0]] = 0.0;
        seed_image[[3, 2, 1]] = 1.0;
        seed_image[[0, 2, 1]] = 0.25;
        let params = GradyWeights::new(0.12, 0.02).unwrap();
        let config = RandomWalkerConfig::default();
        let mut expected = Array3::<f64>::zeros((volume[0], volume[1], volume[2]));
        random_walker_binary_seed_image_into(
            input.view(),
            seed_image.view(),
            params,
            config,
            expected.view_mut(),
        )
        .unwrap();

        for block in [[2, 2, 1], [1, 3, 2]] {
            let grid = BlockGrid::new(volume, block).unwrap();
            let mut builder = PlanBuilder::new(volume, Dtype::F64, grid);
            append_random_walker_binary_seed_image_phases(
                &mut builder,
                "rw.rows",
                Lifecycle::DeleteOnExit,
                RandomWalkerImages {
                    seed_image: 0,
                    intensity: ImageId::supplied(0).index(),
                },
                params,
                config,
            )
            .unwrap();
            let assembly = builder.finish().unwrap();
            let env = ArrayEnvironment::with_inputs(
                Voxels::from(seed_image.clone()),
                vec![Voxels::from(input.clone())],
                &assembly.decomposition,
                [2, 2, 1],
            )
            .unwrap();

            execute_phases(
                "rw solve",
                &assembly.workflow,
                &assembly.decomposition,
                &Hints::default(),
                &env,
                &[],
                &assembly.work(),
            )
            .unwrap();

            let actual = env.output().view::<f64>().unwrap().to_owned();
            for (got, want) in actual.iter().zip(expected.iter()) {
                assert!((got - want).abs() < 1.0e-9, "got {got}, want {want}");
            }
        }
    }

    #[test]
    fn sparse_row_phase_matches_data_layer_when_decomposed() {
        let volume = [4, 3, 2];
        let input = Array3::from_shape_fn((volume[0], volume[1], volume[2]), |(i, j, k)| {
            (i * 7 + j * 3 + k) as f64
        });
        let mut seed_image = Array3::<f64>::from_elem((volume[0], volume[1], volume[2]), f64::NAN);
        seed_image[[0, 0, 0]] = 0.0;
        seed_image[[3, 2, 1]] = 1.0;
        seed_image[[0, 2, 1]] = 0.25;
        let seeds = seed_probabilities_from_image(seed_image.view()).unwrap();
        let params = GradyWeights::new(0.15, 0.02).unwrap();
        let weights = RandomWalkerWeights::grady(input.view(), params).unwrap();
        let system = assemble_random_walker_system(&weights, seeds.view()).unwrap();
        let expected = expected_sparse_records(&system, volume);
        let mut answers = Vec::new();

        for block in [[4, 3, 2], [2, 2, 1], [1, 3, 2]] {
            let mut builder =
                PlanBuilder::new(volume, Dtype::F64, BlockGrid::new(volume, block).unwrap());
            append_random_walker_row_id_phase(
                &mut builder,
                RandomWalkerRowIdOp::new("rw-row-ids", volume).unwrap(),
                BlockGrid::new(volume, block).unwrap(),
            )
            .unwrap();
            let rows = RandomWalkerRowsOp::new(
                "rw-rows",
                "rw.rows",
                Lifecycle::Persistent,
                volume,
                RandomWalkerImages {
                    seed_image: 0,
                    intensity: ImageId::supplied(0).index(),
                },
                params,
            )
            .unwrap();
            builder.fragments(rows).unwrap();
            let assembly = builder.finish().unwrap();
            let env = ArrayEnvironment::with_inputs(
                Voxels::from(seed_image.clone()),
                vec![Voxels::from(input.clone())],
                &assembly.decomposition,
                [2, 2, 1],
            )
            .unwrap();
            execute_phases(
                "rw sparse rows",
                &assembly.workflow,
                &assembly.decomposition,
                &Hints::default(),
                &env,
                &[],
                &assembly.work(),
            )
            .unwrap();
            let grid = &assembly.decomposition.phases[1].grid;
            let counts = grid.blocks_per_axis();
            let mut fragments = Vec::new();
            for i in 0..counts[0] {
                for j in 0..counts[1] {
                    for k in 0..counts[2] {
                        let block = [i, j, k];
                        let bytes = env
                            .read_sidecar("rw.rows", 1, block)
                            .unwrap()
                            .unwrap_or_default();
                        fragments.push((block, bytes));
                    }
                }
            }
            let table = collect_random_walker_rows(volume, fragments).unwrap();
            answers.push(sparse_records(&table, volume));
        }

        for answer in answers {
            assert_eq!(answer, expected);
        }
    }

    #[test]
    fn grady_weight_phase_matches_data_layer_when_decomposed() {
        let volume = [4, 3, 2];
        let input = Array3::from_shape_fn((volume[0], volume[1], volume[2]), |(i, j, k)| {
            (i * 7 + j * 3 + k) as f64
        });
        let params = GradyWeights::new(0.1, 0.02).unwrap();
        let output = packed_weight_shape(volume);
        let mut expected = Array3::<f64>::zeros((output[0], output[1], output[2]));
        grady_weights_packed_into(input.view(), params, expected.view_mut()).unwrap();

        let mut builder =
            PlanBuilder::new(volume, Dtype::F64, BlockGrid::new(volume, volume).unwrap());
        append_grady_weight_phase(
            &mut builder,
            GradyWeightOp::new("rw-weights", volume, params).unwrap(),
            BlockGrid::new(output, [2, 2, 2]).unwrap(),
        )
        .unwrap();
        let assembly = builder.finish().unwrap();
        let env = ArrayEnvironment::for_decomposition(
            Voxels::from(input),
            &assembly.decomposition,
            [2, 2, 2],
        )
        .unwrap();
        execute(
            "rw weights",
            &assembly.workflow,
            &assembly.decomposition,
            &Hints::default(),
            &env,
        )
        .unwrap();

        assert_eq!(env.output().view::<f64>().unwrap(), &expected);
    }

    fn dense_solve(rows: &[Vec<(usize, f64)>], rhs: &[f64]) -> Vec<f64> {
        let n = rows.len();
        let mut a = vec![vec![0.0; n]; n];
        for (row, entries) in rows.iter().enumerate() {
            for &(col, value) in entries {
                a[row][col] += value;
            }
        }
        let mut b = rhs.to_vec();
        for pivot in 0..n {
            let mut best = pivot;
            for row in pivot + 1..n {
                if a[row][pivot].abs() > a[best][pivot].abs() {
                    best = row;
                }
            }
            assert!(a[best][pivot].abs() > 1.0e-12);
            a.swap(pivot, best);
            b.swap(pivot, best);

            let scale = a[pivot][pivot];
            for col in pivot..n {
                a[pivot][col] /= scale;
            }
            b[pivot] /= scale;

            for row in 0..n {
                if row == pivot {
                    continue;
                }
                let factor = a[row][pivot];
                for col in pivot..n {
                    a[row][col] -= factor * a[pivot][col];
                }
                b[row] -= factor * b[pivot];
            }
        }
        b
    }
}
