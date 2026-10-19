// SPDX-License-Identifier: MIT
//
// Original work for this crate. The flood is `super::watershed`'s (and through
// it scikit-image's, BSD-3-Clause); this file only decides where it runs.
//
// A tiled watershed: an over-segmentation that never needs the whole volume.
// ===========================================================================
//
// What it is for
// --------------
// `super::watershed` declares itself a planning barrier, and its header says
// why: the exact answer is a function of one global queue. For a *final*
// segmentation that is the right contract. For **supervoxels** it is not the
// contract anybody needs: an over-segmentation is consumed by a region graph
// and a partition (`ops::rag`, `ops::rag_phases`, `ops::agglomerate`), which
// merge across any boundary that is not a real one. What matters is that a
// supervoxel never *crosses* a real boundary; where a basin is cut in two by a
// tile seam, the graph puts it back together.
//
// So the volume is cut into fixed tiles. Each tile is flooded with a halo
// around it, from the regional minima of a seed field found **inside that
// buffer**, and keeps the labels of its own voxels. The resident set is one
// tile plus its halo, whatever the volume.
//
// Why the seeds are found per tile
// --------------------------------
// The obvious alternative — global seeds, flooded per tile — leaks. A tile
// holding part of basin A but not A's seed hands that part to whichever seed
// *is* in the buffer, across the membrane between them: exactly the crossing a
// supervoxel must not make. Minima found inside the buffer cannot do that. The
// part of A in the tile descends towards A's seed, so its lowest voxel lies on
// the buffer edge, and that is a regional minimum *of the buffer*: it seeds the
// part separately. Every global regional minimum inside the buffer is still
// one, so away from the buffer edge the seeds are the global ones. The halo
// keeps the edge-made seeds out of the core where it can.
//
// The answer depends on the tile grid, by contract
// -------------------------------------------------
// The tile is a parameter of the answer, not of the execution: the op refuses
// to run on a block grid other than its tiles ([`TiledWatershedOp`] reaches
// exactly the halo and checks that the buffer it is handed is one tile and its
// halo). [`append_tiled_watershed`] regrids the plan to the tiles first. The
// answer is then the same at any worker count and equal to the resident
// reference [`tiled_watershed`].
//
// Global seeds, when there are any
// --------------------------------
// Seeds found per tile cut every basin a seam crosses into two ids, and the
// two halves of a basin in an ambiguous region (membrane-like values on both
// sides of the seam) are exactly the pair a partition has no reason to rejoin.
// So the op can also take **global seeds** (ids from a planned seed phase):
// they keep their ids, so a basin whose seed both tiles reach is one id on
// both sides. Buffer minima are then used only where they are needed — a
// minimum without a global seed that touches a cut face of the buffer is the
// lowest point of a basin whose seed is out of reach, and it gets an edge seed
// of its own, numbered from `edge_seed_base + tile_index * ids_per_tile`.
//
// Labels
// ------
// Without global seeds, basins present in a tile's core are numbered `1..=n`
// in raster order of their first core voxel, and offset by
// `tile_index * ids_per_tile`, so ids are unique across tiles without any
// communication. They are sparse; the region graph and the relabelling after a
// partition take sparse ids. `0` is a voxel outside the mask, or one no seed
// below `seed_below` could reach.

use std::collections::HashMap;

use ndarray::{s, Array3, ArrayView3, ArrayViewMut3};

use crate::assemble::{ImageId, Phase, PlanBuilder};
use crate::dtype::Dtype;
use crate::error::{ensure, Error, Result};
use crate::geometry::BlockGrid;
use crate::op::{Anchor, BlockOp, Chain, SourceInput, SourceInputs};
use crate::reach::Reach;
use crate::voxels::Voxels;

use super::components::{label_members_into_with, offset_by, Connectivity};
use super::regional::regional_maxima_with;
use super::watershed::{seeded_watershed_into, Separation};

/// What a [`TiledWatershedOp`] computes.
#[derive(Debug, Clone, PartialEq)]
pub struct TiledWatershedConfig {
    /// The tiles, `[z, y, x]`. The answer is a function of them.
    pub tile: [usize; 3],
    /// Voxels read around a tile, `[z, y, x]`; each smaller than the tile.
    pub halo: [usize; 3],
    /// Seeds are the regional minima of the seed field whose value is below
    /// this (`f64::INFINITY` for all of them).
    pub seed_below: f64,
    /// Connectivity of the minima plateaux and of the seed components.
    pub connectivity: Connectivity,
    pub separation: Separation,
    /// Ids reserved per tile. A tile with more basins in its core is refused.
    pub ids_per_tile: u32,
    /// With global seeds: where the per-tile ids of the edge seeds start.
    /// Global seed ids must lie below it.
    pub edge_seed_base: u32,
}

impl TiledWatershedConfig {
    pub fn new(tile: [usize; 3], halo: [usize; 3]) -> Result<Self> {
        let config = Self {
            tile,
            halo,
            seed_below: f64::INFINITY,
            connectivity: Connectivity::Faces,
            separation: Separation::Adjacent,
            ids_per_tile: 1 << 16,
            edge_seed_base: 1 << 31,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn with_seed_below(mut self, seed_below: f64) -> Self {
        self.seed_below = seed_below;
        self
    }

    pub fn with_separation(mut self, separation: Separation) -> Self {
        self.separation = separation;
        self
    }

    pub fn with_ids_per_tile(mut self, ids_per_tile: u32) -> Self {
        self.ids_per_tile = ids_per_tile;
        self
    }

    fn validate(&self) -> Result<()> {
        for axis in 0..3 {
            ensure!(
                self.tile[axis] > 0,
                "tiled watershed: a tile needs a positive extent on every axis, got {:?}",
                self.tile
            );
            ensure!(
                self.halo[axis] < self.tile[axis],
                "tiled watershed: the halo {:?} must be smaller than the tile {:?} on every \
                 axis, so that a buffer names its tile",
                self.halo,
                self.tile
            );
        }
        ensure!(
            self.ids_per_tile > 0,
            "tiled watershed: ids_per_tile must be positive"
        );
        ensure!(
            !self.seed_below.is_nan(),
            "tiled watershed: seed_below is NaN"
        );
        Ok(())
    }

    fn tiles_per_axis(&self, volume: [usize; 3]) -> [usize; 3] {
        std::array::from_fn(|axis| volume[axis].div_ceil(self.tile[axis]))
    }

    /// The tile a buffer at `at` with `shape` is, and its core within the
    /// buffer: `(tile index per axis, core start, core shape)`.
    fn core_of(
        &self,
        at: &Anchor,
        shape: [usize; 3],
    ) -> Result<([usize; 3], [usize; 3], [usize; 3])> {
        let mut index = [0; 3];
        let mut start = [0; 3];
        let mut extent = [0; 3];
        for axis in 0..3 {
            let (offset, volume) = (at.offset[axis], at.volume[axis]);
            let (tile, halo) = (self.tile[axis], self.halo[axis]);
            let core = if offset == 0 { 0 } else { offset + halo };
            let end = (core + tile).min(volume);
            let expected = (end + halo).min(volume) - offset;
            if core % tile != 0 || core >= volume || shape[axis] != expected {
                return Err(Error::InvalidArgument(format!(
                    "tiled watershed: a buffer of {shape:?} at {:?} in {:?} is not one tile of \
                     {:?} with its halo {:?}. The answer is defined per tile, so the plan must \
                     run this op on the tile grid (`append_tiled_watershed` regrids it).",
                    at.offset, at.volume, self.tile, self.halo
                )));
            }
            index[axis] = core / tile;
            start[axis] = core - offset;
            extent[axis] = end - core;
        }
        Ok((index, start, extent))
    }
}

/// The tiled watershed of one buffer — a tile and its halo, at `at` — into
/// `out` (the buffer's shape). The core's voxels carry the answer; the halo's
/// carry the same ids where a core basin reaches them and `0` elsewhere, and
/// are not part of it.
///
/// `seed_field` defaults to `cost`; `mask`, when given, bounds both the seeds
/// and the flood.
pub fn tiled_watershed_into(
    cost: ArrayView3<'_, f64>,
    seed_field: Option<ArrayView3<'_, f64>>,
    seeds: Option<ArrayView3<'_, u32>>,
    mask: Option<ArrayView3<'_, bool>>,
    config: &TiledWatershedConfig,
    at: &Anchor,
    mut out: ArrayViewMut3<'_, u32>,
) -> Result<()> {
    config.validate()?;
    let shape = [cost.shape()[0], cost.shape()[1], cost.shape()[2]];
    crate::ops::shapes_agree(&shape, out.shape(), "tiled watershed: out")?;
    if let Some(field) = &seed_field {
        crate::ops::shapes_agree(&shape, field.shape(), "tiled watershed: seed field")?;
    }
    if let Some(mask) = &mask {
        crate::ops::shapes_agree(&shape, mask.shape(), "tiled watershed: mask")?;
    }
    if let Some(seeds) = &seeds {
        crate::ops::shapes_agree(&shape, seeds.shape(), "tiled watershed: seeds")?;
    }
    let (index, core_start, core_shape) = config.core_of(at, shape)?;
    let field = seed_field.unwrap_or(cost);
    ensure!(
        field.iter().all(|v| !v.is_nan()) && cost.iter().all(|v| !v.is_nan()),
        "tiled watershed: the cost and the seed field must not hold NaN"
    );

    let tiles = config.tiles_per_axis(at.volume);
    let linear = (index[0] * tiles[1] + index[1]) * tiles[2] + index[2];

    if let Some(global) = seeds {
        // Global seeds keep their ids, so a basin whose seed both tiles see is
        // one id on both sides of the seam. A buffer minimum without a global
        // seed that touches a cut face of the buffer is the lowest point of a
        // basin whose seed lies outside: it gets an edge seed of its own, so
        // that part is not flooded across a wall by a seed that is inside.
        ensure!(
            global.iter().all(|&id| id < config.edge_seed_base),
            "tiled watershed: global seed ids must be below edge_seed_base {}",
            config.edge_seed_base
        );
        let edges = edge_seed_components(field, global, mask, config, at, shape);
        let mut next = 0u64;
        let base =
            u64::from(config.edge_seed_base) + linear as u64 * u64::from(config.ids_per_tile);
        let mut all = global.to_owned();
        for component in &edges {
            next += 1;
            ensure!(
                next <= u64::from(config.ids_per_tile),
                "tiled watershed: tile {index:?} has more than {} edge seeds; raise ids_per_tile",
                config.ids_per_tile
            );
            let id = base + next;
            ensure!(
                id <= u64::from(u32::MAX),
                "tiled watershed: {} tiles of {} edge ids above {} do not fit u32 labels",
                tiles.iter().product::<usize>(),
                config.ids_per_tile,
                config.edge_seed_base
            );
            for &p in component {
                all[p] = id as u32;
            }
        }
        return seeded_watershed_into(cost, all.view(), mask, config.separation, out);
    }

    let negated = field.mapv(|v| -v);
    let minima = regional_maxima_with(negated.view(), config.connectivity)?;
    let below = |[z, y, x]: [usize; 3]| {
        minima[[z, y, x]]
            && field[[z, y, x]] < config.seed_below
            && mask.as_ref().is_none_or(|m| m[[z, y, x]])
    };
    let mut seeds = Array3::<u32>::zeros(shape);
    label_members_into_with(shape, config.connectivity, below, seeds.view_mut())?;
    let mut basins = Array3::<u32>::zeros(shape);
    seeded_watershed_into(
        cost,
        seeds.view(),
        mask,
        config.separation,
        basins.view_mut(),
    )?;

    let base = linear as u64 * u64::from(config.ids_per_tile);
    let mut ids: HashMap<u32, u32> = HashMap::new();
    let core = basins.slice(s![
        core_start[0]..core_start[0] + core_shape[0],
        core_start[1]..core_start[1] + core_shape[1],
        core_start[2]..core_start[2] + core_shape[2]
    ]);
    for &basin in core.iter() {
        if basin != 0 && !ids.contains_key(&basin) {
            let local = ids.len() as u32 + 1;
            ensure!(
                local <= config.ids_per_tile,
                "tiled watershed: tile {index:?} has more than {} basins; raise ids_per_tile",
                config.ids_per_tile
            );
            let id = base + u64::from(local);
            ensure!(
                id <= u64::from(u32::MAX),
                "tiled watershed: {} tiles of {} ids do not fit u32 labels; use larger tiles or \
                 fewer ids per tile",
                tiles.iter().product::<usize>(),
                config.ids_per_tile
            );
            ids.insert(basin, id as u32);
        }
    }
    out.zip_mut_with(&basins, |o, b| *o = ids.get(b).copied().unwrap_or(0));
    Ok(())
}

/// The edge seeds of one buffer: the components of the voxels that are a
/// regional minimum of `field` within the buffer, below `seed_below`, inside
/// `mask` and without a global seed, that touch a cut face of the buffer.
/// Ordered by their first voxel in raster order, which is the order a raster
/// labelling of every such component would number them in.
///
/// Searched from the cut faces only. A component lies inside one plateau (two
/// adjacent minima with different values cannot both be minima), so walking it
/// from a face voxel visits only that plateau, and whether a plateau is a
/// minimum is decided once, by walking it, the first time one of its voxels is
/// asked about. Taking the regional minima of the whole buffer to the same end
/// was about a sixth of a cell-detection run's CPU.
fn edge_seed_components(
    field: ArrayView3<'_, f64>,
    global: ArrayView3<'_, u32>,
    mask: Option<ArrayView3<'_, bool>>,
    config: &TiledWatershedConfig,
    at: &Anchor,
    shape: [usize; 3],
) -> Vec<Vec<[usize; 3]>> {
    const UNKNOWN: u8 = 0;
    const MINIMUM: u8 = 1;
    const NOT_MINIMUM: u8 = 2;
    const TAKEN: u8 = 4;
    let flat = |[z, y, x]: [usize; 3]| (z * shape[1] + y) * shape[2] + x;
    let offsets = config.connectivity.offsets();
    let mut state = vec![UNKNOWN; shape.iter().product()];
    let mut plateau: Vec<[usize; 3]> = Vec::new();
    // Whether `p`'s plateau is a regional minimum of the buffer, walking the
    // plateau once and recording the answer on every voxel of it.
    let mut is_minimum = |p: [usize; 3], state: &mut Vec<u8>| -> bool {
        match state[flat(p)] & 3 {
            MINIMUM => return true,
            NOT_MINIMUM => return false,
            _ => {}
        }
        let level = field[p];
        let mut minimum = true;
        plateau.clear();
        plateau.push(p);
        // Marked while walking so that a voxel is pushed once; the mark is
        // replaced by the answer below.
        state[flat(p)] |= MINIMUM;
        let mut next = 0;
        while next < plateau.len() {
            let here = plateau[next];
            next += 1;
            for &by in offsets {
                let Some(to) = offset_by(here, by, shape) else {
                    continue;
                };
                let value = field[to];
                if value < level {
                    minimum = false;
                } else if value == level && state[flat(to)] & 3 == UNKNOWN {
                    state[flat(to)] |= MINIMUM;
                    plateau.push(to);
                }
            }
        }
        if !minimum {
            for &q in &plateau {
                state[flat(q)] = (state[flat(q)] & !3) | NOT_MINIMUM;
            }
        }
        minimum
    };
    let candidate = |p: [usize; 3]| {
        global[p] == 0 && field[p] < config.seed_below && mask.as_ref().is_none_or(|m| m[p])
    };

    let mut components: Vec<(usize, Vec<[usize; 3]>)> = Vec::new();
    let mut stack: Vec<[usize; 3]> = Vec::new();
    let mut start = |p: [usize; 3], state: &mut Vec<u8>| {
        if state[flat(p)] & TAKEN != 0 || !candidate(p) || !is_minimum(p, state) {
            return;
        }
        state[flat(p)] |= TAKEN;
        let mut members = vec![p];
        let mut first = flat(p);
        stack.push(p);
        while let Some(here) = stack.pop() {
            for &by in offsets {
                let Some(to) = offset_by(here, by, shape) else {
                    continue;
                };
                if state[flat(to)] & TAKEN != 0 || !candidate(to) || !is_minimum(to, state) {
                    continue;
                }
                state[flat(to)] |= TAKEN;
                first = first.min(flat(to));
                members.push(to);
                stack.push(to);
            }
        }
        components.push((first, members));
    };
    for axis in 0..3 {
        let low = at.offset[axis] > 0;
        let high = at.offset[axis] + shape[axis] < at.volume[axis];
        let planes = [(low, 0), (high, shape[axis] - 1)];
        let (u, v) = match axis {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        for (cut, plane) in planes {
            if !cut {
                continue;
            }
            for i in 0..shape[u] {
                for j in 0..shape[v] {
                    let mut p = [0usize; 3];
                    p[axis] = plane;
                    p[u] = i;
                    p[v] = j;
                    start(p, &mut state);
                }
            }
        }
    }
    components.sort_unstable_by_key(|(first, _)| *first);
    components.into_iter().map(|(_, members)| members).collect()
}

/// The whole volume, tile by tile: the resident reference for the op.
pub fn tiled_watershed(
    cost: ArrayView3<'_, f64>,
    seed_field: Option<ArrayView3<'_, f64>>,
    seeds: Option<ArrayView3<'_, u32>>,
    mask: Option<ArrayView3<'_, bool>>,
    config: &TiledWatershedConfig,
) -> Result<Array3<u32>> {
    config.validate()?;
    let volume = [cost.shape()[0], cost.shape()[1], cost.shape()[2]];
    let tiles = config.tiles_per_axis(volume);
    let mut out = Array3::<u32>::zeros(volume);
    for tz in 0..tiles[0] {
        for ty in 0..tiles[1] {
            for tx in 0..tiles[2] {
                let tile = [tz, ty, tx];
                let core: [usize; 3] = std::array::from_fn(|a| tile[a] * config.tile[a]);
                let lo: [usize; 3] =
                    std::array::from_fn(|a| core[a].saturating_sub(config.halo[a]));
                let hi: [usize; 3] = std::array::from_fn(|a| {
                    ((core[a] + config.tile[a]).min(volume[a]) + config.halo[a]).min(volume[a])
                });
                let window = s![lo[0]..hi[0], lo[1]..hi[1], lo[2]..hi[2]];
                let mut buffer =
                    Array3::<u32>::zeros((hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]));
                tiled_watershed_into(
                    cost.slice(window),
                    seed_field.as_ref().map(|f| f.slice(window)),
                    seeds.as_ref().map(|g| g.slice(window)),
                    mask.as_ref().map(|m| m.slice(window)),
                    config,
                    &Anchor::new(lo, volume),
                    buffer.view_mut(),
                )?;
                let end: [usize; 3] =
                    std::array::from_fn(|a| (core[a] + config.tile[a]).min(volume[a]));
                out.slice_mut(s![core[0]..end[0], core[1]..end[1], core[2]..end[2]])
                    .assign(&buffer.slice(s![
                        core[0] - lo[0]..end[0] - lo[0],
                        core[1] - lo[1]..end[1] - lo[1],
                        core[2] - lo[2]..end[2] - lo[2]
                    ]));
            }
        }
    }
    Ok(out)
}

/// [`tiled_watershed_into`] as a block op over an `f64` cost, with the seed
/// field and the mask read as source images at the same reach.
#[derive(Debug, Clone)]
pub struct TiledWatershedOp {
    name: &'static str,
    seed_field: Option<ImageId>,
    seeds: Option<ImageId>,
    mask: Option<ImageId>,
    config: TiledWatershedConfig,
}

impl TiledWatershedOp {
    pub fn new(name: &'static str, config: TiledWatershedConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            name,
            seed_field: None,
            seeds: None,
            mask: None,
            config,
        })
    }

    /// Seeds from the minima of this `f64` image instead of the cost.
    pub fn seeded_by(mut self, field: ImageId) -> Self {
        self.seed_field = Some(field);
        self
    }

    /// Flood from these global seeds (`u32` ids below `edge_seed_base`, `0`
    /// for none), adding edge seeds only where a basin's seed is out of reach.
    pub fn with_seeds(mut self, seeds: ImageId) -> Self {
        self.seeds = Some(seeds);
        self
    }

    /// Seeds and flood only where this `bool` image is set.
    pub fn within(mut self, mask: ImageId) -> Self {
        self.mask = Some(mask);
        self
    }

    pub fn config(&self) -> &TiledWatershedConfig {
        &self.config
    }
}

impl BlockOp for TiledWatershedOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        self.config.halo[axis]
    }

    fn reach_spec(&self, _volume: [usize; 3]) -> Reach {
        Reach::symmetric(self.config.halo)
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        let reach = Reach::symmetric(self.config.halo);
        let mut inputs = Vec::new();
        if let Some(field) = self.seed_field {
            inputs.push(SourceInput::new(field, reach.clone()).holding(Dtype::F64));
        }
        if let Some(seeds) = self.seeds {
            inputs.push(SourceInput::new(seeds, reach.clone()).holding(Dtype::U32));
        }
        if let Some(mask) = self.mask {
            inputs.push(SourceInput::new(mask, reach).holding(Dtype::Bool));
        }
        inputs
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        dtype == Dtype::F64
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::U32
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, at: &Anchor) -> Result<()> {
        if self.seed_field.is_some() || self.seeds.is_some() || self.mask.is_some() {
            return Err(Error::InvalidArgument(format!(
                "{}: reads source images, so it is applied through `apply_with`",
                self.name
            )));
        }
        tiled_watershed_into(
            input.view::<f64>()?,
            None,
            None,
            None,
            &self.config,
            at,
            out.view_mut::<u32>()?,
        )
    }

    fn apply_with(
        &self,
        input: &Voxels,
        sources: SourceInputs<'_>,
        out: &mut Voxels,
        at: &Anchor,
    ) -> Result<()> {
        let field = match self.seed_field {
            Some(image) => Some(sources.get(image)?.view::<f64>()?),
            None => None,
        };
        let seeds = match self.seeds {
            Some(image) => Some(sources.get(image)?.view::<u32>()?),
            None => None,
        };
        let mask = match self.mask {
            Some(image) => Some(sources.get(image)?.view::<bool>()?),
            None => None,
        };
        tiled_watershed_into(
            input.view::<f64>()?,
            field,
            seeds,
            mask,
            &self.config,
            at,
            out.view_mut::<u32>()?,
        )
    }
}

/// Regrid `builder` to the op's tiles and append it as one pixel phase over
/// `cost` (the current image). The grid stays on the tiles afterwards.
pub fn append_tiled_watershed(
    builder: &mut PlanBuilder,
    volume: [usize; 3],
    op: TiledWatershedOp,
) -> Result<Phase> {
    let tile: [usize; 3] = std::array::from_fn(|a| op.config.tile[a].min(volume[a]).max(1));
    builder.regrid(BlockGrid::new(volume, tile)?);
    builder.pixels(Chain::op(op))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::ArrayEnvironment;
    use crate::strategy::{execute_phases, Hints};

    fn hash01(i: usize, j: usize, k: usize) -> f64 {
        let mut x = (i as u64 * 73_856_093) ^ (j as u64 * 19_349_663) ^ (k as u64 * 83_492_791);
        x ^= x >> 17;
        x = x.wrapping_mul(0xed5a_d4bb);
        x ^= x >> 11;
        (x % 10_000) as f64 / 10_000.0
    }

    /// Cells on a jittered lattice: membrane high on the cell walls.
    fn membrane(shape: [usize; 3], cell: usize) -> Array3<f64> {
        Array3::from_shape_fn(shape, |(i, j, k)| {
            let wall = |v: usize, phase: usize| ((v + phase) % cell == 0) as u8 as f64;
            let w = wall(i, 1)
                .max(wall(j + (i / cell) * 3, 2))
                .max(wall(k + (j / cell) * 5, 0));
            w + 0.2 * hash01(i, j, k)
        })
    }

    #[test]
    fn the_planned_op_is_the_resident_reference_at_any_worker_count() {
        let shape = [13, 37, 29];
        let cost = membrane(shape, 7);
        let field = cost.mapv(|v| v * 0.5);
        let mask = cost.mapv(|v| v < 1.15);
        let config = TiledWatershedConfig::new([6, 16, 12], [2, 4, 3])
            .unwrap()
            .with_seed_below(0.15)
            .with_ids_per_tile(500);
        let expected = tiled_watershed(
            cost.view(),
            Some(field.view()),
            None,
            Some(mask.view()),
            &config,
        )
        .unwrap();
        assert!(expected.iter().filter(|&&v| v != 0).count() > shape.iter().product::<usize>() / 2);
        for workers in [1, 3] {
            let mut builder =
                PlanBuilder::new(shape, Dtype::F64, BlockGrid::new(shape, [5, 9, 7]).unwrap());
            let op = TiledWatershedOp::new("tiled", config.clone())
                .unwrap()
                .seeded_by(ImageId::supplied(0))
                .within(ImageId::supplied(1));
            let phase = append_tiled_watershed(&mut builder, shape, op).unwrap();
            let assembly = builder.finish().unwrap();
            let env = ArrayEnvironment::with_inputs(
                Voxels::from(cost.clone()),
                vec![Voxels::from(field.clone()), Voxels::from(mask.clone())],
                &assembly.decomposition,
                [4, 8, 8],
            )
            .unwrap();
            let hints = Hints {
                concurrency: workers,
                ..Hints::default()
            };
            execute_phases(
                "tiled",
                &assembly.workflow,
                &assembly.decomposition,
                &hints,
                &env,
                &[],
                &assembly.work(),
            )
            .unwrap();
            let got = env.image(phase.image().unwrap().index());
            assert_eq!(
                got.view::<u32>().unwrap(),
                expected.view(),
                "workers {workers}"
            );
        }
    }

    #[test]
    fn ids_are_unique_per_tile_and_basins_do_not_cross_walls() {
        let shape = [8, 40, 40];
        // walls without noise: each cell interior is one plateau, one seed
        let cost = membrane(shape, 8).mapv(|v| v.floor());
        let config = TiledWatershedConfig::new([8, 20, 20], [0, 5, 5])
            .unwrap()
            .with_seed_below(0.3);
        let labels = tiled_watershed(cost.view(), None, None, None, &config).unwrap();
        // every id belongs to one tile
        let mut tile_of: HashMap<u32, [usize; 3]> = HashMap::new();
        for ((i, j, k), &id) in labels.indexed_iter() {
            if id == 0 {
                continue;
            }
            let t = [i / 8, j / 20, k / 20];
            assert_eq!(*tile_of.entry(id).or_insert(t), t, "id {id} spans tiles");
        }
        // a cell interior (away from walls and tile seams) is one label
        let labels = &labels;
        let interior: Vec<u32> = (2..6)
            .flat_map(|i| (11..14).map(move |j| labels[[i, j, 3]]))
            .collect();
        assert!(
            interior.iter().all(|&v| v == interior[0] && v != 0),
            "{interior:?}"
        );
    }

    /// With global seeds, a basin is one id across tile seams wherever its
    /// seed is in reach, and cell interiors carry the global watershed's ids.
    #[test]
    fn global_seeds_keep_their_ids_across_seams() {
        let shape = [8, 48, 48];
        let cost = membrane(shape, 8).mapv(|v| v.floor());
        let minima = regional_maxima_with(cost.mapv(|v| -v).view(), Connectivity::Faces).unwrap();
        let mut seeds = Array3::<u32>::zeros(shape);
        label_members_into_with(
            shape,
            Connectivity::Faces,
            |p| minima[p] && cost[p] < 0.5,
            seeds.view_mut(),
        )
        .unwrap();
        let global = crate::ops::watershed::seeded_watershed(
            cost.view(),
            seeds.view(),
            None,
            Separation::Adjacent,
        )
        .unwrap();
        let config = TiledWatershedConfig::new([8, 16, 16], [0, 8, 8])
            .unwrap()
            .with_seed_below(0.5);
        let tiled = tiled_watershed(cost.view(), None, Some(seeds.view()), None, &config).unwrap();
        for ((i, j, k), &c) in cost.indexed_iter() {
            if c == 0.0 {
                assert_eq!(
                    tiled[[i, j, k]],
                    global[[i, j, k]],
                    "interior voxel {:?}",
                    (i, j, k)
                );
            }
        }
        assert!(
            tiled.iter().all(|&v| v != 0 && v < config.edge_seed_base),
            "no edge seeds needed"
        );

        // planned = resident with global seeds too
        let mut builder =
            PlanBuilder::new(shape, Dtype::F64, BlockGrid::new(shape, [8, 7, 9]).unwrap());
        let op = TiledWatershedOp::new("tiled", config.clone())
            .unwrap()
            .with_seeds(ImageId::supplied(0));
        let phase = append_tiled_watershed(&mut builder, shape, op).unwrap();
        let assembly = builder.finish().unwrap();
        let env = ArrayEnvironment::with_inputs(
            Voxels::from(cost.clone()),
            vec![Voxels::from(seeds.clone())],
            &assembly.decomposition,
            [4, 8, 8],
        )
        .unwrap();
        execute_phases(
            "tiled",
            &assembly.workflow,
            &assembly.decomposition,
            &Hints::default(),
            &env,
            &[],
            &assembly.work(),
        )
        .unwrap();
        assert_eq!(
            env.image(phase.image().unwrap().index())
                .view::<u32>()
                .unwrap(),
            tiled.view()
        );
    }

    /// A basin whose seed is out of a tile's reach gets an edge seed there
    /// rather than being flooded from a seed behind a wall.
    #[test]
    fn a_basin_out_of_reach_gets_an_edge_seed_not_a_neighbour() {
        // two slabs along x separated by a wall at x = 20, each descending to
        // its seed (x = 0 and x = 30)
        let shape = [1, 4, 40];
        let cost = Array3::from_shape_fn(shape, |(_, _, k)| match k {
            20 => 1.0,
            k if k < 20 => 0.01 * k as f64,
            k => 0.01 * k.abs_diff(30) as f64,
        });
        let mut seeds = Array3::<u32>::zeros(shape);
        seeds[[0, 0, 30]] = 7; // right slab
        seeds[[0, 0, 0]] = 3; // left slab
        let config = TiledWatershedConfig::new([1, 4, 16], [0, 0, 4])
            .unwrap()
            .with_seed_below(0.5);
        let tiled = tiled_watershed(cost.view(), None, Some(seeds.view()), None, &config).unwrap();
        // tile [32, 40) holds right-slab voxels only, seeded globally
        assert!((32..40).all(|k| tiled[[0, 0, k]] == 7));
        // tile [16, 32): x 21..32 is right slab (seed 7 in reach), x 16..20 left slab with
        // its seed out of reach: an edge seed, never 7
        assert!((21..32).all(|k| tiled[[0, 0, k]] == 7));
        assert!(
            (16..20).all(|k| tiled[[0, 0, k]] != 7 && tiled[[0, 0, k]] >= config.edge_seed_base)
        );
    }

    /// The face search finds the components a raster labelling of the whole
    /// buffer's minima finds, in the same order, on fields with plateaux,
    /// masks and global seeds, under every connectivity and set of cut faces.
    #[test]
    fn the_face_search_finds_the_edge_seeds_the_whole_buffer_does() {
        let shape = [9, 11, 13];
        let mut checked = 0;
        for connectivity in [
            Connectivity::Faces,
            Connectivity::FacesAndEdges,
            Connectivity::FacesEdgesAndCorners,
        ] {
            for levels in [3.0, 6.0, 40.0] {
                // Quantised, so there are plateaux of every size.
                let field = Array3::from_shape_fn(shape, |(i, j, k)| {
                    (hash01(i, j, k) * levels).floor() - 0.25 * (i % 3) as f64
                });
                let mask = Array3::from_shape_fn(shape, |(i, j, k)| hash01(k, i, j) < 0.85);
                let global = Array3::from_shape_fn(shape, |(i, j, k)| {
                    u32::from(hash01(j, k, i) < 0.03) * (1 + (i * 100 + j * 10 + k) as u32)
                });
                let mut config = TiledWatershedConfig::new([3, 3, 3], [1, 1, 1])
                    .unwrap()
                    .with_seed_below(levels - 1.0);
                config.connectivity = connectivity;
                for offset in [[0, 0, 0], [2, 0, 3], [0, 4, 0], [1, 1, 1]] {
                    let volume = [shape[0] + 2, shape[1] + 4, shape[2] + 3];
                    let at = Anchor::new(offset, volume);

                    let negated = field.mapv(|v| -v);
                    let minima = regional_maxima_with(negated.view(), connectivity).unwrap();
                    let mut labels = Array3::<u32>::zeros(shape);
                    let count = label_members_into_with(
                        shape,
                        connectivity,
                        |p| global[p] == 0 && minima[p] && field[p] < levels - 1.0 && mask[p],
                        labels.view_mut(),
                    )
                    .unwrap();
                    let cut = |axis: usize, i: usize| {
                        (i == 0 && at.offset[axis] > 0)
                            || (i + 1 == shape[axis]
                                && at.offset[axis] + shape[axis] < volume[axis])
                    };
                    let mut expected = vec![Vec::new(); count as usize + 1];
                    let mut touches = vec![false; count as usize + 1];
                    for ((z, y, x), &l) in labels.indexed_iter() {
                        expected[l as usize].push([z, y, x]);
                        if l != 0 && (cut(0, z) || cut(1, y) || cut(2, x)) {
                            touches[l as usize] = true;
                        }
                    }
                    let expected: Vec<Vec<[usize; 3]>> = (1..=count as usize)
                        .filter(|&l| touches[l])
                        .map(|l| expected[l].clone())
                        .collect();
                    let mut found = edge_seed_components(
                        field.view(),
                        global.view(),
                        Some(mask.view()),
                        &config,
                        &at,
                        shape,
                    );
                    for members in &mut found {
                        members.sort_unstable();
                    }
                    assert_eq!(found, expected, "{connectivity:?}, {levels}, {offset:?}");
                    checked += expected.len();
                }
            }
        }
        assert!(checked > 20, "only {checked} edge components were compared");
    }

    #[test]
    fn a_buffer_off_the_tile_grid_is_refused() {
        let shape = [8, 32, 32];
        let cost = membrane(shape, 8);
        let config = TiledWatershedConfig::new([8, 16, 16], [0, 4, 4]).unwrap();
        let mut out = Array3::<u32>::zeros((8, 20, 20));
        let window = s![0..8, 6..26, 6..26];
        let refused = tiled_watershed_into(
            cost.slice(window),
            None,
            None,
            None,
            &config,
            &Anchor::new([0, 6, 6], shape),
            out.view_mut(),
        );
        assert!(refused.is_err());
        assert!(
            TiledWatershedConfig::new([8, 16, 16], [0, 16, 4]).is_err(),
            "halo >= tile"
        );
    }
}
