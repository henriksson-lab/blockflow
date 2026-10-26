// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Fusion: the placed tiles written into one volume.
//!
//! The volume is cut into planes along the wobble axis. In each plane every
//! tile whose slice there is usable sits at its own placed position, and each
//! output voxel is a function of the tiles covering it, in ascending tile
//! order. [`Method::Interpolation`] is a weighted mean whose weight is one plus
//! the voxel's distance to the nearest in-plane face **of its own tile**, so
//! the seam is a ramp across the overlap that crosses at its midline (each
//! weight is measured from a different tile's border; measured in the
//! overlap's own frame both tiles would weigh the same). The `+ 1` keeps every
//! weight positive, so a voxel on a tile's face still contributes.
//!
//! The volume's extent is the bounding box of every placed slice of every tile,
//! with the lower corner at `min(lower, 0)`. A voxel no usable tile covers is
//! zero.
//!
//! An output voxel depends on the tiles at exactly that voxel, so any block of
//! the output can be computed alone and [`blocked`] is identical to the whole
//! volume on every lattice. The tiles sit at different offsets per tile (and
//! per slice), which no `blockflow` phase can declare as a source, so the
//! lattice here is a `BlockGrid` driven directly, in parallel; and
//! [`crate::planned`] runs the same kernel slab by slab over tiles read from
//! disk, holding only the boxes each slab needs.

use blockflow::error::{Error, Result};
use blockflow::geometry::BlockGrid;
use blockflow::region::Region;
use blockflow::voxels::{VoxelElement, Voxels};
use blockflow::Dtype;
use ndarray::{Array3, ArrayView3, ArrayViewMut3};
use rayon::prelude::*;

use crate::place::PlacedLayout;
use crate::wobbly::SourceStatus;

/// What is written where tiles overlap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// The weighted mean, with [`FuseParams::weighting`]'s weights.
    Interpolation,
    Max,
    Min,
    /// The unweighted mean.
    Mean,
}

/// The weight a voxel of a tile carries in [`Method::Interpolation`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Weighting {
    /// One plus the distance to the nearest in-plane face of the tile.
    Pyramid,
    /// Every voxel weighs one.
    Uniform,
}

/// The fusion's parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FuseParams {
    pub method: Method,
    pub weighting: Weighting,
}

impl Default for FuseParams {
    fn default() -> Self {
        Self {
            method: Method::Interpolation,
            weighting: Weighting::Pyramid,
        }
    }
}

/// One placed tile: its shape, where each of its planes along the wobble axis
/// goes, and whether each plane may be used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacedSource {
    shape: [usize; 3],
    coordinate: i64,
    positions: Vec<[i64; 3]>,
    usable: Vec<bool>,
}

impl PlacedSource {
    /// `positions[k]` is where plane `k` goes; its wobble-axis coordinate must
    /// be `positions[0][axis] + k`. `status[k] >= 0` marks a usable plane.
    pub fn new(
        shape: [usize; 3],
        axis: usize,
        positions: Vec<[i64; 3]>,
        status: Vec<i32>,
    ) -> Result<Self> {
        if axis >= 3 {
            return Err(Error::invalid(format!(
                "the wobble axis is {axis} and a volume has three"
            )));
        }
        if shape.contains(&0) {
            return Err(Error::invalid(format!(
                "a placed tile of shape {shape:?} has an empty axis"
            )));
        }
        if positions.len() != shape[axis] {
            return Err(Error::ShapeMismatch {
                expected: vec![shape[axis]],
                got: vec![positions.len()],
            });
        }
        if status.len() != positions.len() {
            return Err(Error::ShapeMismatch {
                expected: vec![positions.len()],
                got: vec![status.len()],
            });
        }
        let coordinate = positions[0][axis];
        if let Some(plane) =
            (0..positions.len()).find(|&k| positions[k][axis] != coordinate + k as i64)
        {
            return Err(Error::invalid(format!(
                "plane {plane} of this tile is placed at {} on the wobble axis but the tile starts \
                 at {coordinate}; plane k must sit at start + k",
                positions[plane][axis]
            )));
        }
        Ok(Self {
            shape,
            coordinate,
            positions,
            usable: status.iter().map(|&s| s >= SourceStatus::VALID).collect(),
        })
    }

    /// A tile whose planes are all usable.
    pub fn all_valid(shape: [usize; 3], axis: usize, positions: Vec<[i64; 3]>) -> Result<Self> {
        let status = vec![SourceStatus::VALID; positions.len()];
        Self::new(shape, axis, positions, status)
    }

    /// A tile placed rigidly: every plane at `position` shifted along the axis.
    pub fn rigid(shape: [usize; 3], axis: usize, position: [i64; 3]) -> Result<Self> {
        let positions = (0..shape[axis] as i64)
            .map(|k| {
                let mut at = position;
                at[axis] += k;
                at
            })
            .collect();
        Self::all_valid(shape, axis, positions)
    }

    pub fn shape(&self) -> [usize; 3] {
        self.shape
    }

    /// The first wobble-axis coordinate this tile occupies.
    pub fn coordinate(&self) -> i64 {
        self.coordinate
    }

    pub fn position(&self, plane: usize) -> [i64; 3] {
        self.positions[plane]
    }

    /// Whether this tile contributes at wobble-axis `coordinate`.
    pub fn is_valid(&self, coordinate: i64) -> bool {
        let plane = coordinate - self.coordinate;
        (0..self.usable.len() as i64).contains(&plane) && self.usable[plane as usize]
    }

    /// The per-axis bounds over every plane, usable or not.
    fn bounds(&self, axis: usize) -> ([i64; 3], [i64; 3]) {
        let mut lower = self.positions[0];
        let mut upper = self.positions[0];
        for position in &self.positions {
            for a in 0..3 {
                lower[a] = lower[a].min(position[a]);
                upper[a] = upper[a].max(position[a]);
            }
        }
        lower[axis] = self.coordinate;
        upper[axis] = self.coordinate;
        for (bound, &extent) in upper.iter_mut().zip(&self.shape) {
            *bound += extent as i64;
        }
        (lower, upper)
    }
}

/// Every tile's placement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Placement {
    axis: usize,
    sources: Vec<PlacedSource>,
}

impl Placement {
    pub fn new(axis: usize, sources: Vec<PlacedSource>) -> Result<Self> {
        if sources.is_empty() {
            return Err(Error::invalid("a placement with no tiles has no volume"));
        }
        if axis >= 3 {
            return Err(Error::invalid(format!(
                "the wobble axis is {axis} and a volume has three"
            )));
        }
        Ok(Self { axis, sources })
    }

    /// The placement a [`PlacedLayout`] describes, its status included: a
    /// tile's isolated slices are left out of the fusion.
    pub fn from_placed(placed: &PlacedLayout, shapes: &[[usize; 3]], axis: usize) -> Result<Self> {
        if placed.positions.len() != shapes.len() || placed.status.len() != shapes.len() {
            return Err(Error::invalid(format!(
                "the placement holds {} tiles and {} shapes were given",
                placed.positions.len(),
                shapes.len()
            )));
        }
        let sources = placed
            .positions
            .iter()
            .zip(&placed.status)
            .zip(shapes)
            .map(|((positions, status), &shape)| {
                PlacedSource::new(
                    shape,
                    axis,
                    positions
                        .iter()
                        .map(|row| [row[0] as i64, row[1] as i64, row[2] as i64])
                        .collect(),
                    status.clone(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Self::new(axis, sources)
    }

    /// Every tile placed rigidly at its position.
    pub fn from_rigid(
        positions: &[[isize; 3]],
        shapes: &[[usize; 3]],
        axis: usize,
    ) -> Result<Self> {
        let sources = positions
            .iter()
            .zip(shapes)
            .map(|(position, &shape)| PlacedSource::rigid(shape, axis, position.map(|v| v as i64)))
            .collect::<Result<Vec<_>>>()?;
        Self::new(axis, sources)
    }

    pub fn axis(&self) -> usize {
        self.axis
    }

    pub fn sources(&self) -> &[PlacedSource] {
        &self.sources
    }

    /// The volume's lower corner: `min(lower, 0)` per axis.
    pub fn origin(&self) -> [i64; 3] {
        let mut origin = [0i64; 3];
        for source in &self.sources {
            let (lower, _) = source.bounds(self.axis);
            for a in 0..3 {
                origin[a] = origin[a].min(lower[a]);
            }
        }
        origin
    }

    /// The volume's shape: from [`Self::origin`] to the highest upper bound.
    pub fn shape(&self) -> Result<[usize; 3]> {
        let origin = self.origin();
        let mut upper = [i64::MIN; 3];
        for source in &self.sources {
            let (_, high) = source.bounds(self.axis);
            for a in 0..3 {
                upper[a] = upper[a].max(high[a]);
            }
        }
        let mut shape = [0usize; 3];
        for a in 0..3 {
            if upper[a] <= origin[a] {
                return Err(Error::invalid(format!(
                    "axis {a} of this placement runs from {} to {}, which is not a volume",
                    origin[a], upper[a]
                )));
            }
            shape[a] = (upper[a] - origin[a]) as usize;
        }
        Ok(shape)
    }
}

fn plane_axes(axis: usize) -> [usize; 2] {
    match axis {
        0 => [1, 2],
        1 => [0, 2],
        _ => [0, 1],
    }
}

fn weight(local: [usize; 2], shape: [usize; 2], weighting: Weighting) -> f64 {
    match weighting {
        Weighting::Uniform => 1.0,
        Weighting::Pyramid => {
            let nearest = (0..2)
                .map(|a| local[a].min(shape[a] - 1 - local[a]))
                .min()
                .unwrap_or(0);
            (nearest + 1) as f64
        }
    }
}

/// The whole volume, by the planned fusion ([`crate::planned`]) over views
/// of the tiles: slab by slab along the wobble axis, each slab's chunks in
/// parallel into one reused slab buffer, under the memory cap; nothing is held
/// beside the tiles and the volume but that slab. The same voxels as
/// [`blocked`] on any lattice.
pub fn resident(params: &FuseParams, tiles: &[Voxels], placement: &Placement) -> Result<Voxels> {
    let shape = placement.shape()?;
    let dtype = check_tiles(tiles, placement)?;
    let region = Region::new(&[0, 0, 0], &shape);
    let axis = placement.axis();
    let chunk = blockflow::zarr_env::chunk_for_extent(shape, dtype);
    let (rows, _) = crate::planned::fused_slab_rows(placement, &region, &[], axis, chunk, dtype);
    blockflow::dtype_dispatch!(
        dtype,
        |Element| {
            let views = tiles
                .iter()
                .map(|tile| tile.view::<Element>())
                .collect::<Result<Vec<_>>>()?;
            let mut out = Array3::<Element>::from_elem(shape, Element::from_f64(0.0));
            let mut at = 0;
            crate::planned::fuse_planned_boxes::<Element, [ArrayView3<'_, Element>]>(
                params,
                placement,
                &views[..],
                &region,
                axis,
                rows,
                chunk,
                false,
                |slab| {
                    let n = slab.len_of(ndarray::Axis(axis));
                    out.slice_axis_mut(ndarray::Axis(axis), ndarray::Slice::from(at..at + n))
                        .assign(&slab);
                    at += n;
                    Ok(())
                },
            )?;
            Ok(Element::wrap(out))
        },
        f16 => Err(Error::invalid("float16 tiles cannot be fused"))
    )
}

/// The volume assembled block by block over `grid`, blocks in parallel, each
/// written into the volume as it is done.
pub fn blocked(
    params: &FuseParams,
    tiles: &[Voxels],
    placement: &Placement,
    grid: &BlockGrid,
) -> Result<Voxels> {
    let shape = placement.shape()?;
    if grid.volume() != shape {
        return Err(Error::ShapeMismatch {
            expected: shape.to_vec(),
            got: grid.volume().to_vec(),
        });
    }
    let dtype = check_tiles(tiles, placement)?;
    let out = std::sync::Mutex::new(Voxels::zeros(dtype, shape)?);
    grid.cores().into_par_iter().try_for_each(|core| {
        let block = fuse_region(params, tiles, placement, &core.core)?;
        out.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .assign_region(&core.core, &block)
    })?;
    Ok(out
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner))
}

fn check_tiles(tiles: &[Voxels], placement: &Placement) -> Result<Dtype> {
    if tiles.len() != placement.sources().len() {
        return Err(Error::ShapeMismatch {
            expected: vec![placement.sources().len()],
            got: vec![tiles.len()],
        });
    }
    let dtype = tiles[0].dtype();
    for (index, (tile, source)) in tiles.iter().zip(placement.sources()).enumerate() {
        if tile.dtype() != dtype {
            return Err(Error::invalid(format!(
                "tile {index} holds {} and tile 0 holds {}; fusion needs one element type",
                tile.dtype().numpy_name(),
                dtype.numpy_name()
            )));
        }
        if tile.shape() != source.shape() {
            return Err(Error::ShapeMismatch {
                expected: source.shape().to_vec(),
                got: tile.shape().to_vec(),
            });
        }
    }
    Ok(dtype)
}

/// One box of the output volume (in output coordinates), computed alone.
pub fn fuse_region(
    params: &FuseParams,
    tiles: &[Voxels],
    placement: &Placement,
    region: &Region,
) -> Result<Voxels> {
    region.check_within(&placement.shape()?, "fusion output region")?;
    let dtype = check_tiles(tiles, placement)?;
    blockflow::dtype_dispatch!(
        dtype,
        |Element| {
            let views = tiles
                .iter()
                .map(|tile| tile.view::<Element>())
                .collect::<Result<Vec<_>>>()?;
            Ok(Element::wrap(fuse_typed(params, &views, placement, region)))
        },
        f16 => Err(Error::invalid("float16 tiles cannot be fused"))
    )
}

fn fuse_typed<T: VoxelElement>(
    params: &FuseParams,
    tiles: &[ArrayView3<T>],
    placement: &Placement,
    region: &Region,
) -> Array3<T> {
    fuse_with(
        params,
        |tile, at| tiles[tile][at].into_f64(),
        placement,
        region,
    )
}

/// The fusion of `region` (output coordinates), with `sample(tile, at)` the
/// value of tile `tile` at its own voxel `at` (tile coordinates, wobble-axis
/// index counted from the placement's first plane of that tile).
///
/// `sample` is asked only for voxels [`needed_box`] reports for `region`, in
/// ascending tile order per output voxel, so a caller holding only that box
/// of each tile gets the same answer as one holding every tile whole.
pub(crate) fn fuse_with<T: VoxelElement>(
    params: &FuseParams,
    sample: impl Fn(usize, [usize; 3]) -> f64,
    placement: &Placement,
    region: &Region,
) -> Array3<T> {
    let mut out = Array3::from_elem(
        (region.shape[0], region.shape[1], region.shape[2]),
        T::from_f64(0.0),
    );
    fuse_into(params, sample, placement, region, out.view_mut());
    out
}

/// [`fuse_with`], into `out` (of `region`'s shape), which it overwrites.
pub(crate) fn fuse_into<T: VoxelElement>(
    params: &FuseParams,
    sample: impl Fn(usize, [usize; 3]) -> f64,
    placement: &Placement,
    region: &Region,
    mut out: ArrayViewMut3<'_, T>,
) {
    debug_assert_eq!(out.shape(), &region.shape[..]);
    out.fill(T::from_f64(0.0));
    let origin = placement.origin();
    let axis = placement.axis();
    let plane = plane_axes(axis);
    let floating = matches!(T::DTYPE, Dtype::F32 | Dtype::F64);
    let lower: [i64; 3] = std::array::from_fn(|a| origin[a] + region.start[a] as i64);
    let upper: [i64; 3] = std::array::from_fn(|a| lower[a] + region.shape[a] as i64);
    let mut covering = Vec::with_capacity(placement.sources().len());
    let mut row = Vec::with_capacity(placement.sources().len());
    for k in 0..region.shape[axis] {
        let coordinate = lower[axis] + k as i64;
        // The tiles usable in this plane that reach into the region, in
        // ascending order, with their plane index, in-plane position and size.
        // A tile outside the region's plane would cover none of its voxels, so
        // leaving it out changes nothing but the time.
        let active = placement
            .sources()
            .iter()
            .enumerate()
            .filter(|(_, source)| source.is_valid(coordinate))
            .filter_map(|(index, source)| {
                let tile_plane = (coordinate - source.coordinate()) as usize;
                let at = source.position(tile_plane);
                let at = [at[plane[0]], at[plane[1]]];
                let shape = source.shape();
                let size = [shape[plane[0]], shape[plane[1]]];
                let reaches = (0..2)
                    .all(|a| at[a] < upper[plane[a]] && at[a] + size[a] as i64 > lower[plane[a]]);
                reaches.then_some((index, tile_plane, at, size))
            })
            .collect::<Vec<_>>();
        if active.is_empty() {
            continue;
        }
        for u in 0..region.shape[plane[0]] {
            let global_u = lower[plane[0]] + u as i64;
            // The tiles covering this row, still in ascending order.
            row.clear();
            row.extend(active.iter().filter_map(|&(index, tile_plane, at, size)| {
                let local = global_u - at[0];
                (0..size[0] as i64).contains(&local).then_some((
                    index,
                    tile_plane,
                    local as usize,
                    at[1],
                    size,
                ))
            }));
            if row.is_empty() {
                continue;
            }
            for v in 0..region.shape[plane[1]] {
                let global_v = lower[plane[1]] + v as i64;
                covering.clear();
                for &(index, tile_plane, local_u, at_v, size) in &row {
                    let local_v = global_v - at_v;
                    if (0..size[1] as i64).contains(&local_v) {
                        let local = [local_u, local_v as usize];
                        let mut index3 = [0usize; 3];
                        index3[axis] = tile_plane;
                        index3[plane[0]] = local[0];
                        index3[plane[1]] = local[1];
                        let value = sample(index, index3);
                        covering.push((value, weight(local, size, params.weighting)));
                    }
                }
                if covering.is_empty() {
                    continue;
                }
                let value = blend(params.method, &covering);
                let mut at = [0usize; 3];
                at[axis] = k;
                at[plane[0]] = u;
                at[plane[1]] = v;
                out[at] = T::from_f64(if floating { value } else { value.round() });
            }
        }
    }
}

/// The box of each tile (tile coordinates, half-open `start..stop`) that
/// [`fuse_with`] reads for `region`, or `None` for a tile it does not read.
pub(crate) fn needed_box(
    placement: &Placement,
    region: &Region,
) -> Vec<Option<([usize; 3], [usize; 3])>> {
    let origin = placement.origin();
    let axis = placement.axis();
    let plane = plane_axes(axis);
    let lower: [i64; 3] = std::array::from_fn(|a| origin[a] + region.start[a] as i64);
    let upper: [i64; 3] = std::array::from_fn(|a| lower[a] + region.shape[a] as i64);
    placement
        .sources()
        .iter()
        .map(|source| {
            let shape = source.shape();
            let mut start = [usize::MAX; 3];
            let mut stop = [0usize; 3];
            let first = (lower[axis] - source.coordinate()).max(0);
            let last = (upper[axis] - source.coordinate()).min(shape[axis] as i64);
            for tile_plane in first..last {
                if !source.is_valid(source.coordinate() + tile_plane) {
                    continue;
                }
                let at = source.position(tile_plane as usize);
                let mut lo = [0usize; 2];
                let mut hi = [0usize; 2];
                let mut reaches = true;
                for a in 0..2 {
                    let size = shape[plane[a]] as i64;
                    let l = (lower[plane[a]] - at[plane[a]]).max(0);
                    let h = (upper[plane[a]] - at[plane[a]]).min(size);
                    if l >= h {
                        reaches = false;
                        break;
                    }
                    lo[a] = l as usize;
                    hi[a] = h as usize;
                }
                if !reaches {
                    continue;
                }
                let tile_plane = tile_plane as usize;
                start[axis] = start[axis].min(tile_plane);
                stop[axis] = stop[axis].max(tile_plane + 1);
                for a in 0..2 {
                    start[plane[a]] = start[plane[a]].min(lo[a]);
                    stop[plane[a]] = stop[plane[a]].max(hi[a]);
                }
            }
            (start[axis] < stop[axis]).then_some((start, stop))
        })
        .collect()
}

fn blend(method: Method, covering: &[(f64, f64)]) -> f64 {
    match method {
        Method::Max => covering
            .iter()
            .map(|c| c.0)
            .fold(f64::NEG_INFINITY, f64::max),
        Method::Min => covering.iter().map(|c| c.0).fold(f64::INFINITY, f64::min),
        Method::Mean => covering.iter().map(|c| c.0).sum::<f64>() / covering.len() as f64,
        Method::Interpolation => {
            let (numerator, denominator) = covering
                .iter()
                .fold((0.0, 0.0), |(n, d), &(value, w)| (n + value * w, d + w));
            numerator / denominator
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pyramid_is_one_plus_the_distance_to_the_nearest_face() {
        let shape = [5usize, 3];
        let expected = [[1, 1, 1], [1, 2, 1], [1, 2, 1], [1, 2, 1], [1, 1, 1]];
        for (i, row) in expected.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                assert_eq!(
                    weight([i, j], shape, Weighting::Pyramid),
                    value as f64,
                    "({i}, {j})"
                );
                assert_eq!(
                    weight([i, j], shape, Weighting::Pyramid),
                    weight([4 - i, 2 - j], shape, Weighting::Pyramid),
                    "symmetric"
                );
            }
        }
    }

    #[test]
    fn the_status_rule_admits_fixed_and_refuses_isolated() {
        let positions: Vec<[i64; 3]> = (0..4).map(|k| [0, 0, k]).collect();
        let source = PlacedSource::new([2, 2, 4], 2, positions, vec![0, 2, -1, -2])
            .expect("a placed source");
        assert!(source.is_valid(0));
        assert!(source.is_valid(1));
        assert!(!source.is_valid(2));
        assert!(!source.is_valid(3));
        assert!(!source.is_valid(-1));
        assert!(!source.is_valid(4));
    }

    #[test]
    fn a_placement_whose_axis_column_is_not_a_range_is_refused() {
        let message = PlacedSource::all_valid([2, 2, 3], 2, vec![[0, 0, 5], [0, 0, 6], [0, 0, 8]])
            .expect_err("a refusal")
            .to_string();
        assert!(message.contains("plane 2"), "{message}");
    }

    #[test]
    fn the_extent_is_the_placed_bounding_box() {
        let shape = [4usize, 3, 2];
        let placement = Placement::new(
            2,
            vec![
                PlacedSource::all_valid(shape, 2, vec![[0, 0, 0], [2, 1, 1]]).expect("a source"),
                PlacedSource::all_valid(shape, 2, vec![[3, 0, 1], [3, 0, 2]]).expect("a source"),
            ],
        )
        .expect("a placement");
        assert_eq!(placement.origin(), [0, 0, 0]);
        assert_eq!(placement.shape().expect("a shape"), [7, 4, 3]);
    }

    #[test]
    fn a_negative_corner_is_kept() {
        let shape = [4usize, 3, 1];
        let placement = Placement::new(
            2,
            vec![
                PlacedSource::all_valid(shape, 2, vec![[-2, 0, 0]]).expect("a source"),
                PlacedSource::all_valid(shape, 2, vec![[1, 0, 0]]).expect("a source"),
            ],
        )
        .expect("a placement");
        assert_eq!(placement.origin(), [-2, 0, 0]);
        assert_eq!(placement.shape().expect("a shape"), [7, 3, 1]);
        let tiles = vec![
            Voxels::U16(Array3::from_elem((4, 3, 1), 7)),
            Voxels::U16(Array3::from_elem((4, 3, 1), 9)),
        ];
        let out = resident(&FuseParams::default(), &tiles, &placement).expect("a fusion");
        let view = out.view::<u16>().expect("u16");
        assert_eq!(
            view[[0, 0, 0]],
            7,
            "the space below zero holds the first tile"
        );
        assert_eq!(view[[6, 0, 0]], 9);
    }

    #[test]
    fn tiles_of_mixed_element_types_are_refused() {
        let placement = Placement::new(
            2,
            vec![
                PlacedSource::all_valid([2, 2, 1], 2, vec![[0, 0, 0]]).expect("a source"),
                PlacedSource::all_valid([2, 2, 1], 2, vec![[1, 0, 0]]).expect("a source"),
            ],
        )
        .expect("a placement");
        let tiles = vec![
            Voxels::zeros(Dtype::U16, [2, 2, 1]).expect("a tile"),
            Voxels::zeros(Dtype::F32, [2, 2, 1]).expect("a tile"),
        ];
        let message = resident(&FuseParams::default(), &tiles, &placement)
            .expect_err("a refusal")
            .to_string();
        assert!(message.contains("one element type"), "{message}");
    }
}
