// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Tile stitching: layout fit, rigid and per-slice alignment, and fusion of
//! overlapping light-sheet tiles into one OME-Zarr volume.
//!
//! The chain, one module per arrow:
//!
//! ```text
//! tiles + grid + overlap
//!    | layout: nominal positions from the grid       (layout)
//!    | rigid:  max-projection correlation per pair   (rigid)
//!    | layout: least-squares placement               (layout)
//!    v
//! rigid positions
//!    | per-slice measurement per pair                (wobbly)
//!    | per-slice and whole-axis placement            (place)
//!    v
//! one position per tile per slice
//!    | weighted-mean fusion                          (fuse)
//!    v
//! fused volume
//! ```
//!
//! Volumes are `(z, y, x)`. The tiles are tiled along x and y (`grid_axes =
//! [2, 1]`) and wobble along z (`wobble_axis = 0`) by default; every module
//! takes the axes as parameters.
//!
//! Alignment reads boxes (`tiles`): the rigid step a `depth`-thick slab at
//! each facing end, the per-slice step the overlap cuts. From stored tiles
//! ([`align_stored`]) only the boxes of the pairs in flight are held, and the
//! pairs in flight are as many as the memory cap leaves room for; where a
//! pair's slabs and cuts share stored chunks, the rigid step reads both as one
//! and keeps the cuts, leased, for the per-slice step. Fusion is planned
//! (`planned`): slab by slab, reading only the tile boxes each slab needs, and
//! written slab by slab into core's multiscale writer (`pyramid`). Resident
//! tiles ([`align`], [`stitch`], [`fuse::resident`]) go through the same
//! boxes, as views.

pub mod fuse;
pub mod io;
pub mod layout;
pub mod place;
pub mod plane;
pub mod planned;
pub mod pyramid;
pub mod relayout;
pub mod rigid;
pub mod solve;
pub mod tile_dir;
pub mod tiles;
pub mod wobbly;

use blockflow::error::{Error, Result};
use blockflow::voxels::{VoxelElement, Voxels};
use ndarray::ArrayView3;

use crate::fuse::{FuseParams, Placement};
use crate::layout::{RigidPlacementParams, TileLayout};
use crate::place::{PlaceParams, PlacedLayout, WobblyLayout};
use crate::rigid::RigidParams;
use crate::tiles::{StoredTiles, TileBoxes};
use crate::wobbly::MeasureParams;

/// Every parameter of the chain.
#[derive(Debug, Clone)]
pub struct StitchParams {
    pub rigid: RigidParams,
    pub rigid_placement: RigidPlacementParams,
    /// Run the per-slice alignment. Off, every tile is fused at its rigid
    /// position in every slice.
    pub wobbly: bool,
    pub measure: MeasureParams,
    pub place: PlaceParams,
    pub fuse: FuseParams,
    /// The axis along which tiles wobble (the stage's travel axis).
    pub wobble_axis: usize,
}

impl Default for StitchParams {
    fn default() -> Self {
        Self {
            rigid: RigidParams::default(),
            rigid_placement: RigidPlacementParams::default(),
            wobbly: true,
            measure: MeasureParams::default(),
            place: PlaceParams::default(),
            fuse: FuseParams::default(),
            wobble_axis: 0,
        }
    }
}

/// What a stitching run produced.
#[derive(Debug, Clone)]
pub struct Stitched {
    /// The layout with its alignments measured and its rigid positions placed.
    pub layout: TileLayout,
    /// The per-slice measurement, when it ran.
    pub wobbly: Option<WobblyLayout>,
    /// The per-slice placement, when it ran.
    pub placed: Option<PlacedLayout>,
    /// What the fusion used.
    pub placement: Placement,
    pub fused: Voxels,
}

/// Align `layout` against `tiles` (in place) and place it rigidly.
pub fn align_rigid(tiles: &[Voxels], layout: &mut TileLayout, params: &StitchParams) -> Result<()> {
    struct Align<'a> {
        layout: &'a mut TileLayout,
        params: &'a RigidParams,
    }
    impl TileVisitor for Align<'_> {
        type Out = ();
        fn visit<T: VoxelElement>(self, views: &[ArrayView3<'_, T>]) -> Result<()> {
            rigid::align_layout(self.layout, views, self.params)
        }
    }
    visit_tiles(
        tiles,
        Align {
            layout: &mut *layout,
            params: &params.rigid,
        },
    )?;
    layout.place_in_place(&params.rigid_placement)
}

/// The per-slice placement for a rigidly placed layout.
pub fn place_wobbly(
    tiles: &[Voxels],
    layout: &TileLayout,
    params: &StitchParams,
) -> Result<PlacedLayout> {
    place::place_layout(&measure_wobbly(tiles, layout, params)?, &params.place)
}

/// The per-slice measurement of every alignment of a rigidly placed layout.
pub fn measure_wobbly(
    tiles: &[Voxels],
    layout: &TileLayout,
    params: &StitchParams,
) -> Result<WobblyLayout> {
    struct Measure<'a> {
        layout: &'a TileLayout,
        params: &'a StitchParams,
    }
    impl TileVisitor for Measure<'_> {
        type Out = WobblyLayout;
        fn visit<T: VoxelElement>(self, views: &[ArrayView3<'_, T>]) -> Result<WobblyLayout> {
            let pairs = self
                .layout
                .alignments()
                .iter()
                .map(|alignment| (alignment.pre, alignment.post))
                .collect::<Vec<_>>();
            wobbly::measure_layout(
                self.layout.positions(),
                &pairs,
                views,
                self.params.wobble_axis,
                &self.params.measure,
            )
        }
    }
    visit_tiles(tiles, Measure { layout, params })
}

/// What the alignment produced, before fusion.
#[derive(Debug, Clone)]
pub struct Aligned {
    /// The layout with its alignments measured and its rigid positions placed.
    pub layout: TileLayout,
    /// The per-slice measurement, when it ran.
    pub wobbly: Option<WobblyLayout>,
    /// The per-slice placement, when it ran.
    pub placed: Option<PlacedLayout>,
    /// What a fusion would use.
    pub placement: Placement,
}

/// The chain up to the placement: rigid, then (unless `params.wobbly` is off)
/// per-slice measurement and placement.
pub fn align(tiles: &[Voxels], layout: TileLayout, params: &StitchParams) -> Result<Aligned> {
    if tiles.len() != layout.sources().len() {
        return Err(Error::invalid(format!(
            "the layout has {} tiles and {} volumes were given",
            layout.sources().len(),
            tiles.len()
        )));
    }
    for (index, (tile, source)) in tiles.iter().zip(layout.sources()).enumerate() {
        if tile.shape() != source.shape {
            return Err(Error::invalid(format!(
                "tile {index} is {:?} and the layout says {:?}",
                tile.shape(),
                source.shape
            )));
        }
    }
    struct Align<'a> {
        layout: TileLayout,
        params: &'a StitchParams,
    }
    impl TileVisitor for Align<'_> {
        type Out = Aligned;
        fn visit<T: VoxelElement>(self, views: &[ArrayView3<'_, T>]) -> Result<Aligned> {
            align_boxes(views, self.layout, self.params)
        }
    }
    visit_tiles(tiles, Align { layout, params })
}

/// [`align`], reading only the boxes each pair needs from the stored tiles:
/// no tile is held whole, and the pairs in flight are as many as the memory
/// cap leaves room for.
pub fn align_stored(
    tiles: &StoredTiles,
    layout: TileLayout,
    params: &StitchParams,
) -> Result<Aligned> {
    if tiles.sources.len() != layout.sources().len() {
        return Err(Error::invalid(format!(
            "the layout has {} tiles and {} were given",
            layout.sources().len(),
            tiles.sources.len()
        )));
    }
    for (index, (shape, source)) in tiles.shapes.iter().zip(layout.sources()).enumerate() {
        if *shape != source.shape {
            return Err(Error::invalid(format!(
                "tile {index} is {shape:?} and the layout says {:?}",
                source.shape
            )));
        }
    }
    blockflow::dtype_dispatch!(
        tiles.dtype,
        |Element| align_boxes::<Element, _>(tiles, layout, params),
        f16 => Err(Error::invalid("float16 tiles are not supported"))
    )
}

/// The chain up to the placement over any source of tile boxes.
fn align_boxes<T: VoxelElement, B: TileBoxes<T> + ?Sized>(
    tiles: &B,
    mut layout: TileLayout,
    params: &StitchParams,
) -> Result<Aligned> {
    let shapes = (0..tiles.len())
        .map(|tile| tiles.shape(tile))
        .collect::<Vec<_>>();
    let pairs = layout
        .alignments()
        .iter()
        .map(|alignment| (alignment.pre, alignment.post))
        .collect::<Vec<_>>();
    let measured = params.wobbly && !pairs.is_empty();
    // The per-slice step cuts much of what the rigid step reads (all of it,
    // for narrow tiles), so the rigid step reads ahead the boxes the cuts will
    // take once the rigid positions are placed, wherever that decodes less
    // than reading the two apart.
    let ahead = if measured {
        wobbly::cut_boxes_within(
            layout.positions(),
            &pairs,
            &shapes,
            params.wobble_axis,
            &params.measure,
            &read_ahead_margins(&layout, &params.rigid),
        )
    } else {
        Vec::new()
    };
    let kept = rigid::align_layout_keeping(&mut layout, tiles, &params.rigid, &ahead)?;
    layout.place_in_place(&params.rigid_placement)?;
    let (wobbly, placed, placement) = if measured {
        let wobbly = wobbly::measure_layout_kept(
            layout.positions(),
            &pairs,
            tiles,
            params.wobble_axis,
            &params.measure,
            kept,
        )?;
        let placed = place::place_layout(&wobbly, &params.place)?;
        let placement = Placement::from_placed(&placed, &shapes, params.wobble_axis)?;
        (Some(wobbly), Some(placed), placement)
    } else {
        let placement = Placement::from_rigid(layout.positions(), &shapes, params.wobble_axis)?;
        (None, None, placement)
    };
    Ok(Aligned {
        layout,
        wobbly,
        placed,
        placement,
    })
}

/// How far each pair's position difference may move between the nominal and
/// the rigidly placed layout, for reading its cuts ahead: on the two axes the
/// rigid step corrects, its search range; on the tiling axis, which it leaves
/// alone, none; and on every axis a few voxels for what the placement's least
/// squares spread. A pair that moves further has its cuts read again.
fn read_ahead_margins(layout: &TileLayout, params: &RigidParams) -> Vec<[(isize, isize); 3]> {
    const SPREAD: isize = 2;
    (0..layout.alignments().len())
        .map(|index| {
            let tiling = layout.tiling_axis(index);
            std::array::from_fn(|axis| {
                let (low, high) = if tiling == Some(axis) {
                    (0, 0)
                } else {
                    params.max_shifts[axis]
                };
                (low - SPREAD, high + SPREAD)
            })
        })
        .collect()
}

/// The whole chain, from tiles and a nominal layout to the fused volume.
pub fn stitch(tiles: &[Voxels], layout: TileLayout, params: &StitchParams) -> Result<Stitched> {
    let Aligned {
        layout,
        wobbly,
        placed,
        placement,
    } = align(tiles, layout, params)?;
    let fused = fuse::resident(&params.fuse, tiles, &placement)?;
    Ok(Stitched {
        layout,
        wobbly,
        placed,
        placement,
        fused,
    })
}

/// Something run on typed views of every tile. The alignment arithmetic is in
/// `f64`, but widening a whole tile costs eight bytes a voxel, so the steps take
/// views of the stored type and widen only the slabs and cuts they read.
trait TileVisitor {
    type Out;
    fn visit<T: VoxelElement>(self, views: &[ArrayView3<'_, T>]) -> Result<Self::Out>;
}

fn visit_tiles<V: TileVisitor>(tiles: &[Voxels], visitor: V) -> Result<V::Out> {
    let dtype = tiles
        .first()
        .ok_or_else(|| Error::invalid("no tiles"))?
        .dtype();
    if let Some(index) = tiles.iter().position(|tile| tile.dtype() != dtype) {
        return Err(Error::invalid(format!(
            "tile {index} holds {} and tile 0 holds {}; the tiles must share one element type",
            tiles[index].dtype().numpy_name(),
            dtype.numpy_name()
        )));
    }
    blockflow::dtype_dispatch!(
        dtype,
        |Element| {
            let views = tiles
                .iter()
                .map(|tile| tile.view::<Element>())
                .collect::<Result<Vec<_>>>()?;
            visitor.visit(&views)
        },
        f16 => Err(Error::invalid("float16 tiles are not supported"))
    )
}
