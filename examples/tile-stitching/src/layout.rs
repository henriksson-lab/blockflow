// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The tile layout and the rigid placement: how pairwise offsets become one
//! position per tile.
//!
//! A layout is a set of tiles, a position for each, and one [`Alignment`] per
//! pair of neighbours: grid neighbours, or tiles whose boxes overlap
//! ([`TileLayout::from_boxes`]). The rigid correlation ([`crate::rigid`]) writes each
//! alignment's shift; [`place`] turns the displacements into positions:
//!
//! 1. **Filter.** Alignments with `quality > min_quality` are kept, strictly, so
//!    a pair the background check refused (`quality = -inf`) is dropped rather
//!    than placed at zero.
//! 2. **Components.** The kept alignments split the tiles into connected
//!    components, each placed on its own. A lone tile keeps its position.
//! 3. **Place.** Least squares over the component, or a walk along a
//!    breadth-first tree. On a tree the two agree; round a cycle whose offsets
//!    do not close, least squares spreads the inconsistency and the tree ignores
//!    the edges it did not walk.
//! 4. **Pin.** The component is translated so one source (its middle one by
//!    default) stays where it was.
//! 5. **Lower to origin.** The per-axis minimum over every tile is subtracted.

use std::collections::VecDeque;

use blockflow::error::{Error, Result};

use crate::solve::{connected_components, fit_positions, round_even};

/// How the pairwise displacements become absolute positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementMethod {
    /// Least squares over the whole component.
    Optimization,
    /// Accumulate displacements along a breadth-first tree from the pinned
    /// source.
    Tree,
}

/// Which source of a component is held where it started.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixedSource {
    /// `members[(len - 1) / 2]`.
    Middle,
    /// The component's lowest-numbered source.
    First,
}

/// One tile of the grid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileSource {
    /// The tile's shape in voxels.
    pub shape: [usize; 3],
    /// Where the tile sits on the acquisition grid, `(0, 0)`, `(1, 0)`, ...
    /// Empty for a layout from boxes when the tile has no grid index.
    pub tile_position: Vec<isize>,
}

/// One pair of grid neighbours and the offset between them.
///
/// `displacement` is `post.position - pre.position + shift`: the alignment
/// carries the absolute offset, and the shift is the correction to the grid's
/// guess. [`TileLayout::set_shift`] is the only writer, so the two agree.
#[derive(Debug, Clone, PartialEq)]
pub struct Alignment {
    pub pre: usize,
    pub post: usize,
    pub displacement: [isize; 3],
    /// `-inf` until the correlation runs, which is also what the background
    /// check returns and therefore what [`place`]'s filter drops.
    pub quality: f64,
}

/// A grid of tiles, their positions, and the alignments between neighbours.
#[derive(Debug, Clone)]
pub struct TileLayout {
    sources: Vec<TileSource>,
    grid_axes: Vec<usize>,
    alignments: Vec<Alignment>,
    /// Per alignment, the volume axis its tiles are tiled along.
    tiling_axes: Vec<usize>,
    positions: Vec<[isize; 3]>,
}

impl TileLayout {
    /// The layout a regular grid implies.
    ///
    /// `grid_axes[d]` is the volume axis that grid axis `d` runs along, so a
    /// `(z, y, x)` tile tiled in x then y has `grid_axes = [2, 1]`. `overlaps[d]`
    /// is the nominal overlap along grid axis `d` (one value is used for every
    /// axis). Each tile with a neighbour at `+1` on grid axis `d` gets an
    /// alignment of `shape - overlap` along `grid_axes[d]`, and the positions
    /// are walked out of those from tile 0 and lowered to the origin.
    pub fn on_grid(
        sources: Vec<TileSource>,
        overlaps: &[isize],
        grid_axes: &[usize],
    ) -> Result<Self> {
        let tile_dim = check_sources(&sources, grid_axes)?;
        if overlaps.len() != tile_dim && overlaps.len() != 1 {
            return Err(Error::invalid(format!(
                "the grid has {tile_dim} axes and {} overlaps were given; give one per grid axis \
                 or one for all",
                overlaps.len()
            )));
        }
        let mut alignments = Vec::new();
        let mut tiling_axes = Vec::new();
        for (grid_axis, &axis) in grid_axes.iter().enumerate() {
            let overlap = overlaps[grid_axis.min(overlaps.len() - 1)];
            for (pre, post) in neighbours(&sources, grid_axis) {
                let mut displacement = [0isize; 3];
                displacement[axis] = sources[pre].shape[axis] as isize - overlap;
                alignments.push(Alignment {
                    pre,
                    post,
                    displacement,
                    quality: f64::NEG_INFINITY,
                });
                tiling_axes.push(axis);
            }
        }
        let n = sources.len();
        let mut positions = walk_tree(n, &alignments, &(0..n).collect::<Vec<_>>(), 0)?;
        lower_to_origin(&mut positions);
        Ok(Self {
            sources,
            grid_axes: grid_axes.to_vec(),
            alignments,
            tiling_axes,
            positions,
        })
    }

    /// A layout whose nominal positions are given (stage coordinates, in
    /// voxels) rather than derived from an overlap. Alignments are still
    /// between grid neighbours, with the given positions as their guess.
    pub fn with_positions(
        sources: Vec<TileSource>,
        positions: Vec<[isize; 3]>,
        grid_axes: &[usize],
    ) -> Result<Self> {
        check_sources(&sources, grid_axes)?;
        if positions.len() != sources.len() {
            return Err(Error::invalid(format!(
                "{} tiles and {} positions",
                sources.len(),
                positions.len()
            )));
        }
        let mut alignments = Vec::new();
        let mut tiling_axes = Vec::new();
        for (grid_axis, &axis) in grid_axes.iter().enumerate() {
            for (pre, post) in neighbours(&sources, grid_axis) {
                alignments.push(Alignment {
                    pre,
                    post,
                    displacement: std::array::from_fn(|axis| {
                        positions[post][axis] - positions[pre][axis]
                    }),
                    quality: f64::NEG_INFINITY,
                });
                tiling_axes.push(axis);
            }
        }
        Ok(Self {
            sources,
            grid_axes: grid_axes.to_vec(),
            alignments,
            tiling_axes,
            positions,
        })
    }

    /// A layout whose nominal positions are given, with neighbours found from
    /// the tiles' boxes at those positions instead of from grid indices.
    ///
    /// Two tiles are neighbours along tiling axis `a` (one of `tiling_axes`,
    /// the in-plane volume axes) when their boxes overlap along `a`, they are
    /// offset along `a`, and on every other tiling axis they are offset by
    /// less than half a tile and relatively less than along `a`. So a
    /// regular grid gives its row and column neighbours and no diagonals,
    /// and a staggered or partial mosaic gives the pairs that share a seam.
    /// The tile further along `a` is the alignment's `post`. Grid indices
    /// are not used (a source's `tile_position` may be empty).
    pub fn from_boxes(
        sources: Vec<TileSource>,
        positions: Vec<[isize; 3]>,
        tiling_axes: &[usize],
    ) -> Result<Self> {
        if sources.is_empty() {
            return Err(Error::invalid("a layout needs at least one tile"));
        }
        if tiling_axes.is_empty() || tiling_axes.len() > 3 || tiling_axes.iter().any(|&a| a > 2) {
            return Err(Error::invalid(format!(
                "tiling axes {tiling_axes:?}: one to three volume axes"
            )));
        }
        if positions.len() != sources.len() {
            return Err(Error::invalid(format!(
                "{} tiles and {} positions",
                sources.len(),
                positions.len()
            )));
        }
        let mut alignments = Vec::new();
        let mut alignment_axes = Vec::new();
        for &axis in tiling_axes {
            for i in 0..sources.len() {
                for j in 0..sources.len() {
                    // i is `pre`: j lies further along `axis`
                    let d: [isize; 3] = std::array::from_fn(|a| positions[j][a] - positions[i][a]);
                    if d[axis] <= 0 || d[axis] >= sources[i].shape[axis] as isize {
                        continue;
                    }
                    let relative = |a: usize| {
                        d[a].unsigned_abs() as f64
                            / sources[i].shape[a].max(sources[j].shape[a]).max(1) as f64
                    };
                    let along = relative(axis);
                    let aside = tiling_axes
                        .iter()
                        .filter(|&&other| other != axis)
                        .all(|&other| relative(other) < 0.5 && relative(other) < along);
                    if aside {
                        alignments.push(Alignment {
                            pre: i,
                            post: j,
                            displacement: d,
                            quality: f64::NEG_INFINITY,
                        });
                        alignment_axes.push(axis);
                    }
                }
            }
        }
        Ok(Self {
            sources,
            grid_axes: tiling_axes.to_vec(),
            alignments,
            tiling_axes: alignment_axes,
            positions,
        })
    }

    pub fn sources(&self) -> &[TileSource] {
        &self.sources
    }

    /// The volume axis each grid axis runs along.
    pub fn grid_axes(&self) -> &[usize] {
        &self.grid_axes
    }

    pub fn alignments(&self) -> &[Alignment] {
        &self.alignments
    }

    /// The volume axis alignment `index`'s tiles are tiled along.
    pub fn tiling_axis(&self, index: usize) -> Option<usize> {
        self.tiling_axes.get(index).copied()
    }

    pub fn positions(&self) -> &[[isize; 3]] {
        &self.positions
    }

    /// Write a correlation result into alignment `index`.
    pub fn set_shift(&mut self, index: usize, shift: [isize; 3], quality: f64) -> Result<()> {
        let alignment = self
            .alignments
            .get_mut(index)
            .ok_or_else(|| Error::invalid(format!("no alignment {index}")))?;
        let pre = self.positions[alignment.pre];
        let post = self.positions[alignment.post];
        for axis in 0..3 {
            alignment.displacement[axis] = post[axis] - pre[axis] + shift[axis];
        }
        alignment.quality = quality;
        Ok(())
    }

    /// The shift alignment `index` carries: its displacement less the current
    /// positions' difference.
    pub fn shift(&self, index: usize) -> Option<[isize; 3]> {
        let alignment = self.alignments.get(index)?;
        let pre = self.positions[alignment.pre];
        let post = self.positions[alignment.post];
        Some(std::array::from_fn(|axis| {
            alignment.displacement[axis] - (post[axis] - pre[axis])
        }))
    }

    /// The positions [`place`] gives this layout.
    pub fn place(&self, params: &RigidPlacementParams) -> Result<Vec<[isize; 3]>> {
        place(
            &self.positions,
            &self.alignments,
            self.sources.len(),
            params,
        )
    }

    /// [`Self::place`], applied.
    pub fn place_in_place(&mut self, params: &RigidPlacementParams) -> Result<()> {
        self.positions = self.place(params)?;
        Ok(())
    }
}

fn check_sources(sources: &[TileSource], grid_axes: &[usize]) -> Result<usize> {
    if sources.is_empty() {
        return Err(Error::invalid("a layout needs at least one tile"));
    }
    let tile_dim = grid_axes.len();
    if tile_dim == 0 || tile_dim > 3 {
        return Err(Error::invalid(format!(
            "a grid has one to three axes, not {tile_dim}"
        )));
    }
    if grid_axes.iter().any(|&axis| axis > 2) {
        return Err(Error::invalid(format!(
            "grid axes {grid_axes:?} name a volume axis beyond 2"
        )));
    }
    if let Some(bad) = sources
        .iter()
        .position(|source| source.tile_position.len() != tile_dim)
    {
        return Err(Error::invalid(format!(
            "tile {bad} has a grid position of {} axes and the grid has {tile_dim}",
            sources[bad].tile_position.len()
        )));
    }
    Ok(tile_dim)
}

/// Every `(pre, post)` pair where `post` sits one step after `pre` on
/// `grid_axis`.
fn neighbours(sources: &[TileSource], grid_axis: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for (pre, source) in sources.iter().enumerate() {
        let mut neighbour = source.tile_position.clone();
        neighbour[grid_axis] += 1;
        if let Some(post) = sources
            .iter()
            .position(|other| other.tile_position == neighbour)
        {
            pairs.push((pre, post));
        }
    }
    pairs
}

/// Filter, split into components, place each, then lower.
pub fn place(
    positions: &[[isize; 3]],
    alignments: &[Alignment],
    n_sources: usize,
    params: &RigidPlacementParams,
) -> Result<Vec<[isize; 3]>> {
    let kept = alignments
        .iter()
        .filter(|alignment| alignment.quality > params.min_quality)
        .cloned()
        .collect::<Vec<_>>();
    let edges = kept.iter().map(|a| (a.pre, a.post)).collect::<Vec<_>>();
    let component_of = connected_components(&edges, n_sources);
    let n_components = component_of.iter().copied().max().map_or(0, |max| max + 1);

    let mut placed = positions.to_vec();
    for component in 0..n_components {
        let members = (0..n_sources)
            .filter(|source| component_of[*source] == component)
            .collect::<Vec<_>>();
        let component_alignments = kept
            .iter()
            .filter(|alignment| component_of[alignment.pre] == component)
            .cloned()
            .collect::<Vec<_>>();
        if component_alignments.is_empty() {
            continue;
        }
        let fixed = match params.fixed_source {
            FixedSource::Middle => members.len().saturating_sub(1) / 2,
            FixedSource::First => 0,
        };
        let solved = match params.method {
            PlacementMethod::Optimization => {
                let index_of = local_index(&members);
                let edges = component_alignments
                    .iter()
                    .map(|a| Ok((index_of(a.pre)?, index_of(a.post)?)))
                    .collect::<Result<Vec<_>>>()?;
                let displacements = component_alignments
                    .iter()
                    .map(|a| a.displacement.iter().map(|&d| d as f64).collect())
                    .collect::<Vec<Vec<f64>>>();
                fit_positions(members.len(), &edges, &displacements, 3)?
                    .into_iter()
                    .map(|row| std::array::from_fn(|axis| round_even(row[axis])))
                    .collect::<Vec<[isize; 3]>>()
            }
            PlacementMethod::Tree => {
                walk_tree(members.len(), &component_alignments, &members, fixed)?
            }
        };
        let fixed_source = members[fixed];
        for (local, &source) in members.iter().enumerate() {
            for axis in 0..3 {
                placed[source][axis] =
                    solved[local][axis] - solved[fixed][axis] + positions[fixed_source][axis];
            }
        }
    }
    if params.lower_to_origin {
        lower_to_origin(&mut placed);
    }
    Ok(placed)
}

fn local_index(members: &[usize]) -> impl Fn(usize) -> Result<usize> + '_ {
    move |source| {
        members
            .iter()
            .position(|member| *member == source)
            .ok_or_else(|| {
                Error::invalid(format!(
                    "alignment names source {source}, which is not in this component"
                ))
            })
    }
}

/// Subtract the per-axis minimum, so the lower corner is the origin.
pub fn lower_to_origin(positions: &mut [[isize; 3]]) {
    let Some(&first) = positions.first() else {
        return;
    };
    let mut lowest = first;
    for position in positions.iter() {
        for axis in 0..3 {
            lowest[axis] = lowest[axis].min(position[axis]);
        }
    }
    for position in positions.iter_mut() {
        for axis in 0..3 {
            position[axis] -= lowest[axis];
        }
    }
}

/// Breadth-first from `fixed` over the doubled edge set, accumulating
/// displacements. Edges are visited in alignment order, forward before reverse,
/// so on a cycle the path taken is the first one the walk reaches.
fn walk_tree(
    n_members: usize,
    alignments: &[Alignment],
    members: &[usize],
    fixed: usize,
) -> Result<Vec<[isize; 3]>> {
    let index_of = local_index(members);
    let mut adjacency = vec![Vec::new(); n_members];
    for alignment in alignments {
        let pre = index_of(alignment.pre)?;
        let post = index_of(alignment.post)?;
        adjacency[pre].push((post, alignment.displacement));
        adjacency[post].push((pre, alignment.displacement.map(|d| -d)));
    }
    let mut solved = vec![[0isize; 3]; n_members];
    let mut seen = vec![false; n_members];
    seen[fixed] = true;
    let mut queue = VecDeque::from([fixed]);
    while let Some(node) = queue.pop_front() {
        for &(neighbour, displacement) in &adjacency[node] {
            if seen[neighbour] {
                continue;
            }
            seen[neighbour] = true;
            for axis in 0..3 {
                solved[neighbour][axis] = solved[node][axis] + displacement[axis];
            }
            queue.push_back(neighbour);
        }
    }
    Ok(solved)
}

/// The rigid placement's parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigidPlacementParams {
    pub method: PlacementMethod,
    pub fixed_source: FixedSource,
    pub lower_to_origin: bool,
    /// Alignments at or below this quality are dropped. `-inf` keeps every pair
    /// the correlation scored and drops the ones the background check refused.
    pub min_quality: f64,
}

impl Default for RigidPlacementParams {
    fn default() -> Self {
        Self {
            method: PlacementMethod::Optimization,
            fixed_source: FixedSource::Middle,
            lower_to_origin: true,
            min_quality: f64::NEG_INFINITY,
        }
    }
}
