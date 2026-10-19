// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The per-slice placement: one position per tile per slice from the measured
//! tracks and the rigid positions.
//!
//! ```text
//!  per alignment:  resolve unaligned slices -> smooth over the valid intervals
//!  per slice:      keep the usable alignments -> connected components
//!                  -> least squares within each component
//!  once:           link components of adjacent slices that share a tile
//!                  -> least squares for one shift per component, over the
//!                     whole wobble axis
//!  once:           smooth the positions -> lower to the origin
//! ```
//!
//! The per-slice fit only fixes positions within a component up to a
//! translation; the whole-axis fit chooses those translations so a tile moves
//! as little as possible from one slice to the next:
//!
//! ```text
//!   E = sum_s sum_{k in slice s and s+1} (x_{s,k} + t_s - x_{s+1,k} - t_{s+1})^2
//! ```
//!
//! Its normal equations are a grounded Laplacian over the slices, so the last
//! slice depends on the first: this step is resident along the wobble axis.
//!
//! Those systems often have exact half-integer solutions (a displacement that
//! steps by one between two slices is split evenly), which a solver returns as
//! `k/2 + eps` with `eps` a residual of 1e-13. Rounding that directly would let
//! the residual decide; the solution is snapped to the nearest half first and
//! then rounded half to even, so the answer is a function of the data.

use blockflow::error::{Error, Result};

use crate::solve::{connected_components, fit_positions, round_even, solve};
use crate::wobbly::{
    runs_of, smooth_displacements, ResolutionReport, SmoothParams, SourceStatus, Track, Window,
};

/// How close to a half a solved shift must be to count as one.
pub const TIE_TOLERANCE: f64 = 1e-6;

/// One tile: its rigid position, and how many slices of the wobble axis it
/// spans.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceSlab {
    pub position: Vec<isize>,
    pub height: usize,
}

/// One pairwise alignment and its track.
#[derive(Debug, Clone, PartialEq)]
pub struct AlignmentSlot {
    pub pre: usize,
    pub post: usize,
    pub track: Track,
}

/// The rigid layout plus one track per alignment.
#[derive(Debug, Clone, PartialEq)]
pub struct WobblyLayout {
    pub sources: Vec<SourceSlab>,
    pub alignments: Vec<AlignmentSlot>,
    /// The wobble axis.
    pub axis: usize,
}

impl WobblyLayout {
    /// The lowest wobble-axis coordinate of any tile.
    pub fn lower(&self) -> isize {
        self.sources
            .iter()
            .map(|source| source.position[self.axis])
            .min()
            .unwrap_or(0)
    }

    /// The number of slices the placement runs over.
    pub fn n_slices(&self) -> usize {
        let upper = self
            .sources
            .iter()
            .map(|source| source.position[self.axis] + source.height as isize)
            .max()
            .unwrap_or(0);
        (upper - self.lower()).max(0) as usize
    }
}

fn drop_axis(position: &[isize], axis: usize) -> Vec<isize> {
    position
        .iter()
        .enumerate()
        .filter_map(|(dim, &value)| (dim != axis).then_some(value))
        .collect()
}

fn insert_axis(plane: &[isize], axis: usize, value: isize) -> Vec<isize> {
    let mut full = Vec::with_capacity(plane.len() + 1);
    full.extend_from_slice(&plane[..axis]);
    full.push(value);
    full.extend_from_slice(&plane[axis..]);
    full
}

/// The placement's parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct PlaceParams {
    /// Alignment slices at or below this quality are not used.
    pub min_quality: f64,
    /// The smoothing of each track's displacements, within its valid intervals.
    pub smooth: SmoothParams,
    /// The smoothing of each tile's placed positions along the wobble axis.
    pub smooth_optimized: Option<SmoothParams>,
    /// Interpolate the positions of a tile's isolated slices between their
    /// placed neighbours and fuse them, instead of leaving them out.
    pub fix_isolated: bool,
    pub lower_to_origin: bool,
}

impl Default for PlaceParams {
    fn default() -> Self {
        Self {
            min_quality: f64::NEG_INFINITY,
            smooth: SmoothParams {
                window: Some(Window::Bartlett),
                window_length: 100,
                binary: None,
            },
            smooth_optimized: Some(SmoothParams {
                window: Some(Window::Bartlett),
                window_length: 20,
                binary: Some(10),
            }),
            fix_isolated: true,
            lower_to_origin: true,
        }
    }
}

/// One position per tile per slice, and the status that says which mean
/// anything.
#[derive(Debug, Clone, PartialEq)]
pub struct PlacedLayout {
    /// `positions[source][slice]`, full dimension, with the wobble-axis
    /// coordinate at `axis`.
    pub positions: Vec<Vec<Vec<isize>>>,
    /// `status[source][slice]`, one of [`SourceStatus`]'s values.
    pub status: Vec<Vec<i32>>,
    pub report: PlacementReport,
}

/// What the placement did.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PlacementReport {
    pub n_slices: usize,
    /// Components of more than one tile, summed over the slices.
    pub clusters: usize,
    /// Connected groups of clusters over the whole axis.
    pub cluster_components: usize,
    /// `(source, slice)` pairs no usable alignment reached.
    pub isolated: usize,
    /// Unknowns of the whole-axis fit, summed over dimensions.
    pub optimiser_unknowns: usize,
    /// How many of them were exact half-integer ties.
    pub tie_shifts: usize,
    /// One per alignment.
    pub resolution: Vec<ResolutionReport>,
}

/// Place every tile in every slice.
pub fn place_layout(layout: &WobblyLayout, params: &PlaceParams) -> Result<PlacedLayout> {
    let axis = layout.axis;
    let n_sources = layout.sources.len();
    let n_slices = layout.n_slices();
    let ndim = layout.sources.first().map_or(0, |s| s.position.len());
    if n_sources == 0 || n_slices == 0 {
        return Err(Error::invalid(
            "an empty wobbly layout has nothing to place",
        ));
    }
    if axis >= ndim || layout.sources.iter().any(|s| s.position.len() != ndim) {
        return Err(Error::invalid(format!(
            "wobble axis {axis} and the tiles' positions disagree about the dimension"
        )));
    }
    let lower = layout.lower();
    let plane_ndim = ndim - 1;
    let plane_positions = layout
        .sources
        .iter()
        .map(|source| drop_axis(&source.position, axis))
        .collect::<Vec<_>>();
    let pairs = layout
        .alignments
        .iter()
        .map(|alignment| (alignment.pre, alignment.post))
        .collect::<Vec<_>>();
    if let Some(&(pre, post)) = pairs
        .iter()
        .find(|&&(pre, post)| pre >= n_sources || post >= n_sources)
    {
        return Err(Error::invalid(format!(
            "alignment ({pre}, {post}) names a tile outside a layout of {n_sources}"
        )));
    }

    // Per slice and alignment: the smoothed displacement if usable.
    let n_alignments = layout.alignments.len();
    let mut usable = vec![vec![None::<Vec<isize>>; n_alignments]; n_slices];
    let mut resolution = Vec::with_capacity(n_alignments);
    for (index, alignment) in layout.alignments.iter().enumerate() {
        let mut track = alignment.track.clone();
        resolution.push(track.resolve_rejected());
        let valids = track.valids(params.min_quality);
        let smoothed = smooth_displacements(&track.displacements, &valids, params.smooth)?;
        let start = layout.sources[alignment.pre].position[axis]
            .max(layout.sources[alignment.post].position[axis])
            - lower;
        for (local, displacement) in smoothed.into_iter().enumerate() {
            let slice = start + local as isize;
            if valids[local] && (0..n_slices as isize).contains(&slice) {
                usable[slice as usize][index] = Some(displacement);
            }
        }
    }

    // Per slice: components and the fit within each.
    let mut per_slice_positions = Vec::with_capacity(n_slices);
    let mut per_slice_components = Vec::with_capacity(n_slices);
    for row in &usable {
        let (positions, components) = place_slice(row, &plane_positions, &pairs)?;
        per_slice_positions.push(positions);
        per_slice_components.push(components);
    }

    // A tile in a component of its own at a slice is either alone by geometry
    // (no alignment partner spans that slice) or isolated (a partner does, but
    // no usable alignment reached it). Either way it drops out of the
    // whole-axis fit and its position there is filled from its own neighbouring
    // slices; an isolated slice is also left out of the fusion unless
    // `fix_isolated` is set.
    let mut status = layout
        .sources
        .iter()
        .map(|source| vec![SourceStatus::VALID; source.height])
        .collect::<Vec<_>>();
    let mut alone = layout
        .sources
        .iter()
        .map(|source| vec![false; source.height])
        .collect::<Vec<_>>();
    let spans = |source: usize, coordinate: isize| {
        let slab = &layout.sources[source];
        (slab.position[axis]..slab.position[axis] + slab.height as isize).contains(&coordinate)
    };
    let mut isolated = 0;
    for (slice, components) in per_slice_components.iter().enumerate() {
        let coordinate = slice as isize + lower;
        for component in components.iter().filter(|c| c.len() == 1) {
            let source = component[0];
            if !spans(source, coordinate) {
                continue;
            }
            let local = (coordinate - layout.sources[source].position[axis]) as usize;
            alone[source][local] = true;
            let partnered = pairs.iter().any(|&(pre, post)| {
                (pre == source && spans(post, coordinate))
                    || (post == source && spans(pre, coordinate))
            });
            if partnered {
                status[source][local] = SourceStatus::ISOLATED;
                isolated += 1;
            }
        }
    }
    let components = per_slice_components
        .into_iter()
        .map(|slice| {
            slice
                .into_iter()
                .filter(|c| c.len() > 1)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let optimised = optimize_slice_positions(&per_slice_positions, &components, plane_ndim)?;

    // Each tile's own span of the whole-axis answer, filled where it was alone
    // and then smoothed within the span.
    let mut per_source = Vec::with_capacity(n_sources);
    for (index, source) in layout.sources.iter().enumerate() {
        let start = (source.position[axis] - lower) as usize;
        let mut span = (start..start + source.height)
            .map(|slice| optimised.positions[slice][index].clone())
            .collect::<Vec<_>>();
        let anchored = alone[index].iter().map(|&a| !a).collect::<Vec<_>>();
        fill_runs(&mut span, &alone[index], &anchored);
        if let Some(smooth) = params.smooth_optimized {
            span = smooth_displacements(&span, &vec![true; span.len()], smooth)?;
        }
        per_source.push(span);
    }
    if params.lower_to_origin {
        let mut minimum = vec![isize::MAX; plane_ndim];
        for position in per_source.iter().flatten() {
            for (dim, &value) in position.iter().enumerate() {
                minimum[dim] = minimum[dim].min(value);
            }
        }
        for position in per_source.iter_mut().flatten() {
            for (dim, value) in position.iter_mut().enumerate() {
                *value -= minimum[dim];
            }
        }
    }
    if params.fix_isolated {
        for (index, source_status) in status.iter_mut().enumerate() {
            let isolated = source_status
                .iter()
                .map(|&s| s == SourceStatus::ISOLATED)
                .collect::<Vec<_>>();
            let usable = source_status
                .iter()
                .map(|&s| s >= SourceStatus::VALID)
                .collect::<Vec<_>>();
            for (slot, filled) in fill_runs(&mut per_source[index], &isolated, &usable)
                .into_iter()
                .enumerate()
            {
                if filled {
                    source_status[slot] = SourceStatus::FIXED;
                }
            }
        }
    }
    let positions = layout
        .sources
        .iter()
        .zip(&per_source)
        .map(|(source, span)| {
            span.iter()
                .enumerate()
                .map(|(local, plane)| {
                    insert_axis(plane, axis, source.position[axis] + local as isize)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    Ok(PlacedLayout {
        positions,
        status,
        report: PlacementReport {
            n_slices,
            clusters: components.iter().map(Vec::len).sum(),
            cluster_components: optimised.components,
            isolated,
            optimiser_unknowns: optimised.unknowns,
            tie_shifts: optimised.ties,
            resolution,
        },
    })
}

/// One slice's positions per source, and the components they were fitted in.
type SlicePlacement = (Vec<Vec<isize>>, Vec<Vec<usize>>);

fn place_slice(
    usable: &[Option<Vec<isize>>],
    positions: &[Vec<isize>],
    pairs: &[(usize, usize)],
) -> Result<SlicePlacement> {
    let mut positions = positions.to_vec();
    let (valid_pairs, valid_displacements): (Vec<_>, Vec<_>) = pairs
        .iter()
        .zip(usable)
        .filter_map(|(&pair, displacement)| displacement.clone().map(|d| (pair, d)))
        .unzip();
    let labels = connected_components(&valid_pairs, positions.len());
    let n_components = labels.iter().copied().max().map_or(0, |top| top + 1);
    let mut component_ids = Vec::with_capacity(n_components);
    for component in 0..n_components {
        let ids = (0..labels.len())
            .filter(|&source| labels[source] == component)
            .collect::<Vec<_>>();
        let (component_pairs, component_displacements): (Vec<_>, Vec<_>) = valid_pairs
            .iter()
            .zip(&valid_displacements)
            .filter(|((pre, _), _)| labels[*pre] == component)
            .map(|(&pair, displacement)| (pair, displacement.clone()))
            .unzip();
        place_slice_component(
            &mut positions,
            &ids,
            &component_pairs,
            &component_displacements,
        )?;
        component_ids.push(ids);
    }
    Ok((positions, component_ids))
}

/// Fit a component's positions to its displacements, holding its lowest tile
/// where it was.
fn place_slice_component(
    positions: &mut [Vec<isize>],
    nodes: &[usize],
    pairs: &[(usize, usize)],
    displacements: &[Vec<isize>],
) -> Result<()> {
    if pairs.is_empty() {
        return Ok(());
    }
    let slot_of = |node: usize| {
        nodes
            .iter()
            .position(|&other| other == node)
            .ok_or_else(|| Error::invalid("a component lost one of its tiles"))
    };
    let edges = pairs
        .iter()
        .map(|&(pre, post)| Ok((slot_of(pre)?, slot_of(post)?)))
        .collect::<Result<Vec<_>>>()?;
    let ndim = positions[nodes[0]].len();
    let wanted = displacements
        .iter()
        .map(|d| d.iter().map(|&v| v as f64).collect())
        .collect::<Vec<Vec<f64>>>();
    let fitted = fit_positions(nodes.len(), &edges, &wanted, ndim)?;
    let anchor = positions[nodes[0]].clone();
    for (slot, &source) in nodes.iter().enumerate() {
        for dim in 0..ndim {
            positions[source][dim] = round_even(fitted[slot][dim]) + anchor[dim];
        }
    }
    Ok(())
}

struct Optimised {
    positions: Vec<Vec<Vec<isize>>>,
    components: usize,
    unknowns: usize,
    ties: usize,
}

/// One shift per cluster over the whole wobble axis.
fn optimize_slice_positions(
    positions: &[Vec<Vec<isize>>],
    components: &[Vec<Vec<usize>>],
    ndim: usize,
) -> Result<Optimised> {
    let mut positions = positions.to_vec();
    let n_slices = components.len();

    // Cluster ids, numbered slice by slice.
    let mut cluster_base = Vec::with_capacity(n_slices + 1);
    let mut total = 0usize;
    for slice in components {
        cluster_base.push(total);
        total += slice.len();
    }
    cluster_base.push(total);
    let slice_of = |cluster: usize| {
        let slice = cluster_base.partition_point(|&base| base <= cluster) - 1;
        (slice, cluster - cluster_base[slice])
    };
    // Clusters in adjacent slices are joined when they share a tile.
    let mut edges = Vec::new();
    for slice in 1..n_slices {
        for (i, before) in components[slice - 1].iter().enumerate() {
            for (j, after) in components[slice].iter().enumerate() {
                if before.iter().any(|tile| after.contains(tile)) {
                    edges.push((cluster_base[slice - 1] + i, cluster_base[slice] + j));
                }
            }
        }
    }
    let labels = connected_components(&edges, total);
    let n_labels = labels.iter().copied().max().map_or(0, |top| top + 1);
    let mut groups = vec![Vec::new(); n_labels];
    for (cluster, &label) in labels.iter().enumerate() {
        groups[label].push(cluster);
    }
    groups.retain(|group| group.len() > 1);

    let (mut unknowns_total, mut ties_total) = (0, 0);
    for group in &groups {
        // The group's first cluster is grounded; the others are unknowns.
        let unknown_of = |cluster: usize| group[1..].iter().position(|&c| c == cluster);
        let n_unknowns = group.len() - 1;
        let mut matrix = vec![vec![vec![0.0; n_unknowns]; n_unknowns]; ndim];
        let mut rhs = vec![vec![0.0; n_unknowns]; ndim];
        for &(a, b) in &edges {
            if labels[a] != labels[group[0]] {
                continue;
            }
            let ((slice_a, index_a), (slice_b, index_b)) = (slice_of(a), slice_of(b));
            for &tile in &components[slice_a][index_a] {
                if !components[slice_b][index_b].contains(&tile) {
                    continue;
                }
                // (x_a + t_a) - (x_b + t_b) should be zero for a shared tile.
                for dim in 0..ndim {
                    let delta =
                        (positions[slice_a][tile][dim] - positions[slice_b][tile][dim]) as f64;
                    let (ua, ub) = (unknown_of(a), unknown_of(b));
                    if let Some(ua) = ua {
                        matrix[dim][ua][ua] += 1.0;
                        rhs[dim][ua] -= delta;
                        if let Some(ub) = ub {
                            matrix[dim][ua][ub] -= 1.0;
                        }
                    }
                    if let Some(ub) = ub {
                        matrix[dim][ub][ub] += 1.0;
                        rhs[dim][ub] += delta;
                        if let Some(ua) = ua {
                            matrix[dim][ub][ua] -= 1.0;
                        }
                    }
                }
            }
        }
        let mut shifts = vec![vec![0isize; ndim]; n_unknowns];
        for dim in 0..ndim {
            let solution = solve(matrix[dim].clone(), rhs[dim].clone())?;
            unknowns_total += solution.len();
            for (unknown, value) in solution.into_iter().enumerate() {
                let (rounded, tie) = round_snapped(value);
                ties_total += usize::from(tie);
                shifts[unknown][dim] = rounded;
            }
        }
        for (unknown, &cluster) in group[1..].iter().enumerate() {
            let (slice, index) = slice_of(cluster);
            for &tile in &components[slice][index] {
                for dim in 0..ndim {
                    positions[slice][tile][dim] += shifts[unknown][dim];
                }
            }
        }
    }
    Ok(Optimised {
        positions,
        components: groups.len(),
        unknowns: unknowns_total,
        ties: ties_total,
    })
}

/// Snap to the nearest half within [`TIE_TOLERANCE`], then round half to
/// even. Also says whether the value was a tie.
pub fn round_snapped(value: f64) -> (isize, bool) {
    let doubled = (value * 2.0).round();
    let tie = (value * 2.0 - doubled).abs() < 2.0 * TIE_TOLERANCE && doubled.rem_euclid(2.0) == 1.0;
    let snapped = if (value * 2.0 - doubled).abs() < 2.0 * TIE_TOLERANCE {
        doubled / 2.0
    } else {
        value
    };
    (round_even(snapped), tie)
}

/// Fill each run of `fill` slices from the `usable` slices either side:
/// interpolate (rounded) between two, hold one, leave the run when there is
/// neither. Returns which slices were filled.
fn fill_runs(positions: &mut [Vec<isize>], fill: &[bool], usable: &[bool]) -> Vec<bool> {
    let n = fill.len();
    let mut filled = vec![false; n];
    let marks = fill.iter().map(|&f| i32::from(f)).collect::<Vec<_>>();
    for (start, end) in runs_of(&marks, 1) {
        let left = (start > 0 && usable[start - 1]).then(|| start - 1);
        let right = (end < n && usable[end]).then_some(end);
        let values: Vec<Vec<isize>> = match (left, right) {
            (None, None) => continue,
            (Some(from), None) | (None, Some(from)) => vec![positions[from].clone(); end - start],
            (Some(left), Some(right)) => {
                let (low, high) = (positions[left].clone(), positions[right].clone());
                let span = (end - start + 1) as f64;
                (1..=end - start)
                    .map(|step| {
                        low.iter()
                            .zip(&high)
                            .map(|(&low, &high)| {
                                round_even((high - low) as f64 / span * step as f64 + low as f64)
                            })
                            .collect()
                    })
                    .collect()
            }
        };
        for (slot, value) in (start..end).zip(values) {
            positions[slot] = value;
            filled[slot] = true;
        }
    }
    filled
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tie_is_snapped_and_rounded_to_even() {
        assert_eq!(round_snapped(0.5 + 1e-13), (0, true));
        assert_eq!(round_snapped(1.5 - 1e-13), (2, true));
        assert_eq!(round_snapped(-0.5 - 1e-13), (0, true));
        assert_eq!(round_snapped(0.7), (1, false));
        assert_eq!(round_snapped(2.0), (2, false));
    }
}
