// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Branch density: graph vertices splatted into a dense volume by
//! `blockflow::ops::voxelize`, a ball of fixed radius per point.
//!
//! "Branch" is **every vertex of the reduced graph** by default, degree-1
//! endpoints included; [`Selection::Degrees`] restricts it to an inclusive
//! degree range (`Degrees { low: 3, high: 3 }` is exactly degree three, which
//! excludes higher-order junctions). A selected vertex can carry its own
//! radius as its weight, read from the same vertex.
//!
//! Coordinates are rounded to the nearest voxel. A vertex outside the density
//! volume is dropped: `ops::voxelize` requires every point inside the volume,
//! so the part of its ball that would reach in is lost, and the count of
//! dropped vertices is reported.

use blockflow::error::{Error, Result};
use blockflow::geometry::BlockGrid;
use blockflow::ops::voxelize::{ball, voxelize_into};
use blockflow::points::Point;
use blockflow::region::Region;
use blockflow::voxels::Voxels;
use ndarray::Array3;

use super::vessel::VesselGraph;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Selection {
    Every,
    /// Degrees in `low..=high`.
    Degrees {
        low: usize,
        high: usize,
    },
}

impl Selection {
    pub fn degree(exactly: usize) -> Self {
        Self::Degrees {
            low: exactly,
            high: exactly,
        }
    }

    pub fn admits(self, degree: usize) -> bool {
        match self {
            Self::Every => true,
            Self::Degrees { low, high } => (low..=high).contains(&degree),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BranchParams {
    pub selection: Selection,
    /// The vertex column weighting each point, or `None` to count.
    pub weight: Option<String>,
    /// The vertex column of positions in the density volume's voxels.
    pub coordinates: String,
    /// The ball radius, in density voxels per axis.
    pub radius: [usize; 3],
}

impl Default for BranchParams {
    fn default() -> Self {
        Self {
            selection: Selection::Every,
            weight: None,
            coordinates: "coordinates_atlas".to_string(),
            radius: [15, 15, 15],
        }
    }
}

/// The points that reach the density, and how many fell outside it.
#[derive(Debug, Clone, PartialEq)]
pub struct BranchPoints {
    pub points: Vec<Point>,
    pub outside: usize,
}

/// Select, weight and place the vertices.
pub fn branch_points(
    coordinates: &[[f64; 3]],
    degrees: &[usize],
    weights: Option<&[f64]>,
    selection: Selection,
    shape: [usize; 3],
) -> Result<BranchPoints> {
    if degrees.len() != coordinates.len() {
        return Err(Error::invalid(format!(
            "{} vertices and {} degrees.",
            coordinates.len(),
            degrees.len()
        )));
    }
    if let Some(weights) = weights {
        if weights.len() != coordinates.len() {
            return Err(Error::invalid(format!(
                "{} vertices and {} weights.",
                coordinates.len(),
                weights.len()
            )));
        }
    }
    let mut points = Vec::new();
    let mut outside = 0;
    'vertices: for (vertex, at) in coordinates.iter().enumerate() {
        if !selection.admits(degrees[vertex]) {
            continue;
        }
        let weight = weights.map_or(1.0, |weights| weights[vertex]);
        if !weight.is_finite() || at.iter().any(|value| !value.is_finite()) {
            return Err(Error::invalid(format!(
                "vertex {vertex} at {at:?} with weight {weight} is not finite."
            )));
        }
        let mut voxel = [0usize; 3];
        for axis in 0..3 {
            let index = at[axis].round();
            if index < 0.0 || index >= shape[axis] as f64 {
                outside += 1;
                continue 'vertices;
            }
            voxel[axis] = index as usize;
        }
        points.push(Point::weighted(voxel, weight));
    }
    Ok(BranchPoints { points, outside })
}

/// Render points into an `f64` density of `shape`, one ball per point.
pub fn density(points: &[Point], shape: [usize; 3], radius: [usize; 3]) -> Result<Voxels> {
    let grid = BlockGrid::whole(shape)?;
    let mut out = Array3::<f64>::zeros((shape[0], shape[1], shape[2]));
    voxelize_into(
        &[([0, 0, 0], points.to_vec())],
        &grid,
        &ball(radius),
        &Region::whole(&shape),
        out.view_mut(),
    )?;
    Ok(Voxels::from(out))
}

/// The branch density of a graph: [`branch_points`] then [`density`].
/// Returns the density and the number of vertices outside it.
pub fn branch_density(
    graph: &VesselGraph,
    shape: [usize; 3],
    params: &BranchParams,
) -> Result<(Voxels, usize)> {
    let coordinates = graph.points(&params.coordinates)?;
    let weights = match &params.weight {
        None => None,
        Some(name) => {
            let column = graph.vertex(name).ok_or_else(|| {
                Error::invalid(format!(
                    "the branch density is weighted by `{name}` and the graph has no such vertex \
                     column."
                ))
            })?;
            if column.width() != 1 {
                return Err(Error::invalid(format!(
                    "the weight column `{name}` has width {}; a weight is one number.",
                    column.width()
                )));
            }
            Some(column.values())
        }
    };
    let placed = branch_points(
        &coordinates,
        &graph.degrees(),
        weights,
        params.selection,
        shape,
    )?;
    Ok((
        density(&placed.points, shape, params.radius)?,
        placed.outside,
    ))
}
