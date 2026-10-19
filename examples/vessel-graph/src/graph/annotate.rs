// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Atlas columns on the vessel graph.
//!
//! Region, hemisphere and distance to the brain surface are looked up by
//! `atlas_regions::Annotator` at the `coordinates_atlas` of every vertex and
//! every polyline point. The edge gets one column, `distance_to_surface`, and
//! it is the **minimum over the polyline** (`atlas_regions::polyline_minima`),
//! not over the two endpoints: a vessel can dip towards the surface between
//! its branch points, and that dip is what the artery trace's
//! `distance_to_surface_min` is about.

use atlas_regions::{polyline_minima, Annotator, AtlasVolumes};
use blockflow::error::Result;

use super::columns::Column;
use super::vessel::VesselGraph;

/// The column of atlas coordinates the lookups read.
pub const ATLAS_COORDINATES: &str = "coordinates_atlas";

/// Set `annotation` (and `hemisphere`, `distance_to_surface` where the
/// volumes are given) at the vertex and polyline levels, and the polyline
/// minimum of `distance_to_surface` per edge.
pub fn annotate(
    graph: &mut VesselGraph,
    annotator: &Annotator,
    volumes: AtlasVolumes<'_>,
) -> Result<()> {
    let vertices = annotator.columns(volumes, &graph.points(ATLAS_COORDINATES)?)?;
    let along = annotator.columns(volumes, &graph.along_points(ATLAS_COORDINATES)?)?;
    graph.set_vertex("annotation", ids(&vertices.annotation))?;
    graph.set_along("annotation", ids(&along.annotation))?;
    if !vertices.hemisphere.is_empty() {
        graph.set_vertex("hemisphere", ids(&vertices.hemisphere))?;
        graph.set_along("hemisphere", ids(&along.hemisphere))?;
    }
    if !vertices.distance_to_surface.is_empty() {
        let minima = polyline_minima(&along.distance_to_surface, &graph.ranges())?;
        graph.set_vertex(
            "distance_to_surface",
            Column::scalar(vertices.distance_to_surface),
        )?;
        graph.set_along(
            "distance_to_surface",
            Column::scalar(along.distance_to_surface),
        )?;
        graph.set_edge("distance_to_surface", Column::scalar(minima))?;
    }
    Ok(())
}

fn ids(values: &[i64]) -> Column {
    Column::scalar(values.iter().map(|id| *id as f64).collect())
}
