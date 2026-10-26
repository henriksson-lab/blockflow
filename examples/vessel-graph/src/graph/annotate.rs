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
use super::store::Values;
use super::vessel::{Level, VesselGraph};

/// The column of atlas coordinates the lookups read.
pub const ATLAS_COORDINATES: &str = "coordinates_atlas";

/// About how many polyline points are looked up at a time (a quarter of a
/// million): whole polylines, so that each batch's edge minima are its own.
const BATCH_POINTS: usize = 1 << 18;

/// Set `annotation` (and `hemisphere`, `distance_to_surface` where the
/// volumes are given) at the vertex and polyline levels, and the polyline
/// minimum of `distance_to_surface` per edge. The polyline points are looked
/// up a batch of whole polylines at a time, and each column kept in its
/// narrowest store.
pub fn annotate(
    graph: &mut VesselGraph,
    annotator: &Annotator,
    volumes: AtlasVolumes<'_>,
) -> Result<()> {
    let vertices = annotator.columns(volumes, &graph.points(ATLAS_COORDINATES)?)?;
    let polylines = graph.polylines().to_vec();
    let mut annotation = Grown::new();
    let mut hemisphere = Grown::new();
    let mut distance = Grown::new();
    let mut minima = Vec::with_capacity(polylines.len());
    let mut first = 0;
    while first < polylines.len() {
        let start = polylines[first][0] as usize;
        let mut last = first;
        while last + 1 < polylines.len() && polylines[last + 1][1] as usize - start <= BATCH_POINTS
        {
            last += 1;
        }
        let end = polylines[last][1] as usize;
        let points = graph.rows3(Level::Along, ATLAS_COORDINATES, start..end)?;
        let along = annotator.columns(volumes, &points)?;
        drop(points);
        annotation.extend(along.annotation.iter().map(|id| *id as f64))?;
        hemisphere.extend(along.hemisphere.iter().map(|id| *id as f64))?;
        if !vertices.distance_to_surface.is_empty() {
            let ranges: Vec<[usize; 2]> = polylines[first..=last]
                .iter()
                .map(|[a, b]| [*a as usize - start, *b as usize - start])
                .collect();
            minima.extend(polyline_minima(&along.distance_to_surface, &ranges)?);
            distance.extend(along.distance_to_surface.iter().copied())?;
        }
        first = last + 1;
    }
    graph.set_vertex("annotation", ids(&vertices.annotation))?;
    graph.set_values(Level::Along, "annotation", 1, annotation.finish())?;
    if !vertices.hemisphere.is_empty() {
        graph.set_vertex("hemisphere", ids(&vertices.hemisphere))?;
        graph.set_values(Level::Along, "hemisphere", 1, hemisphere.finish())?;
    }
    if !vertices.distance_to_surface.is_empty() {
        graph.set_vertex(
            "distance_to_surface",
            Column::scalar(vertices.distance_to_surface),
        )?;
        graph.set_values(Level::Along, "distance_to_surface", 1, distance.finish())?;
        graph.set_edge("distance_to_surface", Column::scalar(minima))?;
    }
    Ok(())
}

fn ids(values: &[i64]) -> Column {
    Column::scalar(values.iter().map(|id| *id as f64).collect())
}

/// A column grown batch by batch, each batch narrowed on its own and the
/// whole widened to `f64` only if a batch needs it.
struct Grown {
    values: Option<Values>,
}

impl Grown {
    fn new() -> Self {
        Self { values: None }
    }

    fn extend(&mut self, batch: impl Iterator<Item = f64>) -> Result<()> {
        let batch = Values::of(batch.collect());
        self.values = Some(match self.values.take() {
            None => batch,
            Some(values) => values.append(batch),
        });
        Ok(())
    }

    fn finish(self) -> Values {
        self.values.unwrap_or(Values::F64(Vec::new()))
    }
}
