// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The annotated vessel graph: the reduced structure plus named columns at
//! three levels — per vertex, per edge, and per polyline point (`along`).
//!
//! Every setter checks the row count against its level, because a column at
//! the wrong level reads real values off the wrong vessels and nothing later
//! can tell.

use std::collections::BTreeMap;
use std::path::Path;

use blockflow::error::{Error, Result};
use serde_json::{json, Map, Value};

use super::clean::CleanGraph;
use super::columns::{
    along_geometry, onto_edges, through_clean, through_reduce, Aggregate, Column, EdgeFold,
};
use super::component::{largest_component, Component};
use super::raw::RawGraph;
use super::reduce::ReducedGraph;

/// A batch map of `[z, y, x]` points, from the graph's voxels into another
/// frame (an atlas, a coarser lattice). Any
/// `Fn(&[[f64; 3]]) -> Result<Vec<[f64; 3]>>` is one, so a registration's
/// point transform is passed as a closure.
pub trait PointMap {
    fn map_points(&self, points: &[[f64; 3]]) -> Result<Vec<[f64; 3]>>;
}

impl<F> PointMap for F
where
    F: Fn(&[[f64; 3]]) -> Result<Vec<[f64; 3]>>,
{
    fn map_points(&self, points: &[[f64; 3]]) -> Result<Vec<[f64; 3]>> {
        self(points)
    }
}

/// `p * scale + offset` per axis: a resample, or a change between two
/// lattices of one physical frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Affine {
    pub scale: [f64; 3],
    pub offset: [f64; 3],
}

impl Affine {
    pub fn scale(scale: [f64; 3]) -> Self {
        Self {
            scale,
            offset: [0.0; 3],
        }
    }

    /// From the voxels of a lattice at `from` (`physical = translation + index
    /// * scale`) to the voxels of one at `to`.
    pub fn between(from: Lattice, to: Lattice) -> Result<Self> {
        for (what, lattice) in [("source", from), ("target", to)] {
            if lattice
                .scale
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
            {
                return Err(Error::invalid(format!(
                    "the {what} lattice's scale is {:?}; a voxel size is positive.",
                    lattice.scale
                )));
            }
        }
        Ok(Self {
            scale: std::array::from_fn(|axis| from.scale[axis] / to.scale[axis]),
            offset: std::array::from_fn(|axis| {
                (from.translation[axis] - to.translation[axis]) / to.scale[axis]
            }),
        })
    }

    pub fn apply(&self, p: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|axis| p[axis] * self.scale[axis] + self.offset[axis])
    }
}

impl PointMap for Affine {
    fn map_points(&self, points: &[[f64; 3]]) -> Result<Vec<[f64; 3]>> {
        Ok(points.iter().map(|p| self.apply(*p)).collect())
    }
}

/// A voxel lattice in physical space: NGFF's `translation + index * scale`,
/// `[z, y, x]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lattice {
    pub scale: [f64; 3],
    pub translation: [f64; 3],
}

impl Lattice {
    pub fn unit() -> Self {
        Self {
            scale: [1.0; 3],
            translation: [0.0; 3],
        }
    }

    pub fn physical(&self, index: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|axis| self.translation[axis] + index[axis] * self.scale[axis])
    }
}

/// A raw-vertex column to carry onto the vessel graph, and how.
#[derive(Debug, Clone, PartialEq)]
pub struct Carried {
    pub name: String,
    pub column: Column,
    /// How a clique centre folds its members.
    pub clique: Aggregate,
    /// How the column folds onto edges; `None` carries it to vertices and
    /// polyline points only. Only scalar columns fold onto edges.
    pub edge: Option<EdgeFold>,
}

impl Carried {
    /// A measurement folded by its maximum at cliques and along edges.
    pub fn max(name: impl Into<String>, column: Column) -> Self {
        let edge = (column.width() == 1).then_some(EdgeFold::Reduce(Aggregate::Max));
        Self {
            name: name.into(),
            column,
            clique: Aggregate::Max,
            edge,
        }
    }

    /// A 0/1 channel mask: maximum at cliques, a two-thirds vote along edges.
    pub fn binary(name: impl Into<String>, column: Column) -> Self {
        Self {
            name: name.into(),
            column,
            clique: Aggregate::Max,
            edge: Some(EdgeFold::BINARY),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VesselGraph {
    volume: [usize; 3],
    n_vertices: usize,
    edges: Vec<[u32; 2]>,
    polylines: Vec<[u32; 2]>,
    n_points: usize,
    vertex: BTreeMap<String, Column>,
    edge: BTreeMap<String, Column>,
    along: BTreeMap<String, Column>,
}

impl VesselGraph {
    /// The reduced graph with `coordinates` at the vertex and polyline levels
    /// and every carried column at every level its rule reaches.
    pub fn assemble(
        raw: &RawGraph,
        cleaned: &CleanGraph,
        reduced: &ReducedGraph,
        carried: &[Carried],
    ) -> Result<Self> {
        let geometry = reduced.require_geometry()?;
        let mut graph = Self {
            volume: reduced.volume(),
            n_vertices: reduced.n_vertices(),
            edges: reduced.edges().to_vec(),
            polylines: geometry.ranges(),
            n_points: geometry.n_points(),
            vertex: BTreeMap::new(),
            edge: BTreeMap::new(),
            along: BTreeMap::new(),
        };
        graph.set_vertex(
            "coordinates",
            Column::rows3(&as_points(reduced.coordinates())),
        )?;
        graph.set_along(
            "coordinates",
            Column::rows3(&as_points(&reduced.geometry_coordinates(cleaned)?)),
        )?;
        for carry in carried {
            let on_clean = through_clean(raw, cleaned, &carry.column, carry.clique)?;
            graph.set_vertex(&carry.name, through_reduce(reduced, &on_clean)?)?;
            graph.set_along(&carry.name, along_geometry(reduced, &on_clean)?)?;
            if let Some(fold) = carry.edge {
                graph.set_edge(&carry.name, onto_edges(reduced, &on_clean, fold)?)?;
            }
        }
        Ok(graph)
    }

    pub fn volume(&self) -> [usize; 3] {
        self.volume
    }

    pub fn n_vertices(&self) -> usize {
        self.n_vertices
    }

    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn n_points(&self) -> usize {
        self.n_points
    }

    pub fn edges(&self) -> &[[u32; 2]] {
        &self.edges
    }

    /// `[start, end)` of each edge's polyline in the `along` columns.
    pub fn polylines(&self) -> &[[u32; 2]] {
        &self.polylines
    }

    pub fn ranges(&self) -> Vec<[usize; 2]> {
        self.polylines
            .iter()
            .map(|range| [range[0] as usize, range[1] as usize])
            .collect()
    }

    pub fn degrees(&self) -> Vec<usize> {
        let mut degree = vec![0usize; self.n_vertices];
        for [a, b] in &self.edges {
            degree[*a as usize] += 1;
            degree[*b as usize] += 1;
        }
        degree
    }

    pub fn vertex(&self, name: &str) -> Option<&Column> {
        self.vertex.get(name)
    }

    pub fn edge(&self, name: &str) -> Option<&Column> {
        self.edge.get(name)
    }

    pub fn along(&self, name: &str) -> Option<&Column> {
        self.along.get(name)
    }

    pub fn vertex_names(&self) -> Vec<&str> {
        self.vertex.keys().map(String::as_str).collect()
    }

    pub fn edge_names(&self) -> Vec<&str> {
        self.edge.keys().map(String::as_str).collect()
    }

    pub fn along_names(&self) -> Vec<&str> {
        self.along.keys().map(String::as_str).collect()
    }

    pub fn set_vertex(&mut self, name: &str, column: Column) -> Result<()> {
        check_rows("vertex", name, column.len(), self.n_vertices)?;
        self.vertex.insert(name.to_string(), column);
        Ok(())
    }

    pub fn set_edge(&mut self, name: &str, column: Column) -> Result<()> {
        check_rows("edge", name, column.len(), self.edges.len())?;
        self.edge.insert(name.to_string(), column);
        Ok(())
    }

    pub fn set_along(&mut self, name: &str, column: Column) -> Result<()> {
        check_rows("polyline", name, column.len(), self.n_points)?;
        self.along.insert(name.to_string(), column);
        Ok(())
    }

    /// A width-3 vertex column as points.
    pub fn points(&self, name: &str) -> Result<Vec<[f64; 3]>> {
        self.named("vertex", name, self.vertex.get(name))?
            .to_rows3()
    }

    /// A width-3 polyline column as points.
    pub fn along_points(&self, name: &str) -> Result<Vec<[f64; 3]>> {
        self.named("polyline", name, self.along.get(name))?
            .to_rows3()
    }

    fn named<'a>(&self, level: &str, name: &str, column: Option<&'a Column>) -> Result<&'a Column> {
        column.ok_or_else(|| {
            Error::invalid(format!(
                "this graph has no {level} column `{name}`; it has vertex {:?}, edge {:?}, \
                 polyline {:?}.",
                self.vertex_names(),
                self.edge_names(),
                self.along_names()
            ))
        })
    }

    /// The graph restricted to its largest connected component, every column
    /// carried.
    pub fn largest_component(&self) -> Result<Self> {
        self.gathered(&largest_component(self.n_vertices, &self.edges)?)
    }

    /// The graph restricted to `component`, which must be a component of it.
    pub fn gathered(&self, component: &Component) -> Result<Self> {
        if component.source_vertices() != self.n_vertices
            || component.source_edges() != self.edges.len()
        {
            return Err(Error::invalid(format!(
                "a component of a {}-vertex, {}-edge graph applied to one with {} and {}.",
                component.source_vertices(),
                component.source_edges(),
                self.n_vertices,
                self.edges.len()
            )));
        }
        let (polylines, rows) = component.gather_ranges(&self.polylines)?;
        let mut graph = Self {
            volume: self.volume,
            n_vertices: component.n_vertices(),
            edges: component.connectivity().to_vec(),
            polylines,
            n_points: rows.len(),
            vertex: BTreeMap::new(),
            edge: BTreeMap::new(),
            along: BTreeMap::new(),
        };
        for (name, column) in &self.vertex {
            graph.set_vertex(name, column.gather(component.vertices())?)?;
        }
        for (name, column) in &self.edge {
            graph.set_edge(name, column.gather(component.edges())?)?;
        }
        for (name, column) in &self.along {
            graph.set_along(name, column.gather(&rows)?)?;
        }
        Ok(graph)
    }

    /// Map the `coordinates` vertex and polyline columns through `map` into a
    /// new pair of columns called `name`. This is where a registration plugs
    /// in: anything that maps a batch of `[z, y, x]` points is a [`PointMap`].
    pub fn transform_coordinates<M: PointMap + ?Sized>(
        &mut self,
        name: &str,
        map: &M,
    ) -> Result<()> {
        let vertices = self.points("coordinates")?;
        let along = self.along_points("coordinates")?;
        let mapped_vertices = map.map_points(&vertices)?;
        let mapped_along = map.map_points(&along)?;
        for (level, given, mapped) in [
            ("vertex", vertices.len(), mapped_vertices.len()),
            ("polyline", along.len(), mapped_along.len()),
        ] {
            if given != mapped {
                return Err(Error::invalid(format!(
                    "a point map returned {mapped} points for {given} {level} points."
                )));
            }
        }
        self.set_vertex(name, Column::rows3(&mapped_vertices))?;
        self.set_along(name, Column::rows3(&mapped_along))
    }

    /// Each edge's length along its polyline, in voxels and in physical units
    /// (`spacing` per voxel, `[z, y, x]`): the sum of the polyline's segment
    /// lengths.
    pub fn edge_lengths(&self, spacing: [f64; 3]) -> Result<(Vec<f64>, Vec<f64>)> {
        let points = self.along_points("coordinates")?;
        let mut voxels = Vec::with_capacity(self.edges.len());
        let mut physical = Vec::with_capacity(self.edges.len());
        for [start, end] in self.ranges() {
            let (mut plain, mut scaled) = (0.0f64, 0.0f64);
            for pair in points[start..end].windows(2) {
                let step: [f64; 3] = std::array::from_fn(|axis| pair[1][axis] - pair[0][axis]);
                plain += step.iter().map(|d| d * d).sum::<f64>().sqrt();
                scaled += (0..3)
                    .map(|axis| (step[axis] * spacing[axis]).powi(2))
                    .sum::<f64>()
                    .sqrt();
            }
            voxels.push(plain);
            physical.push(scaled);
        }
        Ok((voxels, physical))
    }

    /// Set the `length` (voxels) and `length_units` (physical) edge columns
    /// from [`Self::edge_lengths`].
    pub fn set_lengths(&mut self, spacing: [f64; 3]) -> Result<()> {
        let (voxels, physical) = self.edge_lengths(spacing)?;
        self.set_edge("length", Column::scalar(voxels))?;
        self.set_edge("length_units", Column::scalar(physical))
    }

    /// `coordinates_atlas` as the plain resample from this graph's volume to
    /// `atlas_shape`: `p * atlas_shape / volume` per axis. A registration
    /// would replace this with [`Self::transform_coordinates`].
    pub fn rescale_to_atlas(&mut self, atlas_shape: [usize; 3]) -> Result<()> {
        let factor = resample_factor(self.volume, atlas_shape)?;
        self.transform_coordinates("coordinates_atlas", &Affine::scale(factor))
    }

    /// Give every radius column (name containing `radii` or `radius`, not
    /// already ending in `_atlas`) an `_atlas` twin in atlas voxel units, at
    /// every level it exists at. A name containing `units` is in µm; any other
    /// is in source voxels. Negative values (the not-found radius) are kept.
    pub fn scale_radii(&mut self, spacing: [f64; 3], atlas_shape: [usize; 3]) -> Result<()> {
        let factors = Factors::new(spacing, self.volume, atlas_shape)?;
        self.scale_radii_with(&factors)
    }

    /// [`Self::scale_radii`] with the conversions given, as a registration
    /// derives them from the images' frames rather than from shapes.
    pub fn scale_radii_with(&mut self, factors: &Factors) -> Result<()> {
        let selected: Vec<String> = self
            .vertex
            .keys()
            .filter(|name| {
                (name.contains("radii") || name.contains("radius")) && !name.ends_with("_atlas")
            })
            .cloned()
            .collect();
        for name in selected {
            let twin = format!("{name}_atlas");
            let micrometres = name.contains("units");
            for level in [&mut self.vertex, &mut self.edge, &mut self.along] {
                if let Some(column) = level.get(&name) {
                    let scaled = factors.scale(column, micrometres)?;
                    level.insert(twin.clone(), scaled);
                }
            }
        }
        Ok(())
    }

    /// The graph as JSON: the structure, then every column as
    /// `{"width": w, "values": [...]}` by level and name.
    pub fn to_json(&self) -> Value {
        let level = |columns: &BTreeMap<String, Column>| -> Value {
            let mut map = Map::new();
            for (name, column) in columns {
                map.insert(
                    name.clone(),
                    json!({"width": column.width(), "values": column.values()}),
                );
            }
            Value::Object(map)
        };
        json!({
            "format": "vessel-graph",
            "version": 1,
            "volume": self.volume,
            "n_vertices": self.n_vertices,
            "n_points": self.n_points,
            "edges": self.edges,
            "polylines": self.polylines,
            "vertex": level(&self.vertex),
            "edge": level(&self.edge),
            "along": level(&self.along),
        })
    }

    /// The inverse of [`Self::to_json`]. `null` values (non-finite numbers)
    /// read back as NaN.
    pub fn from_json(value: &Value) -> Result<Self> {
        let bad = |what: &str| Error::invalid(format!("a vessel-graph JSON document {what}"));
        if value.get("format").and_then(Value::as_str) != Some("vessel-graph") {
            return Err(bad("must have \"format\": \"vessel-graph\""));
        }
        let usize_of = |value: &Value| value.as_u64().map(|v| v as usize);
        let volume: Vec<usize> = value
            .get("volume")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("has no volume"))?
            .iter()
            .map(usize_of)
            .collect::<Option<_>>()
            .ok_or_else(|| bad("has a non-integer volume"))?;
        let volume: [usize; 3] = volume
            .try_into()
            .map_err(|_| bad("has a volume that is not 3-D"))?;
        let pairs = |key: &str| -> Result<Vec<[u32; 2]>> {
            value
                .get(key)
                .and_then(Value::as_array)
                .ok_or_else(|| bad(&format!("has no {key}")))?
                .iter()
                .map(|pair| {
                    let pair = pair.as_array().filter(|pair| pair.len() == 2);
                    let at = |index: usize| pair.and_then(|p| p[index].as_u64()).map(|v| v as u32);
                    match (at(0), at(1)) {
                        (Some(a), Some(b)) => Ok([a, b]),
                        _ => Err(bad(&format!("has a malformed entry in {key}"))),
                    }
                })
                .collect()
        };
        let mut graph = Self {
            volume,
            n_vertices: value
                .get("n_vertices")
                .and_then(usize_of)
                .ok_or_else(|| bad("has no n_vertices"))?,
            edges: pairs("edges")?,
            polylines: pairs("polylines")?,
            n_points: value
                .get("n_points")
                .and_then(usize_of)
                .ok_or_else(|| bad("has no n_points"))?,
            vertex: BTreeMap::new(),
            edge: BTreeMap::new(),
            along: BTreeMap::new(),
        };
        for key in ["vertex", "edge", "along"] {
            let Some(columns) = value.get(key).and_then(Value::as_object) else {
                return Err(bad(&format!("has no {key} columns")));
            };
            for (name, column) in columns {
                let width = column
                    .get("width")
                    .and_then(usize_of)
                    .ok_or_else(|| bad(&format!("column {name} has no width")))?;
                let values: Vec<f64> = column
                    .get("values")
                    .and_then(Value::as_array)
                    .ok_or_else(|| bad(&format!("column {name} has no values")))?
                    .iter()
                    .map(|value| value.as_f64().unwrap_or(f64::NAN))
                    .collect();
                let column = Column::new(width, values)?;
                match key {
                    "vertex" => graph.set_vertex(name, column)?,
                    "edge" => graph.set_edge(name, column)?,
                    _ => graph.set_along(name, column)?,
                }
            }
        }
        Ok(graph)
    }

    pub fn write_json(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let text = serde_json::to_string(&self.to_json())
            .map_err(|error| Error::invalid(format!("serialising the vessel graph: {error}")))?;
        std::fs::write(path, text)
            .map_err(|error| Error::invalid(format!("writing {}: {error}", path.display())))
    }

    pub fn read_json(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|error| Error::invalid(format!("reading {}: {error}", path.display())))?;
        let value: Value = serde_json::from_str(&text)
            .map_err(|error| Error::invalid(format!("parsing {}: {error}", path.display())))?;
        Self::from_json(&value)
    }
}

fn check_rows(level: &str, name: &str, rows: usize, wanted: usize) -> Result<()> {
    if rows != wanted {
        return Err(Error::invalid(format!(
            "`{name}` was set as a {level} column of {rows} rows on a graph with {wanted} of those."
        )));
    }
    Ok(())
}

fn as_points(coordinates: &[[u32; 3]]) -> Vec<[f64; 3]> {
    coordinates.iter().map(|at| at.map(f64::from)).collect()
}

/// `atlas_shape / volume` per axis.
pub fn resample_factor(volume: [usize; 3], atlas_shape: [usize; 3]) -> Result<[f64; 3]> {
    let mut factor = [0.0f64; 3];
    for axis in 0..3 {
        if volume[axis] == 0 || atlas_shape[axis] == 0 {
            return Err(Error::invalid(format!(
                "a resample between {volume:?} and {atlas_shape:?} has a zero extent on axis \
                 {axis}."
            )));
        }
        factor[axis] = atlas_shape[axis] as f64 / volume[axis] as f64;
    }
    Ok(factor)
}

/// The unit conversions from source voxels and µm to atlas voxels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Factors {
    /// Atlas voxels per source voxel, per axis.
    pub resample: [f64; 3],
    /// µm per atlas voxel, per axis.
    pub atlas_spacing: [f64; 3],
    pub spacing_mean: f64,
    pub atlas_spacing_mean: f64,
}

impl Factors {
    pub fn new(spacing: [f64; 3], volume: [usize; 3], atlas_shape: [usize; 3]) -> Result<Self> {
        for (axis, value) in spacing.iter().enumerate() {
            if !value.is_finite() || *value <= 0.0 {
                return Err(Error::invalid(format!(
                    "the spacing on axis {axis} is {value}; µm per voxel is positive and finite."
                )));
            }
        }
        let resample = resample_factor(volume, atlas_shape)?;
        let mut atlas_spacing = [0.0f64; 3];
        for axis in 0..3 {
            atlas_spacing[axis] = spacing[axis] / resample[axis];
        }
        let mean = |v: [f64; 3]| (v[0] + v[1] + v[2]) / 3.0;
        Ok(Self {
            resample,
            atlas_spacing,
            spacing_mean: mean(spacing),
            atlas_spacing_mean: mean(atlas_spacing),
        })
    }

    /// A scalar radius scales by the mean physical size of a voxel; a
    /// per-axis one axis by axis.
    pub fn scale(&self, column: &Column, micrometres: bool) -> Result<Column> {
        let keep = |value: f64, scaled: f64| if value < 0.0 { value } else { scaled };
        match column.width() {
            1 => {
                let factor = if micrometres {
                    1.0 / self.atlas_spacing_mean
                } else {
                    self.spacing_mean / self.atlas_spacing_mean
                };
                Ok(Column::scalar(
                    column
                        .values()
                        .iter()
                        .map(|v| keep(*v, v * factor))
                        .collect(),
                ))
            }
            3 => {
                let factor: [f64; 3] = if micrometres {
                    self.atlas_spacing.map(|spacing| 1.0 / spacing)
                } else {
                    self.resample
                };
                let rows: Vec<[f64; 3]> = column
                    .to_rows3()?
                    .into_iter()
                    .map(|row| [0, 1, 2].map(|axis| keep(row[axis], row[axis] * factor[axis])))
                    .collect();
                Ok(Column::rows3(&rows))
            }
            width => Err(Error::invalid(format!(
                "a radius column of width {width}; radii are scalar or per axis."
            ))),
        }
    }
}
