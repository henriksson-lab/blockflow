// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The annotated vessel graph: the reduced structure plus named columns at
//! three levels — per vertex, per edge, and per polyline point (`along`).
//!
//! Every setter checks the row count against its level, because a column at
//! the wrong level reads real values off the wrong vessels and nothing later
//! can tell.
//!
//! **How the columns are held.** A column measured on the raw graph (the
//! coordinates, the radii, the channel signals) is not copied to the vertex
//! and polyline levels: those levels hold a cleaned-vertex id per row, and
//! the column is read through it — off the raw graph's own array for a
//! surviving voxel, off the few values folded at a clique centre otherwise
//! ([`Carried`]). The radii are held as the walk's stop per raw vertex, one
//! `u32` for all four radius arrays. A column set on the graph is held in the
//! narrowest store that gives back every value bit for bit (`f32`, `i32`,
//! flags), and the `_atlas` radius twins and the lattice changes are read
//! through the column they scale. Every column still reads as `f64`, the same
//! numbers in the same order, so the tables and the JSON are what they were;
//! [`VesselGraph::vertex`], [`VesselGraph::edge`] and [`VesselGraph::along`]
//! hand out a column as `f64` rows, made on first asking and kept.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::{Arc, OnceLock};

use blockflow::budget::Lease;
use blockflow::error::{Error, Result};
use serde_json::{json, Map, Value};

use super::clean::CleanGraph;
use super::columns::{Aggregate, Column, EdgeFold, RawColumn};
use super::component::{largest_component, Component};
use super::raw::RawGraph;
use super::reduce::ReducedGraph;
use super::store::Values;

/// How many points a batch of a [`PointMap`] holds: a quarter of a million.
const POINT_BATCH: usize = 1 << 18;

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
    pub column: RawColumn,
    /// How a clique centre folds its members.
    pub clique: Aggregate,
    /// How the column folds onto edges; `None` carries it to vertices and
    /// polyline points only. Only scalar columns fold onto edges.
    pub edge: Option<EdgeFold>,
}

impl Carried {
    /// A measurement folded by its maximum at cliques and along edges.
    pub fn max(name: impl Into<String>, column: impl Into<RawColumn>) -> Self {
        let column = column.into();
        let edge = (column.width() == 1).then_some(EdgeFold::Reduce(Aggregate::Max));
        Self {
            name: name.into(),
            column,
            clique: Aggregate::Max,
            edge,
        }
    }

    /// A 0/1 channel mask: maximum at cliques, a two-thirds vote along edges.
    pub fn binary(name: impl Into<String>, column: impl Into<RawColumn>) -> Self {
        Self {
            name: name.into(),
            column: column.into(),
            clique: Aggregate::Max,
            edge: Some(EdgeFold::BINARY),
        }
    }
}

/// What the carried columns of a graph read, per cleaned vertex: a surviving
/// raw vertex's row, or the values a clique centre folded from its members.
/// Shared by every level and by the graphs gathered from one assembly.
#[derive(Debug)]
struct Source {
    /// Raw id of each cleaned vertex below its length; the rest are centres.
    survivors: Arc<Vec<u32>>,
    columns: Vec<SourceColumn>,
}

#[derive(Debug)]
struct SourceColumn {
    width: usize,
    raw: RawSource,
    /// `width` values per kept centre, in cleaned-vertex order.
    centres: Vec<f64>,
}

#[derive(Debug)]
enum RawSource {
    Column(RawColumn),
    /// The raw graph's coordinates, read as `f64`.
    Coordinates(Arc<Vec<[u32; 3]>>),
}

impl Source {
    #[inline]
    fn value(&self, column: usize, clean: u32, axis: usize) -> f64 {
        let source = &self.columns[column];
        match self.survivors.get(clean as usize) {
            Some(&raw) => match &source.raw {
                RawSource::Column(values) => values.value(raw as usize, axis),
                RawSource::Coordinates(coordinates) => f64::from(coordinates[raw as usize][axis]),
            },
            None => source.centres[(clean as usize - self.survivors.len()) * source.width + axis],
        }
    }

    fn bytes(&self) -> usize {
        let mut bytes = self.survivors.len() * 4;
        for column in &self.columns {
            bytes += column.centres.len() * 8;
            bytes += match &column.raw {
                RawSource::Column(values) => values.bytes(),
                RawSource::Coordinates(coordinates) => coordinates.len() * 12,
            };
        }
        bytes
    }
}

/// A column as the graph holds it.
#[derive(Debug)]
enum Entry {
    /// Its own values, `width` a row.
    Dense { width: usize, values: Values },
    /// Column `column` of the [`Source`], read through the level's cleaned
    /// vertex ids.
    Carried { column: usize, width: usize },
    /// `base` times `factors[axis]`, negative values (the not-found radius)
    /// kept as they are: an `_atlas` radius twin.
    Scaled { base: Arc<Entry>, factors: [f64; 3] },
    /// `base` through an affine map per axis: a change of lattice.
    Mapped { base: Arc<Entry>, affine: Affine },
}

/// What an entry of one level reads through.
#[derive(Clone, Copy)]
struct Reader<'a> {
    source: Option<&'a Source>,
    map: &'a [u32],
}

impl Entry {
    fn width(&self) -> usize {
        match self {
            Self::Dense { width, .. } | Self::Carried { width, .. } => *width,
            Self::Scaled { base, .. } | Self::Mapped { base, .. } => base.width(),
        }
    }

    #[inline]
    fn value(&self, reader: Reader<'_>, row: usize, axis: usize) -> f64 {
        match self {
            Self::Dense { width, values } => values.get(row * width + axis),
            Self::Carried { column, .. } => reader
                .source
                .expect("a carried column has a source")
                .value(*column, reader.map[row], axis),
            Self::Scaled { base, factors } => {
                let value = base.value(reader, row, axis);
                if value < 0.0 {
                    value
                } else {
                    value * factors[axis]
                }
            }
            Self::Mapped { base, affine } => {
                base.value(reader, row, axis) * affine.scale[axis] + affine.offset[axis]
            }
        }
    }

    /// The bytes it holds of its own.
    fn bytes(&self) -> usize {
        match self {
            Self::Dense { values, .. } => values.bytes(),
            _ => 0,
        }
    }

    /// The same column restricted to `rows` of its level: the dense values
    /// gathered, the rest read through the gathered level (an entry shared
    /// by two columns stays shared, through `memo`).
    fn gathered(
        self: &Arc<Self>,
        rows: &[u32],
        memo: &mut HashMap<*const Entry, Arc<Entry>>,
    ) -> Arc<Entry> {
        if let Some(done) = memo.get(&Arc::as_ptr(self)) {
            return Arc::clone(done);
        }
        let gathered = match self.as_ref() {
            Self::Dense { width, values } => Arc::new(Self::Dense {
                width: *width,
                values: values.gather(*width, rows),
            }),
            Self::Carried { .. } => Arc::clone(self),
            Self::Scaled { base, factors } => Arc::new(Self::Scaled {
                base: base.gathered(rows, memo),
                factors: *factors,
            }),
            Self::Mapped { base, affine } => Arc::new(Self::Mapped {
                base: base.gathered(rows, memo),
                affine: *affine,
            }),
        };
        memo.insert(Arc::as_ptr(self), Arc::clone(&gathered));
        gathered
    }
}

/// A column and its `f64` rows, made the first time they are asked for.
#[derive(Debug, Clone)]
struct Slot {
    entry: Arc<Entry>,
    rows: Arc<OnceLock<Column>>,
}

impl Slot {
    fn new(entry: Entry) -> Self {
        Self::of(Arc::new(entry))
    }

    fn of(entry: Arc<Entry>) -> Self {
        Self {
            entry,
            rows: Arc::new(OnceLock::new()),
        }
    }
}

/// The three levels a column lives at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Level {
    Vertex,
    Edge,
    Along,
}

impl Level {
    fn name(self) -> &'static str {
        match self {
            Self::Vertex => "vertex",
            Self::Edge => "edge",
            Self::Along => "polyline",
        }
    }
}

/// One column of one level, read row by row as `f64`.
#[derive(Clone, Copy)]
pub(crate) struct ColumnRead<'a> {
    entry: &'a Entry,
    reader: Reader<'a>,
    rows: usize,
}

impl ColumnRead<'_> {
    pub(crate) fn width(&self) -> usize {
        self.entry.width()
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    pub(crate) fn value(&self, row: usize, axis: usize) -> f64 {
        self.entry.value(self.reader, row, axis)
    }

    /// The whole column as `f64` rows.
    pub(crate) fn to_column(self) -> Column {
        let width = self.width();
        let mut values = Vec::with_capacity(self.rows * width);
        for row in 0..self.rows {
            for axis in 0..width {
                values.push(self.value(row, axis));
            }
        }
        Column::new(width, values).expect("a column has a width")
    }
}

#[derive(Debug, Clone)]
pub struct VesselGraph {
    volume: [usize; 3],
    n_vertices: usize,
    edges: Vec<[u32; 2]>,
    polylines: Vec<[u32; 2]>,
    n_points: usize,
    /// What the carried columns read; `None` for a graph read from JSON.
    source: Option<Arc<Source>>,
    /// The cleaned vertex id of each vertex and of each polyline point
    /// (empty without a source).
    vertex_map: Vec<u32>,
    point_map: Vec<u32>,
    vertex: BTreeMap<String, Slot>,
    edge: BTreeMap<String, Slot>,
    along: BTreeMap<String, Slot>,
    /// The memory budget's lease on what the graph holds, if one was taken.
    lease: Option<Arc<Lease>>,
}

impl VesselGraph {
    /// Hold `lease` (of the process's memory budget) for as long as this
    /// graph, or a graph gathered from it, lives.
    pub fn hold(&mut self, lease: Lease) {
        self.lease = Some(Arc::new(lease));
    }

    /// The reduced graph with `coordinates` at the vertex and polyline levels
    /// and every carried column at every level its rule reaches: a surviving
    /// vertex keeps its row, a clique centre folds its members
    /// ([`super::columns::through_clean`]), a reduced vertex is a subset, a
    /// polyline point a gather and an edge a fold over its polyline
    /// ([`super::columns::onto_edges`]).
    pub fn assemble(
        raw: &RawGraph,
        cleaned: &CleanGraph,
        reduced: &ReducedGraph,
        carried: &[Carried],
    ) -> Result<Self> {
        let geometry = reduced.require_geometry()?;
        if reduced.source_vertices() != cleaned.n_vertices() {
            return Err(Error::invalid(format!(
                "a property column of {} rows was handed a reduction of a {}-vertex graph.",
                cleaned.n_vertices(),
                reduced.source_vertices()
            )));
        }
        let centres: Vec<&[u32]> = cleaned
            .centre_cliques()
            .iter()
            .map(|clique| cleaned.cliques()[*clique as usize].members())
            .collect();
        let mut columns = vec![SourceColumn {
            width: 3,
            raw: RawSource::Coordinates(raw.shared_coordinates()),
            centres: cleaned
                .centre_cliques()
                .iter()
                .flat_map(|clique| cleaned.cliques()[*clique as usize].centre())
                .map(f64::from)
                .collect(),
        }];
        for carry in carried {
            let column = &carry.column;
            if column.len() != raw.n_vertices() {
                return Err(Error::invalid(format!(
                    "a property column of {} rows was handed a raw graph of {} vertices.",
                    column.len(),
                    raw.n_vertices()
                )));
            }
            let width = column.width();
            let mut folded = Vec::with_capacity(centres.len() * width);
            for members in &centres {
                for axis in 0..width {
                    folded.push(
                        carry
                            .clique
                            .fold(members.iter().map(|m| column.value(*m as usize, axis)))
                            .ok_or_else(|| Error::invalid("a clique with no member"))?,
                    );
                }
            }
            columns.push(SourceColumn {
                width,
                raw: RawSource::Column(column.clone()),
                centres: folded,
            });
        }
        let source = Arc::new(Source {
            survivors: cleaned.shared_survivors(),
            columns,
        });
        let mut graph = Self {
            volume: reduced.volume(),
            n_vertices: reduced.n_vertices(),
            edges: reduced.edges().to_vec(),
            polylines: geometry.ranges(),
            n_points: geometry.n_points(),
            source: Some(Arc::clone(&source)),
            vertex_map: reduced.sources().to_vec(),
            point_map: geometry.points().to_vec(),
            vertex: BTreeMap::new(),
            edge: BTreeMap::new(),
            along: BTreeMap::new(),
            lease: None,
        };
        let carry = |column: usize| {
            Arc::new(Entry::Carried {
                column,
                width: source.columns[column].width,
            })
        };
        graph.insert(Level::Vertex, "coordinates", Slot::of(carry(0)));
        graph.insert(Level::Along, "coordinates", Slot::of(carry(0)));
        for (index, carry_spec) in carried.iter().enumerate() {
            let entry = carry(index + 1);
            graph.insert(
                Level::Vertex,
                &carry_spec.name,
                Slot::of(Arc::clone(&entry)),
            );
            graph.insert(Level::Along, &carry_spec.name, Slot::of(entry));
            if let Some(fold) = carry_spec.edge {
                let values = onto_edges(&source, index + 1, fold, geometry)?;
                graph.set_edge(&carry_spec.name, Column::scalar(values))?;
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

    fn level(&self, level: Level) -> &BTreeMap<String, Slot> {
        match level {
            Level::Vertex => &self.vertex,
            Level::Edge => &self.edge,
            Level::Along => &self.along,
        }
    }

    fn rows_of(&self, level: Level) -> usize {
        match level {
            Level::Vertex => self.n_vertices,
            Level::Edge => self.edges.len(),
            Level::Along => self.n_points,
        }
    }

    fn reader(&self, level: Level) -> Reader<'_> {
        Reader {
            source: self.source.as_deref(),
            map: match level {
                Level::Vertex => &self.vertex_map,
                Level::Edge => &[],
                Level::Along => &self.point_map,
            },
        }
    }

    fn read<'a>(&'a self, level: Level, slot: &'a Slot) -> ColumnRead<'a> {
        ColumnRead {
            entry: &slot.entry,
            reader: self.reader(level),
            rows: self.rows_of(level),
        }
    }

    /// One column of a level, read row by row.
    pub(crate) fn column(&self, level: Level, name: &str) -> Option<ColumnRead<'_>> {
        self.level(level)
            .get(name)
            .map(|slot| self.read(level, slot))
    }

    /// Every column of a level, by name.
    pub(crate) fn columns(&self, level: Level) -> Vec<(&str, ColumnRead<'_>)> {
        self.level(level)
            .iter()
            .map(|(name, slot)| (name.as_str(), self.read(level, slot)))
            .collect()
    }

    fn rows(&self, level: Level, name: &str) -> Option<&Column> {
        self.level(level)
            .get(name)
            .map(|slot| slot.rows.get_or_init(|| self.read(level, slot).to_column()))
    }

    /// A vertex column as `f64` rows (made on first asking, then kept).
    pub fn vertex(&self, name: &str) -> Option<&Column> {
        self.rows(Level::Vertex, name)
    }

    /// An edge column as `f64` rows (made on first asking, then kept).
    pub fn edge(&self, name: &str) -> Option<&Column> {
        self.rows(Level::Edge, name)
    }

    /// A polyline column as `f64` rows (made on first asking, then kept).
    pub fn along(&self, name: &str) -> Option<&Column> {
        self.rows(Level::Along, name)
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

    fn insert(&mut self, level: Level, name: &str, slot: Slot) {
        let map = match level {
            Level::Vertex => &mut self.vertex,
            Level::Edge => &mut self.edge,
            Level::Along => &mut self.along,
        };
        map.insert(name.to_string(), slot);
    }

    /// Set `values` (`width` a row) as a column of `level`.
    pub(crate) fn set_values(
        &mut self,
        level: Level,
        name: &str,
        width: usize,
        values: Values,
    ) -> Result<()> {
        if width == 0 || !values.len().is_multiple_of(width) {
            return Err(Error::invalid(format!(
                "{} values do not divide into rows of {width}.",
                values.len()
            )));
        }
        check_rows(
            level.name(),
            name,
            values.len() / width,
            self.rows_of(level),
        )?;
        self.insert(level, name, Slot::new(Entry::Dense { width, values }));
        Ok(())
    }

    fn set(&mut self, level: Level, name: &str, column: Column) -> Result<()> {
        let width = column.width();
        check_rows(level.name(), name, column.len(), self.rows_of(level))?;
        self.set_values(level, name, width, Values::of(column.into_values()))
    }

    pub fn set_vertex(&mut self, name: &str, column: Column) -> Result<()> {
        self.set(Level::Vertex, name, column)
    }

    pub fn set_edge(&mut self, name: &str, column: Column) -> Result<()> {
        self.set(Level::Edge, name, column)
    }

    pub fn set_along(&mut self, name: &str, column: Column) -> Result<()> {
        self.set(Level::Along, name, column)
    }

    /// Rows `range` of a width-3 column of `level` as points.
    pub(crate) fn rows3(
        &self,
        level: Level,
        name: &str,
        range: std::ops::Range<usize>,
    ) -> Result<Vec<[f64; 3]>> {
        let column = self.named(level, name)?;
        if column.width() != 3 {
            return Err(Error::invalid(format!(
                "a column of width {} is not a list of 3-vectors.",
                column.width()
            )));
        }
        Ok(range
            .map(|row| std::array::from_fn(|axis| column.value(row, axis)))
            .collect())
    }

    /// A width-3 vertex column as points.
    pub fn points(&self, name: &str) -> Result<Vec<[f64; 3]>> {
        self.rows3(Level::Vertex, name, 0..self.n_vertices)
    }

    /// A width-3 polyline column as points.
    pub fn along_points(&self, name: &str) -> Result<Vec<[f64; 3]>> {
        self.rows3(Level::Along, name, 0..self.n_points)
    }

    pub(crate) fn named(&self, level: Level, name: &str) -> Result<ColumnRead<'_>> {
        self.column(level, name).ok_or_else(|| {
            Error::invalid(format!(
                "this graph has no {} column `{name}`; it has vertex {:?}, edge {:?}, \
                 polyline {:?}.",
                level.name(),
                self.vertex_names(),
                self.edge_names(),
                self.along_names()
            ))
        })
    }

    /// The bytes the graph holds: its structure, its maps and columns, and
    /// the source its carried columns read.
    pub fn bytes(&self) -> usize {
        let mut bytes = self.edges.len() * 8
            + self.polylines.len() * 8
            + (self.vertex_map.len() + self.point_map.len()) * 4
            + self.source.as_ref().map_or(0, |source| source.bytes());
        for level in [&self.vertex, &self.edge, &self.along] {
            for slot in level.values() {
                bytes += slot.entry.bytes();
                bytes += slot.rows.get().map_or(0, |rows| rows.values().len() * 8);
            }
        }
        bytes
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
        let gather = |map: &[u32], rows: &[u32]| -> Vec<u32> {
            match map.is_empty() {
                true => Vec::new(),
                false => rows.iter().map(|row| map[*row as usize]).collect(),
            }
        };
        let mut graph = Self {
            volume: self.volume,
            n_vertices: component.n_vertices(),
            edges: component.connectivity().to_vec(),
            polylines,
            n_points: rows.len(),
            source: self.source.clone(),
            vertex_map: gather(&self.vertex_map, component.vertices()),
            point_map: gather(&self.point_map, &rows),
            vertex: BTreeMap::new(),
            edge: BTreeMap::new(),
            along: BTreeMap::new(),
            lease: self.lease.clone(),
        };
        let mut memo = HashMap::new();
        for (level, kept) in [
            (Level::Vertex, component.vertices()),
            (Level::Edge, component.edges()),
            (Level::Along, rows.as_slice()),
        ] {
            memo.clear();
            for (name, slot) in self.level(level) {
                let entry = slot.entry.gathered(kept, &mut memo);
                graph.insert(level, name, Slot::of(entry));
            }
        }
        Ok(graph)
    }

    /// Map the `coordinates` vertex and polyline columns through `map` into a
    /// new pair of columns called `name`. This is where a registration plugs
    /// in: anything that maps a batch of `[z, y, x]` points is a [`PointMap`].
    /// The polyline points are mapped a batch of about a million at a time,
    /// so a map must map each point on its own (a lattice change, a
    /// registration's point transform).
    pub fn transform_coordinates<M: PointMap + ?Sized>(
        &mut self,
        name: &str,
        map: &M,
    ) -> Result<()> {
        let vertices = self.points("coordinates")?;
        let mapped_vertices = map.map_points(&vertices)?;
        check_mapped("vertex", vertices.len(), mapped_vertices.len())?;
        drop(vertices);
        let mut along = Vec::with_capacity(self.n_points * 3);
        for start in (0..self.n_points).step_by(POINT_BATCH) {
            let end = (start + POINT_BATCH).min(self.n_points);
            let points = self.rows3(Level::Along, "coordinates", start..end)?;
            let mapped = map.map_points(&points)?;
            check_mapped("polyline", points.len(), mapped.len())?;
            along.extend(mapped.iter().flatten());
        }
        self.set_vertex(name, Column::rows3(&mapped_vertices))?;
        self.set_values(Level::Along, name, 3, Values::of(along))
    }

    /// [`Self::transform_coordinates`] through an affine map per axis,
    /// read through `coordinates` rather than held: the same numbers.
    pub fn transform_coordinates_affine(&mut self, name: &str, affine: Affine) -> Result<()> {
        for level in [Level::Vertex, Level::Along] {
            let base = self.named(level, "coordinates")?;
            if base.width() != 3 {
                return Err(Error::invalid(format!(
                    "a column of width {} is not a list of 3-vectors.",
                    base.width()
                )));
            }
            let base = Arc::clone(&self.level(level)["coordinates"].entry);
            self.insert(level, name, Slot::new(Entry::Mapped { base, affine }));
        }
        Ok(())
    }

    /// Each edge's length along its polyline, in voxels and in physical units
    /// (`spacing` per voxel, `[z, y, x]`): the sum of the polyline's segment
    /// lengths.
    pub fn edge_lengths(&self, spacing: [f64; 3]) -> Result<(Vec<f64>, Vec<f64>)> {
        let coordinates = self.named(Level::Along, "coordinates")?;
        if coordinates.width() != 3 {
            return Err(Error::invalid(format!(
                "a column of width {} is not a list of 3-vectors.",
                coordinates.width()
            )));
        }
        let point =
            |row: usize| -> [f64; 3] { std::array::from_fn(|axis| coordinates.value(row, axis)) };
        let mut voxels = Vec::with_capacity(self.edges.len());
        let mut physical = Vec::with_capacity(self.edges.len());
        for [start, end] in self.ranges() {
            let (mut plain, mut scaled) = (0.0f64, 0.0f64);
            for row in start + 1..end.max(start + 1) {
                let pair = [point(row - 1), point(row)];
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
        self.transform_coordinates_affine("coordinates_atlas", Affine::scale(factor))
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
    /// derives them from the images' frames rather than from shapes. The
    /// twins are read through the columns they scale.
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
            for level in [Level::Vertex, Level::Edge, Level::Along] {
                if let Some(slot) = self.level(level).get(&name) {
                    let base = Arc::clone(&slot.entry);
                    let factors = factors.per_axis(base.width(), micrometres)?;
                    self.insert(level, &twin, Slot::new(Entry::Scaled { base, factors }));
                }
            }
        }
        Ok(())
    }

    /// The graph as JSON: the structure, then every column as
    /// `{"width": w, "values": [...]}` by level and name.
    pub fn to_json(&self) -> Value {
        let level = |level: Level| -> Value {
            let mut map = Map::new();
            for (name, column) in self.columns(level) {
                let column = column.to_column();
                map.insert(
                    name.to_string(),
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
            "vertex": level(Level::Vertex),
            "edge": level(Level::Edge),
            "along": level(Level::Along),
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
            source: None,
            vertex_map: Vec::new(),
            point_map: Vec::new(),
            vertex: BTreeMap::new(),
            edge: BTreeMap::new(),
            along: BTreeMap::new(),
            lease: None,
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

    /// [`Self::to_json`] written to `path`, streamed: the same text, without
    /// the document tree or a column's `f64` rows in memory.
    pub fn write_json(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let file = std::fs::File::create(path)
            .map_err(|error| Error::invalid(format!("writing {}: {error}", path.display())))?;
        let mut out = std::io::BufWriter::new(file);
        serde_json::to_writer(&mut out, &JsonGraph(self))
            .map_err(|error| Error::invalid(format!("serialising the vessel graph: {error}")))?;
        std::io::Write::flush(&mut out)
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

/// Two graphs are equal when their structure and every column's values are,
/// however the columns are held.
impl PartialEq for VesselGraph {
    fn eq(&self, other: &Self) -> bool {
        if self.volume != other.volume
            || self.n_vertices != other.n_vertices
            || self.n_points != other.n_points
            || self.edges != other.edges
            || self.polylines != other.polylines
        {
            return false;
        }
        [Level::Vertex, Level::Edge, Level::Along]
            .into_iter()
            .all(|level| {
                let (mine, theirs) = (self.columns(level), other.columns(level));
                mine.len() == theirs.len()
                    && mine.iter().zip(&theirs).all(|((a, x), (b, y))| {
                        a == b
                            && x.width() == y.width()
                            && x.rows() == y.rows()
                            && (0..x.rows()).all(|row| {
                                (0..x.width()).all(|axis| x.value(row, axis) == y.value(row, axis))
                            })
                    })
            })
    }
}

/// A scalar source column folded onto every reduced edge over its whole
/// polyline: [`super::columns::onto_edges`] on the column as the cleaned
/// graph would hold it.
fn onto_edges(
    source: &Source,
    column: usize,
    fold: EdgeFold,
    geometry: &super::reduce::EdgeGeometry,
) -> Result<Vec<f64>> {
    let width = source.columns[column].width;
    if width != 1 {
        return Err(Error::invalid(format!(
            "a column of width {width} was handed to the vertex-to-edge fold, which folds \
             scalars; carry a wide column along the geometry instead."
        )));
    }
    if let EdgeFold::Vote { .. } = fold {
        let cleaned = source.survivors.len() + source.columns[column].centres.len();
        for clean in 0..cleaned as u32 {
            let value = source.value(column, clean, 0);
            if value != 0.0 && value != 1.0 {
                return Err(Error::invalid(format!(
                    "{value} is neither 0 nor 1, so this column is not a set of flags."
                )));
            }
        }
    }
    let mut values = Vec::with_capacity(geometry.n_edges());
    for edge in 0..geometry.n_edges() {
        let path = geometry.path(edge);
        let along = path.iter().map(|point| source.value(column, *point, 0));
        values.push(match fold {
            EdgeFold::Reduce(aggregate) => aggregate
                .fold(along)
                .ok_or_else(|| Error::invalid(format!("edge {edge} has an empty polyline")))?,
            EdgeFold::Vote { fraction } => {
                let mean = along.sum::<f64>() / path.len() as f64;
                f64::from(u8::from(mean >= fraction))
            }
        });
    }
    Ok(values)
}

fn check_rows(level: &str, name: &str, rows: usize, wanted: usize) -> Result<()> {
    if rows != wanted {
        return Err(Error::invalid(format!(
            "`{name}` was set as a {level} column of {rows} rows on a graph with {wanted} of those."
        )));
    }
    Ok(())
}

fn check_mapped(level: &str, given: usize, mapped: usize) -> Result<()> {
    if given != mapped {
        return Err(Error::invalid(format!(
            "a point map returned {mapped} points for {given} {level} points."
        )));
    }
    Ok(())
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

    /// The factor per axis a radius column of `width` scales by: a scalar
    /// radius by the mean physical size of a voxel (the same factor on every
    /// axis), a per-axis one axis by axis.
    pub fn per_axis(&self, width: usize, micrometres: bool) -> Result<[f64; 3]> {
        match width {
            1 => Ok([if micrometres {
                1.0 / self.atlas_spacing_mean
            } else {
                self.spacing_mean / self.atlas_spacing_mean
            }; 3]),
            3 => Ok(if micrometres {
                self.atlas_spacing.map(|spacing| 1.0 / spacing)
            } else {
                self.resample
            }),
            width => Err(Error::invalid(format!(
                "a radius column of width {width}; radii are scalar or per axis."
            ))),
        }
    }

    /// A radius column scaled by [`Self::per_axis`], negative values (the
    /// not-found radius) kept.
    pub fn scale(&self, column: &Column, micrometres: bool) -> Result<Column> {
        let factors = self.per_axis(column.width(), micrometres)?;
        let width = column.width();
        let values = column
            .values()
            .iter()
            .enumerate()
            .map(|(index, value)| {
                if *value < 0.0 {
                    *value
                } else {
                    value * factors[index % width]
                }
            })
            .collect();
        Column::new(width, values)
    }
}

/// [`VesselGraph::to_json`]'s document as a serialiser: the same keys in the
/// same order, each value serialised as `json!` would have built it (serde's
/// `f64` serialiser writes a non-finite number as `null`, as `json!` makes it),
/// a column's values read as they are written.
struct JsonGraph<'a>(&'a VesselGraph);

/// One level's columns, `{name: {"width": w, "values": [...]}}` by name.
struct JsonLevel<'a>(&'a VesselGraph, Level);

struct JsonColumn<'a>(ColumnRead<'a>);

struct JsonValues<'a>(ColumnRead<'a>);

impl serde::Serialize for JsonGraph<'_> {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let graph = self.0;
        let mut map = serializer.serialize_map(Some(10))?;
        map.serialize_entry("format", "vessel-graph")?;
        map.serialize_entry("version", &1)?;
        map.serialize_entry("volume", &graph.volume)?;
        map.serialize_entry("n_vertices", &graph.n_vertices)?;
        map.serialize_entry("n_points", &graph.n_points)?;
        map.serialize_entry("edges", &graph.edges)?;
        map.serialize_entry("polylines", &graph.polylines)?;
        map.serialize_entry("vertex", &JsonLevel(graph, Level::Vertex))?;
        map.serialize_entry("edge", &JsonLevel(graph, Level::Edge))?;
        map.serialize_entry("along", &JsonLevel(graph, Level::Along))?;
        map.end()
    }
}

impl serde::Serialize for JsonLevel<'_> {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let columns = self.0.columns(self.1);
        let mut map = serializer.serialize_map(Some(columns.len()))?;
        for (name, column) in columns {
            map.serialize_entry(name, &JsonColumn(column))?;
        }
        map.end()
    }
}

impl serde::Serialize for JsonColumn<'_> {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(Some(2))?;
        map.serialize_entry("width", &self.0.width())?;
        map.serialize_entry("values", &JsonValues(self.0))?;
        map.end()
    }
}

impl serde::Serialize for JsonValues<'_> {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let column = self.0;
        let width = column.width();
        let mut seq = serializer.serialize_seq(Some(column.rows() * width))?;
        for row in 0..column.rows() {
            for axis in 0..width {
                seq.serialize_element(&column.value(row, axis))?;
            }
        }
        seq.end()
    }
}
