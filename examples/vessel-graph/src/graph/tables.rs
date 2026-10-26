// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The vessel graph as three NGFF object tables under the image's `tables/`,
//! the layout a viewer discovers:
//!
//! * `<name>-vertices` — one row per vertex, identity `vertex_id`;
//! * `<name>-edges` — one row per edge, identity `edge_id`, with `source` and
//!   `target` vertex ids;
//! * `<name>-edge-points` — one row per polyline point, identity `point_id`,
//!   with the `edge_id` it belongs to and its `index` along the edge.
//!
//! An object table holds one scalar per row and column, so a polyline cannot
//! be a column of the edge table; it is the points table instead, and the edge
//! row says where its points are (`point_start`, `point_count`: the point ids
//! `point_start .. point_start + point_count`, in order from `source` to
//! `target`).
//!
//! Every table is spatially indexed by `z, y, x` in level-0 voxels of the
//! image (`coordinate_space` `../../`): a vertex at its position, an edge at
//! its polyline's middle point, a point at itself. Physical positions are
//! `z_um, y_um, x_um` (`translation + index * scale`). Every graph column is
//! carried: a scalar as itself, a per-axis one as `<name>_z`, `<name>_y`,
//! `<name>_x`. Region and hemisphere ids are `int64`, flags `uint8`, the rest
//! `float64`.
//!
//! The tables declare themselves in `extensions["vessel_graph"]`:
//! `{"version": 1, "role": "vertices" | "edges" | "edge_points", "vertices":
//! "../<name>-vertices", "edges": ..., "edge_points": ...}`, so a viewer can
//! find the other two from any one. A graph in atlas space through a
//! registration also records it there as `"atlas"` (the registration and
//! atlas paths, the atlas shape, voxel size and window, and the sample frame).

use std::fs;
use std::path::{Path, PathBuf};

use blockflow::error::{Error, Result};
use blockflow::object_table::{
    ColumnRole, ColumnSpec, CoordinateColumn, DType, ObjectTableBuilder, ObjectValue,
    SpatialIndexSpec, TableReader, TableSpec,
};
use blockflow::ome_zarr::ensure_zarr_group;
use serde_json::json;

use super::vessel::{ColumnRead, Lattice, Level, VesselGraph};

/// The spatial index's tile, `[z, y, x]`.
const TILE_ZYX: [u64; 3] = [32, 256, 256];

/// What the tables need to know about the image.
#[derive(Debug, Clone, PartialEq)]
pub struct TableGeometry {
    /// Level-0 shape, `[z, y, x]`.
    pub shape: [usize; 3],
    /// Level-0 voxel to physical.
    pub lattice: Lattice,
    /// The physical unit, as the image declares it.
    pub unit: String,
    /// The atlas space `coordinates_atlas` is in, recorded as
    /// `extensions["vessel_graph"]["atlas"]` (a registration's description).
    pub atlas: Option<serde_json::Value>,
}

/// Where the three tables went.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphTables {
    pub vertices: PathBuf,
    pub edges: PathBuf,
    pub edge_points: PathBuf,
}

impl GraphTables {
    /// The three table roots of graph `name` under `tables`.
    pub fn under(tables: &Path, name: &str) -> Self {
        Self {
            vertices: tables.join(format!("{name}-vertices")),
            edges: tables.join(format!("{name}-edges")),
            edge_points: tables.join(format!("{name}-edge-points")),
        }
    }
}

/// How a graph column is stored.
fn dtype_of(name: &str) -> (DType, ColumnRole) {
    match name {
        "annotation" | "hemisphere" => (DType::I64, ColumnRole::Category),
        "artery" | "vein" | "artery_binary" | "vein_binary" => (DType::U8, ColumnRole::Category),
        name if name.ends_with("_raw") => (DType::F64, ColumnRole::Intensity),
        _ => (DType::F64, ColumnRole::Measurement),
    }
}

fn value(dtype: DType, value: f64) -> ObjectValue {
    match dtype {
        DType::I64 => ObjectValue::I64(if value.is_finite() { value as i64 } else { 0 }),
        DType::U8 => ObjectValue::U8(u8::from(value != 0.0 && !value.is_nan())),
        _ => ObjectValue::F64(value),
    }
}

/// The unit of a graph column, by its name.
fn unit_of(name: &str, unit: &str) -> Option<String> {
    if name.contains("units") {
        Some(unit.to_string())
    } else if name == "length" || name.starts_with("radii") || name.starts_with("coordinates") {
        Some("voxel".to_string())
    } else {
        None
    }
}

/// How much memory the table writer may hold, and where it spills.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableMemory {
    /// The bytes the buffered rows of one table may take
    /// ([`ObjectTableBuilder::row_bytes`] each); `u64::MAX` keeps every row
    /// in memory.
    pub buffer_bytes: u64,
    /// Where the sorted runs of a table larger than the buffer go (removed
    /// afterwards). Needed only with a finite buffer.
    pub temporary: Option<PathBuf>,
}

impl TableMemory {
    /// Every row in memory.
    pub fn unbounded() -> Self {
        Self {
            buffer_bytes: u64::MAX,
            temporary: None,
        }
    }
}

/// What the largest of the three tables takes when every row is buffered
/// (`ObjectTableBuilder::row_bytes` a row): what a writer needs to sort it in
/// memory without a run on disk.
pub fn table_buffer_bytes(graph: &VesselGraph) -> u64 {
    let width = |level: Level, skipped: usize| -> usize {
        graph
            .columns(level)
            .iter()
            .map(|(_, column)| column.width())
            .sum::<usize>()
            .saturating_sub(skipped)
    };
    // The identity and six positions, the integer columns, the graph's.
    [
        (graph.n_vertices(), 7 + 1 + width(Level::Vertex, 3)),
        (graph.n_edges(), 7 + 4 + width(Level::Edge, 0)),
        (graph.n_points(), 7 + 2 + width(Level::Along, 3)),
    ]
    .iter()
    .map(|(rows, columns)| {
        (*rows as u64).saturating_mul(ObjectTableBuilder::row_bytes_of(*columns))
    })
    .max()
    .unwrap_or(0)
}

/// A table's columns: the identity, the voxel position and the physical
/// one, then whatever is pushed.
struct Columns {
    specs: Vec<ColumnSpec>,
}

impl Columns {
    fn new(identity: &str, unit: &str) -> Self {
        let mut specs = vec![ColumnSpec::new(identity, DType::U64, ColumnRole::Identity)];
        for axis in ["z", "y", "x"] {
            let mut spec = ColumnSpec::new(axis, DType::F64, ColumnRole::Coordinate);
            spec.unit = Some("voxel".into());
            spec.display_name = Some(format!("{axis} (level-0 voxel)"));
            specs.push(spec);
        }
        for axis in ["z", "y", "x"] {
            let mut spec =
                ColumnSpec::new(format!("{axis}_um"), DType::F64, ColumnRole::Measurement);
            spec.unit = Some(unit.to_string());
            spec.display_name = Some(format!("{axis} (physical)"));
            specs.push(spec);
        }
        Self { specs }
    }

    fn push(&mut self, name: &str, dtype: DType, role: ColumnRole, unit: Option<String>) {
        let mut spec = ColumnSpec::new(name, dtype, role);
        spec.unit = unit;
        self.specs.push(spec);
    }

    fn integer(&mut self, name: &str, dtype: DType) {
        self.push(name, dtype, ColumnRole::Measurement, None);
    }

    /// Every column of one graph level, but `skip`: their table columns
    /// declared, and the columns returned with the dtype their values are
    /// written as.
    fn graph_columns<'a>(
        &mut self,
        columns: Vec<(&'a str, ColumnRead<'a>)>,
        skip: &[&str],
        unit: &str,
    ) -> Result<Vec<(ColumnRead<'a>, DType)>> {
        let mut out = Vec::new();
        for (name, column) in columns {
            if skip.contains(&name) {
                continue;
            }
            let (dtype, role) = dtype_of(name);
            let names: Vec<String> = match column.width() {
                1 => vec![name.to_string()],
                3 => ["z", "y", "x"]
                    .iter()
                    .map(|axis| format!("{name}_{axis}"))
                    .collect(),
                width => {
                    return Err(Error::invalid(format!(
                        "the graph column `{name}` has width {width}; a table takes scalar or \
                         per-axis columns."
                    )))
                }
            };
            for column_name in &names {
                if self.specs.iter().any(|spec| &spec.name == column_name) {
                    return Err(Error::invalid(format!(
                        "the graph column `{name}` would write the table column `{column_name}`, \
                         which is already there."
                    )));
                }
                self.push(column_name, dtype, role, unit_of(name, unit));
            }
            out.push((column, dtype));
        }
        Ok(out)
    }

    /// The table's builder, its rows to be pushed in any order.
    fn builder(
        self,
        identity: &str,
        geometry: &TableGeometry,
        extension: serde_json::Value,
        memory: &TableMemory,
    ) -> Result<ObjectTableBuilder> {
        let shape = geometry.shape;
        let spatial_index = SpatialIndexSpec {
            coordinates: ["z", "y", "x"]
                .into_iter()
                .map(|axis| CoordinateColumn {
                    axis: axis.into(),
                    column: axis.into(),
                })
                .collect(),
            grid_shape: (0..3)
                .map(|axis| (shape[axis] as u64).div_ceil(TILE_ZYX[axis]).max(1))
                .collect(),
            tile_shape: TILE_ZYX.to_vec(),
            tile_order: "row_major".into(),
            within_tile_order: "lexicographic_coordinates_then_identity".into(),
            downsample: None,
        };
        let mut spec = TableSpec::new(0, 4096, "../../", identity, self.specs, spatial_index);
        spec.extensions.insert("vessel_graph".into(), extension);
        ObjectTableBuilder::new(spec, memory.temporary.as_ref(), memory.buffer_bytes)
    }
}

/// One row's values, built in place.
struct Row {
    values: Vec<ObjectValue>,
}

impl Row {
    /// The identity, the voxel position and the physical one.
    fn start(&mut self, id: usize, at: [f64; 3], lattice: &Lattice) {
        self.values.clear();
        self.values.push(ObjectValue::U64(id as u64));
        self.values.extend(at.iter().map(|v| ObjectValue::F64(*v)));
        let physical = lattice.physical(at);
        self.values
            .extend(physical.iter().map(|v| ObjectValue::F64(*v)));
    }

    fn push(&mut self, value: ObjectValue) {
        self.values.push(value);
    }

    /// Row `row` of every graph column, a value per axis.
    fn graph_columns(&mut self, columns: &[(ColumnRead<'_>, DType)], row: usize) {
        for (column, dtype) in columns {
            for axis in 0..column.width() {
                self.values.push(value(*dtype, column.value(row, axis)));
            }
        }
    }
}

/// Write `graph` as its three tables under `tables` (created as a Zarr group
/// if absent), replacing any earlier tables of the same `name`. Coordinates
/// are the graph's `coordinates` column, level-0 voxels of the image. Every
/// row is held in memory; [`write_graph_tables_within`] bounds that.
pub fn write_graph_tables(
    graph: &VesselGraph,
    tables: &Path,
    name: &str,
    geometry: &TableGeometry,
) -> Result<GraphTables> {
    write_graph_tables_within(graph, tables, name, geometry, &TableMemory::unbounded())
}

/// [`write_graph_tables`] with each table's buffered rows held to `memory`:
/// the rows are made one at a time from the graph's columns and pushed to a
/// [`ObjectTableBuilder`], which spills sorted runs past its buffer. The
/// tables are the same files whatever the buffer.
pub fn write_graph_tables_within(
    graph: &VesselGraph,
    tables: &Path,
    name: &str,
    geometry: &TableGeometry,
    memory: &TableMemory,
) -> Result<GraphTables> {
    let roots = GraphTables::under(tables, name);
    ensure_zarr_group(tables)?;
    let extension = |role: &str| {
        let mut extension = json!({
            "version": 1,
            "role": role,
            "graph": name,
            "vertices": format!("../{name}-vertices"),
            "edges": format!("../{name}-edges"),
            "edge_points": format!("../{name}-edge-points"),
        });
        if let Some(atlas) = &geometry.atlas {
            extension["atlas"] = atlas.clone();
        }
        extension
    };
    let unit = geometry.unit.as_str();
    let lattice = geometry.lattice;
    let mut row = Row { values: Vec::new() };
    let point = |column: &ColumnRead<'_>, at: usize| -> [f64; 3] {
        std::array::from_fn(|axis| column.value(at, axis))
    };

    // Vertices.
    let coordinates = graph.named(Level::Vertex, "coordinates")?;
    check_points(&coordinates)?;
    let mut columns = Columns::new("vertex_id", unit);
    columns.integer("degree", DType::U32);
    let carried = columns.graph_columns(graph.columns(Level::Vertex), &["coordinates"], unit)?;
    let mut table = columns.builder("vertex_id", geometry, extension("vertices"), memory)?;
    table.reserve(graph.n_vertices());
    let degrees = graph.degrees();
    for (vertex, degree) in degrees.iter().enumerate() {
        row.start(vertex, point(&coordinates, vertex), &lattice);
        row.push(ObjectValue::U32(*degree as u32));
        row.graph_columns(&carried, vertex);
        table.push(&row.values)?;
    }
    drop(degrees);
    finish(table, &roots.vertices)?;

    // Edges, each at its polyline's middle point.
    let along = graph.named(Level::Along, "coordinates")?;
    check_points(&along)?;
    let ranges = graph.polylines();
    let mut columns = Columns::new("edge_id", unit);
    columns.integer("source", DType::U64);
    columns.integer("target", DType::U64);
    columns.integer("point_start", DType::U64);
    columns.integer("point_count", DType::U32);
    let carried = columns.graph_columns(graph.columns(Level::Edge), &[], unit)?;
    let mut table = columns.builder("edge_id", geometry, extension("edges"), memory)?;
    table.reserve(graph.n_edges());
    for (edge, ([start, end], [source, target])) in ranges.iter().zip(graph.edges()).enumerate() {
        let (start, end) = (*start as usize, *end as usize);
        row.start(edge, point(&along, start + (end - start) / 2), &lattice);
        row.push(ObjectValue::U64(u64::from(*source)));
        row.push(ObjectValue::U64(u64::from(*target)));
        row.push(ObjectValue::U64(start as u64));
        row.push(ObjectValue::U32((end - start) as u32));
        row.graph_columns(&carried, edge);
        table.push(&row.values)?;
    }
    finish(table, &roots.edges)?;

    // Polyline points.
    let mut columns = Columns::new("point_id", unit);
    columns.integer("edge_id", DType::U64);
    columns.integer("index", DType::U32);
    let carried = columns.graph_columns(graph.columns(Level::Along), &["coordinates"], unit)?;
    let mut table = columns.builder("point_id", geometry, extension("edge_points"), memory)?;
    table.reserve(graph.n_points());
    for (edge, [start, end]) in ranges.iter().enumerate() {
        for at in *start as usize..*end as usize {
            row.start(at, point(&along, at), &lattice);
            row.push(ObjectValue::U64(edge as u64));
            row.push(ObjectValue::U32((at - *start as usize) as u32));
            row.graph_columns(&carried, at);
            table.push(&row.values)?;
        }
    }
    finish(table, &roots.edge_points)?;
    Ok(roots)
}

fn check_points(column: &ColumnRead<'_>) -> Result<()> {
    if column.width() != 3 {
        return Err(Error::invalid(format!(
            "a column of width {} is not a list of 3-vectors.",
            column.width()
        )));
    }
    Ok(())
}

/// Write a table at `root`, replacing what is there.
fn finish(table: ObjectTableBuilder, root: &Path) -> Result<TableReader> {
    if root.exists() {
        fs::remove_dir_all(root).map_err(Error::backend)?;
    }
    table.finish(root)
}
