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
    write_object_rows, ColumnRole, ColumnSpec, CoordinateColumn, DType, ObjectValue,
    SpatialIndexSpec, TableReader, TableSpec,
};
use blockflow::ome_zarr::ensure_zarr_group;
use serde_json::json;

use super::columns::Column;
use super::vessel::{Lattice, VesselGraph};

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

/// A table under construction: its columns and, per row, its values.
struct Builder {
    columns: Vec<ColumnSpec>,
    rows: Vec<Vec<ObjectValue>>,
}

impl Builder {
    fn new(identity: &str, rows: usize, unit: &str) -> Self {
        let mut columns = vec![ColumnSpec::new(identity, DType::U64, ColumnRole::Identity)];
        for axis in ["z", "y", "x"] {
            let mut spec = ColumnSpec::new(axis, DType::F64, ColumnRole::Coordinate);
            spec.unit = Some("voxel".into());
            spec.display_name = Some(format!("{axis} (level-0 voxel)"));
            columns.push(spec);
        }
        for axis in ["z", "y", "x"] {
            let mut spec =
                ColumnSpec::new(format!("{axis}_um"), DType::F64, ColumnRole::Measurement);
            spec.unit = Some(unit.to_string());
            spec.display_name = Some(format!("{axis} (physical)"));
            columns.push(spec);
        }
        Self {
            columns,
            rows: (0..rows).map(|_| Vec::new()).collect(),
        }
    }

    /// The identity, the voxel position and the physical one, per row.
    fn positions(&mut self, positions: &[[f64; 3]], lattice: &Lattice) {
        for (row, (values, at)) in self.rows.iter_mut().zip(positions).enumerate() {
            let physical = lattice.physical(*at);
            values.push(ObjectValue::U64(row as u64));
            values.extend(at.iter().map(|v| ObjectValue::F64(*v)));
            values.extend(physical.iter().map(|v| ObjectValue::F64(*v)));
        }
    }

    fn push(&mut self, name: &str, dtype: DType, role: ColumnRole, unit: Option<String>) {
        let mut spec = ColumnSpec::new(name, dtype, role);
        spec.unit = unit;
        self.columns.push(spec);
    }

    fn integers(&mut self, name: &str, dtype: DType, values: impl Iterator<Item = u64>) {
        self.push(name, dtype, ColumnRole::Measurement, None);
        for (row, value) in self.rows.iter_mut().zip(values) {
            row.push(match dtype {
                DType::U32 => ObjectValue::U32(value as u32),
                _ => ObjectValue::U64(value),
            });
        }
    }

    /// Every column of one graph level, but `skip`.
    fn graph_columns<'a>(
        &mut self,
        columns: impl Iterator<Item = (&'a str, &'a Column)>,
        skip: &[&str],
        unit: &str,
    ) -> Result<()> {
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
                if self.columns.iter().any(|spec| &spec.name == column_name) {
                    return Err(Error::invalid(format!(
                        "the graph column `{name}` would write the table column `{column_name}`, \
                         which is already there."
                    )));
                }
                self.push(column_name, dtype, role, unit_of(name, unit));
            }
            for (index, row) in self.rows.iter_mut().enumerate() {
                for v in column.row(index) {
                    row.push(value(dtype, *v));
                }
            }
        }
        Ok(())
    }

    fn write(
        self,
        root: &Path,
        identity: &str,
        geometry: &TableGeometry,
        extension: serde_json::Value,
    ) -> Result<TableReader> {
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
        let mut spec = TableSpec::new(0, 4096, "../../", identity, self.columns, spatial_index);
        spec.extensions.insert("vessel_graph".into(), extension);
        if root.exists() {
            fs::remove_dir_all(root).map_err(Error::backend)?;
        }
        write_object_rows(root, spec, self.rows)
    }
}

/// Write `graph` as its three tables under `tables` (created as a Zarr group
/// if absent), replacing any earlier tables of the same `name`. Coordinates
/// are the graph's `coordinates` column, level-0 voxels of the image.
pub fn write_graph_tables(
    graph: &VesselGraph,
    tables: &Path,
    name: &str,
    geometry: &TableGeometry,
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

    // Vertices.
    let vertices = graph.points("coordinates")?;
    let mut table = Builder::new("vertex_id", vertices.len(), unit);
    table.positions(&vertices, &lattice);
    table.integers(
        "degree",
        DType::U32,
        graph.degrees().into_iter().map(|d| d as u64),
    );
    table.graph_columns(
        graph
            .vertex_names()
            .into_iter()
            .filter_map(|n| graph.vertex(n).map(|c| (n, c))),
        &["coordinates"],
        unit,
    )?;
    table.write(
        &roots.vertices,
        "vertex_id",
        geometry,
        extension("vertices"),
    )?;

    // Edges, each at its polyline's middle point.
    let along = graph.along_points("coordinates")?;
    let ranges = graph.ranges();
    let middles: Vec<[f64; 3]> = ranges
        .iter()
        .map(|[start, end]| along[start + (end - start) / 2])
        .collect();
    let mut table = Builder::new("edge_id", graph.n_edges(), unit);
    table.positions(&middles, &lattice);
    table.integers(
        "source",
        DType::U64,
        graph.edges().iter().map(|e| u64::from(e[0])),
    );
    table.integers(
        "target",
        DType::U64,
        graph.edges().iter().map(|e| u64::from(e[1])),
    );
    table.integers(
        "point_start",
        DType::U64,
        ranges.iter().map(|r| r[0] as u64),
    );
    table.integers(
        "point_count",
        DType::U32,
        ranges.iter().map(|r| (r[1] - r[0]) as u64),
    );
    table.graph_columns(
        graph
            .edge_names()
            .into_iter()
            .filter_map(|n| graph.edge(n).map(|c| (n, c))),
        &[],
        unit,
    )?;
    table.write(&roots.edges, "edge_id", geometry, extension("edges"))?;

    // Polyline points.
    let mut edge_of = vec![0u64; along.len()];
    let mut index_of = vec![0u64; along.len()];
    for (edge, [start, end]) in ranges.iter().enumerate() {
        for point in *start..*end {
            edge_of[point] = edge as u64;
            index_of[point] = (point - start) as u64;
        }
    }
    let mut table = Builder::new("point_id", along.len(), unit);
    table.positions(&along, &lattice);
    table.integers("edge_id", DType::U64, edge_of.into_iter());
    table.integers("index", DType::U32, index_of.into_iter());
    table.graph_columns(
        graph
            .along_names()
            .into_iter()
            .filter_map(|n| graph.along(n).map(|c| (n, c))),
        &["coordinates"],
        unit,
    )?;
    table.write(
        &roots.edge_points,
        "point_id",
        geometry,
        extension("edge_points"),
    )?;
    Ok(roots)
}
