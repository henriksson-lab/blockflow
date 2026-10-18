// SPDX-License-Identifier: MIT

//! The three per-field tables, as native NGFF object tables.
//!
//! | table | one row per | index |
//! |---|---|---|
//! | `tables/<layer>` | detection | `(t, [z,] y, x)`, time tile 1, time not downsampled |
//! | `tables/<layer>-tracks` | maximal unbranched track | its first detection's `(t, [z,] y, x)` |
//! | `tables/<layer>-links` | graph edge | its source detection's `(t, [z,] y, x)` |
//!
//! The downstream contract: group `tables/<layer>` by `track_id`, order by
//! `t_index`, and plot any measurement column.

use std::collections::BTreeMap;
use std::path::Path;

use blockflow::object_table::{
    write_object_rows, ColumnRole, ColumnSpec, CoordinateColumn, DType, ObjectValue,
    SpatialIndexSpec, TableSpec,
};
use blockflow::track::{DetectionSet, Track, TrackGraph};
use blockflow::{Error, Result};
use ngff_label_map::{
    DomainId, IdScope, IdentityDomain, IdentityGraphMetadata, RelationDescriptor, RelationKind,
    UnmappedBehavior,
};
use serde_json::json;

use crate::frame::FrameRow;

const SPATIAL_TILE: u64 = 512;
const ROW_CHUNK: u64 = 65536;

fn domain(layer: &str, role: &str) -> DomainId {
    DomainId::new(format!("{layer}.{role}")).expect("a layer name already valid for an NGFF path")
}

/// Identity contract shared by the label raster and its detections table. Detection IDs are
/// globally unique because frame segmentation reserves a disjoint numeric interval per frame.
pub fn identity_graph(layer: &str) -> IdentityGraphMetadata {
    let detection = domain(layer, "detections");
    let track = domain(layer, "tracks");
    let lineage = domain(layer, "lineages");
    let domains = [
        (
            detection.clone(),
            IdentityDomain {
                role: "detection".into(),
                scope: IdScope::Dataset,
            },
        ),
        (
            track.clone(),
            IdentityDomain {
                role: "track".into(),
                scope: IdScope::Dataset,
            },
        ),
        (
            lineage.clone(),
            IdentityDomain {
                role: "lineage".into(),
                scope: IdScope::Dataset,
            },
        ),
    ]
    .into_iter()
    .collect();
    let mut tracking_parameters = serde_json::Map::new();
    tracking_parameters.insert("table".into(), json!(format!("../../tables/{layer}")));
    tracking_parameters.insert("source_column".into(), json!("detection_id"));
    tracking_parameters.insert("target_column".into(), json!("track_id"));
    let mut lineage_parameters = serde_json::Map::new();
    lineage_parameters.insert("links".into(), json!(format!("../../tables/{layer}-links")));
    IdentityGraphMetadata {
        version: 1,
        stored_domain: detection.clone(),
        default_display_domain: detection.clone(),
        domains,
        relations: vec![
            RelationDescriptor {
                name: "tracking".into(),
                kind: RelationKind::Column,
                from: detection,
                to: track.clone(),
                editable: false,
                unmapped: UnmappedBehavior::Missing,
                parameters: tracking_parameters,
            },
            RelationDescriptor {
                name: "lineage".into(),
                kind: RelationKind::GraphComponent,
                from: track,
                to: lineage,
                editable: false,
                unmapped: UnmappedBehavior::Missing,
                parameters: lineage_parameters,
            },
        ],
    }
}

fn index(has_z: bool, extent: [usize; 3], frames: u64, prefix: &str) -> SpatialIndexSpec {
    let mut axes = vec!["t"];
    if has_z {
        axes.push("z");
    }
    axes.extend(["y", "x"]);
    let mut tile_shape = vec![1];
    let mut grid_shape = vec![frames.max(1)];
    let mut downsample = vec![false];
    if has_z {
        tile_shape.push(extent[0] as u64);
        grid_shape.push(1);
        downsample.push(true);
    }
    for axis in [1, 2] {
        tile_shape.push(SPATIAL_TILE);
        grid_shape.push((extent[axis] as u64).div_ceil(SPATIAL_TILE).max(1));
        downsample.push(true);
    }
    SpatialIndexSpec {
        coordinates: axes
            .iter()
            .map(|axis| CoordinateColumn {
                axis: (*axis).into(),
                column: format!("{prefix}{axis}"),
            })
            .collect(),
        tile_shape,
        grid_shape,
        tile_order: "row_major".into(),
        within_tile_order: "lexicographic_coordinates_then_identity".into(),
        downsample: Some(downsample),
    }
}

fn coordinate_columns(has_z: bool, prefix: &str) -> Vec<ColumnSpec> {
    let mut columns = vec![ColumnSpec::new(
        format!("{prefix}t"),
        DType::F64,
        ColumnRole::Coordinate,
    )];
    if has_z {
        columns.push(ColumnSpec::new(
            format!("{prefix}z"),
            DType::F64,
            ColumnRole::Coordinate,
        ));
    }
    columns.push(ColumnSpec::new(
        format!("{prefix}y"),
        DType::F64,
        ColumnRole::Coordinate,
    ));
    columns.push(ColumnSpec::new(
        format!("{prefix}x"),
        DType::F64,
        ColumnRole::Coordinate,
    ));
    columns
}

fn coordinates(has_z: bool, t: u64, at: [f64; 3]) -> Vec<ObjectValue> {
    let mut values = vec![ObjectValue::F64(t as f64)];
    if has_z {
        values.push(ObjectValue::F64(at[0]));
    }
    values.push(ObjectValue::F64(at[1]));
    values.push(ObjectValue::F64(at[2]));
    values
}

fn optional(value: Option<f64>) -> ObjectValue {
    ObjectValue::F64(value.unwrap_or(f64::NAN))
}

/// Shape of the per-field tables.
pub struct TableContext<'a> {
    pub has_z: bool,
    pub extent: [usize; 3],
    pub spacing: [f64; 3],
    /// Grid length along `t`: one past the largest `t_index`.
    pub frames: u64,
    pub layer: &'a str,
    pub elapsed: &'a BTreeMap<u64, Option<f64>>,
    /// Whether `labels/<layer>-tracks` is written.
    pub track_labels: bool,
}

pub fn write_detections(
    root: &Path,
    context: &TableContext<'_>,
    rows: &[FrameRow],
    track_of: &BTreeMap<u64, u64>,
    lineage_of: &BTreeMap<u64, u64>,
) -> Result<()> {
    let mut columns = vec![ColumnSpec::new(
        "detection_id",
        DType::U64,
        ColumnRole::Identity,
    )];
    columns.extend(coordinate_columns(context.has_z, ""));
    let measurement = |name: &str| ColumnSpec::new(name, DType::F64, ColumnRole::Measurement);
    let intensity = |name: &str| ColumnSpec::new(name, DType::F64, ColumnRole::Intensity);
    columns.extend([
        ColumnSpec::new("track_id", DType::U64, ColumnRole::Measurement),
        ColumnSpec::new("lineage_id", DType::U64, ColumnRole::Measurement),
        ColumnSpec::new("t_index", DType::U64, ColumnRole::Measurement),
        measurement("elapsed_seconds"),
        ColumnSpec::new("area_pixels", DType::U64, ColumnRole::Measurement),
        measurement("area_physical"),
        measurement("equivalent_diameter"),
        measurement("eccentricity"),
        measurement("major_axis_length"),
        measurement("minor_axis_length"),
        measurement("orientation_yx"),
        intensity("segment_mean"),
        intensity("segment_min"),
        intensity("segment_max"),
        intensity("measure_sum"),
        intensity("measure_mean"),
        intensity("measure_min"),
        intensity("measure_max"),
    ]);
    let pixel_area = if context.has_z {
        context.spacing.iter().product::<f64>()
    } else {
        context.spacing[1] * context.spacing[2]
    };
    let mut values = Vec::with_capacity(rows.len());
    for row in rows {
        let track = *track_of.get(&row.detection_id).ok_or_else(|| {
            Error::invalid(format!("detection {} has no track", row.detection_id))
        })?;
        let lineage = *lineage_of
            .get(&track)
            .ok_or_else(|| Error::invalid(format!("track {track} has no lineage")))?;
        let mut v = vec![ObjectValue::U64(row.detection_id)];
        v.extend(coordinates(context.has_z, row.t_index, row.centroid));
        v.extend([
            ObjectValue::U64(track),
            ObjectValue::U64(lineage),
            ObjectValue::U64(row.t_index),
            optional(context.elapsed.get(&row.t_index).copied().flatten()),
            ObjectValue::U64(row.area),
            ObjectValue::F64(row.area as f64 * pixel_area),
            ObjectValue::F64(row.equivalent_diameter),
            optional(row.eccentricity),
            optional(row.major_axis_length),
            optional(row.minor_axis_length),
            optional(row.orientation_yx),
            ObjectValue::F64(row.segment_mean),
            ObjectValue::F64(row.segment_min),
            ObjectValue::F64(row.segment_max),
            ObjectValue::F64(row.measure_sum),
            optional(row.measure_mean),
            optional(row.measure_min),
            optional(row.measure_max),
        ]);
        values.push(v);
    }
    let mut spec = TableSpec::new(
        0,
        ROW_CHUNK,
        "../../",
        "detection_id",
        columns,
        index(context.has_z, context.extent, context.frames, ""),
    );
    spec.region = Some(format!("../../labels/{}", context.layer));
    let layer = context.layer;
    spec.extensions.insert(
        "tracking".into(),
        json!({
            "version": 1,
            "role": "detections",
            "track_column": "track_id",
            "time_column": "t_index",
            "elapsed_column": "elapsed_seconds",
            "tracks": format!("../{layer}-tracks"),
            "links": format!("../{layer}-links"),
            "summary": format!("../../tracking/{layer}/tracking-summary.json"),
            "label": format!("../../labels/{layer}"),
            "track_label": context.track_labels.then(|| format!("../../labels/{layer}-tracks")),
        }),
    );
    spec.extensions.insert(
        "identity_graph".into(),
        serde_json::to_value(identity_graph(layer)).expect("identity graph serializes"),
    );
    write_object_rows(root, spec, values)?;
    Ok(())
}

pub fn write_tracks(
    root: &Path,
    context: &TableContext<'_>,
    detections: &DetectionSet,
    tracks: &[Track],
    lineage_of: &BTreeMap<u64, u64>,
) -> Result<()> {
    let mut columns = vec![ColumnSpec::new(
        "track_id",
        DType::U64,
        ColumnRole::Identity,
    )];
    columns.extend(coordinate_columns(context.has_z, "start_"));
    let u64_column = |name: &str| ColumnSpec::new(name, DType::U64, ColumnRole::Measurement);
    let u8_column = |name: &str| ColumnSpec::new(name, DType::U8, ColumnRole::Measurement);
    columns.extend([
        u64_column("lineage_id"),
        u64_column("t_start"),
        u64_column("t_end"),
        u64_column("first_id"),
        u64_column("last_id"),
        u64_column("n_detections"),
        u64_column("n_parents"),
        u64_column("parent_track_id"),
        u64_column("n_children"),
        u8_column("start_reason"),
        u8_column("end_reason"),
        u8_column("start_near_border"),
        u8_column("end_near_border"),
        u8_column("likely_entered_fov"),
        u8_column("likely_left_fov"),
    ]);
    let mut values = Vec::with_capacity(tracks.len());
    for track in tracks {
        let first = detections
            .get(track.first_id())
            .ok_or_else(|| Error::invalid("track names an unknown detection"))?;
        let mut v = vec![ObjectValue::U64(track.track_id)];
        v.extend(coordinates(context.has_z, track.t_start, first.native));
        v.extend([
            ObjectValue::U64(lineage_of[&track.track_id]),
            ObjectValue::U64(track.t_start),
            ObjectValue::U64(track.t_end),
            ObjectValue::U64(track.first_id()),
            ObjectValue::U64(track.last_id()),
            ObjectValue::U64(track.detections.len() as u64),
            ObjectValue::U64(track.parents.len() as u64),
            ObjectValue::U64(track.parents.first().copied().unwrap_or(0)),
            ObjectValue::U64(track.children.len() as u64),
            ObjectValue::U8(track.events.start_reason.code()),
            ObjectValue::U8(track.events.end_reason.code()),
            ObjectValue::U8(track.events.start_border.code()),
            ObjectValue::U8(track.events.end_border.code()),
            ObjectValue::U8(u8::from(track.events.likely_entered_fov())),
            ObjectValue::U8(u8::from(track.events.likely_left_fov())),
        ]);
        values.push(v);
    }
    let mut spec = TableSpec::new(
        0,
        ROW_CHUNK,
        "../../",
        "track_id",
        columns,
        index(context.has_z, context.extent, context.frames, "start_"),
    );
    spec.region = Some(format!("../../labels/{}", context.layer));
    spec.extensions.insert(
        "tracking".into(),
        json!({"version": 1, "role": "tracks", "detections": format!("../{}", context.layer)}),
    );
    write_object_rows(root, spec, values)?;
    Ok(())
}

pub fn write_links(
    root: &Path,
    context: &TableContext<'_>,
    detections: &DetectionSet,
    graph: &TrackGraph,
) -> Result<()> {
    let mut columns = vec![ColumnSpec::new("link_id", DType::U64, ColumnRole::Identity)];
    columns.extend(coordinate_columns(context.has_z, "from_"));
    columns.extend([
        ColumnSpec::new("from_id", DType::U64, ColumnRole::Measurement),
        ColumnSpec::new("to_id", DType::U64, ColumnRole::Measurement),
        ColumnSpec::new("dt", DType::U64, ColumnRole::Measurement),
        ColumnSpec::new("elapsed_seconds", DType::F64, ColumnRole::Measurement),
        ColumnSpec::new("cost", DType::F64, ColumnRole::Measurement),
        ColumnSpec::new("kind", DType::U8, ColumnRole::Measurement),
    ]);
    let mut values = Vec::with_capacity(graph.links().len());
    // Links are sorted by (from, to); the 1-based position is a stable id.
    for (index, link) in graph.links().iter().enumerate() {
        let from = detections
            .get(link.from)
            .ok_or_else(|| Error::invalid("link names an unknown detection"))?;
        let mut v = vec![ObjectValue::U64(index as u64 + 1)];
        v.extend(coordinates(context.has_z, from.t_index, from.native));
        v.extend([
            ObjectValue::U64(link.from),
            ObjectValue::U64(link.to),
            ObjectValue::U64(link.dt),
            optional(link.elapsed_seconds),
            ObjectValue::F64(link.cost),
            ObjectValue::U8(link.kind.code()),
        ]);
        values.push(v);
    }
    let mut spec = TableSpec::new(
        0,
        ROW_CHUNK,
        "../../",
        "link_id",
        columns,
        index(context.has_z, context.extent, context.frames, "from_"),
    );
    spec.region = Some(format!("../../labels/{}", context.layer));
    spec.extensions.insert(
        "tracking".into(),
        json!({
            "version": 1,
            "role": "links",
            "detections": format!("../{}", context.layer),
            "kind_codes": {"0": "continue", "1": "divide", "2": "merge", "3": "gap"},
        }),
    );
    write_object_rows(root, spec, values)?;
    Ok(())
}
