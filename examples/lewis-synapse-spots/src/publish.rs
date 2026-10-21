// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The segmentation, installed in the input OME-Zarr image beside the raw data:
//!
//! | path | content |
//! |---|---|
//! | `labels/<layer>-spots` | one label per spot (the watershed basins) |
//! | `tables/<layer>-spots` | one row per spot; `in_synapse` marks those on the analysed synapse |
//! | `labels/<layer>-synapse` | the filled synapse mask, one label per component |
//! | `tables/<layer>-synapse` | one row, the analysed synapse: its shape, intensity and per-synapse features |
//! | `rozt:imageMetadata` | `condition` (SLB1..4), `field`, `roi`, parsed from the image name |
//!
//! The synapse table holds only the analysed component, so every value in it
//! is defined: a cross-image collection (ROZT `analysis.build-collection`)
//! takes whole tables and refuses missing values. The other mask components
//! (neighbouring cells, debris) stay in the label image without rows.
//!
//! Spot labels are the peak's linear voxel index plus one (see
//! [`crate::seeds`]), so they are sparse; the tables' `label_id` is that label.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use blockflow::object_table::{
    write_object_rows, ColumnRole, ColumnSpec, CoordinateColumn, DType, ObjectValue,
    SpatialIndexSpec, TableSpec,
};
use blockflow::image_metadata::{
    self, ImageMetadataDocument, ImageMetadataProperty, ImageMetadataProvenance, SCHEMA_VERSION,
};
use blockflow::ome_zarr::{publish_label, LabelPublication, OmeVolumePyramid};
use blockflow::{Error, Result};

use crate::{features, CellRow, SpotRow};

/// Spatial-index tiles: one plane, 256 x 256.
const TILE: [u64; 3] = [1, 256, 256];

pub fn spots_name(layer: &str) -> String {
    format!("{layer}-spots")
}

pub fn synapse_name(layer: &str) -> String {
    format!("{layer}-synapse")
}

fn col(name: &str, dtype: DType, role: ColumnRole, unit: Option<&str>) -> ColumnSpec {
    let mut c = ColumnSpec::new(name, dtype, role);
    c.unit = unit.map(str::to_owned);
    c
}

fn measurement(name: &str, unit: Option<&str>) -> ColumnSpec {
    col(name, DType::F32, ColumnRole::Measurement, unit)
}

/// `label_id` and the level-0 voxel centroid that the spatial index uses.
fn identity_and_centroid() -> Vec<ColumnSpec> {
    vec![
        ColumnSpec::new("label_id", DType::U64, ColumnRole::Identity),
        ColumnSpec::new("centroid_z", DType::F32, ColumnRole::Coordinate),
        ColumnSpec::new("centroid_y", DType::F32, ColumnRole::Coordinate),
        ColumnSpec::new("centroid_x", DType::F32, ColumnRole::Coordinate),
    ]
}

fn spec(volume: [usize; 3], label: &str, columns: Vec<ColumnSpec>) -> TableSpec {
    let spatial_index = SpatialIndexSpec {
        coordinates: ["z", "y", "x"]
            .into_iter()
            .map(|axis| CoordinateColumn {
                axis: axis.into(),
                column: format!("centroid_{axis}"),
            })
            .collect(),
        tile_shape: TILE.to_vec(),
        grid_shape: (0..3)
            .map(|a| (volume[a] as u64).div_ceil(TILE[a]))
            .collect(),
        tile_order: "row_major".into(),
        within_tile_order: "lexicographic_coordinates_then_identity".into(),
        downsample: None,
    };
    let mut s = TableSpec::new(0, 2048, "../../", "label_id", columns, spatial_index);
    s.region = Some(format!("../../labels/{label}"));
    s
}

fn centroid(yx: [f64; 2]) -> [ObjectValue; 3] {
    [
        ObjectValue::F32(0.0),
        ObjectValue::F32(yx[0] as f32),
        ObjectValue::F32(yx[1] as f32),
    ]
}

/// Write `tables/<layer>-spots` to `root`, staged for [`install`].
pub fn spot_table(
    root: &Path,
    volume: [usize; 3],
    label: &str,
    spots: &[SpotRow],
    synapse_label: u64,
) -> Result<()> {
    let mut columns = identity_and_centroid();
    columns.extend([
        measurement("weighted_centroid_y", None),
        measurement("weighted_centroid_x", None),
        col("area_px", DType::U64, ColumnRole::Measurement, None),
        col("raw_sum", DType::F32, ColumnRole::Intensity, None),
        col("raw_max", DType::F32, ColumnRole::Intensity, None),
        col("response_max", DType::F32, ColumnRole::Intensity, None),
        measurement("eccentricity", None),
        measurement("major_axis_px", None),
        measurement("minor_axis_px", None),
        measurement("edge_distance_px", None),
        col("synapse_label", DType::U64, ColumnRole::Measurement, None),
        col("in_synapse", DType::U8, ColumnRole::Category, None),
    ]);
    let rows = spots
        .iter()
        .map(|s| {
            let mut v = vec![ObjectValue::U64(s.label)];
            v.extend(centroid(s.centroid));
            v.extend([
                ObjectValue::F32(s.weighted_centroid[0] as f32),
                ObjectValue::F32(s.weighted_centroid[1] as f32),
                ObjectValue::U64(s.area),
                ObjectValue::F32(s.raw_sum as f32),
                ObjectValue::F32(s.raw_max as f32),
                ObjectValue::F32(s.response_max as f32),
                ObjectValue::F32(s.eccentricity as f32),
                ObjectValue::F32(s.major_axis as f32),
                ObjectValue::F32(s.minor_axis as f32),
                ObjectValue::F32(s.edge_distance as f32),
                ObjectValue::U64(s.cell_label),
                ObjectValue::U8(u8::from(synapse_label != 0 && s.cell_label == synapse_label)),
            ]);
            v
        })
        .collect();
    write_object_rows(root, spec(volume, label, columns), rows).map(drop)
}

/// Write `tables/<layer>-synapse` to `root`, staged for [`install`]: one row
/// for the analysed synapse, none when the image has none.
pub fn synapse_table(
    root: &Path,
    volume: [usize; 3],
    label: &str,
    synapse: Option<(&CellRow, &[f64])>,
) -> Result<()> {
    let names = features::header();
    let feature_names: Vec<&str> = names.split(',').collect();
    let mut columns = identity_and_centroid();
    columns.extend([
        col("area_px", DType::U64, ColumnRole::Measurement, None),
        col("raw_sum", DType::F32, ColumnRole::Intensity, None),
        col("raw_mean", DType::F32, ColumnRole::Intensity, None),
        // equivalent-disk radius, which sizes a viewer's crop around the synapse
        measurement("radius_px", None),
    ]);
    columns.extend(feature_names.iter().map(|n| measurement(n, None)));
    let rows = synapse
        .map(|(c, features)| {
            let mut v = vec![ObjectValue::U64(c.label)];
            v.extend(centroid(c.centroid));
            v.extend([
                ObjectValue::U64(c.area),
                ObjectValue::F32(c.intensity_sum as f32),
                ObjectValue::F32(c.intensity_mean as f32),
                ObjectValue::F32((c.area as f64 / std::f64::consts::PI).sqrt() as f32),
            ]);
            v.extend(features.iter().map(|&f| ObjectValue::F32(f as f32)));
            v
        })
        .into_iter()
        .collect();
    write_object_rows(root, spec(volume, label, columns), rows).map(drop)
}

/// Install a finished `uint64` level-0 label array and its staged table as
/// `labels/<label>` and `tables/<label>` of `pyramid`'s image.
pub fn install(
    pyramid: &OmeVolumePyramid,
    label: &str,
    level0: &Path,
    table: &Path,
    replace: bool,
) -> Result<()> {
    publish_label(
        pyramid,
        LabelPublication {
            name: label,
            level0,
            image_label: None,
            table: Some(table),
            replace,
            pyramid_blocks: vec![[1, 256, 256]],
        },
        1,
    )
}

/// The bilayer of a condition code from the image name.
fn condition_name(code: &str) -> Option<&'static str> {
    Some(match code {
        "icam1slb" => "SLB1 ICAM-1",
        "icam1ucht1slb" => "SLB2 +anti-CD3",
        "icam1ucht1cd80slb" => "SLB3 +CD80",
        "icam1ucht1cd80cd58slb" => "SLB4 +CD58",
        _ => return None,
    })
}

/// Set `condition`, `field` and `roi` in the image's `rozt:imageMetadata`,
/// keeping any other properties.
pub fn image_metadata(pyramid: &OmeVolumePyramid, condition: &str, field: &str, roi: &str) -> Result<()> {
    let condition = condition_name(condition)
        .ok_or_else(|| Error::invalid(format!("lewis-synapse-spots: unknown condition {condition:?}")))?;
    let number = |text: &str, what: &str| {
        text.trim_start_matches(|c: char| !c.is_ascii_digit())
            .parse::<i64>()
            .map_err(|_| Error::invalid(format!("lewis-synapse-spots: {what} {text:?} is not numbered")))
    };
    let current = image_metadata::read_image_metadata(pyramid.image_root())?;
    let mut properties = current.document.properties;
    let named = |name: &str| Some(name.to_owned());
    properties.insert(
        "condition".into(),
        ImageMetadataProperty::Categorical {
            value: condition.into(),
            display_name: named("Condition"),
            unit: None,
            description: Some("supported lipid bilayer composition".into()),
        },
    );
    for (key, display, text) in [("field", "Field", field), ("roi", "ROI", roi)] {
        properties.insert(
            key.into(),
            ImageMetadataProperty::Integer {
                value: number(text, key)?,
                display_name: named(display),
                unit: None,
                description: None,
            },
        );
    }
    let updated_at_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(Error::backend)?
        .as_millis() as u64;
    let document = ImageMetadataDocument {
        schema_version: SCHEMA_VERSION,
        properties,
        provenance: Some(ImageMetadataProvenance {
            writer: "blockflow-lewis-synapse-spots".into(),
            updated_at_ms,
        }),
    };
    image_metadata::replace_image_metadata(pyramid.image_root(), Some(&current.revision), &document)
        .map(drop)
}
