//! What newvolim shows: object tables paired with the label images, label
//! colours and names, and a thin surface label.
//!
//! Tables follow blockflow's Cellpose 3D finaliser (`label_id`,
//! `centroid_z/y/x` in level-0 voxels, 32×256×256 index tiles) plus our own
//! columns. The `deleted` column is the final decision; newvolim reads it as
//! the default review state, so rejected candidates are hidden in its
//! "included only" mode and a person can restore or delete objects.

use blockflow::object_table::{
    ColumnRole, ColumnSpec, CoordinateColumn, DType, ObjectValue, SpatialIndexSpec, TableSpec,
};

use serde_json::{json, Value as Json};

use crate::cells::LargeObject;
use crate::measure::{Class, Measured};
/// Region labels of the two fitted surfaces (mucus top, epithelium apical
/// surface): the number of surfaces at or above a voxel.
pub const LABEL_MUCUS: u32 = 1;
/// At or below the epithelium apical surface.
pub const LABEL_TISSUE: u32 = 2;

const TILE: [u64; 3] = [32, 256, 256];

fn col(
    name: &str,
    dtype: DType,
    role: ColumnRole,
    unit: Option<&str>,
    display: &str,
) -> ColumnSpec {
    let mut c = ColumnSpec::new(name, dtype, role);
    c.unit = unit.map(str::to_owned);
    c.display_name = Some(display.to_owned());
    c
}

fn spec(volume: [usize; 3], columns: Vec<ColumnSpec>, extension: Json) -> TableSpec {
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
    s.extensions.insert("bjorn".into(), extension);
    s
}

fn identity_and_centroid() -> Vec<ColumnSpec> {
    vec![
        ColumnSpec::new("label_id", DType::U64, ColumnRole::Identity),
        ColumnSpec::new("centroid_z", DType::F32, ColumnRole::Coordinate),
        ColumnSpec::new("centroid_y", DType::F32, ColumnRole::Coordinate),
        ColumnSpec::new("centroid_x", DType::F32, ColumnRole::Coordinate),
    ]
}

fn codes<T: Copy>(all: &[T], code: impl Fn(T) -> u8, name: impl Fn(T) -> &'static str) -> Json {
    Json::Object(
        all.iter()
            .map(|&v| (code(v).to_string(), json!(name(v))))
            .collect(),
    )
}

pub fn bead_table_spec(volume: [usize; 3]) -> TableSpec {
    use crate::measure::Manual;
    let um = Some("µm");
    let mut columns = identity_and_centroid();
    columns.extend([
        col(
            "deleted",
            DType::U8,
            ColumnRole::Category,
            None,
            "rejected (final)",
        ),
        col(
            "class",
            DType::U8,
            ColumnRole::Category,
            None,
            "automatic class",
        ),
        col(
            "manual",
            DType::U8,
            ColumnRole::Category,
            None,
            "manual decision",
        ),
        col(
            "vertical_dist_um",
            DType::F32,
            ColumnRole::Measurement,
            um,
            "vertical distance to epithelium top",
        ),
        col(
            "euclidean_dist_um",
            DType::F32,
            ColumnRole::Measurement,
            um,
            "3D distance to epithelium top",
        ),
        col(
            "relative_depth",
            DType::F32,
            ColumnRole::Measurement,
            None,
            "depth in mucus (0 top, 1 epithelium)",
        ),
        col(
            "mucus_thickness_um",
            DType::F32,
            ColumnRole::Measurement,
            um,
            "mucus thickness here",
        ),
        col(
            "mucus_top_um",
            DType::F32,
            ColumnRole::Measurement,
            um,
            "mucus top z",
        ),
        col(
            "epi_top_um",
            DType::F32,
            ColumnRole::Measurement,
            um,
            "epithelium top z",
        ),
        col(
            "response",
            DType::F32,
            ColumnRole::Measurement,
            None,
            "DoG response",
        ),
        col(
            "cblue_peak",
            DType::F32,
            ColumnRole::Intensity,
            None,
            "C-Blue at peak",
        ),
        col(
            "z_support",
            DType::F32,
            ColumnRole::Measurement,
            None,
            "PSF z support",
        ),
        col(
            "xy_support",
            DType::F32,
            ColumnRole::Measurement,
            None,
            "PSF xy support",
        ),
        col(
            "large_object",
            DType::U64,
            ColumnRole::Measurement,
            None,
            "touching large AF488 object",
        ),
    ]);
    let ext = json!({
        "version": 1,
        "role": "bead_candidates",
        "deleted": "final decision: 1 = not counted as a bead",
        "class": codes(&Class::ALL, Class::code, Class::as_str),
        "manual": codes(&[Manual::None, Manual::Accepted, Manual::Rejected], Manual::code, |m| match m {
            Manual::None => "none",
            m => m.as_str(),
        }),
    });
    spec(volume, columns, ext)
}

/// One bead-table row.
pub fn bead_values(m: &Measured) -> Vec<ObjectValue> {
    let c = &m.c;
    vec![
        ObjectValue::U64(c.id),
        ObjectValue::F32(c.z as f32),
        ObjectValue::F32(c.y as f32),
        ObjectValue::F32(c.x as f32),
        ObjectValue::U8(u8::from(!m.kept())),
        ObjectValue::U8(m.class.code()),
        ObjectValue::U8(m.manual.code()),
        ObjectValue::F32(m.vertical_um as f32),
        ObjectValue::F32(m.euclidean_um as f32),
        ObjectValue::F32(m.relative_depth as f32),
        ObjectValue::F32(m.mucus_thickness_um as f32),
        ObjectValue::F32(m.h_mucus_um as f32),
        ObjectValue::F32(m.h_epi_um as f32),
        ObjectValue::F32(c.response as f32),
        ObjectValue::F32(c.raw_intensity as f32),
        ObjectValue::F32(c.z_support as f32),
        ObjectValue::F32(c.xy_support as f32),
        ObjectValue::U64(m.large_object as u64),
    ]
}

pub fn object_table_spec(volume: [usize; 3]) -> TableSpec {
    let mut columns = identity_and_centroid();
    columns.extend([
        col("deleted", DType::U8, ColumnRole::Category, None, "rejected"),
        ColumnSpec::new("volume_voxels", DType::U64, ColumnRole::Measurement),
        col(
            "volume_um3",
            DType::F32,
            ColumnRole::Measurement,
            Some("µm³"),
            "volume",
        ),
        col(
            "af488_mean",
            DType::F32,
            ColumnRole::Intensity,
            None,
            "AF488 mean",
        ),
        col(
            "cblue_mean",
            DType::F32,
            ColumnRole::Intensity,
            None,
            "C-Blue mean",
        ),
        col(
            "n_beads",
            DType::U32,
            ColumnRole::Measurement,
            None,
            "bead candidates touching",
        ),
    ]);
    let ext = json!({
        "version": 1,
        "role": "large_af488_objects",
        "description": "large bright AF488 objects in the mucus, probably dying cells; beads touching one are ignored",
        "deleted": "always 0 as written; newvolim review may set it",
    });
    spec(volume, columns, ext)
}

/// One large-object-table row.
pub fn object_values(o: &LargeObject) -> Vec<ObjectValue> {
    vec![
        ObjectValue::U64(o.id as u64),
        ObjectValue::F32(o.centroid[0] as f32),
        ObjectValue::F32(o.centroid[1] as f32),
        ObjectValue::F32(o.centroid[2] as f32),
        ObjectValue::U8(0),
        ObjectValue::U64(o.voxels),
        ObjectValue::F32(o.volume_um3 as f32),
        ObjectValue::F32(o.mean_af488 as f32),
        ObjectValue::F32(o.mean_blue as f32),
        ObjectValue::U32(o.n_beads),
    ]
}

/// NGFF `image-label` colours and names.
pub fn label_style(entries: &[(u32, &str, [u8; 4])]) -> Json {
    json!({
        "colors": entries.iter().map(|(v, _, c)| json!({"label-value": v, "rgba": c})).collect::<Vec<_>>(),
        "properties": entries.iter().map(|(v, n, _)| json!({"label-value": v, "name": n})).collect::<Vec<_>>(),
    })
}

pub fn layer_style() -> Json {
    label_style(&[
        (LABEL_MUCUS, "mucus", [255, 200, 0, 70]),
        (LABEL_TISSUE, "tissue", [220, 60, 220, 50]),
    ])
}

pub fn surface_style() -> Json {
    label_style(&[
        (1, "mucus top", [255, 230, 0, 255]),
        (2, "epithelium top", [255, 255, 255, 255]),
    ])
}
