// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Writing the results: the cell rows as an NGFF object table attached to the
//! image they were detected in (`<image>/tables/<name>/`, the layout a viewer
//! discovers), the per-region counts as CSV, and the density as a one-level
//! OME-Zarr image.

use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

use blockflow::error::{Error, Result};
use blockflow::object_table::{
    write_object_rows, ColumnRole, ColumnSpec, CoordinateColumn, DType, ObjectValue,
    SpatialIndexSpec, TableReader, TableSpec,
};
use blockflow::voxels::Voxels;
use blockflow::zarr_env::{PhysicalFrame, PyramidSpec};
use blockflow::ZarrEnvironment;
use serde_json::{Map, Value};

use atlas_regions::Ontology;

use crate::collapse::RegionRow;
use crate::detect::CellRow;

/// What turns a level-0 voxel index into a physical position.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageGeometry {
    /// Level-0 shape, `[z, y, x]`.
    pub shape_zyx: [usize; 3],
    /// Level-0 `scale`, `[z, y, x]`.
    pub scale_zyx: [f64; 3],
    /// Level-0 `translation`, `[z, y, x]`; zero when absent.
    pub translation_zyx: [f64; 3],
    /// The spatial axes' unit as the image declares it.
    pub unit: String,
}

impl ImageGeometry {
    /// NGFF's `translation + index * scale`: the voxel's centre.
    pub fn physical(&self, voxel_zyx: [f64; 3]) -> [f64; 3] {
        [0, 1, 2].map(|axis| self.translation_zyx[axis] + voxel_zyx[axis] * self.scale_zyx[axis])
    }
}

/// The spatial index's tile, `[z, y, x]`. Independent of the compute grid.
const TILE_ZYX: [u64; 3] = [32, 256, 256];

fn column(
    name: &str,
    dtype: DType,
    role: ColumnRole,
    unit: Option<&str>,
    display: &str,
) -> ColumnSpec {
    let mut spec = ColumnSpec::new(name, dtype, role);
    spec.unit = unit.map(str::to_string);
    spec.display_name = Some(display.to_string());
    spec
}

/// Which atlas columns a table carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtlasColumns {
    None,
    /// `region`, `hemisphere`, `distance_to_surface` (µm).
    Labels,
    /// `region`, `hemisphere`, `distance_to_surface_voxels` (atlas voxels)
    /// and the atlas position `atlas_z`, `atlas_y`, `atlas_x`.
    Registered,
}

impl AtlasColumns {
    /// The distance column's name: in µm on an annotation in the sample's
    /// frame, in atlas voxels through a registration.
    pub fn distance_column(self) -> &'static str {
        match self {
            Self::Registered => "distance_to_surface_voxels",
            _ => "distance_to_surface",
        }
    }

    fn of(rows: &[CellRow]) -> Self {
        if rows
            .iter()
            .any(|row| row.atlas.is_some_and(|atlas| atlas.atlas_at.is_some()))
        {
            Self::Registered
        } else if rows.iter().any(|row| row.atlas.is_some()) {
            Self::Labels
        } else {
            Self::None
        }
    }
}

/// The table's columns, in storage order. The atlas columns are present only
/// when `atlas` asks for them.
pub fn cell_columns(unit: &str, atlas: AtlasColumns) -> Vec<ColumnSpec> {
    let mut columns = vec![
        column(
            "label_id",
            DType::U64,
            ColumnRole::Identity,
            None,
            "label id",
        ),
        column(
            "z",
            DType::F64,
            ColumnRole::Coordinate,
            Some("voxel"),
            "z (level-0 voxel)",
        ),
        column(
            "y",
            DType::F64,
            ColumnRole::Coordinate,
            Some("voxel"),
            "y (level-0 voxel)",
        ),
        column(
            "x",
            DType::F64,
            ColumnRole::Coordinate,
            Some("voxel"),
            "x (level-0 voxel)",
        ),
        column(
            "z_um",
            DType::F64,
            ColumnRole::Measurement,
            Some(unit),
            "z (physical)",
        ),
        column(
            "y_um",
            DType::F64,
            ColumnRole::Measurement,
            Some(unit),
            "y (physical)",
        ),
        column(
            "x_um",
            DType::F64,
            ColumnRole::Measurement,
            Some(unit),
            "x (physical)",
        ),
        column(
            "size",
            DType::U64,
            ColumnRole::Measurement,
            Some("voxel"),
            "size (voxels)",
        ),
        column(
            "source",
            DType::F64,
            ColumnRole::Intensity,
            None,
            "raw intensity (max over basin)",
        ),
        column(
            "background",
            DType::F64,
            ColumnRole::Intensity,
            None,
            "background-removed intensity (max over basin)",
        ),
        column(
            "passes_filter",
            DType::U8,
            ColumnRole::Category,
            None,
            "passes filter",
        ),
    ];
    if atlas != AtlasColumns::None {
        columns.extend([
            column(
                "region",
                DType::I64,
                ColumnRole::Category,
                None,
                "atlas region id",
            ),
            column(
                "hemisphere",
                DType::I64,
                ColumnRole::Category,
                None,
                "hemisphere id",
            ),
            if atlas == AtlasColumns::Registered {
                column(
                    atlas.distance_column(),
                    DType::F64,
                    ColumnRole::Measurement,
                    Some("voxel"),
                    "distance to the brain surface (atlas voxels)",
                )
            } else {
                column(
                    atlas.distance_column(),
                    DType::F64,
                    ColumnRole::Measurement,
                    Some("micrometer"),
                    "distance to the brain surface",
                )
            },
        ]);
    }
    if atlas == AtlasColumns::Registered {
        for axis in ["z", "y", "x"] {
            columns.push(column(
                &format!("atlas_{axis}"),
                DType::F64,
                ColumnRole::Measurement,
                Some("voxel"),
                &format!("{axis} (atlas level-0 voxel)"),
            ));
        }
    }
    columns
}

fn values(
    row: &CellRow,
    origin: [usize; 3],
    geometry: &ImageGeometry,
    atlas: AtlasColumns,
) -> Vec<ObjectValue> {
    let global = [0, 1, 2].map(|axis| (row.at[axis] + origin[axis]) as f64);
    let physical = geometry.physical(global);
    let mut values = vec![
        ObjectValue::U64(row.label_id),
        ObjectValue::F64(global[0]),
        ObjectValue::F64(global[1]),
        ObjectValue::F64(global[2]),
        ObjectValue::F64(physical[0]),
        ObjectValue::F64(physical[1]),
        ObjectValue::F64(physical[2]),
        ObjectValue::U64(row.size),
        ObjectValue::F64(row.source),
        ObjectValue::F64(row.background),
        ObjectValue::U8(u8::from(row.passes_filter)),
    ];
    let columns = row.atlas.unwrap_or(crate::detect::CellAtlas {
        region: 0,
        hemisphere: -1,
        distance_to_surface: f64::NAN,
        atlas_at: None,
    });
    if atlas != AtlasColumns::None {
        values.extend([
            ObjectValue::I64(columns.region),
            ObjectValue::I64(columns.hemisphere),
            ObjectValue::F64(columns.distance_to_surface),
        ]);
    }
    if atlas == AtlasColumns::Registered {
        let at = columns.atlas_at.unwrap_or([f64::NAN; 3]);
        values.extend(at.map(ObjectValue::F64));
    }
    values
}

/// Write `rows` as an object table at `root` (which must not exist), with
/// `attributes` merged into its root metadata, a `table.csv` of the same rows
/// beside the arrays, and `region` naming the label image when there is one.
///
/// `origin` is the processed volume's offset in the image, so coordinates are
/// level-0 voxels of the whole image and a crop's table lands where the crop
/// was taken.
pub fn write_cell_table(
    root: &Path,
    rows: &[CellRow],
    origin: [usize; 3],
    geometry: &ImageGeometry,
    region: Option<&str>,
    attributes: &Map<String, Value>,
) -> Result<TableReader> {
    write_cell_table_named(root, rows, origin, geometry, region, attributes, None)
}

/// [`write_cell_table`], with the region tree that names the `region`
/// column: its acronym and name are added to `table.csv` (the object table
/// holds numbers only; a viewer names regions from the atlas's label
/// properties).
pub fn write_cell_table_named(
    root: &Path,
    rows: &[CellRow],
    origin: [usize; 3],
    geometry: &ImageGeometry,
    region: Option<&str>,
    attributes: &Map<String, Value>,
    ontology: Option<&Ontology>,
) -> Result<TableReader> {
    if root.exists() {
        return Err(Error::invalid(format!(
            "{} exists; it is not overwritten here",
            root.display()
        )));
    }
    let shape = geometry.shape_zyx;
    for row in rows {
        if (0..3).any(|axis| row.at[axis] + origin[axis] >= shape[axis]) {
            return Err(Error::invalid(format!(
                "cell {} at [z, y, x] {:?} + {origin:?} is outside the image's {shape:?}",
                row.label_id, row.at
            )));
        }
    }
    let atlas = AtlasColumns::of(rows);
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
    let mut spec = TableSpec::new(
        0,
        4096,
        "../../",
        "label_id",
        cell_columns(&geometry.unit, atlas),
        spatial_index,
    );
    spec.region = region.map(str::to_string);
    let projected = rows
        .iter()
        .map(|row| values(row, origin, geometry, atlas))
        .collect();
    let table = write_object_rows(root, spec, projected)?;
    merge_root_attributes(root, attributes)?;
    write_csv(&table, &root.join("table.csv"), atlas, ontology)?;
    Ok(table)
}

fn merge_root_attributes(root: &Path, extra: &Map<String, Value>) -> Result<()> {
    let path = root.join("zarr.json");
    let mut metadata: Value =
        serde_json::from_str(&fs::read_to_string(&path).map_err(Error::backend)?)
            .map_err(Error::backend)?;
    let attributes = metadata
        .get_mut("attributes")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| Error::invalid("an object table root has no attributes"))?;
    for (key, value) in extra {
        attributes.insert(key.clone(), value.clone());
    }
    write_json(&path, &metadata)
}

fn write_json(path: &Path, value: &Value) -> Result<()> {
    let staged = path.with_extension("json.tmp");
    fs::write(
        &staged,
        serde_json::to_string_pretty(value).map_err(Error::backend)?,
    )
    .map_err(Error::backend)?;
    fs::rename(&staged, path).map_err(Error::backend)
}

/// The rows as CSV, read back from the published table so the two cannot
/// disagree.
fn write_csv(
    table: &TableReader,
    path: &Path,
    atlas: AtlasColumns,
    ontology: Option<&Ontology>,
) -> Result<()> {
    let n = table.spec().row_count;
    let ids = table.read_u64("label_id", 0..n).map_err(Error::backend)?;
    let names = [
        "z",
        "y",
        "x",
        "z_um",
        "y_um",
        "x_um",
        "source",
        "background",
    ];
    let floats = names
        .iter()
        .map(|name| table.read_f64(name, 0..n).map_err(Error::backend))
        .collect::<Result<Vec<_>>>()?;
    let sizes = table.read_u64("size", 0..n).map_err(Error::backend)?;
    let flags = table
        .read_u8("passes_filter", 0..n)
        .map_err(Error::backend)?;
    let atlas_columns = if atlas != AtlasColumns::None {
        Some((
            table.read_i64("region", 0..n).map_err(Error::backend)?,
            table.read_i64("hemisphere", 0..n).map_err(Error::backend)?,
            table
                .read_f64(atlas.distance_column(), 0..n)
                .map_err(Error::backend)?,
        ))
    } else {
        None
    };
    let positions = if atlas == AtlasColumns::Registered {
        Some(
            ["atlas_z", "atlas_y", "atlas_x"]
                .iter()
                .map(|name| table.read_f64(name, 0..n).map_err(Error::backend))
                .collect::<Result<Vec<_>>>()?,
        )
    } else {
        None
    };
    let named = ontology.filter(|_| atlas != AtlasColumns::None);
    let mut out = BufWriter::new(fs::File::create(path).map_err(Error::backend)?);
    write!(out, "label_id,{},size,passes_filter", names.join(",")).map_err(Error::backend)?;
    if atlas != AtlasColumns::None {
        write!(out, ",region,hemisphere,{}", atlas.distance_column()).map_err(Error::backend)?;
    }
    if named.is_some() {
        write!(out, ",region_acronym,region_name").map_err(Error::backend)?;
    }
    if positions.is_some() {
        write!(out, ",atlas_z,atlas_y,atlas_x").map_err(Error::backend)?;
    }
    writeln!(out).map_err(Error::backend)?;
    for row in 0..ids.len() {
        write!(out, "{}", ids[row]).map_err(Error::backend)?;
        for column in &floats {
            write!(out, ",{}", column[row]).map_err(Error::backend)?;
        }
        write!(out, ",{},{}", sizes[row], flags[row]).map_err(Error::backend)?;
        if let Some((regions, hemispheres, distances)) = &atlas_columns {
            write!(
                out,
                ",{},{},{}",
                regions[row], hemispheres[row], distances[row]
            )
            .map_err(Error::backend)?;
            if let Some(tree) = named {
                write!(
                    out,
                    ",{},{}",
                    csv_field(tree.acronym(regions[row]).unwrap_or("")),
                    csv_field(tree.name(regions[row]).unwrap_or(""))
                )
                .map_err(Error::backend)?;
            }
        }
        if let Some(positions) = &positions {
            for column in positions {
                write!(out, ",{}", column[row]).map_err(Error::backend)?;
            }
        }
        writeln!(out).map_err(Error::backend)?;
    }
    out.flush().map_err(Error::backend)
}

fn csv_field(text: &str) -> String {
    if text.contains([',', '"', '\n']) {
        format!("\"{}\"", text.replace('"', "\"\""))
    } else {
        text.to_string()
    }
}

/// The per-region table as CSV. Empty fields are absent values.
pub fn write_region_csv(path: &Path, rows: &[RegionRow]) -> Result<()> {
    let optional = |value: Option<f64>| value.map(|v| v.to_string()).unwrap_or_default();
    let mut out = BufWriter::new(fs::File::create(path).map_err(Error::backend)?);
    writeln!(
        out,
        "region,hemisphere,acronym,name,cells,mean_size,volume_voxels,volume_mm3,density_per_mm3"
    )
    .map_err(Error::backend)?;
    for row in rows {
        writeln!(
            out,
            "{},{},{},{},{},{},{},{},{}",
            row.region,
            row.hemisphere,
            csv_field(row.acronym.as_deref().unwrap_or("")),
            csv_field(row.name.as_deref().unwrap_or("")),
            row.cells,
            optional(row.mean_size),
            row.volume_voxels.map(|v| v.to_string()).unwrap_or_default(),
            optional(row.volume_mm3),
            optional(row.density_per_mm3),
        )
        .map_err(Error::backend)?;
    }
    out.flush().map_err(Error::backend)
}

/// Write `voxels` as a one-level OME-Zarr image at `root` (which must not
/// exist), with level-0 `scale` and `translation` (`[z, y, x]`, in `unit`).
pub fn write_ome_image(
    root: &Path,
    voxels: &Voxels,
    scale: [f64; 3],
    translation: [f64; 3],
    unit: &str,
    attributes: &Map<String, Value>,
) -> Result<()> {
    if root.exists() {
        return Err(Error::invalid(format!(
            "{} exists; it is not overwritten here",
            root.display()
        )));
    }
    let shape = voxels.shape();
    let chunk = [0, 1, 2].map(|axis| shape[axis].clamp(1, 128));
    let frame = PhysicalFrame::new(scale, Some(unit)).with_origin(translation);
    let spec = PyramidSpec::new(Vec::new())?
        .with_chunk(chunk)
        .with_physical(frame);
    ZarrEnvironment::build_multiscale(root, voxels, &spec)?;
    // Carry the caller's attributes beside the `ome` block the pyramid wrote.
    let path = root.join("zarr.json");
    let mut metadata: Value = serde_json::from_slice(&fs::read(&path).map_err(Error::backend)?)
        .map_err(Error::backend)?;
    let written = metadata["attributes"]
        .as_object_mut()
        .ok_or_else(|| Error::invalid(format!("{} has no attributes object", path.display())))?;
    for (key, value) in attributes {
        if key != "ome" {
            written.insert(key.clone(), value.clone());
        }
    }
    write_json(&path, &metadata)
}

/// Write `density` as a one-level OME-Zarr image at `root`, on a lattice
/// `factor` times coarser than the image's level 0, as `float32`.
pub fn write_density_image(
    root: &Path,
    density: &Voxels,
    factor: [f64; 3],
    geometry: &ImageGeometry,
    attributes: &Map<String, Value>,
) -> Result<()> {
    let narrow = Voxels::F32(density.view::<f64>()?.mapv(|value| value as f32));
    // A coarse voxel covers `factor` fine ones, so its centre sits half a
    // coarse voxel minus half a fine one past the first fine centre.
    let scale = [0, 1, 2].map(|axis| geometry.scale_zyx[axis] * factor[axis]);
    let translation = [0, 1, 2].map(|axis| {
        geometry.translation_zyx[axis] + 0.5 * (factor[axis] - 1.0) * geometry.scale_zyx[axis]
    });
    write_ome_image(
        root,
        &narrow,
        scale,
        translation,
        &geometry.unit,
        attributes,
    )
}

/// Read the level-0 geometry of the first multiscale image at `root`.
pub fn image_geometry(root: &Path, shape_zyx: [usize; 3]) -> Result<ImageGeometry> {
    let metadata: Value =
        serde_json::from_str(&fs::read_to_string(root.join("zarr.json")).map_err(Error::backend)?)
            .map_err(Error::backend)?;
    let multiscale = metadata
        .pointer("/attributes/ome/multiscales/0")
        .or_else(|| metadata.pointer("/attributes/multiscales/0"))
        .ok_or_else(|| Error::invalid(format!("{} has no multiscales", root.display())))?;
    let axes: Vec<Value> = multiscale
        .get("axes")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let name = |axis: &Value| {
        axis.as_str()
            .or_else(|| axis.get("name").and_then(Value::as_str))
            .unwrap_or("")
            .to_ascii_lowercase()
    };
    let index = |wanted: &str| axes.iter().position(|axis| name(axis) == wanted);
    let spatial = [index("z"), index("y"), index("x")];
    let unit = index("x")
        .and_then(|at| axes[at].get("unit"))
        .and_then(Value::as_str)
        .unwrap_or("voxel")
        .to_string();
    let mut scale = [1.0; 3];
    let mut translation = [0.0; 3];
    if let Some(transforms) = multiscale
        .pointer("/datasets/0/coordinateTransformations")
        .and_then(Value::as_array)
    {
        for transform in transforms {
            let key = match transform.get("type").and_then(Value::as_str) {
                Some("scale") => "scale",
                Some("translation") => "translation",
                _ => continue,
            };
            let Some(vector) = transform.get(key).and_then(Value::as_array) else {
                continue;
            };
            for (slot, axis) in spatial.iter().enumerate() {
                if let Some(value) = axis.and_then(|at| vector.get(at)).and_then(Value::as_f64) {
                    if key == "scale" {
                        scale[slot] = value;
                    } else {
                        translation[slot] = value;
                    }
                }
            }
        }
    }
    Ok(ImageGeometry {
        shape_zyx,
        scale_zyx: scale,
        translation_zyx: translation,
        unit,
    })
}

/// Seconds since the epoch as an ISO 8601 UTC timestamp.
pub fn utc_timestamp(seconds: u64) -> String {
    let days = (seconds / 86_400) as i64;
    let rem = seconds % 86_400;
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        (rem / 60) % 60,
        rem % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_timestamp_is_civil() {
        assert_eq!(utc_timestamp(0), "1970-01-01T00:00:00Z");
        assert_eq!(utc_timestamp(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(utc_timestamp(1_790_000_000), "2026-09-21T14:13:20Z");
    }

    #[test]
    fn a_field_with_a_comma_is_quoted() {
        assert_eq!(csv_field("Field CA1, layer"), "\"Field CA1, layer\"");
        assert_eq!(csv_field("CA1"), "CA1");
    }
}
