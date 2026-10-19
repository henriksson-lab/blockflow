// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The writers: the object table reads back in whole-image voxels and
//! physical units, refuses to overwrite, and carries its provenance; the
//! geometry comes from the image; the region CSV and density image are what
//! they say.

use std::path::PathBuf;

use blockflow::object_table::TableReader;
use blockflow::ome_zarr::OmeVolumePyramid;
use blockflow::voxels::Voxels;
use blockflow::AttachedImage;
use cell_density::collapse::RegionRow;
use cell_density::detect::{CellAtlas, CellRow};
use cell_density::table::{
    image_geometry, write_cell_table, write_density_image, write_region_csv, ImageGeometry,
};
use ndarray::Array3;
use serde_json::{json, Map};

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("cell-density-table-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn geometry() -> ImageGeometry {
    ImageGeometry {
        shape_zyx: [70, 600, 100],
        scale_zyx: [6.0, 1.625, 1.625],
        translation_zyx: [3.0, 0.0, 10.0],
        unit: "micrometer".into(),
    }
}

fn row(label_id: u64, at: [usize; 3], size: u64, passes_filter: bool) -> CellRow {
    CellRow {
        label_id,
        at,
        size,
        source: 1000.0 + size as f64,
        background: 500.0 + size as f64,
        passes_filter,
        atlas: None,
    }
}

#[test]
fn the_table_reads_back_in_whole_image_voxels() {
    let dir = scratch("readback");
    let root = dir.join("tables").join("cells");
    let rows = vec![
        row(7, [2, 10, 60], 51, true),
        row(2, [0, 0, 0], 3, false),
        row(9, [59, 299, 1], 120, true),
    ];
    let mut attributes = Map::new();
    attributes.insert("cell_density".into(), json!({"tool": "test"}));
    let table =
        write_cell_table(&root, &rows, [5, 300, 30], &geometry(), None, &attributes).unwrap();
    let n = table.spec().row_count;
    assert_eq!(n, 3);
    let reopened = TableReader::open(&root).unwrap();
    let ids = reopened.read_u64("label_id", 0..n).unwrap();
    let z = reopened.read_f64("z", 0..n).unwrap();
    let y = reopened.read_f64("y", 0..n).unwrap();
    let x = reopened.read_f64("x", 0..n).unwrap();
    let z_um = reopened.read_f64("z_um", 0..n).unwrap();
    let x_um = reopened.read_f64("x_um", 0..n).unwrap();
    let size = reopened.read_u64("size", 0..n).unwrap();
    let flag = reopened.read_u8("passes_filter", 0..n).unwrap();
    for i in 0..n as usize {
        let original = rows.iter().find(|r| r.label_id == ids[i]).unwrap();
        assert_eq!(z[i], (original.at[0] + 5) as f64);
        assert_eq!(y[i], (original.at[1] + 300) as f64);
        assert_eq!(x[i], (original.at[2] + 30) as f64);
        assert_eq!(z_um[i], 3.0 + z[i] * 6.0);
        assert_eq!(x_um[i], 10.0 + x[i] * 1.625);
        assert_eq!(size[i], original.size);
        assert_eq!(flag[i], u8::from(original.passes_filter));
    }
    let spec = reopened.spec();
    assert_eq!(spec.identity, "label_id");
    let axes: Vec<_> = spec
        .spatial_index
        .coordinates
        .iter()
        .map(|c| (c.axis.as_str(), c.column.as_str()))
        .collect();
    assert_eq!(axes, [("z", "z"), ("y", "y"), ("x", "x")]);
    assert!(spec.columns.iter().all(|column| column.name != "region"));
    let metadata: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(root.join("zarr.json")).unwrap()).unwrap();
    assert_eq!(metadata["attributes"]["cell_density"]["tool"], "test");
    assert!(metadata["attributes"]["object_table"].is_object());
    let csv = std::fs::read_to_string(root.join("table.csv")).unwrap();
    let mut lines = csv.lines();
    assert_eq!(
        lines.next(),
        Some("label_id,z,y,x,z_um,y_um,x_um,source,background,size,passes_filter")
    );
    assert_eq!(lines.count(), 3);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn atlas_columns_are_written_when_the_rows_carry_them() {
    let dir = scratch("atlas");
    let root = dir.join("cells");
    let mut rows = vec![row(1, [1, 1, 1], 60, true), row(2, [2, 2, 2], 70, false)];
    rows[0].atlas = Some(CellAtlas {
        region: 315,
        hemisphere: 255,
        distance_to_surface: 125.0,
        atlas_at: None,
    });
    write_cell_table(
        &root,
        &rows,
        [0; 3],
        &geometry(),
        Some("../../labels/cells"),
        &Map::new(),
    )
    .unwrap();
    let table = TableReader::open(&root).unwrap();
    assert_eq!(table.spec().region.as_deref(), Some("../../labels/cells"));
    let regions = table.read_i64("region", 0..2).unwrap();
    let ids = table.read_u64("label_id", 0..2).unwrap();
    let distances = table.read_f64("distance_to_surface", 0..2).unwrap();
    for i in 0..2 {
        if ids[i] == 1 {
            assert_eq!((regions[i], distances[i]), (315, 125.0));
        } else {
            assert_eq!(regions[i], 0, "a row with no lookup is outside the atlas");
            assert!(distances[i].is_nan());
        }
    }
    let header = std::fs::read_to_string(root.join("table.csv")).unwrap();
    assert!(header
        .lines()
        .next()
        .unwrap()
        .ends_with(",region,hemisphere,distance_to_surface"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_published_table_is_not_overwritten_and_a_stray_row_is_refused() {
    let dir = scratch("refuse");
    let root = dir.join("cells");
    let rows = vec![row(1, [0, 0, 0], 60, true)];
    write_cell_table(&root, &rows, [0; 3], &geometry(), None, &Map::new()).unwrap();
    assert!(write_cell_table(&root, &rows, [0; 3], &geometry(), None, &Map::new()).is_err());
    let outside = vec![row(1, [70, 0, 0], 60, true)];
    assert!(write_cell_table(
        &dir.join("other"),
        &outside,
        [0; 3],
        &geometry(),
        None,
        &Map::new()
    )
    .is_err());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn the_geometry_is_read_from_the_image() {
    let dir = scratch("geometry");
    let metadata = json!({
        "zarr_format": 3, "node_type": "group",
        "attributes": {"ome": {"version": "0.5", "multiscales": [{
            "axes": [
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"}
            ],
            "datasets": [{"path": "0", "coordinateTransformations": [
                {"type": "scale", "scale": [1.0, 1.6, 1.625, 1.625]},
                {"type": "translation", "translation": [0.0, 8.0, 4.0, 2.0]}
            ]}]
        }]}}
    });
    std::fs::write(dir.join("zarr.json"), metadata.to_string()).unwrap();
    let geometry = image_geometry(&dir, [10, 20, 30]).unwrap();
    assert_eq!(geometry.scale_zyx, [1.6, 1.625, 1.625]);
    assert_eq!(geometry.translation_zyx, [8.0, 4.0, 2.0]);
    assert_eq!(geometry.unit, "micrometer");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn the_region_csv_quotes_names_and_leaves_absences_empty() {
    let dir = scratch("regions");
    let path = dir.join("regions.csv");
    write_region_csv(
        &path,
        &[
            RegionRow {
                region: 382,
                hemisphere: 0,
                name: Some("Field CA1, stratum".into()),
                acronym: Some("CA1".into()),
                cells: 4,
                mean_size: Some(80.5),
                volume_voxels: Some(128),
                volume_mm3: Some(2.0),
                density_per_mm3: Some(2.0),
            },
            RegionRow {
                region: 7,
                hemisphere: -1,
                name: None,
                acronym: None,
                cells: 0,
                mean_size: None,
                volume_voxels: None,
                volume_mm3: None,
                density_per_mm3: None,
            },
        ],
    )
    .unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(
        lines,
        [
            "region,hemisphere,acronym,name,cells,mean_size,volume_voxels,volume_mm3,density_per_mm3",
            "382,0,CA1,\"Field CA1, stratum\",4,80.5,128,2,2",
            "7,-1,,,0,,,,",
        ]
    );
    std::fs::remove_dir_all(dir).unwrap();
}

/// The density image opens as an OME-Zarr volume whose voxel centres sit over
/// the fine voxels they summarise.
#[test]
fn the_density_image_is_an_ome_zarr_volume_on_the_coarse_lattice() {
    let dir = scratch("density");
    let root = dir.join("density.ome.zarr");
    let mut values = Array3::<f64>::zeros((3, 4, 5));
    values[[1, 2, 3]] = 7.0;
    write_density_image(
        &root,
        &Voxels::F64(values),
        [2.0, 4.0, 4.0],
        &geometry(),
        &Map::new(),
    )
    .unwrap();
    let pyramid = OmeVolumePyramid::open(&root, 0, 0).unwrap();
    assert_eq!(pyramid.levels()[0].shape, [3, 4, 5]);
    assert_eq!(pyramid.voxel_size(), [12.0, 6.5, 6.5]);
    let back = AttachedImage::at(root.join("0")).read().unwrap();
    assert_eq!(back.view::<f32>().unwrap()[[1, 2, 3]], 7.0);
    let reopened = image_geometry(&root, [3, 4, 5]).unwrap();
    // Coarse voxel 0 covers fine voxels 0 and 1 in z, centred at 0.5.
    assert_eq!(
        reopened.translation_zyx,
        [3.0 + 0.5 * 6.0, 1.5 * 1.625, 10.0 + 1.5 * 1.625]
    );
    std::fs::remove_dir_all(dir).unwrap();
}
