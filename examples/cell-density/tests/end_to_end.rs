// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The command line on a synthetic OME-Zarr image with an annotation in its
//! frame: what is published, that the label image and the table agree, and
//! that two block grids give byte-identical tables.

use std::fs;
use std::path::{Path, PathBuf};

use blockflow::object_table::TableReader;
use blockflow::ome_zarr::OmeVolumePyramid;
use blockflow::synthetic::{Scene, SceneSpec};
use blockflow::voxels::Voxels;
use blockflow::AttachedImage;
use cell_density::cli::{run, Args};
use cell_density::table::write_ome_image;
use clap::Parser;
use ndarray::Array3;
use serde_json::Map;

const SHAPE: [usize; 3] = [32, 64, 64];
const VOXEL_UM: [f64; 3] = [4.0, 2.0, 2.0];

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("cell-density-e2e-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// An image of 24 objects in raw counts over a floor of 100, an annotation
/// on a 2 x 4 x 4 coarser grid (region 10 in the left half of `x`, 20 in the
/// right, 0 in a one-voxel rim), hemispheres, and a tree.
fn fixture(dir: &Path) -> (PathBuf, PathBuf, PathBuf, PathBuf) {
    let scene = Scene::new(
        SceneSpec::clean(SHAPE, 5)
            .with_objects(24)
            .with_radius(2.5, 3.5),
    )
    .unwrap();
    let counts = scene
        .render()
        .intensity
        .mapv(|value| (100.0 + value * 2048.0).round() as u16);
    let image = dir.join("image.ome.zarr");
    write_ome_image(
        &image,
        &Voxels::U16(counts),
        VOXEL_UM,
        [0.0; 3],
        "micrometer",
        &Map::new(),
    )
    .unwrap();

    let atlas_shape = [16, 16, 16];
    let annotation = Array3::<u16>::from_shape_fn(atlas_shape, |(z, y, x)| {
        let rim = [z, y, x].iter().any(|c| *c == 0 || *c == 15);
        if rim {
            0
        } else if x < 8 {
            10
        } else {
            20
        }
    });
    let hemispheres =
        Array3::<u8>::from_shape_fn(atlas_shape, |(_, _, x)| if x < 8 { 0 } else { 255 });
    let atlas = VOXEL_UM.map(|um| um * 4.0);
    let annotation_root = dir.join("annotation.ome.zarr");
    write_ome_image(
        &annotation_root,
        &Voxels::U16(annotation),
        atlas,
        [0.0; 3],
        "micrometer",
        &Map::new(),
    )
    .unwrap();
    let hemisphere_root = dir.join("hemispheres.ome.zarr");
    write_ome_image(
        &hemisphere_root,
        &Voxels::U8(hemispheres),
        atlas,
        [0.0; 3],
        "micrometer",
        &Map::new(),
    )
    .unwrap();
    let tree = dir.join("tree.jsonl");
    fs::write(
        &tree,
        "{\"id\": 1, \"st_level\": 0, \"parent_structure_id\": null, \"name\": \"root\", \"acronym\": \"root\"}\n\
         {\"id\": 10, \"st_level\": 1, \"parent_structure_id\": 1, \"name\": \"left, west\", \"acronym\": \"L\"}\n\
         {\"id\": 20, \"st_level\": 1, \"parent_structure_id\": 1, \"name\": \"right\", \"acronym\": \"R\"}\n",
    )
    .unwrap();
    (
        image,
        annotation_root.join("0"),
        hemisphere_root.join("0"),
        tree,
    )
}

fn args(
    image: &Path,
    annotation: &Path,
    hemispheres: &Path,
    tree: &Path,
    block: &str,
    density: &Path,
) -> Args {
    Args::parse_from([
        "cell-density",
        "--input",
        image.to_str().unwrap(),
        "--in-place",
        "--block",
        block,
        "--labels",
        "--overwrite",
        "--size-min",
        "20",
        "--annotation",
        annotation.to_str().unwrap(),
        "--hemispheres",
        hemispheres.to_str().unwrap(),
        "--ontology",
        tree.to_str().unwrap(),
        "--all-regions",
        "--crust-um",
        "8",
        "--density",
        density.to_str().unwrap(),
        "--density-radius-um",
        "16",
    ])
}

#[test]
fn the_command_publishes_cells_labels_regions_and_density() {
    let dir = scratch("publish");
    let (image, annotation, hemispheres, tree) = fixture(&dir);
    let density = dir.join("density.ome.zarr");
    let report = run(&args(
        &image,
        &annotation,
        &hemispheres,
        &tree,
        "16,32,32",
        &density,
    ))
    .unwrap();
    let cells = report["cells"]["count"].as_u64().unwrap();
    let filtered = report["filtered"]["count"].as_u64().unwrap();
    assert!(cells >= 20, "{report}");
    assert!(
        filtered > 0 && filtered < cells,
        "the crust and the size band must bite: {report}"
    );

    // The full table is paired with the label image, and every label value is
    // a row's identity.
    let table = TableReader::open(image.join("tables/cell-density")).unwrap();
    assert_eq!(table.spec().row_count, cells);
    assert_eq!(
        table.spec().region.as_deref(),
        Some("../../labels/cell-density")
    );
    let ids = table.read_u64("label_id", 0..cells).unwrap();
    let labels = AttachedImage::at(image.join("labels/cell-density/0"))
        .read()
        .unwrap();
    let labels = labels.view::<u64>().unwrap();
    let present: std::collections::BTreeSet<u64> =
        labels.iter().copied().filter(|label| *label != 0).collect();
    assert_eq!(present, ids.iter().copied().collect());
    let sizes = table.read_u64("size", 0..cells).unwrap();
    for (id, size) in ids.iter().zip(&sizes) {
        assert_eq!(
            labels.iter().filter(|label| **label == *id).count() as u64,
            *size
        );
    }
    // The atlas columns: the two halves of x are the two regions.
    let x = table.read_f64("x", 0..cells).unwrap();
    let regions = table.read_i64("region", 0..cells).unwrap();
    for (x, region) in x.iter().zip(&regions) {
        if *region != 0 {
            assert_eq!(*region, if *x < 32.0 { 10 } else { 20 }, "x = {x}");
        }
    }

    // The filtered table holds exactly the passing rows.
    let subset = TableReader::open(image.join("tables/cell-density-filtered")).unwrap();
    assert_eq!(subset.spec().row_count, filtered);
    assert!(subset
        .read_u8("passes_filter", 0..filtered)
        .unwrap()
        .iter()
        .all(|flag| *flag == 1));

    // The region table: every tree id on both hemispheres, the filtered cells
    // counted once.
    let regions = fs::read_to_string(image.join("tables/cell-density/regions.csv")).unwrap();
    let lines: Vec<&str> = regions.lines().collect();
    assert!(lines[0].starts_with("region,hemisphere,acronym,name,cells"));
    assert!(regions.contains("\"left, west\""));
    let counted: u64 = lines[1..]
        .iter()
        .map(|line| {
            line.split(',')
                .nth(4)
                .unwrap()
                .parse::<u64>()
                .unwrap_or_else(|_| {
                    // A quoted name with a comma shifts the fields by one.
                    line.split(',').nth(5).unwrap().parse().unwrap()
                })
        })
        .sum();
    assert_eq!(counted, filtered);

    // The density image is on the annotation's grid and carries one ball per
    // filtered cell.
    let pyramid = OmeVolumePyramid::open(&density, 0, 0).unwrap();
    assert_eq!(pyramid.levels()[0].shape, [16, 16, 16]);
    let values = AttachedImage::at(density.join("0")).read().unwrap();
    assert!(values
        .view::<f32>()
        .unwrap()
        .iter()
        .any(|value| *value > 0.0));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn two_block_grids_publish_identical_tables() {
    let dir = scratch("grids");
    let (image, annotation, hemispheres, tree) = fixture(&dir);
    let mut tables = Vec::new();
    for block in ["32,64,64", "8,24,40"] {
        let density = dir.join(format!("density-{block}"));
        run(&args(
            &image,
            &annotation,
            &hemispheres,
            &tree,
            block,
            &density,
        ))
        .unwrap();
        tables.push([
            fs::read_to_string(image.join("tables/cell-density/table.csv")).unwrap(),
            fs::read_to_string(image.join("tables/cell-density-filtered/table.csv")).unwrap(),
            fs::read_to_string(image.join("tables/cell-density/regions.csv")).unwrap(),
        ]);
    }
    assert_eq!(tables[0], tables[1]);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn an_existing_output_is_refused_without_overwrite() {
    let dir = scratch("refuse");
    let (image, ..) = fixture(&dir);
    let base = [
        "cell-density",
        "--input",
        image.to_str().unwrap(),
        "--in-place",
    ];
    run(&Args::parse_from(base)).unwrap();
    let failed = run(&Args::parse_from(base)).unwrap_err();
    assert!(failed.to_string().contains("--overwrite"), "{failed}");
    // No output named at all is refused rather than written into the input.
    let failed = run(&Args::parse_from([
        "cell-density",
        "--input",
        image.to_str().unwrap(),
    ]))
    .unwrap_err();
    assert!(failed.to_string().contains("--in-place"), "{failed}");
    fs::remove_dir_all(dir).unwrap();
}
