// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Cells through a synthetic registration into a synthetic prepared atlas,
//! where every expected atlas coordinate is known in closed form:
//!
//! - detection image: 4 x 2 x 2 µm voxels at the origin;
//! - registration sample (first fixed image): 8 µm voxels translated by
//!   4 µm, so the detection-to-sample scale and offset are not trivial;
//! - `--sample-offset-um 16,-8,24` places the detection image in it;
//! - one elastix affine `out = A p + t` (physical, no rotation centre) into
//!   a 16^3 atlas of 10 µm voxels.
//!
//! So a detection voxel `i` is expected at atlas voxel
//! `(A (i * s_d + o) + t) / 10`, and its labels are the atlas's at that
//! voxel; the test checks the coordinates, labels, the distance in atlas
//! voxels, the crust in voxels and in µm, region volumes, density grid, a
//! two-step chain, and the refusal of a mismatched atlas.

use std::fs;
use std::path::{Path, PathBuf};

use atlas_regions::atlas_zarr::{write_prepared_atlas, WriteAtlasParams};
use atlas_regions::{AtlasStack, Ontology, OntologyLevel, Orientation, PreparedWindow, Rounding};
use blockflow::object_table::TableReader;
use blockflow::synthetic::{Scene, SceneSpec};
use blockflow::voxels::Voxels;
use blockflow::AttachedImage;
use blockflow_atlas_registration::atlas_space::{ImageFrame, OpenParams, RegisteredAtlas};
use cell_density::cli::{run, Args};
use cell_density::table::{image_geometry, write_ome_image};
use clap::Parser;
use ndarray::Array3;
use serde_json::{json, Map};

const SHAPE: [usize; 3] = [32, 64, 64];
const VOXEL_UM: [f64; 3] = [4.0, 2.0, 2.0];
const SAMPLE_SHAPE: [usize; 3] = [20, 20, 20];
const SAMPLE_UM: f64 = 8.0;
const SAMPLE_TRANSLATION: [f64; 3] = [4.0, 4.0, 4.0];
const OFFSET_UM: [f64; 3] = [16.0, -8.0, 24.0];
const ATLAS: usize = 16;
const ATLAS_UM: f64 = 10.0;
const A: [f64; 3] = [1.0, 0.8, 1.25];
const T: [f64; 3] = [5.0, 12.0, -7.0];

const TREE: &str = r#"{"msg": [{"id": 0, "name": "universe", "acronym": "universe",
  "st_level": null, "children": [
  {"id": 997, "name": "root", "acronym": "root", "st_level": 0, "children": [
    {"id": 10, "name": "left, west", "acronym": "L", "st_level": 1, "children": [
      {"id": 11, "name": "left upper", "acronym": "L1", "st_level": 2, "children": []}]},
    {"id": 20, "name": "right", "acronym": "R", "st_level": 1, "children": []}]}]}]}"#;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "cell-density-registered-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// Region by atlas voxel: a one-voxel rim of 0, `x < 8` left (11 where also
/// `y < 6`, else 10), `x >= 8` right (20).
fn region(z: usize, y: usize, x: usize) -> u32 {
    let last = ATLAS - 1;
    if [z, y, x].iter().any(|c| *c == 0 || *c == last) {
        0
    } else if x < 8 {
        if y < 6 {
            11
        } else {
            10
        }
    } else {
        20
    }
}

/// Hemisphere by atlas `z`: 255 below 8, 0 above (0 is a hemisphere).
fn hemisphere(z: usize) -> u32 {
    if z < 8 {
        255
    } else {
        0
    }
}

/// Distance to the rim in voxels, as the reference ships its field.
fn distance(z: usize, y: usize, x: usize) -> f32 {
    let last = ATLAS - 1;
    [z, y, x]
        .iter()
        .map(|c| (*c).min(last - *c) as f32)
        .fold(f32::INFINITY, f32::min)
}

fn write_atlas(root: &Path, shape: [usize; 3]) {
    let s = (shape[0], shape[1], shape[2]);
    let stack = AtlasStack {
        annotation: Voxels::U32(Array3::from_shape_fn(s, |(z, y, x)| region(z, y, x))),
        reference: Some(Voxels::U16(Array3::from_shape_fn(s, |(z, y, x)| {
            (z + y + x) as u16
        }))),
        hemispheres: Some(Voxels::U32(Array3::from_shape_fn(s, |(z, _, _)| {
            hemisphere(z)
        }))),
        distance_to_surface: Some(Voxels::F32(Array3::from_shape_fn(s, |(z, y, x)| {
            distance(z, y, x)
        }))),
    };
    let window = PreparedWindow {
        orientation: Orientation::IDENTITY,
        source_shape: shape,
        start: [0; 3],
        shape,
    };
    let ontology = Ontology::from_ontology_json(TREE, OntologyLevel::TreeDepth).unwrap();
    write_prepared_atlas(
        root,
        &stack,
        &window,
        &ontology,
        &WriteAtlasParams {
            voxel_size_um: [ATLAS_UM; 3],
            name: "synthetic".into(),
            replace: false,
            workers: 2,
        },
    )
    .unwrap();
}

/// One registration step: `registration.json` and an elastix affine
/// `out = diag(a) p + t` in physical `z, y, x`.
fn write_step(
    dir: &Path,
    fixed: ([usize; 3], f64, [f64; 3]),
    moving: ([usize; 3], f64, [f64; 3]),
    a: [f64; 3],
    t: [f64; 3],
) {
    fs::create_dir_all(dir).unwrap();
    let side = |(shape, scale, translation): ([usize; 3], f64, [f64; 3])| {
        json!({"path": null, "channel": 0, "level": 0, "level0_shape": shape,
               "level0_frame": {"scale": vec![scale; 3], "translation": translation}})
    };
    fs::write(
        dir.join("registration.json"),
        serde_json::to_string_pretty(&json!({
            "transform": "TransformParameters.0.txt",
            "fixed": side(fixed),
            "moving": side(moving),
        }))
        .unwrap(),
    )
    .unwrap();
    let (shape, scale, translation) = fixed;
    fs::write(
        dir.join("TransformParameters.0.txt"),
        format!(
            "(Transform \"AffineTransform\")\n(NumberOfParameters 12)\n\
             (TransformParameters {} 0 0 0 {} 0 0 0 {} {} {} {})\n\
             (InitialTransformParametersFileName \"NoInitialTransform\")\n\
             (HowToCombineTransforms \"Compose\")\n(CenterOfRotationPoint 0.0 0.0 0.0)\n\
             (Size {} {} {})\n(Index 0 0 0)\n(Spacing {scale} {scale} {scale})\n\
             (Origin {} {} {})\n(Direction 1 0 0 0 1 0 0 0 1)\n\
             (FixedImageDimension 3)\n(MovingImageDimension 3)\n",
            a[0],
            a[1],
            a[2],
            t[0],
            t[1],
            t[2],
            shape[0],
            shape[1],
            shape[2],
            translation[0],
            translation[1],
            translation[2]
        ),
    )
    .unwrap();
}

fn sample() -> ([usize; 3], f64, [f64; 3]) {
    (SAMPLE_SHAPE, SAMPLE_UM, SAMPLE_TRANSLATION)
}

fn atlas_side() -> ([usize; 3], f64, [f64; 3]) {
    ([ATLAS; 3], ATLAS_UM, [0.0; 3])
}

fn image(dir: &Path) -> PathBuf {
    let scene = Scene::new(
        SceneSpec::clean(SHAPE, 9)
            .with_objects(40)
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
    image
}

/// The closed-form atlas voxel of detection voxel `i` through one affine.
fn expected(i: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|a| (A[a] * (i[a] * VOXEL_UM[a] + OFFSET_UM[a]) + T[a]) / ATLAS_UM)
}

fn args(image: &Path, registration: &Path, atlas: &Path, extra: &[&str]) -> Args {
    let mut all = vec![
        "cell-density",
        "--input",
        image.to_str().unwrap(),
        "--in-place",
        "--block",
        "16,32,32",
        "--workers",
        "2",
        "--size-min",
        "20",
        "--registration",
        registration.to_str().unwrap(),
        "--atlas",
        atlas.to_str().unwrap(),
        "--sample-offset-um",
        "16,-8,24",
        "--overwrite",
    ];
    all.extend_from_slice(extra);
    Args::parse_from(all)
}

fn column_f64(table: &TableReader, name: &str) -> Vec<f64> {
    table.read_f64(name, 0..table.spec().row_count).unwrap()
}

fn column_i64(table: &TableReader, name: &str) -> Vec<i64> {
    table.read_i64(name, 0..table.spec().row_count).unwrap()
}

#[test]
fn cells_land_where_the_registration_puts_them_and_are_labelled_there() {
    let dir = scratch("labels");
    let image = image(&dir);
    let registration = dir.join("registration");
    write_step(&registration, sample(), atlas_side(), A, T);
    let atlas = dir.join("atlas.ome.zarr");
    write_atlas(&atlas, [ATLAS; 3]);
    let density = dir.join("density.ome.zarr");
    let report = run(&args(
        &image,
        &registration,
        &atlas,
        &[
            "--crust-voxels",
            "2",
            "--density",
            density.to_str().unwrap(),
            "--density-radius-voxels",
            "0",
            "--density-weight",
            "source",
        ],
    ))
    .unwrap();
    let tables = image.join("tables");
    let all = TableReader::open(tables.join("cell-density")).unwrap();
    let (z, y, x) = (
        column_f64(&all, "z"),
        column_f64(&all, "y"),
        column_f64(&all, "x"),
    );
    let at = [
        column_f64(&all, "atlas_z"),
        column_f64(&all, "atlas_y"),
        column_f64(&all, "atlas_x"),
    ];
    let regions = column_i64(&all, "region");
    let hemispheres = column_i64(&all, "hemisphere");
    let distances = column_f64(&all, "distance_to_surface_voxels");
    let sizes = all.read_u64("size", 0..all.spec().row_count).unwrap();
    let passes = all
        .read_u8("passes_filter", 0..all.spec().row_count)
        .unwrap();
    let sources = column_f64(&all, "source");
    assert!(z.len() >= 30, "{} cells", z.len());
    let (mut inside, mut outside, mut crusted) = (0, 0, 0);
    let mut density_mass = 0.0;
    for row in 0..z.len() {
        let want = expected([z[row], y[row], x[row]]);
        for a in 0..3 {
            assert!(
                (at[a][row] - want[a]).abs() < 1e-9,
                "row {row} axis {a}: {} vs {}",
                at[a][row],
                want[a]
            );
        }
        let voxel = want.map(|v| v.round_ties_even());
        let in_atlas = voxel.iter().all(|v| *v >= 0.0 && *v < ATLAS as f64);
        let band = (20..900).contains(&sizes[row]);
        if in_atlas {
            inside += 1;
            let [vz, vy, vx] = voxel.map(|v| v as usize);
            assert_eq!(regions[row], i64::from(region(vz, vy, vx)), "row {row}");
            assert_eq!(hemispheres[row], i64::from(hemisphere(vz)), "row {row}");
            let depth = f64::from(distance(vz, vy, vx));
            assert_eq!(distances[row], depth, "row {row}");
            // The reference's crust: strictly deeper than 2 voxels.
            assert_eq!(passes[row] == 1, band && depth > 2.0, "row {row}");
            if band && depth <= 2.0 {
                crusted += 1;
            }
            if passes[row] == 1 {
                density_mass += sources[row];
            }
        } else {
            outside += 1;
            assert_eq!((regions[row], hemispheres[row]), (0, -1), "row {row}");
            // The distance is read at the nearest edge voxel, the rim here.
            assert_eq!(distances[row], 0.0, "row {row}");
            assert_eq!(passes[row], 0, "the edge is in the crust");
        }
    }
    assert!(
        inside > 10 && outside > 0 && crusted > 0,
        "{inside} {outside} {crusted}"
    );
    assert_eq!(report["atlas_mapping"]["cells_outside_atlas"], outside);

    // Region counts and volumes from the atlas.
    let csv = fs::read_to_string(tables.join("cell-density").join("regions.csv")).unwrap();
    let mut lines = csv.lines();
    assert_eq!(
        lines.next(),
        Some("region,hemisphere,acronym,name,cells,mean_size,volume_voxels,volume_mm3,density_per_mm3")
    );
    let mut counted = 0;
    for line in lines {
        let fields: Vec<&str> = line.split(',').collect();
        let (id, hemi): (u32, i64) = (fields[0].parse().unwrap(), fields[1].parse().unwrap());
        counted += fields[fields.len() - 5].parse::<u64>().unwrap();
        if hemi < 0 {
            continue;
        }
        let truth = (0..ATLAS)
            .flat_map(|z| (0..ATLAS).flat_map(move |y| (0..ATLAS).map(move |x| (z, y, x))))
            .filter(|&(z, y, x)| region(z, y, x) == id && i64::from(hemisphere(z)) == hemi)
            .count();
        let voxels: u64 = fields[fields.len() - 3].parse().unwrap();
        assert_eq!(voxels, truth as u64, "{line}");
        let mm3: f64 = fields[fields.len() - 2].parse().unwrap();
        assert!((mm3 - truth as f64 * 1e-6).abs() < 1e-12, "{line}");
    }
    assert_eq!(counted, passes.iter().filter(|p| **p == 1).count() as u64);
    assert!(csv.contains(",L1,left upper,") || csv.contains(",L,\"left, west\","));
    let sidecar: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(tables.join("cell-density").join("regions.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(sidecar["labels"]["frame"], "atlas");

    // The density is on the atlas grid, in its frame, weighted by intensity.
    let geometry = image_geometry(&density, [ATLAS; 3]).unwrap();
    assert_eq!(geometry.scale_zyx, [ATLAS_UM; 3]);
    let values = AttachedImage::at(density.join("0")).read().unwrap();
    assert_eq!(values.shape(), [ATLAS; 3]);
    let total: f64 = values.widened().sum();
    assert!(
        (total - density_mass).abs() < 1e-3 * density_mass.max(1.0),
        "{total} vs {density_mass}"
    );

    // The tree's names are in the CSV, and a level collapses 11 onto 10.
    let table_csv = fs::read_to_string(tables.join("cell-density").join("table.csv")).unwrap();
    assert!(table_csv.lines().next().unwrap().ends_with(
        ",region,hemisphere,distance_to_surface_voxels,region_acronym,region_name,atlas_z,atlas_y,\
         atlas_x"
    ));
    let metadata: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(tables.join("cell-density").join("zarr.json")).unwrap(),
    )
    .unwrap();
    let recorded = &metadata["attributes"]["cell_density"];
    assert_eq!(
        recorded["filter"]["distance_to_surface_unit"],
        "atlas voxel"
    );
    assert_eq!(recorded["filter"]["crust_voxels"], 2.0);
    assert_eq!(recorded["filter"]["crust_um"], 20.0);
    assert_eq!(
        recorded["registration"]["distance_to_surface_unit"],
        "atlas voxel"
    );

    // The crust in µm reads the voxel distance through the voxel's edge.
    run(&args(&image, &registration, &atlas, &["--crust-um", "25"])).unwrap();
    let crusted_um = TableReader::open(tables.join("cell-density")).unwrap();
    let passes_um = crusted_um
        .read_u8("passes_filter", 0..crusted_um.spec().row_count)
        .unwrap();
    for row in 0..z.len() {
        let band = (20..900).contains(&sizes[row]);
        assert_eq!(
            passes_um[row] == 1,
            band && distances[row] * ATLAS_UM > 25.0,
            "row {row}"
        );
    }
    run(&args(&image, &registration, &atlas, &["--level", "2"])).unwrap();
    let collapsed = TableReader::open(tables.join("cell-density")).unwrap();
    let collapsed_regions = column_i64(&collapsed, "region");
    assert!(
        regions.contains(&11),
        "the fixture must exercise the collapse"
    );
    assert!(!collapsed_regions.contains(&11));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_chain_composes_and_floor_rounding_is_the_references() {
    let dir = scratch("chain");
    let image = image(&dir);
    let chain = dir.join("chain");
    // Sample -> a middle image (shifted 8 µm in z) -> atlas.
    let middle = (SAMPLE_SHAPE, SAMPLE_UM, [0.0; 3]);
    write_step(
        &chain.join("step0"),
        sample(),
        middle,
        [1.0; 3],
        [8.0, 0.0, 0.0],
    );
    write_step(&chain.join("step1"), middle, atlas_side(), A, T);
    fs::write(
        chain.join("chain.json"),
        r#"{"schema": 1, "steps": ["step0", "step1"]}"#,
    )
    .unwrap();
    let atlas = dir.join("atlas.ome.zarr");
    write_atlas(&atlas, [ATLAS; 3]);
    let geometry = image_geometry(&image, SHAPE).unwrap();
    let registered = RegisteredAtlas::open(
        &chain,
        &OpenParams {
            atlas: Some(atlas.clone()),
            ..OpenParams::default()
        },
    )
    .unwrap();
    let to_sample = registered
        .image_to_sample(
            &ImageFrame::new(
                geometry.scale_zyx,
                geometry.translation_zyx,
                Some(&geometry.unit),
            ),
            OFFSET_UM,
        )
        .unwrap();
    assert_eq!(registered.transform.steps(), 2);
    let points = [[3.0, 10.0, 20.0], [17.0, 40.0, 5.0], [31.0, 63.0, 63.0]];
    let mapped = registered.map(&to_sample, &points);
    for (point, got) in points.iter().zip(&mapped.atlas) {
        let want: [f64; 3] = [0, 1, 2].map(|a| {
            let shifted = point[a] * VOXEL_UM[a] + OFFSET_UM[a] + if a == 0 { 8.0 } else { 0.0 };
            (A[a] * shifted + T[a]) / ATLAS_UM
        });
        for a in 0..3 {
            assert!((got[a] - want[a]).abs() < 1e-9, "{got:?} vs {want:?}");
        }
        let floor = want.map(|v| v.floor());
        let voxel = registered.voxel(*got, Rounding::Floor);
        if floor.iter().all(|v| *v >= 0.0 && *v < ATLAS as f64) {
            assert_eq!(voxel, Some(floor.map(|v| v as usize)));
        } else {
            assert_eq!(voxel, None);
        }
    }
    // The sample grid: `(i * s_d + o - t_s) / s_s`.
    assert_eq!(to_sample.scale, [0.5, 0.25, 0.25]);
    assert_eq!(to_sample.offset, [1.5, -1.5, 2.5]);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn an_atlas_other_than_the_moving_image_is_refused() {
    let dir = scratch("refused");
    let image = image(&dir);
    let registration = dir.join("registration");
    write_step(&registration, sample(), atlas_side(), A, T);
    let atlas = dir.join("atlas.ome.zarr");
    write_atlas(&atlas, [ATLAS, ATLAS, ATLAS + 2]);
    let failed = run(&args(&image, &registration, &atlas, &[])).unwrap_err();
    assert!(
        failed
            .to_string()
            .contains("moving image; label with the atlas"),
        "{failed}"
    );

    // The right shape, but a reference that is not the moving image.
    let atlas = dir.join("atlas-right-shape.ome.zarr");
    write_atlas(&atlas, [ATLAS; 3]);
    let moving = dir.join("moving.ome.zarr");
    write_ome_image(
        &moving,
        &Voxels::U16(Array3::zeros((ATLAS, ATLAS, ATLAS))),
        [ATLAS_UM; 3],
        [0.0; 3],
        "micrometer",
        &Map::new(),
    )
    .unwrap();
    let sidecar = registration.join("registration.json");
    let mut json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&sidecar).unwrap()).unwrap();
    json["moving"]["path"] = json!(moving);
    fs::write(&sidecar, json.to_string()).unwrap();
    let failed = run(&args(&image, &registration, &atlas, &[])).unwrap_err();
    assert!(
        failed
            .to_string()
            .contains("differs from the registration's moving image"),
        "{failed}"
    );
    run(&args(
        &image,
        &registration,
        &atlas,
        &["--skip-atlas-check"],
    ))
    .unwrap();
    fs::remove_dir_all(dir).unwrap();
}
