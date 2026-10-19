// SPDX-License-Identifier: MIT
// Original work for this crate.

//! `graph --registration` and `tract --registration` on a synthetic image, a
//! synthetic prepared atlas and a hand-written registration whose answer is
//! known: an affine (a scale on z and a translation) in physical units, with a
//! registration sample on a lattice twice as coarse as the image, so the
//! change of lattice from the image to the sample is exercised too. Also the
//! refusals of an atlas that is not the registration's, and `--crop`.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use atlas_regions::atlas_zarr::{write_prepared_atlas, WriteAtlasParams};
use atlas_regions::{AtlasStack, Ontology, OntologyLevel, PreparedWindow};
use blockflow::object_table::TableReader;
use blockflow::ome_zarr::OmeVolumePyramid;
use blockflow::voxels::Voxels;
use blockflow_atlas_registration::atlas_space::{ImageFrame, OpenParams, RegisteredAtlas};
use blockflow_vessel_graph::graph::cli::{run_graph, GraphArgs};
use blockflow_vessel_graph::graph::VesselGraph;
use blockflow_vessel_graph::segment::cli::{run_segment, run_tract, SegmentArgs, TractArgs};
use blockflow_vessel_graph::segment::synthetic::vessel_volume;
use clap::Parser;
use ndarray::Array3;
use zarrs::array::{data_type, ArrayBuilder, ArraySubset, FillValue};
use zarrs::filesystem::FilesystemStore;

/// The image: `[z, y, x]` voxels of 2 x 1 x 1 µm from the origin.
const SHAPE: [usize; 3] = [20, 40, 48];
const IMAGE_SCALE: [f64; 3] = [2.0, 1.0, 1.0];
/// The registration's sample: the same physical box on a lattice twice as
/// coarse.
const SAMPLE_SHAPE: [usize; 3] = [10, 20, 24];
const SAMPLE_SCALE: [f64; 3] = [4.0, 2.0, 2.0];
/// The atlas.
const ATLAS_SHAPE: [usize; 3] = [8, 20, 24];
const ATLAS_SCALE: [f64; 3] = [4.0, 2.0, 2.0];
/// The registration, sample physical to atlas physical: `A p + t`.
const A_DIAGONAL: [f64; 3] = [0.8, 1.0, 1.0];
const T: [f64; 3] = [8.0, -2.0, 2.0];

const TREE: &str = r#"{"msg": [{"id": 997, "name": "root", "acronym": "root",
  "color_hex_triplet": "FFFFFF", "st_level": 0, "children": [
    {"id": 8, "name": "grey", "acronym": "grey", "color_hex_triplet": "BFDAE3", "st_level": 1,
     "children": [{"id": 567, "name": "Cerebrum", "acronym": "CH",
                   "color_hex_triplet": "B0F0FF", "st_level": 2, "children": []}]}]}]}"#;

#[derive(Parser)]
struct Graph {
    #[command(flatten)]
    args: GraphArgs,
}

#[derive(Parser)]
struct Segment {
    #[command(flatten)]
    args: SegmentArgs,
}

#[derive(Parser)]
struct Tract {
    #[command(flatten)]
    args: TractArgs,
}

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("vessel-graph-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// Image voxels to atlas voxels, as the test states the registration.
fn expected_atlas(image: [f64; 3], origin: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|a| {
        let physical = origin[a] + image[a] * IMAGE_SCALE[a];
        (A_DIAGONAL[a] * physical + T[a]) / ATLAS_SCALE[a]
    })
}

fn region_at(voxel: [usize; 3]) -> u32 {
    if voxel[1] < 2 {
        0
    } else if voxel[0] < 4 {
        567
    } else {
        8
    }
}

/// A one-level `t, c, z, y, x` image of one `uint16` channel.
fn write_image(root: &Path, channel: &Voxels) {
    fs::create_dir_all(root.join("0")).unwrap();
    let attrs = serde_json::json!({"ome": {"version": "0.5", "multiscales": [{
        "axes": [
            {"name": "t", "type": "time"}, {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"}],
        "datasets": [{"path": "0", "coordinateTransformations": [
            {"type": "scale", "scale": [1.0, 1.0, IMAGE_SCALE[0], IMAGE_SCALE[1], IMAGE_SCALE[2]]}]}]}]}});
    fs::write(
        root.join("zarr.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"zarr_format": 3, "node_type": "group", "attributes": attrs}),
        )
        .unwrap(),
    )
    .unwrap();
    let data: Vec<u16> = channel.view::<u16>().unwrap().iter().copied().collect();
    let store = Arc::new(FilesystemStore::new(root).unwrap());
    let shape = vec![1, 1, SHAPE[0] as u64, SHAPE[1] as u64, SHAPE[2] as u64];
    let array = ArrayBuilder::new(
        shape.clone(),
        vec![1, 1, 16, 32, 32],
        data_type::uint16(),
        FillValue::from(0u16),
    )
    .build(store, "/0")
    .unwrap();
    array.store_metadata().unwrap();
    array
        .store_array_subset(&ArraySubset::new_with_shape(shape), &data)
        .unwrap();
}

/// A prepared atlas of `shape`: regions by z (and 0 at low y), hemispheres
/// by x, distance to surface `y`; `bump` changes one reference voxel.
fn write_atlas(root: &Path, shape: [usize; 3], bump: bool) {
    let dims = (shape[0], shape[1], shape[2]);
    let stack = AtlasStack {
        annotation: Voxels::U32(Array3::from_shape_fn(dims, |(z, y, x)| {
            region_at([z, y, x])
        })),
        reference: Some(Voxels::U16(Array3::from_shape_fn(dims, |(z, y, x)| {
            (z * 1000 + y * 30 + x + usize::from(bump && z == 1 && y == 1 && x == 1)) as u16
        }))),
        hemispheres: Some(Voxels::U8(Array3::from_shape_fn(dims, |(_, _, x)| {
            if x < 12 {
                255
            } else {
                0
            }
        }))),
        distance_to_surface: Some(Voxels::F64(Array3::from_shape_fn(dims, |(_, y, _)| {
            y as f64
        }))),
    };
    let ontology = Ontology::from_ontology_json(TREE, OntologyLevel::TreeDepth).unwrap();
    write_prepared_atlas(
        root,
        &stack,
        &PreparedWindow::identity(shape),
        &ontology,
        &WriteAtlasParams {
            voxel_size_um: ATLAS_SCALE,
            name: "synthetic".into(),
            replace: true,
            workers: 2,
        },
    )
    .unwrap();
}

/// A registration directory: the sample on its coarse lattice, `atlas` as the
/// moving image, and the affine as an elastix file.
fn write_registration(dir: &Path, atlas: &Path) {
    write_step(
        dir,
        (dir.join("sample.ome.zarr"), SAMPLE_SHAPE, SAMPLE_SCALE),
        (atlas.to_path_buf(), ATLAS_SHAPE, ATLAS_SCALE),
        A_DIAGONAL,
        T,
    );
}

type Side = (PathBuf, [usize; 3], [f64; 3]);

/// One registration step from `fixed` to `moving`, `diag(a) p + t` in
/// physical units.
fn write_step(dir: &Path, fixed: Side, moving: Side, a: [f64; 3], t: [f64; 3]) {
    fs::create_dir_all(dir).unwrap();
    let side = |path: String, shape: [usize; 3], scale: [f64; 3]| {
        serde_json::json!({"path": path, "channel": 0, "level": 0, "level0_shape": shape,
            "level0_frame": {"scale": scale, "translation": [0.0, 0.0, 0.0]}})
    };
    let json = serde_json::json!({
        "schema": 1,
        "axis_order": ["z", "y", "x"],
        "transform": "TransformParameters.0.txt",
        "direction": "fixed physical -> moving physical",
        "fixed": side(fixed.0.display().to_string(), fixed.1, fixed.2),
        "moving": side(moving.0.display().to_string(), moving.1, moving.2),
    });
    fs::write(
        dir.join("registration.json"),
        serde_json::to_vec_pretty(&json).unwrap(),
    )
    .unwrap();
    let [a0, a1, a2] = a;
    let [t0, t1, t2] = t;
    fs::write(
        dir.join("TransformParameters.0.txt"),
        format!(
            "(Transform \"AffineTransform\")\n(NumberOfParameters 12)\n\
             (TransformParameters {a0} 0 0 0 {a1} 0 0 0 {a2} {t0} {t1} {t2})\n\
             (InitialTransformParametersFileName \"NoInitialTransform\")\n\
             (HowToCombineTransforms \"Compose\")\n(CenterOfRotationPoint 0.0 0.0 0.0)\n\
             (Size {} {} {})\n(Index 0 0 0)\n(Spacing {} {} {})\n(Origin 0 0 0)\n\
             (Direction 1 0 0 0 1 0 0 0 1)\n(FixedImageDimension 3)\n(MovingImageDimension 3)\n",
            fixed.1[0], fixed.1[1], fixed.1[2], fixed.2[0], fixed.2[1], fixed.2[2]
        ),
    )
    .unwrap();
}

fn segment(image: &Path, dir: &Path, crop: Option<(&str, &Path)>) {
    let work = dir.join("work");
    let mut argv = vec![
        "segment".to_string(),
        "--no-deep-fill".into(),
        "--zarr".into(),
        image.display().to_string(),
        "--clip-range".into(),
        "400,60000".into(),
        "--block".into(),
        "10,20,24".into(),
        "--workers".into(),
        "2".into(),
        "--work".into(),
        work.display().to_string(),
    ];
    if let Some((range, out)) = crop {
        argv.extend([
            "--crop".into(),
            range.into(),
            "--crop-out".into(),
            out.display().to_string(),
        ]);
    }
    run_segment(&Segment::parse_from(argv).args).expect("the segmentation");
}

/// Every vertex and polyline point: atlas coordinates as stated, the region
/// and hemisphere of the atlas voxel they round to, 0 outside the atlas.
fn check_atlas_columns(graph: &VesselGraph, origin: [f64; 3]) -> (usize, usize) {
    let (mut inside, mut outside) = (0, 0);
    for (points, atlas, regions, sides) in [
        (
            graph.points("coordinates").unwrap(),
            graph.points("coordinates_atlas").unwrap(),
            graph.vertex("annotation").unwrap().values().to_vec(),
            graph.vertex("hemisphere").unwrap().values().to_vec(),
        ),
        (
            graph.along_points("coordinates").unwrap(),
            graph.along_points("coordinates_atlas").unwrap(),
            graph.along("annotation").unwrap().values().to_vec(),
            graph.along("hemisphere").unwrap().values().to_vec(),
        ),
    ] {
        for (row, at) in points.iter().enumerate() {
            let want = expected_atlas(*at, origin);
            for axis in 0..3 {
                assert!(
                    (atlas[row][axis] - want[axis]).abs() < 1e-6,
                    "{at:?}: {:?} against {want:?}",
                    atlas[row]
                );
            }
            let index = want.map(|v| v.round_ties_even());
            let within = (0..3).all(|a| index[a] >= 0.0 && index[a] < ATLAS_SHAPE[a] as f64);
            if within {
                let voxel = index.map(|v| v as usize);
                assert_eq!(regions[row], f64::from(region_at(voxel)), "{at:?}");
                assert_eq!(sides[row], if voxel[2] < 12 { 255.0 } else { 0.0 });
                inside += 1;
            } else {
                assert_eq!(regions[row], 0.0, "outside the atlas is region 0");
                outside += 1;
            }
        }
    }
    (inside, outside)
}

#[test]
fn graph_and_tract_in_atlas_space_through_a_registration() {
    let dir = scratch("registered");
    let image = dir.join("image.zarr");
    write_image(&image, &vessel_volume(SHAPE, 31));
    segment(&image, &dir, None);
    let atlas = dir.join("atlas.ome.zarr");
    write_atlas(&atlas, ATLAS_SHAPE, false);
    let registration = dir.join("registration");
    write_registration(&registration, &atlas);

    let out = dir.join("graph.json");
    let density = dir.join("branch-density.ome.zarr");
    let parsed = Graph::parse_from([
        "graph",
        "--zarr",
        image.to_str().unwrap(),
        "--registration",
        registration.to_str().unwrap(),
        "--branch-density",
        density.to_str().unwrap(),
        "--density-radius",
        "1,1,1",
        "--out",
        out.to_str().unwrap(),
    ]);
    let summary = run_graph(&parsed.args).expect("the registered graph run");
    assert!(summary.annotated);
    let graph = VesselGraph::read_json(&out).unwrap();
    let (inside, outside) = check_atlas_columns(&graph, [0.0; 3]);
    assert!(inside > 0, "most of the graph is in the atlas");
    assert!(outside > 0, "the registration moves part of it out");
    let regions = graph.vertex("annotation").unwrap().values();
    assert!(regions.contains(&567.0) && regions.contains(&8.0));
    assert!(summary.atlas_outside.is_some());

    // The distance to surface is the atlas's (`y`, clamped at the edge), and
    // the edge's is its polyline's minimum.
    let atlas_points = graph.points("coordinates_atlas").unwrap();
    let distances = graph.vertex("distance_to_surface").unwrap().values();
    for (at, d) in atlas_points.iter().zip(distances) {
        let y = at[1]
            .round_ties_even()
            .clamp(0.0, ATLAS_SHAPE[1] as f64 - 1.0);
        assert_eq!(*d, y);
    }
    assert!(graph.edge("distance_to_surface").is_some());

    // Radii in atlas voxels: µm over the sample voxel, the reference's rule.
    let um = graph.vertex("radius_units").unwrap().values();
    let in_atlas = graph.vertex("radius_units_atlas").unwrap().values();
    let mean = (SAMPLE_SCALE[0] + SAMPLE_SCALE[1] + SAMPLE_SCALE[2]) / 3.0;
    for (r, s) in um.iter().zip(in_atlas) {
        if *r >= 0.0 {
            assert!((r / mean - s).abs() < 1e-9);
        }
    }

    // The branch density on the atlas grid, with the atlas's frame.
    let pyramid = OmeVolumePyramid::open(&density, 0, 0).unwrap();
    assert_eq!(pyramid.voxel_size(), ATLAS_SCALE);
    let field = pyramid.attached_level(0).unwrap().read().unwrap();
    assert_eq!(field.shape(), ATLAS_SHAPE);
    let total: f64 = field.widened().sum();
    assert!(
        total > 0.0
            && total <= 7.0 * (graph.n_vertices() - summary.density_outside.unwrap()) as f64
    );

    // The tables carry the atlas columns and say which atlas space they are.
    let tables = summary.tables.unwrap();
    let vertices = TableReader::open(&tables.vertices).unwrap();
    let names: Vec<&str> = vertices
        .spec()
        .columns
        .iter()
        .map(|c| c.name.as_str())
        .collect();
    for wanted in [
        "annotation",
        "hemisphere",
        "distance_to_surface",
        "coordinates_atlas_z",
        "coordinates_atlas_x",
        "radius_units_atlas",
    ] {
        assert!(names.contains(&wanted), "{wanted} in {names:?}");
    }
    let extension = &vertices.spec().extensions["vessel_graph"]["atlas"];
    assert_eq!(extension["atlas_shape"], serde_json::json!(ATLAS_SHAPE));
    assert_eq!(
        extension["atlas"].as_str().unwrap(),
        atlas.display().to_string()
    );

    // `tract` maps its points the same way and voxelises them on the atlas
    // grid.
    let csv = dir.join("tract.csv");
    let tract_density = dir.join("tract-density.ome.zarr");
    let parsed = Tract::parse_from([
        "tract",
        "--zarr",
        image.to_str().unwrap(),
        "--block",
        "10,20,24",
        "--workers",
        "2",
        "--work",
        dir.join("work").to_str().unwrap(),
        "--registration",
        registration.to_str().unwrap(),
        "--points-csv",
        csv.to_str().unwrap(),
        "--atlas-density",
        tract_density.to_str().unwrap(),
        "--density-radius",
        "0,0,0",
    ]);
    let summary = run_tract(&parsed.args).expect("the registered tract run");
    let text = fs::read_to_string(&csv).unwrap();
    let mut lines = text.lines();
    assert_eq!(
        lines.next().unwrap(),
        "z,y,x,atlas_z,atlas_y,atlas_x,annotation,hemisphere"
    );
    let mut rows = 0;
    let mut inside = 0;
    for line in lines {
        let v: Vec<f64> = line.split(',').map(|s| s.parse().unwrap()).collect();
        let want = expected_atlas([v[0], v[1], v[2]], [0.0; 3]);
        for axis in 0..3 {
            assert!((v[3 + axis] - want[axis]).abs() < 1e-6);
        }
        let index = want.map(|x| x.round_ties_even());
        if (0..3).all(|a| index[a] >= 0.0 && index[a] < ATLAS_SHAPE[a] as f64) {
            assert_eq!(v[6], f64::from(region_at(index.map(|x| x as usize))));
            inside += 1;
        } else {
            assert_eq!(v[6], 0.0);
        }
        rows += 1;
    }
    assert_eq!(rows, summary.points);
    assert!(rows > 0);
    // Radius 0: one count per point inside the atlas.
    let field = OmeVolumePyramid::open(&tract_density, 0, 0)
        .unwrap()
        .attached_level(0)
        .unwrap()
        .read()
        .unwrap();
    assert_eq!(field.shape(), ATLAS_SHAPE);
    assert_eq!(field.widened().sum(), inside as f64);
    assert_eq!(summary.atlas_outside, Some(rows - inside));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn an_atlas_that_is_not_the_registrations_is_refused() {
    let dir = scratch("refused");
    let atlas = dir.join("atlas.ome.zarr");
    write_atlas(&atlas, ATLAS_SHAPE, false);
    let registration = dir.join("registration");
    write_registration(&registration, &atlas);
    let open = |atlas: Option<&Path>, check_reference: bool| {
        let params = OpenParams {
            atlas: atlas.map(Path::to_path_buf),
            check_reference,
        };
        RegisteredAtlas::open(&registration, &params)
    };
    assert!(open(None, true).is_ok());

    // Another shape.
    let other = dir.join("other.ome.zarr");
    write_atlas(&other, [8, 20, 23], false);
    let error = open(Some(&other), true).unwrap_err().to_string();
    assert!(
        error.contains("[8, 20, 23]") && error.contains("[8, 20, 24]"),
        "{error}"
    );

    // The same shape, one reference voxel different.
    let bumped = dir.join("bumped.ome.zarr");
    write_atlas(&bumped, ATLAS_SHAPE, true);
    let error = open(Some(&bumped), true).unwrap_err().to_string();
    assert!(error.contains("at 1 voxels"), "{error}");
    assert!(open(Some(&bumped), false).is_ok());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_crop_keeps_the_physical_frame_through_the_registration() {
    let dir = scratch("crop");
    let image = dir.join("image.zarr");
    let volume = vessel_volume(SHAPE, 32);
    write_image(&image, &volume);
    let crop = dir.join("crop.ome.zarr");
    segment(&image, &dir, Some(("2:18,4:,:40", &crop)));

    // The copy is the crop of the source, at the crop's physical origin.
    let pyramid = OmeVolumePyramid::open(&crop, 0, 0).unwrap();
    let level = &pyramid.levels()[0];
    assert_eq!(level.shape, [16, 36, 40]);
    assert_eq!(
        level.coordinate_transformations[1]["translation"],
        serde_json::json!([4.0, 4.0, 0.0])
    );
    let copied = pyramid.attached_level(0).unwrap().read().unwrap();
    let source = volume.view::<u16>().unwrap();
    let copied = copied.view::<u16>().unwrap();
    for ((z, y, x), value) in copied.indexed_iter() {
        assert_eq!(*value, source[[z + 2, y + 4, x]]);
    }
    assert!(crop.join("labels/vessel-skeleton/zarr.json").is_file());

    let atlas = dir.join("atlas.ome.zarr");
    write_atlas(&atlas, ATLAS_SHAPE, false);
    let registration = dir.join("registration");
    write_registration(&registration, &atlas);
    let out = dir.join("graph.json");
    let parsed = Graph::parse_from([
        "graph",
        "--zarr",
        crop.to_str().unwrap(),
        "--registration",
        registration.to_str().unwrap(),
        "--no-tables",
        "--out",
        out.to_str().unwrap(),
    ]);
    run_graph(&parsed.args).expect("the graph of the crop");
    let graph = VesselGraph::read_json(&out).unwrap();
    // Crop voxels at the crop's origin map as the source's voxels would.
    let (inside, _) = check_atlas_columns(&graph, [4.0, 4.0, 0.0]);
    assert!(inside > 0);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_chain_maps_from_its_first_fixed_image() {
    // sample (4 x 2 x 2 µm) -> intermediate (2 x 2 x 2 µm, the identity in
    // physical units) -> atlas (the test's affine): the same answer as the
    // one-step registration, which needs the first step's sample frame.
    let dir = scratch("chain");
    let atlas = dir.join("atlas.ome.zarr");
    write_atlas(&atlas, ATLAS_SHAPE, false);
    let chain = dir.join("chain");
    let middle = (dir.join("middle.ome.zarr"), [20, 20, 24], [2.0, 2.0, 2.0]);
    write_step(
        &chain.join("step0"),
        (dir.join("sample.ome.zarr"), SAMPLE_SHAPE, SAMPLE_SCALE),
        middle.clone(),
        [1.0; 3],
        [0.0; 3],
    );
    write_step(
        &chain.join("step1"),
        middle,
        (atlas.clone(), ATLAS_SHAPE, ATLAS_SCALE),
        A_DIAGONAL,
        T,
    );
    fs::write(chain.join("chain.json"), r#"{"steps": ["step0", "step1"]}"#).unwrap();
    let registered = RegisteredAtlas::open(&chain, &OpenParams::default()).unwrap();
    assert_eq!(registered.transform.steps(), 2);
    let image = ImageFrame::new(IMAGE_SCALE, [0.0; 3], Some("micrometer"));
    let points = [[3.0, 7.0, 11.0], [10.0, 20.0, 30.0], [0.5, 39.0, 0.0]];
    let mapped = registered.map_points(&image, &points).unwrap();
    for (at, got) in points.iter().zip(&mapped) {
        let want = expected_atlas(*at, [0.0; 3]);
        for axis in 0..3 {
            assert!(
                (got[axis] - want[axis]).abs() < 1e-6,
                "{got:?} against {want:?}"
            );
        }
    }
    let _ = fs::remove_dir_all(&dir);
}
