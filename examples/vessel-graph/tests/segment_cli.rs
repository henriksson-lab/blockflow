// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The `segment` and `tract` subcommands end to end on a synthetic OME-Zarr
//! image: the labels they publish are the in-memory answers, voxel for voxel.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use blockflow::object_table::TableReader;
use blockflow::voxels::Voxels;
use blockflow_vessel_graph::graph::cli::{run_graph, GraphArgs};
use blockflow_vessel_graph::graph::{vessel_graph, GraphParams, GraphTables, Inputs, VesselGraph};
use blockflow_vessel_graph::segment::cli::{run_segment, run_tract, SegmentArgs, TractArgs};
use blockflow_vessel_graph::segment::synthetic::vessel_volume;
use blockflow_vessel_graph::segment::{
    segment, tract_map, ChannelParams, ChannelRole, TractParams,
};
use clap::Parser;
use ndarray::Array3;
use zarrs::array::{data_type, Array, ArrayBuilder, ArraySubset, FillValue};
use zarrs::filesystem::FilesystemStore;

const SHAPE: [usize; 3] = [20, 40, 48];

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
    dir
}

/// A one-level `t,c,z,y,x` OME-Zarr 0.5 image of `channels`.
fn write_image(root: &Path, channels: &[Voxels]) {
    fs::create_dir_all(root.join("0")).unwrap();
    let attrs = serde_json::json!({"ome": {"version": "0.5", "multiscales": [{
        "axes": [
            {"name": "t", "type": "time"}, {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"}],
        "datasets": [{"path": "0", "coordinateTransformations": [
            {"type": "scale", "scale": [1.0, 1.0, 2.0, 1.0, 1.0]}]}]}]}});
    fs::write(
        root.join("zarr.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"zarr_format": 3, "node_type": "group", "attributes": attrs}),
        )
        .unwrap(),
    )
    .unwrap();
    let mut data = Vec::new();
    for channel in channels {
        data.extend(channel.view::<u16>().unwrap().iter().copied());
    }
    let store = Arc::new(FilesystemStore::new(root).unwrap());
    let shape = vec![
        1,
        channels.len() as u64,
        SHAPE[0] as u64,
        SHAPE[1] as u64,
        SHAPE[2] as u64,
    ];
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

/// Level 0 of a published label image.
fn read_label(image: &Path, name: &str) -> Array3<u64> {
    let store = Arc::new(FilesystemStore::new(image.join("labels").join(name)).unwrap());
    let array = Array::open(store, "/0").unwrap();
    let shape: Vec<u64> = array.shape().to_vec();
    assert_eq!(shape, SHAPE.map(|n| n as u64).to_vec(), "{name}");
    let values: Vec<u64> = array
        .retrieve_array_subset(&ArraySubset::new_with_shape(shape))
        .unwrap();
    Array3::from_shape_vec((SHAPE[0], SHAPE[1], SHAPE[2]), values).unwrap()
}

fn as_labels(voxels: &Voxels) -> Array3<u64> {
    voxels.widened().mapv(|value| value as u64)
}

/// A one-level `z,y,x` label image `labels/<name>` of `values`, on the
/// lattice `scale`, `translation`.
fn write_label_image(
    image: &Path,
    name: &str,
    values: &Array3<u32>,
    scale: [f64; 3],
    translation: [f64; 3],
) {
    let root = image.join("labels").join(name);
    fs::create_dir_all(root.join("0")).unwrap();
    let attrs = serde_json::json!({"ome": {"version": "0.5", "multiscales": [{
        "axes": [
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"}],
        "datasets": [{"path": "0", "coordinateTransformations": [
            {"type": "scale", "scale": scale},
            {"type": "translation", "translation": translation}]}]}]}});
    fs::write(
        root.join("zarr.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"zarr_format": 3, "node_type": "group", "attributes": attrs}),
        )
        .unwrap(),
    )
    .unwrap();
    let shape: Vec<u64> = values.shape().iter().map(|n| *n as u64).collect();
    let store = Arc::new(FilesystemStore::new(&root).unwrap());
    let array = ArrayBuilder::new(
        shape.clone(),
        shape.clone(),
        data_type::uint32(),
        FillValue::from(0u32),
    )
    .build(store, "/0")
    .unwrap();
    array.store_metadata().unwrap();
    let data: Vec<u32> = values.iter().copied().collect();
    array
        .store_array_subset(&ArraySubset::new_with_shape(shape), &data)
        .unwrap();
}

#[test]
fn segment_publishes_the_in_memory_answer_for_two_channels() {
    let dir = scratch("segment");
    let image = dir.join("image.zarr");
    let channels = [vessel_volume(SHAPE, 21), vessel_volume(SHAPE, 22)];
    write_image(&image, &channels);
    // The fixture's background sits at 1000, the top of the vessel role's
    // clip range; the file moves the vessel channel's range up and leaves the
    // artery channel at its role's defaults.
    let params_path = dir.join("params.json");
    fs::write(
        &params_path,
        r#"{"channels": [{"role": "vessel", "clip_range": [400, 60000]}, {"role": "artery"}]}"#,
    )
    .unwrap();
    // Without the deep filling, which a `deep-fill` build switches on by
    // default and which needs the trained weights.
    let parsed = Segment::parse_from([
        "segment",
        "--no-deep-fill",
        "--zarr",
        image.to_str().unwrap(),
        "--channel",
        "0,1",
        "--params",
        params_path.to_str().unwrap(),
        "--block",
        "10,20,24",
        "--workers",
        "2",
        "--work",
        dir.join("work").to_str().unwrap(),
    ]);
    let summary = run_segment(&parsed.args).expect("the run");
    assert_eq!(summary.mask_label, "vessel-mask");
    assert_eq!(
        summary.channel_mask_labels,
        ["vessel-vessel-mask", "vessel-artery-mask"]
    );

    let (params, roles) = parsed.args.params_and_roles().unwrap();
    assert_eq!(roles, [ChannelRole::Vessel, ChannelRole::Artery]);
    assert_eq!(params.channels[0].binarize.clip.range, (400.0, 60000.0));
    assert_eq!(
        params.channels[1],
        ChannelParams {
            deep_fill: None,
            ..ChannelRole::Artery.channel()
        }
    );
    let expected = segment(channels.to_vec(), &params, [20, 40, 48]).unwrap();
    let mask = read_label(&image, "vessel-mask");
    let skeleton = read_label(&image, "vessel-skeleton");
    assert!(mask.iter().any(|v| *v == 1));
    assert!(mask.iter().all(|v| *v <= 1));
    assert_eq!(mask, as_labels(&expected.mask));
    assert_eq!(skeleton, as_labels(&expected.skeleton));
    let artery = read_label(&image, "vessel-artery-mask");
    assert_eq!(artery, as_labels(&expected.channels[1].processed));
    assert!(
        artery.iter().any(|v| *v == 1),
        "the bright tubes are arteries"
    );
    assert!(
        artery.iter().filter(|v| **v == 1).count() < mask.iter().filter(|v| **v == 1).count(),
        "the artery mask is part of the union"
    );
    assert!(!dir.join("work").exists(), "the work directory is removed");

    // The thinning over stored images, a round per phase, gives the same
    // skeleton as the resident fixed point.
    let stored = Segment::parse_from([
        "segment",
        "--no-deep-fill",
        "--zarr",
        image.to_str().unwrap(),
        "--channel",
        "0,1",
        "--params",
        params_path.to_str().unwrap(),
        "--block",
        "10,20,24",
        "--workers",
        "2",
        "--thin-memory-mb",
        "0",
        "--work",
        dir.join("work").to_str().unwrap(),
    ]);
    run_segment(&stored.args).expect("the run with stored thinning");
    assert_eq!(read_label(&image, "vessel-mask"), mask);
    assert_eq!(read_label(&image, "vessel-skeleton"), skeleton);
    assert_eq!(read_label(&image, "vessel-artery-mask"), artery);
    for leftover in ["work", "work.thin", "work.skeleton"] {
        assert!(!dir.join(leftover).exists(), "{leftover} is removed");
    }

    // `graph` reads back what `segment` published and builds the graph the
    // library builds from the in-memory answer, with the artery mask found by
    // its name.
    let out = dir.join("graph.json");
    let parsed = Graph::parse_from([
        "graph",
        "--zarr",
        image.to_str().unwrap(),
        "--artery-channel",
        "1",
        "--out",
        out.to_str().unwrap(),
    ]);
    let summary = run_graph(&parsed.args).expect("the graph run");
    let mut inputs = Inputs::new(&expected.skeleton, [2.0, 1.0, 1.0]);
    inputs.vessels = Some(&expected.mask);
    let artery_mask = Voxels::from(expected.channels[1].processed.nonzero());
    inputs.artery_mask = Some(&artery_mask);
    inputs.artery_intensity = Some(&channels[1]);
    let in_memory = vessel_graph(&inputs, &GraphParams::default()).unwrap();
    assert!(summary.edges > 0, "the synthetic vessels give a graph");
    assert!(summary.classified, "the artery mask was found");
    assert_eq!(summary.vertices, in_memory.n_vertices());
    assert_eq!(summary.edges, in_memory.n_edges());
    let written = VesselGraph::read_json(&out).unwrap();
    assert_eq!(written.n_edges(), in_memory.n_edges());
    assert_eq!(written.edge("length_units"), in_memory.edge("length_units"));
    assert_eq!(
        written.edge("artery_binary"),
        in_memory.edge("artery_binary")
    );
    assert!(written.edge("artery").is_some() && written.edge("vein").is_some());
    check_tables(&summary.tables.expect("tables by default"), &written);

    // The atlas already in the sample frame, on a lattice twice as coarse in
    // y and x and in z: regions by z, hemispheres by x.
    let atlas_shape = [SHAPE[0] / 2, SHAPE[1] / 2, SHAPE[2] / 2];
    let scale = [4.0, 2.0, 2.0];
    let translation = [1.0, 0.5, 0.5];
    let annotation = Array3::from_shape_fn(atlas_shape, |(z, _, _)| if z < 5 { 10 } else { 20 });
    let hemispheres = Array3::from_shape_fn(atlas_shape, |(_, _, x)| u32::from(x >= 12) + 1);
    write_label_image(&image, "atlas", &annotation, scale, translation);
    write_label_image(&image, "hemispheres", &hemispheres, scale, translation);
    let density = dir.join("branch-density.ome.zarr");
    let parsed = Graph::parse_from([
        "graph",
        "--zarr",
        image.to_str().unwrap(),
        "--annotation-label",
        "atlas",
        "--hemisphere-label",
        "hemispheres",
        "--branch-density",
        density.to_str().unwrap(),
        "--density-radius",
        "1,1,1",
        "--out",
        out.to_str().unwrap(),
    ]);
    let summary = run_graph(&parsed.args).expect("the annotated graph run");
    assert!(summary.annotated);
    assert_eq!(summary.density_outside, Some(0));
    let annotated = VesselGraph::read_json(&out).unwrap();
    let points = annotated.points("coordinates").unwrap();
    let atlas_points = annotated.points("coordinates_atlas").unwrap();
    let regions = annotated.vertex("annotation").unwrap().values();
    let sides = annotated.vertex("hemisphere").unwrap().values();
    for (row, at) in points.iter().enumerate() {
        // Physical position, then the atlas voxel it falls in.
        let physical = [at[0] * 2.0, at[1], at[2]];
        let voxel: [f64; 3] = std::array::from_fn(|a| (physical[a] - translation[a]) / scale[a]);
        for axis in 0..3 {
            assert!((atlas_points[row][axis] - voxel[axis]).abs() < 1e-9);
        }
        let index = voxel.map(|v| v.round().clamp(0.0, 1e9) as usize);
        assert_eq!(regions[row], if index[0] < 5 { 10.0 } else { 20.0 });
        assert_eq!(sides[row], if index[2] >= 12 { 2.0 } else { 1.0 });
    }
    assert!(regions.contains(&10.0) && regions.contains(&20.0));
    let distance = annotated
        .edge("distance_to_surface")
        .expect("the edge minimum");
    assert!(distance.values().iter().all(|d| *d >= 0.0));
    let field = blockflow::ome_zarr::OmeVolumePyramid::open(&density, 0, 0)
        .unwrap()
        .attached_level(0)
        .unwrap()
        .read()
        .unwrap();
    assert_eq!(field.shape(), atlas_shape);
    let total: f64 = field.widened().sum();
    // Every vertex's ball of 7 voxels (radius 1), less what falls off the
    // edge of the lattice.
    assert!(total > 0.0 && total <= 7.0 * annotated.n_vertices() as f64);
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
    ] {
        assert!(names.contains(&wanted), "{wanted} in {names:?}");
    }
    let _ = fs::remove_dir_all(&dir);
}

/// The three tables hold the graph: every edge's polyline is its points'
/// rows, in order, and every column reads back by identity.
fn check_tables(tables: &GraphTables, graph: &VesselGraph) {
    let vertices = TableReader::open(&tables.vertices).unwrap();
    let edges = TableReader::open(&tables.edges).unwrap();
    let points = TableReader::open(&tables.edge_points).unwrap();
    assert_eq!(vertices.spec().row_count, graph.n_vertices() as u64);
    assert_eq!(edges.spec().row_count, graph.n_edges() as u64);
    assert_eq!(points.spec().row_count, graph.n_points() as u64);
    for table in [&vertices, &edges, &points] {
        assert_eq!(table.spec().coordinate_space, "../../");
        assert!(table.spec().extensions.contains_key("vessel_graph"));
    }

    let n = graph.n_edges() as u64;
    let ids = edges.read_u64("edge_id", 0..n).unwrap();
    let sources = edges.read_u64("source", 0..n).unwrap();
    let targets = edges.read_u64("target", 0..n).unwrap();
    let starts = edges.read_u64("point_start", 0..n).unwrap();
    let counts = edges.read_u32("point_count", 0..n).unwrap();
    let lengths = edges.read_f64("length_units", 0..n).unwrap();
    let artery = edges.read_u8("artery", 0..n).unwrap();
    let flags = graph.edge("artery").unwrap().as_flags().unwrap();
    let expected = graph.edge("length_units").unwrap().values();
    for row in 0..n as usize {
        let edge = ids[row] as usize;
        assert_eq!(
            [sources[row] as u32, targets[row] as u32],
            graph.edges()[edge]
        );
        let [start, end] = graph.polylines()[edge];
        assert_eq!((starts[row], counts[row]), (u64::from(start), end - start));
        assert_eq!(lengths[row], expected[edge]);
        assert_eq!(artery[row] == 1, flags[edge]);
    }

    let m = graph.n_points() as u64;
    let point_ids = points.read_u64("point_id", 0..m).unwrap();
    let owners = points.read_u64("edge_id", 0..m).unwrap();
    let order = points.read_u32("index", 0..m).unwrap();
    let z = points.read_f64("z", 0..m).unwrap();
    let along = graph.along_points("coordinates").unwrap();
    for row in 0..m as usize {
        let point = point_ids[row] as usize;
        let [start, end] = graph.polylines()[owners[row] as usize];
        assert!((start as usize..end as usize).contains(&point));
        assert_eq!(order[row] as usize, point - start as usize);
        assert_eq!(z[row], along[point][0]);
    }
}

#[test]
fn tract_publishes_the_in_memory_answer() {
    let dir = scratch("tract");
    let image = dir.join("image.zarr");
    let volume = vessel_volume(SHAPE, 31);
    write_image(&image, std::slice::from_ref(&volume));
    let csv = dir.join("points.csv");
    let parsed = Tract::parse_from([
        "tract",
        "--zarr",
        image.to_str().unwrap(),
        "--block",
        "10,20,24",
        "--percentiles",
        "70,99.9",
        "--sample-every",
        "7",
        "--density-radius",
        "2,2,2",
        "--points-csv",
        csv.to_str().unwrap(),
        "--scale-to",
        "10,20,24",
    ]);
    let summary = run_tract(&parsed.args).expect("the run");
    let params = TractParams {
        percentiles: (70.0, 99.9),
        sample_every: 7,
        density_radius: [2, 2, 2],
        ..TractParams::default()
    };
    let expected = tract_map(volume, &params, [20, 40, 48]).unwrap();
    assert_eq!(summary.range, expected.range);
    assert_eq!(summary.points, expected.points.len());
    assert_eq!(read_label(&image, "tract-mask"), as_labels(&expected.mask));
    assert_eq!(
        read_label(&image, "tract-density"),
        as_labels(&expected.density)
    );
    let rows = fs::read_to_string(&csv).unwrap();
    assert_eq!(rows.lines().count(), expected.points.len() + 1);
    let first = expected.points[0];
    assert_eq!(
        rows.lines().nth(1).unwrap(),
        format!(
            "{},{},{}",
            first[0] as f64 / 2.0,
            first[1] as f64 / 2.0,
            first[2] as f64 / 2.0
        )
    );
    let _ = fs::remove_dir_all(&dir);
}
