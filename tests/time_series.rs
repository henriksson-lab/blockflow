// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **Frames of a time-lapse as ordinary rank-three images.**
//
// A `(t, c, y, x)` image has no `z`. Each frame is exposed as `[1, Y, X]`
// through a virtual singleton axis, segmented by an ordinary plan, and its
// labels are written straight into plane `t` of a `[t, y, x]` label stack —
// produced once, where they will be published. The plane is then read back as
// the label input of a measurement plan over the other channel. Time never
// enters the block grid.

#![cfg(all(feature = "zarr", feature = "model-segment"))]

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use blockflow::assemble::{ImageId, PlanBuilder};
use blockflow::decomposition::Constraints;
use blockflow::geometry::BlockGrid;
use blockflow::label_pyramid::downsample_label_plane;
use blockflow::model_segment::{stub::ThresholdBackend, InstanceSegment};
use blockflow::op::Chain;
use blockflow::ops::collect_rows;
use blockflow::ops::measure::{
    collect_class_a_values, IntensityImage, IntensitySet, Measurements, ShapeSet,
};
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints, Workflow};
use blockflow::table::Value as Cell;
use blockflow::time_series::{LabelStack, TimeSeriesImage};
use blockflow::zarr_env::{Compression, ZarrEnvironment};
use blockflow::zarr_layout::StorageLayout;
use blockflow::{AttachedImage, Dtype};
use serde_json::json;
use zarrs::array::{Array as ZarrArray, ArraySubset};
use zarrs::filesystem::FilesystemStore;

mod support;
use support::scratch::ScratchDir;

const T: usize = 3;
const Y: usize = 16;
const X: usize = 16;

/// A `(t, c, y, x)` u16 movie: channel 0 has one 3x3 square per frame moving
/// one pixel right per frame, channel 1 holds `10 + t` inside it.
fn movie(root: &Path) {
    fs::create_dir_all(root.join("0")).unwrap();
    let group = json!({
        "zarr_format": 3, "node_type": "group",
        "attributes": {"ome": {"version": "0.5", "multiscales": [{
            "axes": [
                {"name": "t", "type": "time", "unit": "second"},
                {"name": "c", "type": "channel"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"}
            ],
            "datasets": [{"path": "0", "coordinateTransformations": [
                {"type": "scale", "scale": [30.0, 1.0, 0.5, 0.5]}
            ]}]
        }]}}
    });
    fs::write(root.join("zarr.json"), serde_json::to_vec(&group).unwrap()).unwrap();
    let metadata = StorageLayout::chunked(vec![1, 1, 8, 8], Compression::None)
        .array_metadata(&[T, 2, Y, X], Dtype::U16, json!(0), None)
        .unwrap();
    fs::write(
        root.join("0/zarr.json"),
        serde_json::to_vec(&metadata).unwrap(),
    )
    .unwrap();
    let store = Arc::new(FilesystemStore::new(root.join("0")).unwrap());
    let array = ZarrArray::open(store, "/").unwrap();
    let mut data = vec![0u16; T * 2 * Y * X];
    for t in 0..T {
        for y in 6..9 {
            for x in (4 + t)..(7 + t) {
                data[((t * 2) * Y + y) * X + x] = 100;
                data[((t * 2 + 1) * Y + y) * X + x] = 10 + t as u16;
            }
        }
    }
    let subset =
        ArraySubset::new_with_start_shape(vec![0; 4], vec![T as u64, 2, Y as u64, X as u64])
            .unwrap();
    array.store_array_subset(&subset, &data).unwrap();
}

fn read_plane(image: AttachedImage, work: &Path) -> Vec<u64> {
    let env = ZarrEnvironment::attach(work, &[image]).unwrap();
    env.image(0)
        .unwrap()
        .view::<u64>()
        .unwrap()
        .iter()
        .copied()
        .collect()
}

/// Segment frame `t` into `stack`'s plane, returning `id -> pixel count` from
/// the rows.
fn segment_frame(
    source: &TimeSeriesImage,
    stack: &LabelStack,
    t: usize,
    work: &Path,
) -> BTreeMap<u64, u64> {
    let frame = source.frame(0, 0, t).unwrap();
    let (dtype, volume) = frame.metadata().unwrap();
    assert_eq!(volume, [1, Y, X]);
    let op = InstanceSegment::new(
        "threshold",
        Arc::new(ThresholdBackend::new(50.0)),
        [0, 4, 4],
        "instances",
        Lifecycle::Persistent,
        Vec::new(),
    )
    .writing_labels()
    .with_id_offset((t * Y * X) as u64);
    let schema = op.schema().unwrap();
    let mut builder = PlanBuilder::new(volume, dtype, BlockGrid::new(volume, [1, 8, 8]).unwrap());
    builder.fragments(op).unwrap();
    let assembly = builder.finish().unwrap();
    let env = ZarrEnvironment::attach(work, &[frame])
        .unwrap()
        .with_output_target(stack.plane(0, t).unwrap())
        .unwrap();
    execute_phases(
        "frame",
        &assembly.workflow,
        &assembly.decomposition,
        &Hints::default(),
        &env,
        &[],
        &assembly.work(),
    )
    .unwrap();
    collect_rows(&env, "instances", assembly.n_phases() - 1, volume, schema)
        .unwrap()
        .iter()
        .map(|row| match (&row.values[0], &row.values[1]) {
            (Cell::U64(id), Cell::U64(count)) => (*id, *count),
            _ => panic!("integer columns"),
        })
        .collect()
}

#[test]
fn frames_write_their_own_planes_and_measure_from_them() {
    let scratch = ScratchDir::new("time-series", "planes");
    let image_root = scratch.path().join("image");
    movie(&image_root);
    let source = TimeSeriesImage::open(&image_root).unwrap();
    assert_eq!(source.n_frames(), T);
    assert_eq!(source.n_channels(), 2);
    assert!(!source.has_z());
    assert_eq!(source.frame_shape(0).unwrap(), [1, Y, X]);
    assert_eq!(source.spacing(0).unwrap(), [1.0, 0.5, 0.5]);
    let timeline = source.timeline().unwrap();
    assert_eq!(timeline[2].elapsed_seconds, Some(60.0));

    let stack =
        LabelStack::create(scratch.path().join(".stack"), &source, [8, 8], "../../").unwrap();
    // Frames out of order, as a concurrent run would finish them.
    for t in [2, 0, 1] {
        let rows = segment_frame(
            &source,
            &stack,
            t,
            &scratch.path().join(format!("work-{t}")),
        );
        assert_eq!(rows.len(), 1);
        let (&id, &count) = rows.iter().next().unwrap();
        assert_eq!(count, 9);
        // The id is the frame offset plus the lowest voxel's raster index.
        assert_eq!(id, (t * Y * X + 6 * X + 4 + t + 1) as u64);
        let plane = read_plane(
            stack.plane(0, t).unwrap(),
            &scratch.path().join(format!("read-{t}")),
        );
        assert_eq!(plane.iter().filter(|&&v| v == id).count(), 9);
        assert_eq!(plane.iter().filter(|&&v| v != 0 && v != id).count(), 0);
    }

    // Measure channel 1 of frame 1 over the staged label plane.
    let labels = stack.plane(0, 1).unwrap();
    let marker = source.frame(0, 1, 1).unwrap();
    let images = [labels, marker];
    let plan = Measurements::for_labels(ImageId::from(0))
        .shape(ShapeSet::basic())
        .intensity(
            IntensityImage::<0>::new(ImageId::supplied(0)).holding(Dtype::U16),
            IntensitySet::standard(),
        )
        .stream("measure")
        .lifecycle(Lifecycle::DeleteOnExit)
        .plan_on_attached(
            &images,
            &Constraints {
                split_axes: vec![1, 2],
                block_candidates: vec![8, 16],
                ..Constraints::default()
            },
        )
        .unwrap();
    let rows = plan.class_a_intensity_rows(0).unwrap();
    let env = ZarrEnvironment::attach(scratch.path().join("measure"), &images).unwrap();
    let workflow = Workflow::new(Chain::sequence(Vec::new()), [1, Y, X], Dtype::U64);
    execute_phases(
        "measure",
        &workflow,
        &plan.decomposition,
        &Hints::default(),
        &env,
        &[],
        &plan.phase_work(),
    )
    .unwrap();
    let values = collect_class_a_values(&env, &rows, [1, Y, X], plan.fixed).unwrap();
    assert_eq!(values.len(), 1);
    assert_eq!(values[0].label, (Y * X + 6 * X + 5 + 1) as u64);
    assert_eq!(values[0].sum, 9.0 * 11.0);
}

#[test]
fn clearing_a_plane_removes_its_chunks_and_hashing_ignores_storage() {
    let scratch = ScratchDir::new("time-series", "clear");
    let image_root = scratch.path().join("image");
    movie(&image_root);
    let source = TimeSeriesImage::open(&image_root).unwrap();
    let stack =
        LabelStack::create(scratch.path().join(".stack"), &source, [8, 8], "../../").unwrap();
    let empty = stack.plane_hash(0, 1).unwrap();
    segment_frame(&source, &stack, 1, &scratch.path().join("work"));
    let written = stack.plane_hash(0, 1).unwrap();
    assert_ne!(empty, written);
    assert!(stack.root().join("0/c/1").is_dir());
    stack.clear_plane(1).unwrap();
    assert!(!stack.root().join("0/c/1").exists());
    assert_eq!(stack.plane_hash(0, 1).unwrap(), empty);
    // Reopening finds the same geometry.
    let reopened = LabelStack::open(stack.root()).unwrap();
    assert_eq!(reopened.n_frames(), T);
    assert_eq!(reopened.frame_shape(0).unwrap(), [1, Y, X]);
}

#[test]
fn an_output_target_must_be_distinct_from_inputs_and_match_the_plan() {
    let scratch = ScratchDir::new("time-series", "refusals");
    let image_root = scratch.path().join("image");
    movie(&image_root);
    let source = TimeSeriesImage::open(&image_root).unwrap();
    // Any plane of an input array is refused as a target.
    let frame = source.frame(0, 0, 0).unwrap();
    let refused = ZarrEnvironment::attach(scratch.path().join("w0"), &[frame])
        .unwrap()
        .with_output_target(source.frame(0, 1, 0).unwrap());
    assert!(refused
        .err()
        .expect("refused")
        .to_string()
        .contains("also an input"));

    // A plan over an 8x8 window cannot write a 16x16 plane.
    let window = source.frame(0, 0, 0).unwrap().window([0, 0, 0], [1, 8, 8]);
    let (dtype, volume) = window.metadata().unwrap();
    let op = InstanceSegment::new(
        "t",
        Arc::new(ThresholdBackend::new(50.0)),
        [0, 4, 4],
        "s",
        Lifecycle::Persistent,
        Vec::new(),
    )
    .writing_labels();
    let mut builder = PlanBuilder::new(volume, dtype, BlockGrid::new(volume, [1, 8, 8]).unwrap());
    builder.fragments(op).unwrap();
    let assembly = builder.finish().unwrap();
    let stack =
        LabelStack::create(scratch.path().join(".stack"), &source, [8, 8], "../../").unwrap();
    let env = ZarrEnvironment::attach(scratch.path().join("w1"), &[window])
        .unwrap()
        .with_output_target(stack.plane(0, 0).unwrap())
        .unwrap();
    let error = execute_phases(
        "bad",
        &assembly.workflow,
        &assembly.decomposition,
        &Hints::default(),
        &env,
        &[],
        &assembly.work(),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("output target"), "{error}");
}

#[test]
fn a_label_plane_pyramid_downsamples_space_only() {
    let scratch = ScratchDir::new("time-series", "pyramid");
    let image_root = scratch.path().join("image");
    movie(&image_root);
    // Add a second level at half resolution to the source metadata.
    let mut group: serde_json::Value =
        serde_json::from_slice(&fs::read(image_root.join("zarr.json")).unwrap()).unwrap();
    group["attributes"]["ome"]["multiscales"][0]["datasets"]
        .as_array_mut()
        .unwrap()
        .push(json!({"path": "1", "coordinateTransformations": [{"type": "scale", "scale": [30.0, 1.0, 1.0, 1.0]}]}));
    fs::write(
        image_root.join("zarr.json"),
        serde_json::to_vec(&group).unwrap(),
    )
    .unwrap();
    fs::create_dir_all(image_root.join("1")).unwrap();
    let metadata = StorageLayout::chunked(vec![1, 1, 8, 8], Compression::None)
        .array_metadata(&[T, 2, Y / 2, X / 2], Dtype::U16, json!(0), None)
        .unwrap();
    fs::write(
        image_root.join("1/zarr.json"),
        serde_json::to_vec(&metadata).unwrap(),
    )
    .unwrap();

    let source = TimeSeriesImage::open(&image_root).unwrap();
    let stack =
        LabelStack::create(scratch.path().join(".stack"), &source, [8, 8], "../../").unwrap();
    assert_eq!(stack.frame_shape(1).unwrap(), [1, Y / 2, X / 2]);
    let rows = segment_frame(&source, &stack, 2, &scratch.path().join("work"));
    let id = *rows.keys().next().unwrap();
    downsample_label_plane(
        stack.plane(0, 2).unwrap(),
        stack.plane(1, 2).unwrap(),
        &scratch.path().join("pyramid"),
        &[[1, 8, 8]],
        2,
    )
    .unwrap();
    let level1 = read_plane(stack.plane(1, 2).unwrap(), &scratch.path().join("read"));
    assert!(level1.iter().any(|&v| v == id));
    assert!(level1.iter().all(|&v| v == 0 || v == id));
    // Other frames of level 1 are untouched.
    let other = read_plane(stack.plane(1, 0).unwrap(), &scratch.path().join("read0"));
    assert!(other.iter().all(|&v| v == 0));
    let group: serde_json::Value =
        serde_json::from_slice(&fs::read(stack.root().join("zarr.json")).unwrap()).unwrap();
    let axes: Vec<&str> = group["attributes"]["ome"]["multiscales"][0]["axes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a["name"].as_str().unwrap())
        .collect();
    assert_eq!(axes, ["t", "y", "x"]);
    assert_eq!(
        group["attributes"]["ome"]["multiscales"][0]["datasets"][1]["coordinateTransformations"][0]
            ["scale"],
        json!([30.0, 1.0, 1.0])
    );
}

#[test]
fn a_single_frame_image_gets_a_stack_with_a_time_axis_of_one() {
    let scratch = ScratchDir::new("time-series", "no-t");
    let root = scratch.path().join("image");
    fs::create_dir_all(root.join("0")).unwrap();
    let group = json!({
        "zarr_format": 3, "node_type": "group",
        "attributes": {"ome": {"version": "0.5", "multiscales": [{
            "axes": [{"name": "c", "type": "channel"}, {"name": "y", "type": "space"}, {"name": "x", "type": "space"}],
            "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 0.5, 0.5]}]}]
        }]}}
    });
    fs::write(root.join("zarr.json"), serde_json::to_vec(&group).unwrap()).unwrap();
    let metadata = StorageLayout::chunked(vec![1, 8, 8], Compression::None)
        .array_metadata(&[2, Y, X], Dtype::U16, json!(0), None)
        .unwrap();
    fs::write(
        root.join("0/zarr.json"),
        serde_json::to_vec(&metadata).unwrap(),
    )
    .unwrap();
    let source = TimeSeriesImage::open(&root).unwrap();
    assert_eq!(source.n_frames(), 1);
    assert_eq!(source.timeline().unwrap()[0].elapsed_seconds, None);
    let stack =
        LabelStack::create(scratch.path().join(".stack"), &source, [8, 8], "../../").unwrap();
    let group: serde_json::Value =
        serde_json::from_slice(&fs::read(stack.root().join("zarr.json")).unwrap()).unwrap();
    let multiscale = &group["attributes"]["ome"]["multiscales"][0];
    let axes: Vec<&str> = multiscale["axes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a["name"].as_str().unwrap())
        .collect();
    assert_eq!(axes, ["t", "y", "x"]);
    assert_eq!(
        multiscale["datasets"][0]["coordinateTransformations"][0]["scale"],
        json!([1, 0.5, 0.5])
    );
    assert_eq!(
        LabelStack::open(stack.root())
            .unwrap()
            .frame_shape(0)
            .unwrap(),
        [1, Y, X]
    );
}
