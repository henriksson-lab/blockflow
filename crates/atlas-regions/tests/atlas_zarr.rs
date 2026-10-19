// SPDX-License-Identifier: MIT
// Original work for this crate.

//! A synthetic atlas through preparation, the OME-Zarr writer and back.

#![cfg(feature = "zarr")]

use std::path::PathBuf;

use atlas_regions::atlas_zarr::{
    open_prepared_atlas, read_volume, write_prepared_atlas, WriteAtlasParams,
};
use atlas_regions::{
    prepare_atlas, AtlasStack, Crop, HemisphereSelection, Ontology, OntologyLevel, Orientation,
    PrepareParams,
};
use blockflow::ome_zarr::OmeVolumePyramid;
use blockflow::voxels::Voxels;
use ndarray::Array3;
use serde_json::Value;

const TREE: &str = r#"{"msg": [{"id": 0, "name": "universe", "acronym": "universe",
  "color_hex_triplet": "000000", "st_level": null, "children": [
  {"id": 997, "name": "root", "acronym": "root", "color_hex_triplet": "FFFFFF", "st_level": 0,
   "children": [
    {"id": 8, "name": "grey", "acronym": "grey", "color_hex_triplet": "BFDAE3", "st_level": 1,
     "children": [{"id": 567, "name": "Cerebrum", "acronym": "CH",
                   "color_hex_triplet": "B0F0FF", "st_level": 2, "children": []}]}]}]}]}"#;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("atlas-regions-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn stack() -> AtlasStack {
    let shape = (64, 36, 34);
    AtlasStack {
        annotation: Voxels::U32(Array3::from_shape_fn(shape, |(z, y, _)| {
            if y < 4 {
                0
            } else if z < 32 {
                567
            } else {
                8
            }
        })),
        reference: Some(Voxels::U16(Array3::from_shape_fn(shape, |(z, y, x)| {
            (z * 1000 + y * 30 + x) as u16
        }))),
        hemispheres: Some(Voxels::U8(Array3::from_shape_fn(shape, |(z, _, _)| {
            if z < 32 {
                255
            } else {
                0
            }
        }))),
        distance_to_surface: Some(Voxels::F64(Array3::from_shape_fn(shape, |(_, y, _)| {
            y as f64 * 0.5
        }))),
    }
}

#[test]
fn a_prepared_atlas_round_trips_through_ome_zarr() {
    let dir = scratch("round-trip");
    let root = dir.join("atlas.ome.zarr");
    let ontology = Ontology::from_ontology_json(TREE, OntologyLevel::TreeDepth).unwrap();
    let params = PrepareParams {
        orientation: Orientation::from_xyz([1, -2, 3]).unwrap(),
        crop: Crop::parse_zyx(":,2:,:").unwrap(),
        hemisphere: Some(HemisphereSelection {
            value: 255,
            margin: 0,
        }),
    };
    let (prepared, window) = prepare_atlas(&stack(), &params).unwrap();
    assert_eq!(prepared.shape(), [32, 34, 34]);
    let written = write_prepared_atlas(
        &root,
        &prepared,
        &window,
        &ontology,
        &WriteAtlasParams {
            voxel_size_um: [25.0; 3],
            name: "synthetic".into(),
            replace: false,
            workers: 2,
        },
    )
    .unwrap();
    assert_eq!(written.regions_present, 1);
    assert_eq!(written.levels, 2);

    let pyramid = OmeVolumePyramid::open(&root, 0, 0).unwrap();
    assert_eq!(pyramid.voxel_size(), [25.0; 3]);
    assert_eq!(pyramid.levels().len(), 2);
    let atlas = open_prepared_atlas(&root, OntologyLevel::TreeDepth).unwrap();
    assert_eq!(atlas.window, window);
    assert_eq!(atlas.voxel_size_um, [25.0; 3]);
    assert_eq!(atlas.ontology.parent(567), Some(8));
    assert_eq!(atlas.ontology.level(567), Some(3));
    assert_eq!(atlas.ontology.structure_level(567), Some(2));
    assert_eq!(atlas.ontology.acronym(567), Some("CH"));
    assert_eq!(atlas.ontology.color(8), Some([0xBF, 0xDA, 0xE3]));
    assert_eq!(atlas.stack.reference, prepared.reference);
    let as_u64 = |v: &Voxels| v.widened_i64("labels").unwrap();
    assert_eq!(
        as_u64(&atlas.stack.annotation),
        as_u64(&prepared.annotation)
    );
    assert_eq!(
        as_u64(atlas.stack.hemispheres.as_ref().unwrap()),
        as_u64(prepared.hemispheres.as_ref().unwrap())
    );
    let distance = atlas.stack.distance_to_surface.unwrap();
    assert!(matches!(distance, Voxels::F32(_)));
    assert_eq!(
        distance.widened(),
        prepared.distance_to_surface.as_ref().unwrap().widened()
    );
    // y is flipped (oriented y = 35 - y) and cropped from 2, so the source's
    // background rows y < 4 end up last: prepared y 30..34.
    let annotation = atlas.stack.annotation.view::<u64>().unwrap();
    assert_eq!(annotation[[0, 0, 0]], 567);
    assert_eq!(annotation[[0, 33, 0]], 0);

    let label: Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("labels/annotation/zarr.json")).unwrap(),
    )
    .unwrap();
    let properties = label
        .pointer("/attributes/image-label/properties")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(properties.len(), 4);
    assert!(label.pointer("/attributes/image-label/colors").is_some());
    let labels = read_volume(&root.join("labels/hemispheres")).unwrap();
    assert_eq!(labels.shape(), [32, 34, 34]);

    // An existing atlas is not replaced unless asked.
    let refused = write_prepared_atlas(
        &root,
        &prepared,
        &window,
        &ontology,
        &WriteAtlasParams {
            voxel_size_um: [25.0; 3],
            name: "synthetic".into(),
            replace: false,
            workers: 1,
        },
    );
    assert!(refused.is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn an_annotation_id_missing_from_the_tree_is_refused() {
    let dir = scratch("missing");
    let mut atlas = stack();
    atlas.annotation = Voxels::U32(Array3::from_elem((20, 20, 20), 12345));
    atlas.reference = Some(Voxels::U16(Array3::zeros((20, 20, 20))));
    atlas.hemispheres = None;
    atlas.distance_to_surface = None;
    let (prepared, window) = prepare_atlas(&atlas, &PrepareParams::default()).unwrap();
    let ontology = Ontology::from_ontology_json(TREE, OntologyLevel::TreeDepth).unwrap();
    let error = write_prepared_atlas(
        &dir.join("atlas.ome.zarr"),
        &prepared,
        &window,
        &ontology,
        &WriteAtlasParams {
            voxel_size_um: [25.0; 3],
            name: "x".into(),
            replace: false,
            workers: 1,
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("12345"), "{error}");
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Needs the prepared Allen atlas written by `atlas-prepare` (see the README);
/// set `ATLAS_REGIONS_PREPARED` to it.
#[test]
#[ignore = "needs the prepared ABA_25um_2017.ome.zarr (set ATLAS_REGIONS_PREPARED)"]
fn the_prepared_allen_atlas_opens_and_labels_points() {
    use atlas_regions::{AnnotateParams, Annotator, AtlasVolumes};
    let root = std::env::var("ATLAS_REGIONS_PREPARED")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from("/husky/henriksson/lightsheet-reference/atlas/ABA_25um_2017.ome.zarr")
        });
    let atlas = open_prepared_atlas(&root, OntologyLevel::TreeDepth).unwrap();
    assert_eq!(atlas.stack.shape(), [456, 528, 320]);
    assert_eq!(atlas.voxel_size_um, [25.0; 3]);
    assert_eq!(atlas.ontology.len(), 1328);
    assert!(atlas.window.orientation.is_identity());
    let stack = &atlas.stack;
    let volumes = AtlasVolumes::new(&stack.annotation)
        .with_hemispheres(stack.hemispheres.as_ref().unwrap())
        .with_distance_to_surface(stack.distance_to_surface.as_ref().unwrap());
    let annotator = Annotator::new(AnnotateParams {
        level: Some(2),
        ..AnnotateParams::default()
    })
    .with_ontology(std::sync::Arc::new(atlas.ontology.clone()));
    // A point deep in the brain and one in the corner outside it.
    let columns = annotator
        .columns(volumes, &[[228.0, 264.0, 160.0], [0.0, 0.0, 0.0]])
        .unwrap();
    assert_ne!(columns.annotation[0], 0);
    assert_eq!(columns.annotation[1], 0);
    assert!(columns.distance_to_surface[0] > 10.0);
    assert_eq!(columns.distance_to_surface[1], 0.0);
}
