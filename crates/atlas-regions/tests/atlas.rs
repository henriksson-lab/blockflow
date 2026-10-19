// SPDX-License-Identifier: MIT
// Original work for this crate.

use std::path::PathBuf;
use std::sync::Arc;

use atlas_regions::{
    distance_to_surface, polyline_minima, AnnotateParams, Annotator, AtlasVolumes, Ontology,
    OntologyLevel,
};
use blockflow::ops::distance::DistanceParams;
use blockflow::voxels::Voxels;
use ndarray::Array3;

/// A synthetic two-hemisphere "brain": a box of background around a labelled
/// block, split on the third axis into region 10 (left) and 11 (right), both
/// under region 2.
fn synthetic_atlas() -> (Voxels, Voxels, Ontology) {
    let shape = (12, 10, 16);
    let annotation = Array3::from_shape_fn(shape, |(x, y, z)| {
        let inside = (2..10).contains(&x) && (2..8).contains(&y) && (2..14).contains(&z);
        match (inside, z < 8) {
            (false, _) => 0u32,
            (true, true) => 10,
            (true, false) => 11,
        }
    });
    let hemispheres = Array3::from_shape_fn(shape, |(_, _, z)| if z < 8 { 255u8 } else { 0 });
    let tree = Ontology::new([
        (0, None, None),
        (2, Some(1), Some(0)),
        (10, Some(3), Some(2)),
        (11, Some(3), Some(2)),
    ])
    .unwrap();
    (Voxels::U32(annotation), Voxels::U8(hemispheres), tree)
}

#[test]
fn points_along_a_polyline_get_regions_sides_and_a_minimum_distance() {
    let (annotation, hemispheres, tree) = synthetic_atlas();
    let distance = distance_to_surface(&annotation, &DistanceParams::default()).unwrap();
    let volumes = AtlasVolumes::new(&annotation)
        .with_hemispheres(&hemispheres)
        .with_distance_to_surface(&distance);

    // One polyline crossing the midline through the middle of the block, and
    // one that leaves the volume.
    let points = [
        [5.0, 5.0, 4.0],
        [5.0, 5.0, 7.6],
        [5.0, 5.0, 11.0],
        [5.0, 1.0, 3.0],
        [5.0, -3.0, 3.0],
    ];
    let columns = Annotator::default().columns(volumes, &points).unwrap();
    assert_eq!(columns.annotation, vec![10, 11, 11, 0, 0]);
    assert_eq!(columns.hemisphere, vec![255, 0, 0, 255, -1]);
    // Depth inside the block is bounded by the nearest face: y in 2..8 at y = 5
    // is 3 from the y = 8 face.
    assert_eq!(columns.distance_to_surface[0], 3.0);
    assert_eq!(columns.distance_to_surface[3], 0.0);
    // Outside the volume the sample is clamped onto the edge, which is
    // background.
    assert_eq!(columns.distance_to_surface[4], 0.0);

    let minima = polyline_minima(&columns.distance_to_surface, &[[0, 3], [3, 5]]).unwrap();
    assert_eq!(minima, vec![3.0, 0.0]);

    let collapsed = Annotator::new(AnnotateParams::default().with_level(Some(1)))
        .with_ontology(Arc::new(tree))
        .regions(&annotation, &points)
        .unwrap();
    assert_eq!(collapsed, vec![2, 2, 2, 0, 0]);
}

/// The shipped Allen region tree.
///
/// Needs `ABA_annotation_last.jsonl`; set `ATLAS_REGIONS_TREE` to its path.
/// The default is where it sat on the development machine.
#[test]
#[ignore = "needs the shipped ABA_annotation_last.jsonl (set ATLAS_REGIONS_TREE)"]
fn the_shipped_allen_tree_loads_and_collapses() {
    let path = std::env::var("ATLAS_REGIONS_TREE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(
                "/husky/otherdataset/teresa/clearmap_em/downloads/atlas/clearmap-work/ABA_annotation_last.jsonl",
            )
        });
    let tree = Ontology::from_jsonl_file(&path).unwrap();
    assert_eq!(tree.len(), 1328);
    assert_eq!(tree.acronym(0), Some("universe"));
    assert_eq!(tree.acronym(8), Some("grey"));

    // Every region collapses at every level and reaches a root, so the tree
    // is closed and acyclic.
    for id in tree.ids().collect::<Vec<_>>() {
        for level in 0..=11 {
            tree.collapse(id, level).unwrap();
        }
        // The file has two roots: `universe` (0) and `root` (997), which has no
        // parent.
        let top = *tree.ancestors(id).unwrap().last().unwrap();
        assert!(top == 0 || top == 997, "{id} ends at {top}");
    }

    // Cortical regions merge onto `grey` at level 1. At level 6 they collapse
    // onto 698 while 477 does not, and at level 8 597 and 605 merge onto 589
    // while 159 and 507 stay.
    for id in [159, 477, 507, 597, 605, 698] {
        assert_eq!(tree.collapse(id, 1).unwrap(), 8, "{id}");
    }
    for id in [159, 507, 597, 605, 698] {
        assert_eq!(tree.collapse(id, 6).unwrap(), 698, "{id}");
    }
    assert_ne!(tree.collapse(477, 6).unwrap(), 698);
    assert_eq!(tree.collapse(597, 8).unwrap(), 589);
    assert_eq!(tree.collapse(605, 8).unwrap(), 589);
    assert_eq!(tree.collapse(159, 8).unwrap(), 159);
    assert_eq!(tree.collapse(507, 8).unwrap(), 507);
}

/// Needs the reference's nested `ABA_annotation_last.json` and the JSON-lines
/// copy beside it; set `ATLAS_REGIONS_NESTED_TREE` to the nested file (the
/// `.jsonl` is looked for next to it, as the reference does).
#[test]
#[ignore = "needs the shipped ABA_annotation_last.json (set ATLAS_REGIONS_NESTED_TREE)"]
fn the_nested_allen_tree_agrees_with_the_json_lines_copy() {
    let nested_path = std::env::var("ATLAS_REGIONS_NESTED_TREE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(
                "/data/henriksson/github/claude/clearmap-rs/ClearMap-py/ClearMap/Resources/Atlas/\
                 ABA_annotation_last.json",
            )
        });
    let nested = Ontology::from_ontology_json_file(&nested_path, OntologyLevel::TreeDepth).unwrap();
    let lines = Ontology::from_jsonl_file(nested_path.with_extension("jsonl")).unwrap();
    assert_eq!(nested.len(), 1328);
    assert_eq!(
        nested.ids().collect::<Vec<_>>(),
        lines.ids().collect::<Vec<_>>()
    );
    for id in nested.ids() {
        assert_eq!(nested.name(id), lines.name(id), "{id}");
        assert_eq!(nested.acronym(id), lines.acronym(id), "{id}");
        assert_eq!(nested.color(id), lines.color(id), "{id}");
        assert_eq!(
            nested.structure_level(id),
            lines.structure_level(id),
            "{id}"
        );
        // The flat copy leaves `root` without a parent; elsewhere they agree.
        if let Some(parent) = lines.parent(id) {
            assert_eq!(nested.parent(id), Some(parent), "{id}");
        }
        assert_eq!(nested.level(id), Some(nested.depth(id).unwrap() as i64));
    }
    assert_eq!(nested.parent(997), Some(0));
    assert_eq!(nested.roots(), vec![0]);
    // The reference's level of `grey` is 2 (universe 0, root 1).
    assert_eq!(nested.level(8), Some(2));
    assert_eq!(nested.collapse(567, 2).unwrap(), 8);
}
