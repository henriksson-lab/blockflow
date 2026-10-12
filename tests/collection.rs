// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **Enumerating plates and manifests.** The relationships between plate,
// acquisition, well and field come from metadata or an explicit manifest —
// never from a directory scan — and every refusal below is a relationship that
// would otherwise be silently invented or merged.

#![cfg(feature = "zarr")]

use std::fs;
use std::path::Path;

use blockflow::collection::{open_manifest, open_plate};
use serde_json::{json, Value};

mod support;
use support::scratch::ScratchDir;

fn group(path: &Path, attributes: Value) {
    fs::create_dir_all(path).unwrap();
    fs::write(
        path.join("zarr.json"),
        serde_json::to_vec(
            &json!({"zarr_format": 3, "node_type": "group", "attributes": attributes}),
        )
        .unwrap(),
    )
    .unwrap();
}

fn plate(root: &Path, acquisitions: Value, images: Value) {
    group(
        root,
        json!({"ome": {"version": "0.5", "plate": {
            "name": "P1",
            "rows": [{"name": "A"}, {"name": "B"}],
            "columns": [{"name": "1"}, {"name": "2"}],
            "wells": [
                {"path": "B/2", "rowIndex": 1, "columnIndex": 1},
                {"path": "A/1", "rowIndex": 0, "columnIndex": 0}
            ],
            "acquisitions": acquisitions,
        }}}),
    );
    for well in ["A/1", "B/2"] {
        group(
            &root.join(well),
            json!({"ome": {"well": {"images": images.clone()}}}),
        );
        for field in ["0", "1"] {
            group(&root.join(well).join(field), json!({}));
        }
    }
}

#[test]
fn a_plate_lists_every_field_in_key_order_with_its_acquisition() {
    let scratch = ScratchDir::new("collection", "plate");
    let root = scratch.path().join("plate.zarr");
    plate(
        &root,
        json!([{"id": 0}, {"id": 1}]),
        json!([{"path": "0", "acquisition": 1}, {"path": "1", "acquisition": 0}]),
    );
    let series = open_plate(&root).unwrap();
    let keys: Vec<_> = series
        .iter()
        .map(|s| {
            (
                s.key.well_row.as_str(),
                s.key.well_column.as_str(),
                s.key.field_id.as_str(),
                s.key.acquisition_id,
            )
        })
        .collect();
    assert_eq!(
        keys,
        // Key order: plate, acquisition, well, field.
        [
            ("A", "1", "1", Some(0)),
            ("B", "2", "1", Some(0)),
            ("A", "1", "0", Some(1)),
            ("B", "2", "0", Some(1))
        ]
    );
    assert!(series.iter().all(|s| s.key.plate_id == "P1"));
    assert_eq!(series[0].image, root.join("A/1/1"));
    assert_eq!(series[1].well_indices, Some((1, 1)));
}

#[test]
fn a_plate_naming_an_undeclared_acquisition_is_refused() {
    let scratch = ScratchDir::new("collection", "acquisition");
    let root = scratch.path().join("plate.zarr");
    plate(
        &root,
        json!([{"id": 0}]),
        json!([{"path": "0", "acquisition": 3}]),
    );
    assert!(open_plate(&root)
        .unwrap_err()
        .to_string()
        .contains("acquisition 3"));
}

#[test]
fn a_manifest_states_relationships_and_refuses_unsafe_or_duplicate_ones() {
    let scratch = ScratchDir::new("collection", "manifest");
    group(&scratch.path().join("a.zarr"), json!({}));
    group(&scratch.path().join("b.zarr"), json!({}));
    let write = |series: Value| {
        let path = scratch.path().join("m.json");
        fs::write(
            &path,
            serde_json::to_vec(&json!({"version": 1, "plate_id": "X", "series": series})).unwrap(),
        )
        .unwrap();
        open_manifest(&path)
    };
    let ok = write(json!([
        {"path": "b.zarr", "well_row": "B", "well_column": 1, "field": 0,
         "channels": {"segment": 1, "measure": 0},
         "timeline": [{"t_index": 0}, {"t_index": 2, "elapsed_seconds": 60.0}]},
        {"path": "a.zarr", "well_row": "A", "well_column": 1, "field": 0}
    ]))
    .unwrap();
    assert_eq!(ok[0].key.well_row, "A");
    assert_eq!(ok[1].channels.unwrap().segment, 1);
    assert_eq!(ok[1].timeline.as_ref().unwrap()[1].t_index, 2);
    assert_eq!(ok[1].key.well_column, "1");

    // The same key twice.
    assert!(write(json!([
        {"path": "a.zarr", "well_row": "A", "well_column": "1", "field": "0"},
        {"path": "b.zarr", "well_row": "A", "well_column": "1", "field": "0"}
    ]))
    .is_err());
    // Two acquisitions of one field are distinct series, not merged.
    assert_eq!(
        write(json!([
            {"path": "a.zarr", "well_row": "A", "well_column": "1", "field": "0", "acquisition": 0},
            {"path": "b.zarr", "well_row": "A", "well_column": "1", "field": "0", "acquisition": 1}
        ]))
        .unwrap()
        .len(),
        2
    );
    // A path leaving the manifest's directory.
    assert!(write(
        json!([{"path": "../a.zarr", "well_row": "A", "well_column": "1", "field": "0"}])
    )
    .is_err());
    // A missing image.
    assert!(
        write(json!([{"path": "c.zarr", "well_row": "A", "well_column": "1", "field": "0"}]))
            .is_err()
    );
}
