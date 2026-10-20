// SPDX-License-Identifier: MIT
//
// Original work for this crate.

//! The whole pipeline on a synthetic image: a tilted mucus top and epithelium,
//! nine beads in the mucus, one large AF488 object. Checks what is found, what
//! is published, and that two block grids give byte-identical output.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use blockflow_bead_layer_distance::{run, Config};
use clap::Parser;
use zarrs::array::{data_type, ArrayBuilder, ArraySubset, FillValue};
use zarrs::filesystem::FilesystemStore;

const NZ: usize = 40;
const NY: usize = 64;
const NX: usize = 64;

fn mucus_top(y: usize) -> usize {
    10 + y / 16
}

fn epithelium_top(x: usize) -> usize {
    26 + x / 32
}

const BEADS: [[usize; 3]; 9] = [
    [16, 8, 8],
    [18, 8, 20],
    [20, 8, 32],
    [18, 20, 8],
    [20, 20, 20],
    [16, 20, 32],
    [20, 32, 8],
    [16, 32, 20],
    [18, 32, 32],
];
const BLOB: [usize; 3] = [18, 48, 48];
/// A bead under the epithelium (stained z 26..29 here): an `in_tissue`
/// candidate whose distances must come out negative.
const BENEATH: [usize; 3] = [34, 48, 16];

fn write_image(root: &Path) {
    fs::create_dir_all(root).unwrap();
    let scale = [1.0, 1.0, 5.0, 2.0, 2.0];
    let attrs = serde_json::json!({"ome": {"version": "0.5", "multiscales": [{
        "axes": [
            {"name": "t", "type": "time"}, {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"}],
        "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": scale}]}]}]}});
    fs::write(
        root.join("zarr.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"zarr_format": 3, "node_type": "group", "attributes": attrs}),
        )
        .unwrap(),
    )
    .unwrap();
    let mut seed = 17u64;
    let mut noise = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) % 31) as f64 - 15.0
    };
    let mut data = vec![0u16; 4 * NZ * NY * NX];
    let at = |c: usize, z: usize, y: usize, x: usize| ((c * NZ + z) * NY + y) * NX + x;
    for z in 0..NZ {
        for y in 0..NY {
            for x in 0..NX {
                let (hm, he) = (mucus_top(y), epithelium_top(x));
                let epithelium = z >= he && z < he + 3;
                let mut blue = 60.0 + noise() + if epithelium { 150.0 } else { 0.0 };
                for b in BEADS.iter().chain([&BENEATH]) {
                    let d = ((z as f64 - b[0] as f64) / 1.0).powi(2)
                        + ((y as f64 - b[1] as f64) / 1.5).powi(2)
                        + ((x as f64 - b[2] as f64) / 1.5).powi(2);
                    blue += 1500.0 * (-0.5 * d).exp();
                }
                let blob = ((z as f64 - BLOB[0] as f64) / 2.0).powi(2)
                    + ((y as f64 - BLOB[1] as f64) / 6.0).powi(2)
                    + ((x as f64 - BLOB[2] as f64) / 6.0).powi(2)
                    <= 1.0;
                let af488 = 60.0
                    + noise()
                    + if epithelium { 2000.0 } else { 0.0 }
                    + if blob { 3000.0 } else { 0.0 };
                let af594 = 60.0 + noise() + if z >= hm && z < hm + 2 { 800.0 } else { 0.0 };
                data[at(0, z, y, x)] = blue.round() as u16;
                data[at(1, z, y, x)] = af488.round() as u16;
                data[at(2, z, y, x)] = af594.round() as u16;
            }
        }
    }
    fs::create_dir_all(root.join("0")).unwrap();
    let store = Arc::new(FilesystemStore::new(root).unwrap());
    let shape = vec![1, 4, NZ as u64, NY as u64, NX as u64];
    let array = ArrayBuilder::new(
        shape.clone(),
        vec![1, 1, 32, 64, 64],
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

fn config(zarr: &Path, out: &Path, block: &str, stained: bool) -> Config {
    let mut args = vec![
        "bead-layer-distance",
        "--zarr",
        zarr.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
        "--workers",
        "2",
        "--block",
        block,
        // nine equal beads have no dim/bright split for log-Otsu to find; a
        // fixed level well above the noise peaks does the same job
        "--bead-min-response",
        "50",
        // the synthetic sheet is 64 x 64 x 3 voxels, 0.25e6 µm³
        "--epithelium-min-volume-um3",
        "50000",
    ];
    if stained {
        args.push("--stained-epithelium");
    }
    Config::parse_from(args)
}

fn scratch() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("bead-layer-distance-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    dir
}

#[test]
fn finds_layers_beads_and_the_large_object_on_any_grid() {
    if cfg!(debug_assertions) {
        eprintln!("skipped: run with --release");
        return;
    }
    let dir = scratch();
    let zarr = dir.join("synthetic.ome.zarr");
    write_image(&zarr);

    let mut outputs = Vec::new();
    for (block, name) in [("32,256,256", "a"), ("16,32,32", "b")] {
        let out = dir.join(name);
        let summary = run(&config(&zarr, &out, block, true)).unwrap();
        assert_eq!(summary.beads, BEADS.len(), "{summary:?}");
        assert_eq!(summary.large_objects, 1, "{summary:?}");
        outputs.push(out);
    }
    for file in [
        "beads.csv",
        "large_af488_objects.csv",
        "height_epi.npy",
        "height_mucus.npy",
    ] {
        assert_eq!(
            fs::read(outputs[0].join(file)).unwrap(),
            fs::read(outputs[1].join(file)).unwrap(),
            "{file} depends on the block grid"
        );
    }

    // The surfaces sit at the first stained slice: the default edge recipe
    // (`relative`) has no position bias. (The original `log` recipe put them
    // two slices above it.)
    const BIAS: f32 = 0.0;
    let epi: ndarray::ArrayD<f32> = blockflow::npy::read_array_file(
        &outputs[0].join("height_epi.npy"),
        blockflow::npy::OrderPolicy::Either,
    )
    .unwrap();
    let mucus: ndarray::ArrayD<f32> = blockflow::npy::read_array_file(
        &outputs[0].join("height_mucus.npy"),
        blockflow::npy::OrderPolicy::Either,
    )
    .unwrap();
    let mut off = 0;
    for y in 0..NY {
        for x in 0..NX {
            off += usize::from((epi[[y, x]] - (epithelium_top(x) as f32 + BIAS)).abs() > 1.0);
            off += usize::from((mucus[[y, x]] - (mucus_top(y) as f32 + BIAS)).abs() > 1.0);
        }
    }
    // (within a slice: columns are pooled 4 px wide, so steps blur over a column)
    assert!(
        off < NY * NX / 50,
        "{off} of {} column heights off",
        2 * NY * NX
    );

    // every bead is measured inside the mucus, above the epithelium
    let csv = fs::read_to_string(outputs[0].join("beads.csv")).unwrap();
    let header: Vec<&str> = csv.lines().next().unwrap().split(',').collect();
    let column = |name: &str| header.iter().position(|h| *h == name).unwrap();
    let kept: Vec<Vec<&str>> = csv
        .lines()
        .skip(1)
        .map(|l| l.split(',').collect::<Vec<_>>())
        .filter(|r| r[column("kept")] == "1")
        .collect();
    let value = |row: &Vec<&str>, name: &str| -> f64 { row[column(name)].parse().unwrap() };
    for row in &kept {
        let depth = value(row, "relative_depth");
        let vertical = value(row, "vertical_dist_um");
        assert!((0.0..1.0).contains(&depth) && vertical > 0.0, "{row:?}");
        // to the fitted tissue region: the height map's nearest point
        assert!(
            (value(row, "surface_dist_um") - value(row, "euclidean_dist_um")).abs() < 1e-6,
            "{row:?}"
        );
        // to the stained epithelium: the drawn top, within a slice
        let p = [value(row, "z"), value(row, "y"), value(row, "x")];
        let mut drawn = f64::INFINITY;
        for y in 0..NY {
            for x in 0..NX {
                let d = (((epithelium_top(x) as f64 - p[0]) * 5.0).powi(2)
                    + ((y as f64 - p[1]) * 2.0).powi(2)
                    + ((x as f64 - p[2]) * 2.0).powi(2))
                .sqrt();
                drawn = drawn.min(d);
            }
        }
        let stained = value(row, "epithelium_dist_um");
        assert!(
            (stained - drawn).abs() <= 5.0,
            "{stained} vs {drawn}: {row:?}"
        );
        assert_eq!(row[column("epithelium_status")], "exact");
    }
    // beneath the epithelium: negative to both
    let beneath = csv
        .lines()
        .skip(1)
        .map(|l| l.split(',').collect::<Vec<_>>())
        .find(|r| [value(r, "z"), value(r, "y"), value(r, "x")] == BENEATH.map(|v| v as f64))
        .expect("the bead beneath the epithelium is a candidate");
    assert_eq!(beneath[column("class")], "in_tissue");
    let expected = -((BENEATH[0] - epithelium_top(BENEATH[2])) as f64) * 5.0;
    for name in ["surface_dist_um", "epithelium_dist_um"] {
        assert!(
            (value(&beneath, name) - expected).abs() <= 5.0,
            "{name}: {} vs {expected}",
            value(&beneath, name)
        );
    }

    // published for a viewer: four labels, two tables paired with them
    let registry: serde_json::Value =
        serde_json::from_slice(&fs::read(zarr.join("labels/zarr.json")).unwrap()).unwrap();
    let mut labels: Vec<String> =
        serde_json::from_value(registry["attributes"]["ome"]["labels"].clone()).unwrap();
    labels.sort();
    assert_eq!(
        labels,
        [
            "bjorn-beads",
            "bjorn-epithelium",
            "bjorn-large-af488",
            "bjorn-layers",
            "bjorn-surfaces"
        ]
    );
    for table in ["bjorn-beads", "bjorn-large-af488"] {
        let meta: serde_json::Value = serde_json::from_slice(
            &fs::read(zarr.join("tables").join(table).join("zarr.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            meta["attributes"]["object_table"]["region"],
            format!("../../labels/{table}")
        );
    }

    // by default (no stained epithelium) a rerun drops its label and columns,
    // and nothing else changes
    let out = dir.join("c");
    run(&config(&zarr, &out, "32,256,256", false)).unwrap();
    let registry: serde_json::Value =
        serde_json::from_slice(&fs::read(zarr.join("labels/zarr.json")).unwrap()).unwrap();
    let mut labels: Vec<String> =
        serde_json::from_value(registry["attributes"]["ome"]["labels"].clone()).unwrap();
    labels.sort();
    assert_eq!(
        labels,
        [
            "bjorn-beads",
            "bjorn-large-af488",
            "bjorn-layers",
            "bjorn-surfaces"
        ]
    );
    assert!(!zarr.join("labels/bjorn-epithelium").exists());
    let plain = fs::read_to_string(out.join("beads.csv")).unwrap();
    assert!(!plain.lines().next().unwrap().contains("epithelium_"));
    let strip = |text: &str| -> Vec<String> {
        text.lines()
            .map(|l| l.split(',').take(26).collect::<Vec<_>>().join(","))
            .collect()
    };
    assert_eq!(strip(&plain), strip(&csv));
    let _ = fs::remove_dir_all(&dir);
}
