// SPDX-License-Identifier: MIT
//
// End-to-end: a sharded, blosc-compressed uint16 OME-Zarr 0.5 image gets an
// intensity pyramid in place, and every new voxel equals a direct reference
// mean over its window of level 0.
#![cfg(feature = "zarr")]

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use blockflow::intensity_pyramid::{build_pyramid, plan_pyramid, Downsample, PyramidOptions};
use blockflow::zarr_env::{BloscCname, BloscShuffle, Compression, ZarrEnvironment};
use blockflow::zarr_layout::StorageLayout;
use blockflow::{AttachedImage, Dtype};
use serde_json::{json, Value as Json};
use zarrs::array::{Array as ZarrArray, ArraySubset};
use zarrs::filesystem::FilesystemStore;

const SHAPE: [usize; 3] = [70, 130, 110];

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("intensity-pyramid-{name}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn value(z: usize, y: usize, x: usize) -> u16 {
    // Varied enough that a wrong window or rounding shows.
    ((z * 7919 + y * 104_729 + x * 31 + (z * y * x) % 977) % 65_000) as u16
}

/// Write a sharded blosc image with a large vendor attribute beside `ome`.
fn write_image(root: &PathBuf) -> String {
    let layout = StorageLayout::sharded(
        vec![16, 32, 32],
        vec![32, 64, 64],
        Compression::Blosc {
            cname: BloscCname::Zstd,
            clevel: 5,
            shuffle: BloscShuffle::Byte,
        },
    );
    let metadata = layout
        .array_metadata(&SHAPE, Dtype::U16, json!(0), Some(json!(["z", "y", "x"])))
        .unwrap();
    fs::create_dir_all(root.join("0")).unwrap();
    fs::write(
        root.join("0/zarr.json"),
        serde_json::to_vec_pretty(&metadata).unwrap(),
    )
    .unwrap();
    let store = Arc::new(FilesystemStore::new(root).unwrap());
    let array = ZarrArray::open(store, "/0").unwrap();
    let mut data = Vec::with_capacity(SHAPE.iter().product());
    for z in 0..SHAPE[0] {
        for y in 0..SHAPE[1] {
            for x in 0..SHAPE[2] {
                data.push(value(z, y, x));
            }
        }
    }
    let subset =
        ArraySubset::new_with_start_shape(vec![0; 3], SHAPE.iter().map(|&v| v as u64).collect())
            .unwrap();
    array.store_array_subset(&subset, data.as_slice()).unwrap();

    let group = r#"{
  "zarr_format": 3,
  "node_type": "group",
  "attributes": {
    "ome": {
      "version": "0.5",
      "multiscales": [
        {
          "name": "test",
          "axes": [
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"}
          ],
          "datasets": [
            {
              "path": "0",
              "coordinateTransformations": [
                {"type": "scale", "scale": [1.0, 0.5, 0.5]}
              ]
            }
          ]
        }
      ]
    },
    "vendor": {"keep": [1, 2, 3], "text": "a ] tricky } string"}
  }
}"#;
    fs::write(root.join("zarr.json"), group).unwrap();
    group.to_owned()
}

/// The direct definition: mean over the clipped window of level 0, rounded
/// half away from zero.
fn reference(factor: [usize; 3], at: [usize; 3]) -> u16 {
    let mut sum = 0u64;
    let mut count = 0u64;
    for z in at[0] * factor[0]..((at[0] + 1) * factor[0]).min(SHAPE[0]) {
        for y in at[1] * factor[1]..((at[1] + 1) * factor[1]).min(SHAPE[1]) {
            for x in at[2] * factor[2]..((at[2] + 1) * factor[2]).min(SHAPE[2]) {
                sum += value(z, y, x) as u64;
                count += 1;
            }
        }
    }
    ((sum as f64 / count as f64) + 0.5).floor() as u16
}

#[test]
fn a_sharded_blosc_image_gets_a_correct_pyramid_in_place() {
    let root = scratch("in-place");
    let original_text = write_image(&root);
    let level0_before = fs::read(root.join("0/zarr.json")).unwrap();

    // The environment attaches a sharded blosc array and reads any region.
    let attached = AttachedImage::at(root.join("0"));
    assert_eq!(attached.metadata().unwrap(), (Dtype::U16, SHAPE));
    let env = ZarrEnvironment::attach(root.join("work"), &[attached]).unwrap();
    assert_eq!(env.chunk_at(0).unwrap(), [16, 32, 32]);
    fs::remove_dir_all(root.join("work")).unwrap();

    let options = PyramidOptions {
        target_size: 10,
        workers: 4,
        ..PyramidOptions::default()
    };
    let plan = plan_pyramid(&root, &options).unwrap();
    // Anisotropic z (1.0 vs 0.5): y, x halve first, then everything.
    let factors: Vec<Vec<usize>> = plan.new_levels.iter().map(|l| l.factor.clone()).collect();
    assert_eq!(
        factors,
        vec![vec![1, 2, 2], vec![2, 2, 2], vec![2, 2, 2], vec![2, 2, 2]]
    );
    let shapes: Vec<Vec<usize>> = plan.new_levels.iter().map(|l| l.shape.clone()).collect();
    assert_eq!(
        shapes,
        vec![
            vec![70, 65, 55],
            vec![35, 33, 28],
            vec![18, 17, 14],
            vec![9, 9, 7]
        ]
    );
    // Layout follows level 0, capped to the level.
    assert_eq!(plan.new_levels[0].layout.shard, Some(vec![32, 64, 64]));
    assert_eq!(plan.new_levels[3].layout.chunk, vec![9, 9, 7]);

    let written = build_pyramid(&plan, &options, &|_| {}).unwrap();
    assert_eq!(written, vec!["1", "2", "3", "4"]);
    assert_eq!(fs::read(root.join("0/zarr.json")).unwrap(), level0_before);

    // Values: level 1 against level 0 directly; deeper levels against the
    // level above them, recomputed here.
    let store = Arc::new(FilesystemStore::new(&root).unwrap());
    let read = |path: &str| -> (Vec<u16>, Vec<usize>) {
        let array = ZarrArray::open(store.clone(), &format!("/{path}")).unwrap();
        let shape: Vec<usize> = array.shape().iter().map(|&v| v as usize).collect();
        let data: Vec<u16> = array.retrieve_array_subset(&array.subset_all()).unwrap();
        (data, shape)
    };
    let (level1, shape1) = read("1");
    for z in 0..shape1[0] {
        for y in 0..shape1[1] {
            for x in 0..shape1[2] {
                let got = level1[(z * shape1[1] + y) * shape1[2] + x];
                assert_eq!(
                    got,
                    reference([1, 2, 2], [z, y, x]),
                    "level 1 at {z},{y},{x}"
                );
            }
        }
    }
    let mut previous = (level1, shape1);
    for path in ["2", "3", "4"] {
        let (data, shape) = read(path);
        let (prev, pshape) = &previous;
        for z in 0..shape[0] {
            for y in 0..shape[1] {
                for x in 0..shape[2] {
                    let mut sum = 0u64;
                    let mut count = 0u64;
                    for zz in 2 * z..(2 * z + 2).min(pshape[0]) {
                        for yy in 2 * y..(2 * y + 2).min(pshape[1]) {
                            for xx in 2 * x..(2 * x + 2).min(pshape[2]) {
                                sum += prev[(zz * pshape[1] + yy) * pshape[2] + xx] as u64;
                                count += 1;
                            }
                        }
                    }
                    let expected = ((sum as f64 / count as f64) + 0.5).floor() as u16;
                    assert_eq!(
                        data[(z * shape[1] + y) * shape[2] + x],
                        expected,
                        "level {path}"
                    );
                }
            }
        }
        previous = (data, shape);
    }

    // Metadata: every level listed with the right scale and translation, and
    // everything else unchanged.
    let after: Json = serde_json::from_slice(&fs::read(root.join("zarr.json")).unwrap()).unwrap();
    let datasets = after["attributes"]["ome"]["multiscales"][0]["datasets"]
        .as_array()
        .unwrap();
    let paths: Vec<&str> = datasets
        .iter()
        .map(|d| d["path"].as_str().unwrap())
        .collect();
    assert_eq!(paths, ["0", "1", "2", "3", "4"]);
    assert_eq!(
        datasets[1]["coordinateTransformations"],
        json!([{"type": "scale", "scale": [1.0, 1.0, 1.0]},
               {"type": "translation", "translation": [0.0, 0.25, 0.25]}])
    );
    assert_eq!(
        datasets[4]["coordinateTransformations"][0]["scale"],
        json!([8.0, 8.0, 8.0])
    );
    // translation(level k+1) = translation(k) + (f - 1) / 2 * scale(k).
    assert_eq!(
        datasets[4]["coordinateTransformations"][1]["translation"],
        json!([3.5, 3.75, 3.75])
    );
    let mut original: Json = serde_json::from_str(&original_text).unwrap();
    original["attributes"]["ome"]["multiscales"][0]["datasets"] = json!(datasets.clone());
    assert_eq!(after, original);
    let text = fs::read_to_string(root.join("zarr.json")).unwrap();
    assert!(text.contains(r#""vendor": {"keep": [1, 2, 3], "text": "a ] tricky } string"}"#));

    // A second run has nothing to do, and the plan reports so.
    let again = plan_pyramid(&root, &options).unwrap();
    assert!(again.new_levels.is_empty());

    // An unlisted leftover level directory is refused without --overwrite.
    let root2 = scratch("conflict");
    write_image(&root2);
    fs::create_dir_all(root2.join("1")).unwrap();
    let plan = plan_pyramid(&root2, &options).unwrap();
    assert_eq!(plan.conflicts, vec!["1"]);
    assert!(build_pyramid(&plan, &options, &|_| {}).is_err());
    let overwrite = PyramidOptions {
        overwrite: true,
        max_levels: Some(2),
        method: Downsample::Max,
        ..options.clone()
    };
    let plan = plan_pyramid(&root2, &overwrite).unwrap();
    assert_eq!(
        build_pyramid(&plan, &overwrite, &|_| {}).unwrap(),
        vec!["1"]
    );
    let (max1, shape1) = {
        let store = Arc::new(FilesystemStore::new(&root2).unwrap());
        let array = ZarrArray::open(store, "/1").unwrap();
        let data: Vec<u16> = array.retrieve_array_subset(&array.subset_all()).unwrap();
        (data, array.shape().to_vec())
    };
    assert_eq!(shape1, vec![70, 65, 55]);
    let expected = (0..2)
        .flat_map(|y| (0..2).map(move |x| value(3, y, x)))
        .max()
        .unwrap();
    assert_eq!(max1[3 * 65 * 55], expected);
}

#[test]
fn a_separate_output_group_links_level_zero_and_leaves_the_source_alone() {
    let root = scratch("source");
    let original = write_image(&root);
    let out = scratch("output").join("out.zarr");
    let options = PyramidOptions {
        target_size: 40,
        workers: 2,
        chunk: Some(vec![8, 16, 16]),
        shard: Some(blockflow::intensity_pyramid::ShardChoice::None),
        compression: Some(Compression::Zstd(3, false)),
        output: Some(out.clone()),
        ..PyramidOptions::default()
    };
    let plan = plan_pyramid(&root, &options).unwrap();
    build_pyramid(&plan, &options, &|_| {}).unwrap();
    assert_eq!(
        fs::read_to_string(root.join("zarr.json")).unwrap(),
        original
    );
    assert!(!root.join("1").exists());
    assert!(out.join("0").is_symlink());
    let level1: Json = serde_json::from_slice(&fs::read(out.join("1/zarr.json")).unwrap()).unwrap();
    assert_eq!(
        level1["chunk_grid"]["configuration"]["chunk_shape"],
        json!([8, 16, 16])
    );
    assert_eq!(level1["codecs"][1]["name"], "zstd");
    let reopened = plan_pyramid(&out, &PyramidOptions::default()).unwrap();
    assert_eq!(reopened.existing.len(), 1 + plan.new_levels.len());
}
