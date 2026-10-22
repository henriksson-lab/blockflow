// SPDX-License-Identifier: MIT
#![cfg(feature = "zarr")]

use blockflow::env::Environment;
use blockflow::voxels::Voxels;
use blockflow::zarr_env::{PhysicalFrame, PyramidSpec, ZarrEnvironment};
use ndarray::Array3;
use serde_json::Value;
use std::path::PathBuf;

fn scratch(name: &str) -> PathBuf {
    let root =
        PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("pyramid-coordinates-{name}"));
    let _ = std::fs::remove_dir_all(&root);
    root
}

fn transform(root: &std::path::Path, level: usize) -> ([f64; 3], [f64; 3]) {
    let metadata: Value =
        serde_json::from_slice(&std::fs::read(root.join("zarr.json")).unwrap()).unwrap();
    let axes = metadata["attributes"]["ome"]["multiscales"][0]["axes"]
        .as_array()
        .unwrap();
    for axis in axes {
        assert_ne!(axis["unit"].as_str(), Some("voxel"));
    }
    let transforms = metadata["attributes"]["ome"]["multiscales"][0]["datasets"][level]
        ["coordinateTransformations"]
        .as_array()
        .unwrap();
    let mut scale = [1.0; 3];
    let mut origin = [0.0; 3];
    for item in transforms {
        let target = if item["type"] == "scale" {
            &mut scale
        } else {
            &mut origin
        };
        let key = if item["type"] == "scale" {
            "scale"
        } else {
            "translation"
        };
        *target = std::array::from_fn(|axis| item[key][axis].as_f64().unwrap());
    }
    (scale, origin)
}

// Independent oracle: average source sample coordinates, rather than calling
// the implementation's phase helper or deriving values from its metadata.
#[test]
fn numeric_ramps_and_impulses_locate_cumulative_full_windows() {
    for calibrated in [false, true] {
        let root = scratch(if calibrated { "physical" } else { "voxel" });
        let origin = if calibrated {
            [-7.0, 5.0, -2.5]
        } else {
            [0.0; 3]
        };
        let pitch = if calibrated {
            [3.0, 0.5, 1.25]
        } else {
            [1.0; 3]
        };
        let world =
            |i: [usize; 3]| std::array::from_fn::<_, 3, _>(|a| origin[a] + pitch[a] * i[a] as f64);
        let signal = |p: [f64; 3]| 2.0 * p[0] + 3.0 * p[1] + 5.0 * p[2];
        let input = Array3::from_shape_fn((9, 11, 13), |(z, y, x)| signal(world([z, y, x])));
        let mut spec = PyramidSpec::new(vec![[2, 3, 2], [2, 1, 3]])
            .unwrap()
            .with_chunk([2, 2, 2]);
        if calibrated {
            spec = spec
                .with_physical(PhysicalFrame::new(pitch, Some("micrometer")).with_origin(origin));
        }
        let pyramid = ZarrEnvironment::build_multiscale(&root, &input.into(), &spec).unwrap();
        for (level, factor) in [[1, 1, 1], [2, 3, 2], [4, 3, 6]].into_iter().enumerate() {
            let env = ZarrEnvironment::attach(
                scratch(&format!("read-{calibrated}-{level}")),
                &[pyramid.level(level).unwrap().clone()],
            )
            .unwrap();
            let image = env.image(0).unwrap();
            assert_eq!(
                image.shape(),
                [9 / factor[0], 11 / factor[1], 13 / factor[2]]
            ); // odd tails deliberately cropped
            let values = image.view::<f64>().unwrap();
            let (scale, translation) = transform(&root, level);
            for ((z, y, x), &actual) in values.indexed_iter() {
                let index = [z, y, x];
                let mut centroid = [0.0; 3];
                let mut count = 0.0;
                for dz in 0..factor[0] {
                    for dy in 0..factor[1] {
                        for dx in 0..factor[2] {
                            let p =
                                world([z * factor[0] + dz, y * factor[1] + dy, x * factor[2] + dx]);
                            for a in 0..3 {
                                centroid[a] += p[a];
                            }
                            count += 1.0;
                        }
                    }
                }
                for a in 0..3 {
                    centroid[a] /= count;
                    assert!(
                        (translation[a] + index[a] as f64 * scale[a] - centroid[a]).abs() < 1e-10
                    );
                }
                assert!((actual - signal(centroid)).abs() < 1e-10);
            }
        }
        // One impulse inside a retained cumulative block; another in a cropped
        // odd tail. The latter must not leak into any lower-resolution sample.
        let root = scratch(&format!("impulse-{calibrated}"));
        let input = Array3::from_shape_fn((9, 11, 13), |i| {
            if i == (3, 2, 5) || i == (8, 10, 12) {
                1.0
            } else {
                0.0
            }
        });
        let pyramid = ZarrEnvironment::build_multiscale(&root, &input.into(), &spec).unwrap();
        let env = ZarrEnvironment::attach(
            scratch(&format!("impulse-read-{calibrated}")),
            &[pyramid.level(2).unwrap().clone()],
        )
        .unwrap();
        let image = env.image(0).unwrap();
        let values = image.view::<f64>().unwrap();
        assert!((values[[0, 0, 0]] - 1.0 / 72.0).abs() < 1e-15);
        assert!((values.sum() - 1.0 / 72.0).abs() < 1e-15);
    }
}

#[test]
fn boolean_first_samples_retain_origin_with_or_without_calibration() {
    for calibrated in [false, true] {
        let root = scratch(&format!("bool-{calibrated}"));
        let origin = if calibrated {
            [-7.0, 5.0, -2.5]
        } else {
            [0.0; 3]
        };
        let pitch = if calibrated {
            [3.0, 0.5, 1.25]
        } else {
            [1.0; 3]
        };
        let input = Array3::from_shape_fn((9, 11, 13), |(z, y, x)| (z * 17 + y * 11 + x) % 5 == 0);
        let mut spec = PyramidSpec::new(vec![[2, 3, 2], [2, 1, 3]]).unwrap();
        if calibrated {
            spec = spec.with_physical(PhysicalFrame::new(pitch, None).with_origin(origin));
        }
        let pyramid =
            ZarrEnvironment::build_multiscale(&root, &Voxels::Bool(input.clone()), &spec).unwrap();
        for (level, factor) in [[1, 1, 1], [2, 3, 2], [4, 3, 6]].into_iter().enumerate() {
            let (scale, translation) = transform(&root, level);
            assert_eq!(translation, origin);
            let env = ZarrEnvironment::attach(
                scratch(&format!("bool-read-{calibrated}-{level}")),
                &[pyramid.level(level).unwrap().clone()],
            )
            .unwrap();
            let image = env.image(0).unwrap();
            for ((z, y, x), &value) in image.view::<bool>().unwrap().indexed_iter() {
                assert_eq!(value, input[[z * factor[0], y * factor[1], x * factor[2]]]);
                let index = [z, y, x];
                for a in 0..3 {
                    assert_eq!(
                        translation[a] + index[a] as f64 * scale[a],
                        origin[a] + index[a] as f64 * factor[a] as f64 * pitch[a]
                    );
                }
            }
        }
    }
}
