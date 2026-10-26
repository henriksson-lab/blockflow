// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The planned graph stages over stored volumes give the resident graph,
//! column for column, on every block grid: one block, blocks that divide the
//! volume, blocks that do not, and blocks smaller than the windows' reach; at
//! the default search limit and at one the thick vessel outgrows.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use blockflow::voxels::Voxels;
use blockflow::zarr_env::AttachedImage;
use blockflow_vessel_graph::graph::planned::{PlanParams, StoredInputs, GRAPH_BYTES_PER_VERTEX};
use blockflow_vessel_graph::graph::{
    vessel_graph, vessel_graph_stored, GraphParams, Inputs, RadiusParams,
};
use ndarray::Array3;
use zarrs::array::{data_type, ArrayBuilder, ArraySubset, FillValue};
use zarrs::filesystem::FilesystemStore;

const SHAPE: [usize; 3] = [36, 52, 44];
const SPACING: [f64; 3] = [1.6, 1.625, 1.625];

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "vessel-graph-planned-{name}-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// Tubes along each axis of radii 1 to 4, a thick one of radius 11 (wider
/// than the small search limit), their centrelines with a few branches and
/// stray voxels as the skeleton, the thick and one thin tube as arteries, one
/// tube as a vein, and pseudo-random channel intensities.
struct Scene {
    skeleton: Array3<bool>,
    vessels: Array3<bool>,
    artery: Array3<bool>,
    vein: Array3<bool>,
    artery_raw: Array3<u16>,
    vein_raw: Array3<u16>,
}

fn scene() -> Scene {
    let shape = (SHAPE[0], SHAPE[1], SHAPE[2]);
    let mut scene = Scene {
        skeleton: Array3::from_elem(shape, false),
        vessels: Array3::from_elem(shape, false),
        artery: Array3::from_elem(shape, false),
        vein: Array3::from_elem(shape, false),
        artery_raw: Array3::zeros(shape),
        vein_raw: Array3::zeros(shape),
    };
    // (axis, the two other coordinates, radius, artery, vein)
    let tubes: [(usize, [f64; 2], f64, bool, bool); 6] = [
        (2, [8.0, 10.0], 1.0, false, false),
        (2, [20.0, 30.0], 2.5, true, false),
        (1, [5.0, 33.0], 1.5, false, true),
        (0, [40.0, 8.0], 4.0, false, false),
        (1, [24.0, 20.0], 11.0, true, false),
        (0, [15.0, 38.0], 2.0, false, false),
    ];
    let mut state = 0x2545_f491_4f6c_dd1du64;
    for ((z, y, x), slot) in scene.vessels.indexed_iter_mut() {
        let at = [z as f64, y as f64, x as f64];
        for (axis, centre, radius, artery, vein) in tubes {
            let others: Vec<usize> = (0..3).filter(|a| *a != axis).collect();
            let distance =
                ((at[others[0]] - centre[0]).powi(2) + (at[others[1]] - centre[1]).powi(2)).sqrt();
            if distance <= radius {
                *slot = true;
                if artery {
                    scene.artery[[z, y, x]] = true;
                }
                if vein {
                    scene.vein[[z, y, x]] = true;
                }
                if distance < 0.5 {
                    scene.skeleton[[z, y, x]] = true;
                }
            }
        }
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        scene.artery_raw[[z, y, x]] = (state % 4096) as u16;
        scene.vein_raw[[z, y, x]] = ((state >> 20) % 1000) as u16;
    }
    // Stray skeleton voxels: inside vessels, at the volume's faces, and one
    // outside every vessel.
    for at in [
        [0, 30, 20],
        [35, 20, 30],
        [17, 51, 20],
        [3, 3, 3],
        [20, 24, 43],
    ] {
        scene.skeleton[at] = true;
    }
    scene
}

/// A `z, y, x` Zarr array at `root`, chunked unevenly.
fn store<T: zarrs::array::Element + Copy>(
    root: &Path,
    values: &Array3<T>,
    data_type: zarrs::array::DataType,
    fill: FillValue,
) -> AttachedImage {
    fs::create_dir_all(root).unwrap();
    let shape: Vec<u64> = SHAPE.iter().map(|n| *n as u64).collect();
    let store = Arc::new(FilesystemStore::new(root).unwrap());
    let array = ArrayBuilder::new(shape.clone(), vec![16, 16, 32], data_type, fill)
        .build(store, "/")
        .unwrap();
    array.store_metadata().unwrap();
    let data: Vec<T> = values.iter().copied().collect();
    array
        .store_array_subset(&ArraySubset::new_with_shape(shape), &data)
        .unwrap();
    AttachedImage::at(root)
}

fn label(root: &Path, mask: &Array3<bool>) -> AttachedImage {
    store(
        root,
        &mask.mapv(u64::from),
        data_type::uint64(),
        FillValue::from(0u64),
    )
}

fn channel(root: &Path, values: &Array3<u16>) -> AttachedImage {
    store(root, values, data_type::uint16(), FillValue::from(0u16))
}

#[test]
fn the_planned_graph_is_the_resident_graph_on_every_grid() {
    let dir = scratch("grids");
    let scene = scene();
    let stored = StoredInputs {
        skeleton: label(&dir.join("skeleton"), &scene.skeleton),
        vessels: label(&dir.join("vessels"), &scene.vessels),
        artery_intensity: Some(channel(&dir.join("artery_raw"), &scene.artery_raw)),
        artery_mask: Some(label(&dir.join("artery"), &scene.artery)),
        vein_intensity: Some(channel(&dir.join("vein_raw"), &scene.vein_raw)),
        vein_mask: Some(label(&dir.join("vein"), &scene.vein)),
        spacing: SPACING,
    };
    let skeleton = Voxels::from(scene.skeleton.clone());
    let vessels = Voxels::from(scene.vessels.clone());
    let artery = Voxels::from(scene.artery.clone());
    let vein = Voxels::from(scene.vein.clone());
    let artery_raw = Voxels::from(scene.artery_raw.clone());
    let vein_raw = Voxels::from(scene.vein_raw.clone());
    let inputs = Inputs {
        vessels: Some(&vessels),
        artery_intensity: Some(&artery_raw),
        artery_mask: Some(&artery),
        vein_intensity: Some(&vein_raw),
        vein_mask: Some(&vein),
        ..Inputs::new(&skeleton, SPACING)
    };
    for max_radius in [[50, 50, 50], [6, 6, 6]] {
        let params = GraphParams {
            radius: RadiusParams {
                max_radius,
                ..RadiusParams::default()
            },
            largest_component: false,
            ..GraphParams::default()
        };
        let resident = vessel_graph(&inputs, &params).unwrap().to_json();
        for (block, pack_block) in [
            (Some([64, 64, 64]), Some([64, 64, 64])),
            (Some([18, 26, 22]), Some([16, 16, 32])),
            (Some([7, 13, 11]), Some([5, 9, 17])),
            (Some([36, 4, 44]), Some([36, 52, 3])),
            (None, None),
        ] {
            let plan = PlanParams {
                block,
                pack_block,
                workers: 4,
                cache_bytes: 16 << 20,
                work: dir.join(format!("work-{block:?}")),
                resident_bytes_per_vertex: GRAPH_BYTES_PER_VERTEX,
            };
            let planned = vessel_graph_stored(&stored, &params, &plan)
                .unwrap()
                .to_json();
            assert!(
                planned == resident,
                "max radius {max_radius:?}, blocks {block:?} / {pack_block:?}: the planned graph \
                 differs from the resident one"
            );
            assert!(!plan.work.exists(), "the scratch directory is removed");
        }
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_volume_of_another_shape_is_refused() {
    let dir = scratch("shapes");
    let scene = scene();
    let small = Array3::from_elem((SHAPE[0], SHAPE[1], SHAPE[2] - 1), 0u16);
    let shape: Vec<u64> = small.shape().iter().map(|n| *n as u64).collect();
    let root = dir.join("small");
    fs::create_dir_all(&root).unwrap();
    let array = ArrayBuilder::new(
        shape.clone(),
        shape.clone(),
        data_type::uint16(),
        FillValue::from(0u16),
    )
    .build(Arc::new(FilesystemStore::new(&root).unwrap()), "/")
    .unwrap();
    array.store_metadata().unwrap();
    let stored = StoredInputs {
        skeleton: label(&dir.join("skeleton"), &scene.skeleton),
        vessels: label(&dir.join("vessels"), &scene.vessels),
        artery_intensity: Some(AttachedImage::at(&root)),
        artery_mask: None,
        vein_intensity: None,
        vein_mask: None,
        spacing: SPACING,
    };
    let plan = PlanParams {
        block: Some([16, 16, 16]),
        pack_block: Some([16, 16, 16]),
        workers: 2,
        cache_bytes: 1 << 20,
        work: dir.join("work"),
        resident_bytes_per_vertex: 0,
    };
    let error = vessel_graph_stored(&stored, &GraphParams::default(), &plan)
        .unwrap_err()
        .to_string();
    assert!(error.contains("artery_raw"), "{error}");
    let _ = fs::remove_dir_all(&dir);
}
