// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The thinning over stored images (`segment::thin`) is the in-plan fixed
//! point (`ThinToFixedPoint`), voxel for voxel, on every grid and at any
//! number of rounds per plan.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use blockflow::voxels::Voxels;
use blockflow::zarr_env::AttachedImage;
use blockflow_vessel_graph::segment::thin::{thin_stored, ThinPlan};
use blockflow_vessel_graph::segment::{mask_steps, MaskSteps, SkeletonParams};
use ndarray::Array3;
use zarrs::array::{data_type, Array, ArrayBuilder, ArraySubset, FillValue};
use zarrs::filesystem::FilesystemStore;

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("vessel-graph-thin-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn store_bool(root: &Path, mask: &Array3<bool>, chunk: [u64; 3]) -> AttachedImage {
    fs::create_dir_all(root).unwrap();
    let shape: Vec<u64> = mask.shape().iter().map(|n| *n as u64).collect();
    let store = Arc::new(FilesystemStore::new(root).unwrap());
    let array = ArrayBuilder::new(
        shape.clone(),
        chunk.to_vec(),
        data_type::bool(),
        FillValue::from(false),
    )
    .build(store, "/")
    .unwrap();
    array.store_metadata().unwrap();
    let data: Vec<bool> = mask.iter().copied().collect();
    array
        .store_array_subset(&ArraySubset::new_with_shape(shape), &data)
        .unwrap();
    AttachedImage::at(root)
}

fn read_bool(root: &Path) -> Array3<bool> {
    let store = Arc::new(FilesystemStore::new(root).unwrap());
    let array = Array::open(store, "/").unwrap();
    let shape: Vec<u64> = array.shape().to_vec();
    let values: Vec<bool> = array
        .retrieve_array_subset(&ArraySubset::new_with_shape(shape.clone()))
        .unwrap();
    Array3::from_shape_vec(
        (shape[0] as usize, shape[1] as usize, shape[2] as usize),
        values,
    )
    .unwrap()
}

/// Solid tubes and a slab: thick enough to take many rounds, thin enough to
/// end in lines and a sheet.
fn shapes(shape: [usize; 3]) -> Array3<bool> {
    let mut mask = Array3::from_elem((shape[0], shape[1], shape[2]), false);
    for ((z, y, x), slot) in mask.indexed_iter_mut() {
        let (zf, yf, xf) = (z as f64, y as f64, x as f64);
        let tube = ((zf - 12.0).powi(2) + (yf - 14.0).powi(2)).sqrt() <= 7.5;
        let tilted = ((zf - 20.0 - 0.3 * xf).powi(2) + (yf - 34.0).powi(2)).sqrt() <= 4.0;
        let blob = ((zf - 22.0).powi(2) + (yf - 30.0).powi(2) + (xf - 40.0).powi(2)).sqrt() <= 13.0;
        let slab = (3..9).contains(&x) && (5..40).contains(&y);
        *slot = (tube || tilted || blob || slab) && z > 0 && y > 0 && x > 0;
    }
    mask
}

fn in_plan(mask: &Array3<bool>, block: [usize; 3]) -> Array3<bool> {
    let params = SkeletonParams {
        delete_border: false,
        ..SkeletonParams::default()
    };
    let (_, _, skeleton) = mask_steps(
        Voxels::from(mask.clone()),
        &MaskSteps {
            post: None,
            skeleton: Some(params),
        },
        block,
    )
    .unwrap();
    skeleton.unwrap().view::<bool>().unwrap().to_owned()
}

#[test]
fn the_stored_thinning_is_the_in_plan_fixed_point() {
    let dir = scratch("synthetic");
    let shape = [40, 48, 56];
    let mask = shapes(shape);
    let expected = in_plan(&mask, [16, 16, 16]);
    assert_eq!(
        expected,
        in_plan(&mask, [40, 48, 56]),
        "the oracle is grid-free"
    );
    assert!(expected.iter().filter(|v| **v).count() > 50);
    let stored = store_bool(&dir.join("mask"), &mask, [16, 16, 16]);
    for (grid, rounds_per_plan) in [
        ([40, 48, 56], 8),
        ([16, 16, 16], 3),
        ([13, 7, 29], 1),
        ([8, 48, 9], 64),
    ] {
        let plan = ThinPlan {
            grid,
            rounds_per_plan,
            workers: 4,
            round_limit: 4096,
            passes: 1,
            work: dir.join(format!("work-{grid:?}")),
            cache_bytes: Some(8 << 20),
        };
        let thinned = thin_stored(&stored, &plan).unwrap();
        assert_eq!(
            read_bool(&thinned.level0),
            expected,
            "grid {grid:?}, {rounds_per_plan} rounds per plan"
        );
        fs::remove_dir_all(&thinned.scratch).unwrap();
        let left: Vec<_> = fs::read_dir(&plan.work)
            .map(|entries| entries.collect())
            .unwrap_or_default();
        assert!(
            left.is_empty(),
            "the plans' scratch directories are removed"
        );
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_round_limit_too_low_is_refused() {
    let dir = scratch("limit");
    let mask = shapes([40, 48, 56]);
    let stored = store_bool(&dir.join("mask"), &mask, [16, 16, 16]);
    let plan = ThinPlan {
        grid: [16, 16, 16],
        rounds_per_plan: 2,
        workers: 2,
        round_limit: 3,
        passes: 1,
        work: dir.join("work"),
        cache_bytes: None,
    };
    let error = thin_stored(&stored, &plan).unwrap_err().to_string();
    assert!(error.contains("did not converge in 3 rounds"), "{error}");
    let _ = fs::remove_dir_all(&dir);
}

/// User plus system CPU seconds of this process.
fn cpu_seconds() -> f64 {
    let stat = fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let tail = stat.rsplit_once(')').map(|(_, tail)| tail).unwrap_or("");
    let fields: Vec<f64> = tail
        .split_whitespace()
        .map(|field| field.parse().unwrap_or(0.0))
        .collect();
    (fields.get(11).unwrap_or(&0.0) + fields.get(12).unwrap_or(&0.0)) / 100.0
}

/// Times the stored thinning of a published mask at several grids and checks
/// every answer against the first. Needs `VESSEL_GRAPH_THIN_MASK`: a label
/// image's level-0 array (`.../labels/vessel-mask/0`).
#[test]
#[ignore = "needs VESSEL_GRAPH_THIN_MASK, a published vessel mask's level-0 array"]
fn thinning_grids_on_a_published_mask() {
    let source = PathBuf::from(std::env::var("VESSEL_GRAPH_THIN_MASK").expect("the mask"));
    let dir = scratch("published");
    let store = Arc::new(FilesystemStore::new(&source).unwrap());
    let array = Array::open(store, "/").unwrap();
    let shape: Vec<u64> = array.shape().to_vec();
    let labels: Vec<u64> = array
        .retrieve_array_subset(&ArraySubset::new_with_shape(shape.clone()))
        .unwrap();
    let dims = [shape[0] as usize, shape[1] as usize, shape[2] as usize];
    let mut mask = Array3::from_shape_vec((dims[0], dims[1], dims[2]), labels)
        .unwrap()
        .mapv(|v| v != 0);
    // The faces cleared, as `segment` does.
    for ((z, y, x), slot) in mask.indexed_iter_mut() {
        if z == 0 || y == 0 || x == 0 || z + 1 == dims[0] || y + 1 == dims[1] || x + 1 == dims[2] {
            *slot = false;
        }
    }
    let stored = store_bool(&dir.join("mask"), &mask, [64, 64, 128]);
    let grids: Vec<[usize; 3]> = std::env::var("VESSEL_GRAPH_THIN_GRIDS")
        .unwrap_or_else(|_| "64,256,256;32,128,128;64,128,128".to_string())
        .split(';')
        .map(|grid| {
            let values: Vec<usize> = grid.split(',').map(|v| v.parse().unwrap()).collect();
            [values[0], values[1], values[2]]
        })
        .collect();
    let mut first: Option<Array3<bool>> = None;
    let started = Instant::now();
    let cpu = cpu_seconds();
    let oracle = in_plan(&mask, grids[0]);
    eprintln!(
        "in-plan fixed point, grid {:?}: {:.1} s, cpu {:.1} s",
        grids[0],
        started.elapsed().as_secs_f64(),
        cpu_seconds() - cpu
    );
    for grid in grids {
        let plan = ThinPlan {
            grid,
            rounds_per_plan: 8,
            workers: rayon::current_num_threads(),
            round_limit: 4096,
            passes: 1,
            work: std::env::var("VESSEL_GRAPH_THIN_WORK")
                .map(PathBuf::from)
                .unwrap_or_else(|_| dir.clone())
                .join(format!("work-{grid:?}")),
            cache_bytes: std::env::var("VESSEL_GRAPH_THIN_CACHE_MB")
                .ok()
                .map(|mb| mb.parse::<u64>().unwrap() << 20),
        };
        let started = Instant::now();
        let cpu = cpu_seconds();
        let thinned = thin_stored(&stored, &plan).unwrap();
        eprintln!(
            "grid {grid:?}: {} rounds, {:.1} s, cpu {:.1} s",
            thinned.rounds,
            started.elapsed().as_secs_f64(),
            cpu_seconds() - cpu,
        );
        let answer = read_bool(&thinned.level0);
        assert_eq!(
            answer, oracle,
            "grid {grid:?} against the in-plan fixed point"
        );
        match &first {
            Some(first) => assert_eq!(&answer, first, "grid {grid:?}"),
            None => first = Some(answer),
        }
        fs::remove_dir_all(&thinned.scratch).unwrap();
    }
    let _ = fs::remove_dir_all(&dir);
}
