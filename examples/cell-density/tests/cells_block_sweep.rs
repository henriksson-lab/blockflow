// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The cut is not part of the answer: every block grid gives the same rows,
//! label ids included, and (ignored, real data) what the cut buys in time.

use std::path::PathBuf;
use std::time::Instant;

use blockflow::ome_zarr::OmeVolumePyramid;
use blockflow::synthetic::{Scene, SceneSpec};
use blockflow::voxels::Voxels;
use blockflow::ZarrEnvironment;
use cell_density::detect::{detect_cells, CellPlan, PlanOptions};
use cell_density::params::{Band, CellParams, DogScales, Maxima};

fn noisy() -> Voxels {
    let scene = Scene::new(
        SceneSpec::new([40, 40, 28], 7)
            .with_objects(16)
            .with_radius(2.5, 4.0)
            .with_touching(0.2, 0.3)
            .with_noise(0.05)
            .with_gradient(0.15, 1.5),
    )
    .expect("a scene");
    Voxels::F64(scene.render().intensity.mapv(|value| value * 2048.0))
}

fn sweep(params: &CellParams) {
    let input = noisy();
    let whole = detect_cells(input.clone(), params, input.shape(), false).expect("one block");
    assert!(!whole.rows.is_empty());
    for block in [
        [16, 16, 12],
        [20, 20, 28],
        [13, 13, 11],
        [40, 40, 8],
        [7, 40, 28],
    ] {
        let cut = detect_cells(input.clone(), params, block, false).expect("a cut run");
        assert_eq!(cut.seeds, whole.seeds, "block {block:?} moved the seeds");
        assert_eq!(cut.rows, whole.rows, "block {block:?} moved the rows");
    }
}

#[test]
fn the_default_detector_does_not_depend_on_the_cut() {
    sweep(&CellParams {
        size: Band::OPEN,
        ..CellParams::default()
    });
}

#[test]
fn the_regional_maxima_with_a_prominence_do_not_depend_on_the_cut() {
    sweep(&CellParams {
        maxima: Maxima::Regional,
        h: 300.0,
        size: Band::OPEN,
        ..CellParams::default()
    });
}

#[test]
fn a_band_pass_does_not_depend_on_the_cut() {
    sweep(&CellParams {
        dog: Some(DogScales {
            sigma: [1.0; 3],
            sigma2: [2.0; 3],
            truncate: 3.0,
        }),
        threshold: 100.0,
        size: Band::OPEN,
        ..CellParams::default()
    });
}

/// The narrow input is the same answer: the opening is exact in `uint16`.
#[test]
fn a_uint16_input_is_the_same_answer_as_its_f64_widening() {
    let wide = noisy();
    let narrow = Voxels::U16(
        wide.view::<f64>()
            .unwrap()
            .mapv(|value| value.round() as u16),
    );
    let params = CellParams {
        size: Band::OPEN,
        ..CellParams::default()
    };
    let from_narrow = detect_cells(narrow.clone(), &params, [16, 16, 12], false).unwrap();
    let from_wide =
        detect_cells(Voxels::F64(narrow.widened()), &params, [16, 16, 12], false).unwrap();
    assert_eq!(from_narrow.rows, from_wide.rows);
}

/// The Brain-39L 05x06 crop used for the README's validation numbers. Needs
/// the converted tile, by default under `/husky/otherdataset/teresa/clearmap_em`
/// (`CELL_DENSITY_TILE` overrides it); a timing, so `--release --ignored`.
fn real_tile() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("CELL_DENSITY_TILE").unwrap_or_else(|_| {
        "/husky/otherdataset/teresa/clearmap_em/osf-sa3x8/tiles/tile_05x06.zarr".to_string()
    }));
    if path.exists() {
        Some(path)
    } else {
        eprintln!("skipping: {} is not on this machine", path.display());
        None
    }
}

#[test]
#[ignore = "reads a real tile (CELL_DENSITY_TILE) and measures time; run with --release --ignored"]
fn what_the_cut_buys_on_a_real_crop() {
    let Some(tile) = real_tile() else { return };
    let pyramid = OmeVolumePyramid::open(&tile, 0, 0).expect("the tile");
    let source = pyramid
        .attached_level(0)
        .unwrap()
        .window([1792, 640, 16], [256, 384, 384]);
    let scratch = std::env::temp_dir().join(format!("cell-density-sweep-{}", std::process::id()));
    let params = CellParams::default();
    let mut reference = None;
    for block in [
        [256, 384, 384],
        [128, 128, 128],
        [64, 128, 128],
        [64, 64, 64],
    ] {
        let (dtype, volume) = source.metadata().unwrap();
        let plan =
            CellPlan::assemble(&params, dtype, volume, block, PlanOptions::default()).unwrap();
        let _ = std::fs::remove_dir_all(&scratch);
        let env = ZarrEnvironment::attach(&scratch, std::slice::from_ref(&source)).unwrap();
        let started = Instant::now();
        plan.execute("sweep", &env, &plan.hints(None, &[]), &[])
            .unwrap();
        let elapsed = started.elapsed().as_secs_f64();
        let run = plan.cells(&env, &params).unwrap();
        eprintln!(
            "block {block:?}: {elapsed:.2} s, {} cells, {} filtered",
            run.rows.len(),
            run.filtered().count()
        );
        match &reference {
            None => reference = Some(run.rows),
            Some(rows) => assert_eq!(&run.rows, rows, "block {block:?} moved the rows"),
        }
    }
    let _ = std::fs::remove_dir_all(&scratch);
}
