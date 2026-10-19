// SPDX-License-Identifier: MIT
// Original work for this crate.

//! One run as the command line drives it: rows flagged by the bands rather
//! than dropped, label ids that are the basin volume's values, and (ignored)
//! a run over a real crop.

use std::path::PathBuf;

use blockflow::ome_zarr::OmeVolumePyramid;
use blockflow::voxels::Voxels;
use blockflow::ZarrEnvironment;
use cell_density::detect::{detect_cells, CellPlan, PlanOptions};
use cell_density::params::{Band, CellParams};
use ndarray::Array3;

/// Two bright blobs of different size over a flat floor, in raw counts.
fn two_blobs() -> Array3<u16> {
    let mut volume = Array3::<u16>::from_elem((12, 40, 40), 100);
    for (centre, radius) in [([6i64, 10, 10], 2i64), ([6, 28, 28], 4)] {
        for ((z, y, x), value) in volume.indexed_iter_mut() {
            let d = [
                z as i64 - centre[0],
                y as i64 - centre[1],
                x as i64 - centre[2],
            ];
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if r2 <= radius * radius {
                *value = 3000 - (r2 as u16) * 50;
            }
        }
    }
    volume
}

#[test]
fn rows_are_flagged_rather_than_dropped() {
    let params = CellParams {
        size: Band::new(Some(100.0), None),
        ..CellParams::default()
    };
    let run = detect_cells(Voxels::U16(two_blobs()), &params, [12, 20, 20], true).unwrap();
    assert_eq!(run.rows.len(), 2, "{:?}", run.rows);
    assert_eq!(
        run.filtered().count(),
        1,
        "only the large blob is 100 voxels or more"
    );
    let basins = run.basins.as_ref().unwrap().view::<u32>().unwrap();
    for row in &run.rows {
        assert_eq!(u64::from(basins[row.at]), row.label_id);
        assert_eq!(row.source, 3000.0, "the raw maximum is the blob's centre");
        assert!(
            row.background > 0.0 && row.background <= 2900.0,
            "{}",
            row.background
        );
        // The basin is its blob: every voxel above the floor by more than the
        // foreground level.
        let voxels = basins
            .iter()
            .filter(|label| u64::from(**label) == row.label_id)
            .count();
        assert_eq!(voxels as u64, row.size);
    }
    // Label ids are ranks in [z, y, x] order, from 1.
    let ids: Vec<u64> = run.rows.iter().map(|row| row.label_id).collect();
    assert_eq!(ids, vec![1, 2]);
    assert!(run.rows[0].at < run.rows[1].at);
}

#[test]
fn an_empty_volume_has_no_cells() {
    let run = detect_cells(
        Voxels::U16(Array3::from_elem((8, 16, 16), 100)),
        &CellParams::default(),
        [8, 8, 8],
        false,
    )
    .unwrap();
    assert_eq!(run.seeds, 0);
    assert!(run.rows.is_empty());
}

#[test]
fn an_invalid_parameter_point_is_refused_before_the_run() {
    let params = CellParams {
        threshold: f64::NAN,
        ..CellParams::default()
    };
    assert!(detect_cells(Voxels::U16(two_blobs()), &params, [12, 20, 20], false).is_err());
}

/// The README's validation crop: Brain-39L tile 05x06, level 0, channel 0,
/// `z 1792..2048, y 640..1024, x 16..400`. Needs the converted tile under
/// `/husky/otherdataset/teresa/clearmap_em` (or `CELL_DENSITY_TILE`).
#[test]
#[ignore = "reads a real tile (CELL_DENSITY_TILE); run with --release --ignored"]
fn a_run_over_a_real_crop() {
    let tile = PathBuf::from(std::env::var("CELL_DENSITY_TILE").unwrap_or_else(|_| {
        "/husky/otherdataset/teresa/clearmap_em/osf-sa3x8/tiles/tile_05x06.zarr".to_string()
    }));
    if !tile.exists() {
        eprintln!("skipping: {} is not on this machine", tile.display());
        return;
    }
    let pyramid = OmeVolumePyramid::open(&tile, 0, 0).unwrap();
    let source = pyramid
        .attached_level(0)
        .unwrap()
        .window([1792, 640, 16], [256, 384, 384]);
    let (dtype, volume) = source.metadata().unwrap();
    let params = CellParams::default();
    let plan = CellPlan::assemble(
        &params,
        dtype,
        volume,
        [64, 128, 128],
        PlanOptions::default(),
    )
    .unwrap();
    let scratch = std::env::temp_dir().join(format!("cell-density-run-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    let env = ZarrEnvironment::attach(&scratch, std::slice::from_ref(&source)).unwrap();
    plan.execute("real crop", &env, &plan.hints(None, &[]), &[])
        .unwrap();
    let run = plan.cells(&env, &params).unwrap();
    drop(env);
    let _ = std::fs::remove_dir_all(&scratch);
    let filtered: Vec<_> = run.filtered().collect();
    eprintln!(
        "{} seeds, {} cells, {} filtered",
        run.seeds,
        run.rows.len(),
        filtered.len()
    );
    // The crop holds tens of thousands of basins, most of them a few voxels,
    // and around 1.5 thousand cells in the default size band.
    assert!(run.rows.len() > 10_000);
    assert!(
        (1_000..3_000).contains(&filtered.len()),
        "{}",
        filtered.len()
    );
    assert!(filtered.iter().all(|row| (50..900).contains(&row.size)));
}
