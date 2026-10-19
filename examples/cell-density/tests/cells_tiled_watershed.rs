// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The tiled watershed: its answer is fixed by the tile, not by the block
//! grid or the worker count; edge seeds leave no row and no voxel behind; and
//! it is close to the whole-volume flood (synthetic, and ignored on the real
//! crop, with timings).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use blockflow::env::ArrayEnvironment;
use blockflow::ome_zarr::OmeVolumePyramid;
use blockflow::synthetic::{Scene, SceneSpec};
use blockflow::voxels::Voxels;
use blockflow::ZarrEnvironment;
use cell_density::detect::{CellPlan, CellRow, PlanOptions};
use cell_density::params::{Band, CellParams, Watershed};
use ndarray::Array3;

/// Many touching objects over noise and a gradient, several tiles of
/// `TILE` on every axis.
fn crowded() -> Voxels {
    let scene = Scene::new(
        SceneSpec::new([48, 96, 80], 11)
            .with_objects(300)
            .with_radius(2.5, 4.5)
            .with_touching(0.3, 0.3)
            .with_noise(0.05)
            .with_gradient(0.15, 1.5),
    )
    .expect("a scene");
    Voxels::F64(scene.render().intensity.mapv(|value| value * 2048.0))
}

const TILE: Watershed = Watershed::Tiled {
    tile: [16, 32, 32],
    halo: [6, 8, 8],
};

fn params(watershed: Watershed) -> CellParams {
    CellParams {
        watershed,
        size: Band::OPEN,
        ..CellParams::default()
    }
}

fn run(
    input: &Voxels,
    params: &CellParams,
    block: [usize; 3],
    workers: usize,
) -> (Vec<CellRow>, Array3<u32>) {
    let plan = CellPlan::assemble(
        params,
        input.dtype(),
        input.shape(),
        block,
        PlanOptions::default(),
    )
    .expect("a plan");
    let env =
        ArrayEnvironment::for_decomposition(input.clone(), &plan.assembly.decomposition, [8, 8, 8])
            .expect("an environment");
    plan.execute(
        "tiled",
        &env,
        &plan.hints(Some(workers), &[plan.labels_image]),
        &[],
    )
    .expect("the run");
    let rows = plan.cells(&env, params).expect("the rows").rows;
    let basins = env
        .try_image(plan.labels_image.index())
        .expect("the basins")
        .view::<u32>()
        .expect("u32 basins")
        .to_owned();
    (rows, basins)
}

/// How two runs' rows compare: `(rows in a, rows in b, common label ids,
/// common rows of identical size, largest size difference over common rows)`.
fn compare(a: &[CellRow], b: &[CellRow]) -> (usize, usize, usize, usize, u64) {
    let b: BTreeMap<u64, &CellRow> = b.iter().map(|row| (row.label_id, row)).collect();
    let (mut common, mut same, mut worst) = (0, 0, 0);
    for row in a {
        if let Some(other) = b.get(&row.label_id) {
            common += 1;
            same += usize::from(other.size == row.size);
            worst = worst.max(other.size.abs_diff(row.size));
        }
    }
    (a.len(), b.len(), common, same, worst)
}

#[test]
fn the_tiled_answer_does_not_depend_on_the_blocks_or_the_workers() {
    let input = crowded();
    let params = params(TILE);
    let (rows, basins) = run(&input, &params, input.shape(), 1);
    assert!(rows.len() > 100, "{} rows", rows.len());
    for block in [[16, 16, 12], [13, 29, 31], [7, 96, 80], [48, 40, 40]] {
        for workers in [1, 4] {
            let (cut_rows, cut_basins) = run(&input, &params, block, workers);
            assert_eq!(cut_rows, rows, "block {block:?}, {workers} workers");
            assert_eq!(cut_basins, basins, "block {block:?}, {workers} workers");
        }
    }
}

#[test]
fn edge_seeds_leave_no_row_and_no_voxel() {
    let input = crowded();
    // A halo of one voxel: basins regularly reach past it, so edge seeds are
    // certain to be made, and must all be cleared.
    let params = params(Watershed::Tiled {
        tile: [16, 32, 32],
        halo: [1, 1, 1],
    });
    let (rows, basins) = run(&input, &params, [16, 32, 32], 3);
    let labels: std::collections::BTreeSet<u64> = rows.iter().map(|row| row.label_id).collect();
    for &value in basins.iter() {
        assert!(
            value == 0 || labels.contains(&u64::from(value)),
            "label {value} is no row's"
        );
    }
    // And they did take voxels from cells: the whole-volume flood labels more.
    let (_, whole) = run(&input, &params_whole(), [16, 32, 32], 3);
    let labelled = |b: &Array3<u32>| b.iter().filter(|&&v| v != 0).count();
    assert!(
        labelled(&basins) < labelled(&whole),
        "a one-voxel halo should cut some basin"
    );
}

fn params_whole() -> CellParams {
    params(Watershed::Whole)
}

#[test]
fn the_tiled_watershed_is_close_to_the_whole_volume_flood() {
    let input = crowded();
    let (whole, whole_basins) = run(&input, &params_whole(), [16, 32, 32], 4);
    let (tiled, tiled_basins) = run(&input, &params(TILE), [16, 32, 32], 4);
    let (n_whole, n_tiled, common, same, worst) = compare(&whole, &tiled);
    let moved = whole_basins
        .iter()
        .zip(tiled_basins.iter())
        .filter(|(a, b)| a != b)
        .count();
    eprintln!(
        "whole {n_whole}, tiled {n_tiled}, common {common}, same size {same}, worst {worst}, \
         {moved} of {} voxels differ",
        whole_basins.len()
    );
    assert_eq!(n_whole, n_tiled, "the same seeds keep a basin");
    assert_eq!(common, n_whole);
    assert!(same * 100 >= common * 95, "{same} of {common} sizes agree");
    assert!(moved * 1000 <= whole_basins.len(), "{moved} voxels moved");

    // Every voxel that moved was dropped, none given to another seed: what a
    // halo too short for a basin costs is that basin's far part, not a leak.
    for (a, b) in whole_basins.iter().zip(tiled_basins.iter()) {
        assert!(a == b || *b == 0, "voxel moved from basin {a} to {b}");
    }

    // With one tile over the volume the two are the same flood.
    let one = Watershed::Tiled {
        tile: input.shape(),
        halo: [0, 0, 0],
    };
    let (single, single_basins) = run(&input, &params(one), [16, 32, 32], 4);
    assert_eq!(single, whole);
    assert_eq!(single_basins, whole_basins);
}

/// The Brain-39L 05x06 crop of the README. Needs the converted tile, by
/// default under `/husky/otherdataset/teresa/clearmap_em` (`CELL_DENSITY_TILE`
/// overrides it); `CELL_DENSITY_WATERSHEDS` lists extra `tile;halo` pairs
/// (`z,y,x;z,y,x` separated by spaces). A timing, so `--release --ignored
/// --nocapture`.
#[test]
#[ignore = "reads a real tile (CELL_DENSITY_TILE) and measures time; run with --release --ignored"]
fn tiled_against_whole_on_a_real_crop() {
    let tile = PathBuf::from(std::env::var("CELL_DENSITY_TILE").unwrap_or_else(|_| {
        "/husky/otherdataset/teresa/clearmap_em/osf-sa3x8/tiles/tile_05x06.zarr".to_string()
    }));
    if !tile.exists() {
        eprintln!("skipping: {} is not on this machine", tile.display());
        return;
    }
    let source = OmeVolumePyramid::open(&tile, 0, 0)
        .unwrap()
        .attached_level(0)
        .unwrap()
        .window([1792, 640, 16], [256, 384, 384]);
    let (dtype, volume) = source.metadata().unwrap();
    let scratch = std::env::temp_dir().join(format!("cell-density-tiled-{}", std::process::id()));
    let run = |watershed: Watershed, block: [usize; 3]| {
        let params = CellParams {
            watershed,
            ..CellParams::default()
        };
        let plan =
            CellPlan::assemble(&params, dtype, volume, block, PlanOptions::default()).unwrap();
        let _ = std::fs::remove_dir_all(&scratch);
        let env = ZarrEnvironment::attach(&scratch, std::slice::from_ref(&source)).unwrap();
        let started = Instant::now();
        plan.execute("tiled", &env, &plan.hints(None, &[plan.labels_image]), &[])
            .unwrap();
        let elapsed = started.elapsed().as_secs_f64();
        let rows = plan.cells(&env, &params).unwrap().rows;
        let basins = env
            .image(plan.labels_image.index())
            .unwrap()
            .view::<u32>()
            .unwrap()
            .to_owned();
        drop(env);
        let _ = std::fs::remove_dir_all(&scratch);
        (rows, basins, elapsed)
    };
    // Two whole-volume references: the barrier op (one exact flood per
    // foreground component), and one tile over the crop (one flood over the
    // whole volume). They break ties between equal costs in different orders.
    let one_tile = Watershed::Tiled {
        tile: volume,
        halo: [0, 0, 0],
    };
    let references: Vec<(&str, Vec<CellRow>, Array3<u32>)> =
        [("whole", Watershed::Whole), ("one tile", one_tile)]
            .into_iter()
            .map(|(name, watershed)| {
                let (rows, basins, seconds) = run(watershed, [64, 128, 128]);
                eprintln!(
                    "{name}: {seconds:.2} s, {} cells, {} filtered",
                    rows.len(),
                    rows.iter().filter(|r| r.passes_filter).count()
                );
                (name, rows, basins)
            })
            .collect();
    let mut configs = vec![Watershed::DEFAULT_TILED];
    if let Ok(extra) = std::env::var("CELL_DENSITY_WATERSHEDS") {
        let triple = |text: &str| -> [usize; 3] {
            let v: Vec<usize> = text.split(',').map(|t| t.parse().unwrap()).collect();
            [v[0], v[1], v[2]]
        };
        for pair in extra.split_whitespace() {
            let (tile, halo) = pair.split_once(';').expect("tile;halo");
            configs.push(Watershed::Tiled {
                tile: triple(tile),
                halo: triple(halo),
            });
        }
    }
    let filtered = |rows: &[CellRow]| -> Vec<CellRow> {
        rows.iter().filter(|r| r.passes_filter).cloned().collect()
    };
    for watershed in configs {
        let (tiled, basins, seconds) = run(watershed, [64, 128, 128]);
        eprintln!(
            "{watershed:?}: {seconds:.2} s, {} cells, {} filtered",
            tiled.len(),
            tiled.iter().filter(|r| r.passes_filter).count()
        );
        for (name, rows, reference) in &references {
            let (mut moved, mut dropped) = (0usize, 0usize);
            for (a, b) in reference.iter().zip(basins.iter()) {
                if a != b {
                    if *b == 0 {
                        dropped += 1;
                    } else {
                        moved += 1;
                    }
                }
            }
            let (_, _, common, same, worst) = compare(rows, &tiled);
            let (n_f, _, f_common, f_same, f_worst) = compare(&filtered(rows), &filtered(&tiled));
            eprintln!(
                "  against {name}: {common} of {} cells common, {same} same size, worst {worst}; \
                 {f_common} of {n_f} filtered common, {f_same} same size, worst {f_worst}; \
                 voxels {dropped} cleared, {moved} relabelled",
                rows.len()
            );
            assert_eq!(common, rows.len(), "every reference basin is kept");
            // Against the one flood over the crop the tiles cost a few
            // hundred voxels (34 248 of 34 251 sizes agree at the default);
            // the barrier op parts from both on its large components.
            if *name == "one tile" && watershed == Watershed::DEFAULT_TILED {
                assert!(
                    same * 1000 >= common * 999,
                    "{same} of {common} sizes agree"
                );
                assert!(
                    f_same * 100 >= f_common * 99,
                    "{f_same} of {f_common} filtered"
                );
                assert!(moved < 100, "{moved} voxels went to another seed");
            }
        }
        if watershed == Watershed::DEFAULT_TILED {
            let (block_rows, block_basins, _) = run(watershed, [128, 128, 128]);
            assert_eq!(block_rows, tiled, "the block grid moved the tiled rows");
            assert_eq!(
                block_basins, basins,
                "the block grid moved the tiled basins"
            );
        }
    }
}
