// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The detector against ground truth: `blockflow::synthetic::Scene` places the
//! objects first and renders them second, so where they are and how many
//! voxels each holds is known by construction.
//!
//! Matching is greedy nearest-first in Chebyshev distance. A seed is a voxel
//! index and a true centre a real position, so the best a detector can do is
//! name the voxel the centre falls in: a distance of at most half a voxel.

use blockflow::iterate::SubstageLimit;
use blockflow::ops::element::{ElementShape, StructuringElement};
use blockflow::ops::reconstruct::flooding_bound;
use blockflow::region::Region;
use blockflow::synthetic::{Scene, SceneSpec};
use blockflow::voxels::Voxels;
use cell_density::detect::{detect_cells, CellPlan, CellRow, PlanOptions};
use cell_density::params::{Band, CellParams, DogScales, Maxima};

/// The scenes render in `0..1.2`; the detector's defaults are in 16-bit
/// counts, so the scenes are put on that scale rather than the defaults on
/// the scene's.
const COUNTS: f64 = 2048.0;

const CLEAN_VOLUME: [usize; 3] = [40, 40, 32];
const NOISY_VOLUME: [usize; 3] = [40, 40, 28];
const BLOCK: [usize; 3] = [16, 16, 12];

/// Eight separated objects of constant brightness on nothing.
fn clean() -> Scene {
    Scene::new(
        SceneSpec::clean(CLEAN_VOLUME, 13)
            .with_objects(8)
            .with_radius(2.5, 4.0),
    )
    .expect("a scene")
}

/// Sixteen objects on a smooth background with per-voxel noise.
fn noisy() -> Scene {
    Scene::new(
        SceneSpec::new(NOISY_VOLUME, 7)
            .with_objects(16)
            .with_radius(2.5, 4.0)
            .with_touching(0.0, 0.0)
            .with_noise(0.05)
            .with_gradient(0.15, 1.5),
    )
    .expect("a scene")
}

fn counts(scene: &Scene) -> Voxels {
    Voxels::F64(scene.render().intensity.mapv(|value| value * COUNTS))
}

/// True centres in voxel-index coordinates (`ObjectRecord::centroid` is in
/// voxel-centre coordinates, half a voxel further on).
fn truth(scene: &Scene) -> Vec<[f64; 3]> {
    scene
        .object_table()
        .into_iter()
        .filter(|record| record.voxels > 0)
        .map(|record| record.centroid.map(|value| value - 0.5))
        .collect()
}

fn open(params: CellParams) -> CellParams {
    CellParams {
        size: Band::OPEN,
        ..params
    }
}

fn run(input: &Voxels, params: &CellParams, block: [usize; 3]) -> Vec<CellRow> {
    detect_cells(input.clone(), params, block, false)
        .expect("the detector runs")
        .rows
}

fn match_up(found: &[CellRow], truth: &[[f64; 3]], radius: f64) -> (Vec<f64>, usize) {
    let mut taken = vec![false; truth.len()];
    let mut distances = Vec::new();
    let mut spurious = 0;
    for row in found {
        let mut best: Option<(usize, f64)> = None;
        for (index, centre) in truth.iter().enumerate() {
            if taken[index] {
                continue;
            }
            let distance = (0..3)
                .map(|axis| (row.at[axis] as f64 - centre[axis]).abs())
                .fold(0.0f64, f64::max);
            if best.is_none_or(|(_, so_far)| distance < so_far) {
                best = Some((index, distance));
            }
        }
        match best {
            Some((index, distance)) if distance <= radius => {
                taken[index] = true;
                distances.push(distance);
            }
            _ => spurious += 1,
        }
    }
    (distances, spurious)
}

fn near(a: &Region, b: &Region, gap: isize) -> bool {
    (0..3).all(|axis| {
        let (a0, a1) = (
            a.start[axis] as isize,
            (a.start[axis] + a.shape[axis]) as isize,
        );
        let (b0, b1) = (
            b.start[axis] as isize,
            (b.start[axis] + b.shape[axis]) as isize,
        );
        a0 - gap <= b1 && b0 - gap <= a1
    })
}

/// Every claim below assumes this: separated objects, each smaller than the
/// 10-voxel background disk in the planes it opens.
#[test]
fn the_fixture_holds_what_it_claims() {
    let scene = clean();
    let table = scene.object_table();
    assert_eq!(table.len(), 8);
    assert!(table.iter().all(|record| record.voxels > 0));
    for (index, a) in table.iter().enumerate() {
        for b in table.iter().skip(index + 1) {
            assert!(
                !near(&a.bounds, &b.bounds, 2),
                "objects {} and {} touch",
                a.id,
                b.id
            );
        }
    }
    for record in &table {
        assert!(record.bounds.shape[1] < 10 && record.bounds.shape[2] < 10);
    }
}

#[test]
fn every_separated_object_is_found_once_at_its_own_centre() {
    let scene = clean();
    let truth = truth(&scene);
    let found = run(&counts(&scene), &open(CellParams::default()), BLOCK);
    assert_eq!(found.len(), truth.len());
    let (distances, spurious) = match_up(&found, &truth, 1.0);
    assert_eq!(spurious, 0);
    assert_eq!(distances.len(), truth.len());
    for distance in distances {
        assert!(
            distance <= 0.5,
            "a cell was found {distance} voxels from its centre"
        );
    }
}

/// The clean objects are constant plateaux above a background of exactly
/// zero once the opening is removed, and the default foreground of 500 sits
/// between that zero and the dimmest object (0.6 x 2048), so each basin is its
/// object voxel for voxel.
#[test]
fn every_objects_basin_is_the_object_voxel_for_voxel() {
    let scene = clean();
    let found = run(&counts(&scene), &open(CellParams::default()), BLOCK);
    let truth: Vec<_> = scene.object_table();
    for row in &found {
        let owner = truth
            .iter()
            .find(|record| {
                (0..3).all(|axis| {
                    let start = record.bounds.start[axis];
                    row.at[axis] >= start && row.at[axis] < start + record.bounds.shape[axis]
                })
            })
            .unwrap_or_else(|| panic!("cell at {:?} is in no object's box", row.at));
        assert_eq!(row.size, owner.voxels, "object {}", owner.id);
        // `source` is the object's own (constant) brightness, and with no
        // background to remove the corrected maximum is the same number.
        assert_eq!(row.source, row.background);
    }
    let distinct: std::collections::BTreeSet<u64> = found.iter().map(|row| row.size).collect();
    assert!(
        distinct.len() > 1,
        "the sizes must be a distribution, not a constant"
    );
    // And the default size band [50, 900) keeps all eight.
    let banded = run(&counts(&scene), &CellParams::default(), BLOCK);
    assert!(banded.iter().all(|row| row.passes_filter));
}

/// What the regional maxima and the prominence are worth on noise: without a
/// prominence every ripple above the foreground is a seed; `h` of a quarter of
/// the brightness scale removes every spurious one and keeps at least four
/// fifths of the objects (the two it loses on this seed are the dimmest).
#[test]
fn the_prominence_removes_the_ripples_and_keeps_the_objects() {
    let scene = noisy();
    let truth = truth(&scene);
    let input = counts(&scene);
    let regional = open(CellParams {
        maxima: Maxima::Regional,
        ..CellParams::default()
    });
    let without_h = run(&input, &regional, BLOCK);
    let with_h = run(
        &input,
        &CellParams {
            h: 0.25 * COUNTS,
            ..regional.clone()
        },
        BLOCK,
    );
    assert!(
        without_h.len() > 2 * truth.len(),
        "{} cells without a prominence",
        without_h.len()
    );
    let (hits, spurious) = match_up(&with_h, &truth, 2.5);
    assert_eq!(
        spurious, 0,
        "the prominence left {spurious} cells that are no object"
    );
    assert!(
        hits.len() * 5 >= truth.len() * 4,
        "{} of {}",
        hits.len(),
        truth.len()
    );

    // The window maximum lands in between: it merges ripples closer than its
    // radius but not those further apart.
    let window = run(&input, &open(CellParams::default()), BLOCK);
    assert!(window.len() < without_h.len() && window.len() >= with_h.len());
}

#[test]
fn the_scales_of_a_band_pass_change_the_answer() {
    let input = counts(&clean());
    let with = |sigma: f64, sigma2: f64, truncate: f64| {
        let params = open(CellParams {
            dog: Some(DogScales {
                sigma: [sigma; 3],
                sigma2: [sigma2; 3],
                truncate,
            }),
            threshold: 100.0,
            ..CellParams::default()
        });
        run(&input, &params, BLOCK)
    };
    let narrow = with(0.6, 1.2, 4.0);
    let wide = with(1.0, 2.0, 4.0);
    let truncated = with(0.6, 1.2, 1.5);
    assert!(!narrow.is_empty());
    assert_ne!(narrow, wide, "the scales are not used");
    assert_ne!(narrow, truncated, "the truncation is not used");
}

/// A flood across the exactly flat (clamped) background takes more substages
/// than the geometric bound, so the detector's guard is four times the bound,
/// and the guard is a refusal rather than a truncated answer.
#[test]
fn the_geometric_substage_bound_is_not_enough_for_this_operation() {
    let input = counts(&noisy());
    let params = open(CellParams {
        dog: Some(DogScales {
            sigma: [1.0; 3],
            sigma2: [2.0; 3],
            truncate: 4.0,
        }),
        maxima: Maxima::Regional,
        h: 0.03 * COUNTS,
        threshold: 0.0,
        ..CellParams::default()
    });
    let element = StructuringElement::from_radius(ElementShape::Ellipsoid, [1, 1, 1]);
    let geometric = flooding_bound(NOISY_VOLUME, &element);

    let plan = |limit: Option<SubstageLimit>| {
        CellPlan::assemble(
            &params,
            input.dtype(),
            input.shape(),
            BLOCK,
            PlanOptions {
                substage_limit: limit,
                ..PlanOptions::default()
            },
        )
        .expect("a plan")
    };
    let execute = |plan: &CellPlan| {
        let env = blockflow::env::ArrayEnvironment::for_decomposition(
            input.clone(),
            &plan.assembly.decomposition,
            [8, 8, 8],
        )
        .expect("an environment");
        plan.execute("cells", &env, &plan.hints(None, &[]), &[])
            .map(|stats| (stats, plan.cells(&env, &params).expect("rows")))
    };
    let (stats, rows) = execute(&plan(None)).expect("the default guard is wide enough");
    let taken = stats
        .substages
        .iter()
        .copied()
        .max()
        .expect("one per phase");
    assert!(
        taken > geometric.substages(),
        "{taken} substages against a bound of {}",
        geometric.substages()
    );
    let refused = execute(&plan(Some(geometric))).expect_err("the bound is too tight");
    assert!(refused.to_string().contains("substage"), "{refused}");
    // A larger guard cannot change a voxel.
    let generous = SubstageLimit::of(geometric.substages() * 40).expect("positive");
    let (_, again) = execute(&plan(Some(generous))).expect("a wider guard");
    assert_eq!(again.rows, rows.rows);
}
