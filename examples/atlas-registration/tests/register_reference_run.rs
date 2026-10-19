// SPDX-License-Identifier: MIT
// Original work for this crate.

// The reference's own parameter files on real data, against the reference's
// elastix run on the same inputs: an autofluorescence volume at 25 µm (fixed)
// and the prepared 25 µm atlas reference (moving), as `reference-clearmap/`
// prepares and runs them. The grid of fixed points both sides carried must
// land within the agreement measured there.
//
//     REGISTRATION_REFERENCE=/husky/henriksson/lightsheet-reference/registration \
//     ALIGNMENT_PARAMETERS=…/ClearMap/Resources/Alignment \
//     cargo test --release -p blockflow-atlas-registration --test register_reference_run -- --ignored --nocapture

use std::path::PathBuf;
use std::time::Instant;

use blockflow_atlas_registration::pipeline::{register, Input, RegisterConfig, Stage};
use blockflow_atlas_registration::stochastic::stochastic_from_elastix_map;
use blockflow_atlas_registration::{ParameterMap, PointTransform};

fn directory(variable: &str) -> PathBuf {
    PathBuf::from(std::env::var(variable).unwrap_or_else(|_| panic!("{variable} is not set")))
}

/// `OutputPoint = [ x y z ]` of a transformix point run, as `[z, y, x]`.
fn elastix_points(path: &PathBuf) -> Vec<[f64; 3]> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| {
            let start = line.find("OutputPoint = [").unwrap() + "OutputPoint = [".len();
            let end = start + line[start..].find(']').unwrap();
            let v: Vec<f64> = line[start..end]
                .split_whitespace()
                .map(|t| t.parse().unwrap())
                .collect();
            [v[2], v[1], v[0]]
        })
        .collect()
}

#[test]
#[ignore = "needs the reference-clearmap inputs and elastix outputs: set REGISTRATION_REFERENCE \
            and ALIGNMENT_PARAMETERS (see reference-clearmap/README.md)"]
fn the_reference_parameter_files_agree_with_elastix_on_real_data() {
    let root = directory("REGISTRATION_REFERENCE");
    let parameters = directory("ALIGNMENT_PARAMETERS");
    let out = std::env::temp_dir().join(format!(
        "atlas-registration-reference-{}",
        std::process::id()
    ));
    let stages = ["align_affine.txt", "align_bspline.txt"]
        .iter()
        .map(|name| {
            let text = std::fs::read_to_string(parameters.join(name)).unwrap();
            Stage::Stochastic(
                stochastic_from_elastix_map(&ParameterMap::parse(&text).unwrap(), 121_212).unwrap(),
            )
        })
        .collect();
    let start = Instant::now();
    let summary = register(&RegisterConfig {
        fixed: Input {
            path: root.join("zarr/autofluorescence_25um.ome.zarr"),
            channel: 0,
            level: None,
        },
        moving: Input {
            path: root.join("zarr/reference_25um.ome.zarr"),
            channel: 0,
            level: None,
        },
        resolution: Some([25.0; 3]),
        stages,
        out: out.clone(),
        warped: None,
        warp_source: None,
        nearest: false,
        block: 64,
        workers: rayon::current_num_threads(),
        warped_stages: false,
        initialisation: Default::default(),
    })
    .unwrap();
    let seconds = start.elapsed().as_secs_f64();

    let mut reader = csv::Reader::from_path(root.join("points/auto_points.csv")).unwrap();
    let points: Vec<[f64; 3]> = reader
        .records()
        .map(|r| {
            let r = r.unwrap();
            [
                r[0].parse().unwrap(),
                r[1].parse().unwrap(),
                r[2].parse().unwrap(),
            ]
        })
        .collect();
    let ours = PointTransform::from_registration_dir(&out)
        .unwrap()
        .apply(&points);
    let theirs = elastix_points(&root.join("clearmap/points.1/outputpoints.txt"));
    let _ = std::fs::remove_dir_all(&out);
    let mut distances: Vec<f64> = ours
        .iter()
        .zip(&theirs)
        .map(|(a, b)| 25.0 * (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>().sqrt())
        .collect();
    distances.sort_by(f64::total_cmp);
    let median = distances[distances.len() / 2];
    let p95 = distances[distances.len() * 95 / 100];
    eprintln!(
        "{} points: median {median:.1} µm, 95th percentile {p95:.1} µm; {seconds:.1} s; stages {:?}",
        distances.len(),
        summary.stages.iter().map(|s| (s.seconds, s.similarity)).collect::<Vec<_>>()
    );
    // Measured at 16 and 51 µm (README); elastix's own affine and B-spline
    // results are 173 and 339 µm apart.
    assert!(
        median < 25.0 && p95 < 75.0,
        "median {median} µm, p95 {p95} µm"
    );
}
