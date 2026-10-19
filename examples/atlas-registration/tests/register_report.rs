// SPDX-License-Identifier: MIT
// Original work for this crate.

use blockflow::npy::{self, Order};
use blockflow::voxels::Voxels;
use blockflow_atlas_registration::compute::{run_with_transform_text, VolumeFitParams};
use blockflow_atlas_registration::report::{
    fixture_report, render_json, require_threshold, synthetic_translation_report,
};
use blockflow_atlas_registration::RegisterParams;
use ndarray::Array3;
use std::path::{Path, PathBuf};

struct Scratch {
    path: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "atlas-registration-register-report-{}-{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("scratch directory");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn write_generated_fixture(scratch: &Scratch, manifest: &str) {
    let fixed = gaussian([3.2, 3.1, 2.0]);
    let moving = gaussian([3.95, 2.75, 2.4]);
    npy::write_voxels_file(&scratch.path().join("fixed.npy"), &fixed, Order::C).expect("fixed npy");
    npy::write_voxels_file(&scratch.path().join("moving.npy"), &moving, Order::C)
        .expect("moving npy");
    let (transform, text) =
        run_with_transform_text(&VolumeFitParams::default(), &fixed, &moving, 4)
            .expect("native generated transform");
    std::fs::write(scratch.path().join("TransformParameters.0.txt"), text).expect("transform");
    let point = transform.apply([3.2, 3.1, 2.0], &RegisterParams::default());
    std::fs::write(
        scratch.path().join("outputpoints.txt"),
        format!(
            "Point 0 ; InputPoint = [ 3.200000 3.100000 2.000000 ] ; OutputPoint = [ {:.6} \
             {:.6} {:.6} ]\n",
            point[0], point[1], point[2]
        ),
    )
    .expect("points");
    std::fs::write(scratch.path().join("elastix.log"), "generated\n").expect("log");
    std::fs::write(
        scratch.path().join("align_affine.txt"),
        "(Transform \"TranslationTransform\")\n(Metric \"AdvancedMeanSquares\")\n",
    )
    .expect("parameter file");
    std::fs::write(scratch.path().join("registration-manifest.json"), manifest).expect("manifest");
}

fn fnv1a64(path: &Path) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in std::fs::read(path).expect("hash input") {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

#[test]
fn registration_report_is_machine_readable_and_checks_threshold() {
    let report = synthetic_translation_report().expect("report computes");
    require_threshold(&report).expect("native synthetic registration stays inside tolerance");

    let json = render_json(&report);
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");

    assert_eq!(parsed["schema"], 1);
    assert_eq!(parsed["fixture"]["kind"], "synthetic_translation_gaussian");
    assert!(parsed["fixture"]["fixed_bytes"].is_null());
    assert!(parsed["fixture"]["moving_bytes"].is_null());
    assert!(parsed["fixture"]["output_points_bytes"].is_null());
    assert!(parsed["fixture"]["log_bytes"].is_null());
    assert_eq!(
        parsed["fixture"]["parameter_file_bytes"]
            .as_array()
            .expect("synthetic parameter byte records")
            .len(),
        0
    );
    assert_eq!(
        parsed["native"]["implementation"],
        "blockflow-atlas-registration"
    );
    assert_eq!(parsed["native"]["model"], "translation");
    assert_eq!(parsed["native"]["metric"], "mean_squares");
    assert!(parsed["native"]["command"]
        .as_str()
        .unwrap()
        .contains("run_with_diagnostics"));
    assert_eq!(parsed["native"]["optimizer_converged_by"], "step_tolerance");
    assert!(parsed["native"]["optimizer_substages"].as_u64().unwrap() >= 1);
    assert!(parsed["native"]["optimizer_final_cost"].as_f64().unwrap() >= 0.0);
    assert!(
        parsed["native"]["optimizer_final_step_norm"]
            .as_f64()
            .unwrap()
            >= 0.0
    );
    assert!(
        parsed["native"]["optimizer_final_cost_delta"]
            .as_f64()
            .unwrap()
            >= 0.0
    );
    assert!(parsed["native"]["wall_seconds"].as_f64().unwrap() >= 0.0);
    if !parsed["native"]["cpu_seconds"].is_null() {
        assert!(parsed["native"]["cpu_seconds"].as_f64().unwrap() >= 0.0);
    }
    assert_eq!(
        parsed["native"]["transform_parameters"]
            .as_array()
            .expect("parameters")
            .len(),
        3
    );
    assert!(parsed["agreement"]["within_tolerance"].as_bool().unwrap());
    assert!(
        parsed["agreement"]["max_abs_error"].as_f64().unwrap()
            <= parsed["agreement"]["tolerance"].as_f64().unwrap()
    );
    let unsupported = parsed["unsupported_differences"]
        .as_array()
        .expect("unsupported differences");
    assert!(unsupported
        .iter()
        .any(|row| row["setting"] == "FinalBSplineInterpolationOrder"));
    assert!(unsupported
        .iter()
        .any(|row| row["setting"] == "NumberOfSpatialSamples"));
    assert!(unsupported
        .iter()
        .any(|row| row["setting"] == "AutomaticParameterEstimation"));
    for row in unsupported {
        assert!(row["setting"].is_string());
        assert!(row["native"].is_string());
    }
    assert!(parsed["measurement_gap"].is_null());
}

#[test]
fn registration_report_can_compare_a_generated_fixture_transform() {
    let scratch = Scratch::new("fixture");
    write_generated_fixture(
        &scratch,
        r#"{
  "provenance": { "kind": "generated" },
  "paths": {
    "fixed_image": "fixed.npy",
    "moving_image": "moving.npy",
    "output_points": "outputpoints.txt",
    "log": "elastix.log"
  },
  "versions": {
    "elastix": "generated-test",
    "transformix": "generated-test"
  },
  "command_line": "generated native transform fixture",
  "parameter_files": ["align_affine.txt"],
  "transform_files": ["TransformParameters.0.txt"],
  "non_identity_probe": {
    "input_point": [3.2, 3.1, 2.0],
    "minimum_abs_displacement": 0.1
  },
  "downstream_atlas_coordinate": {
    "native_point": [10.0, 20.25, 30.0],
    "reference_point": [10.0, 20.0, 30.5]
  },
  "reference_runtime": {
    "wall_seconds": 1.25,
    "cpu_seconds": 1.10,
    "peak_rss_bytes": 123456
  },
  "agreement": { "tolerance": 0.6 }
}
"#,
    );

    let report = fixture_report(scratch.path()).expect("fixture report");
    // The downstream error is part of require_threshold; this fixture keeps a
    // deliberate 0.5 downstream error, so its tolerance must cover that error
    // while the dedicated negative test below proves out-of-threshold failure.
    require_threshold(&report).expect("generated fixture matches native output");
    let json = render_json(&report);
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");

    assert_eq!(parsed["fixture"]["kind"], "elastix_fixture");
    assert_eq!(parsed["fixture"]["provenance"], "generated");
    assert_eq!(parsed["fixture"]["fixed_path"], "fixed.npy");
    assert_eq!(parsed["fixture"]["parameter_files"][0], "align_affine.txt");
    assert_eq!(
        parsed["fixture"]["fixed_bytes"],
        std::fs::metadata(scratch.path().join("fixed.npy"))
            .unwrap()
            .len()
    );
    assert_eq!(
        parsed["fixture"]["fixed_fnv1a64"],
        fnv1a64(&scratch.path().join("fixed.npy"))
    );
    assert_eq!(
        parsed["fixture"]["moving_bytes"],
        std::fs::metadata(scratch.path().join("moving.npy"))
            .unwrap()
            .len()
    );
    assert_eq!(
        parsed["fixture"]["moving_fnv1a64"],
        fnv1a64(&scratch.path().join("moving.npy"))
    );
    assert_eq!(
        parsed["fixture"]["output_points_bytes"],
        std::fs::metadata(scratch.path().join("outputpoints.txt"))
            .unwrap()
            .len()
    );
    assert_eq!(
        parsed["fixture"]["output_points_fnv1a64"],
        fnv1a64(&scratch.path().join("outputpoints.txt"))
    );
    assert_eq!(
        parsed["fixture"]["log_bytes"],
        std::fs::metadata(scratch.path().join("elastix.log"))
            .unwrap()
            .len()
    );
    assert_eq!(
        parsed["fixture"]["log_fnv1a64"],
        fnv1a64(&scratch.path().join("elastix.log"))
    );
    assert_eq!(
        parsed["fixture"]["parameter_file_bytes"][0],
        std::fs::metadata(scratch.path().join("align_affine.txt"))
            .unwrap()
            .len()
    );
    assert_eq!(
        parsed["fixture"]["parameter_file_fnv1a64"][0],
        fnv1a64(&scratch.path().join("align_affine.txt"))
    );
    assert_eq!(parsed["reference"]["implementation"], "elastix fixture");
    assert_eq!(parsed["reference"]["elastix_version"], "generated-test");
    assert_eq!(parsed["reference"]["transformix_version"], "generated-test");
    assert_eq!(
        parsed["reference"]["transform_file_bytes"][0],
        std::fs::metadata(scratch.path().join("TransformParameters.0.txt"))
            .unwrap()
            .len()
    );
    assert_eq!(
        parsed["reference"]["transform_file_fnv1a64"][0],
        fnv1a64(&scratch.path().join("TransformParameters.0.txt"))
    );
    assert_eq!(parsed["reference"]["wall_seconds"], 1.25);
    assert_eq!(parsed["reference"]["cpu_seconds"], 1.10);
    assert!(parsed["native"]["wall_seconds"].as_f64().unwrap() >= 0.0);
    assert!(parsed["native"]["optimizer_substages"].as_u64().unwrap() >= 1);
    assert!(parsed["native"]["optimizer_converged_by"].is_string());
    assert!(parsed["native"]["optimizer_final_cost"].is_number());
    assert!(
        parsed["native"]["cpu_seconds"].is_number() || parsed["native"]["cpu_seconds"].is_null()
    );
    assert!(parsed["agreement"]["within_tolerance"].as_bool().unwrap());
    assert!(
        parsed["agreement"]["max_abs_error"].as_f64().unwrap()
            <= parsed["agreement"]["tolerance"].as_f64().unwrap()
    );
    assert!(parsed["unsupported_differences"]
        .as_array()
        .expect("unsupported differences")
        .iter()
        .any(|row| row["setting"] == "Metric0Weight"));
    assert_eq!(
        parsed["agreement"]["downstream_atlas_coordinate_max_abs_error"],
        0.5
    );
    assert!(parsed["measurement_gap"].is_null());
}

#[test]
fn registration_report_computes_downstream_native_point_from_input() {
    let scratch = Scratch::new("computed-downstream");
    write_generated_fixture(
        &scratch,
        r#"{
  "provenance": { "kind": "generated" },
  "paths": {
    "fixed_image": "fixed.npy",
    "moving_image": "moving.npy",
    "output_points": "outputpoints.txt",
    "log": "elastix.log"
  },
  "versions": {
    "elastix": "generated-test",
    "transformix": "generated-test"
  },
  "command_line": "generated native transform fixture",
  "parameter_files": ["align_affine.txt"],
  "transform_files": ["TransformParameters.0.txt"],
  "non_identity_probe": {
    "input_point": [3.2, 3.1, 2.0],
    "minimum_abs_displacement": 0.1
  },
  "downstream_atlas_coordinate": {
    "input_point": [3.2, 3.1, 2.0],
    "reference_point": [3.95, 2.75, 2.4]
  },
  "reference_runtime": {
    "wall_seconds": 1.25,
    "cpu_seconds": 1.10,
    "peak_rss_bytes": 123456
  },
  "agreement": { "tolerance": 0.08 }
}
"#,
    );

    let report = fixture_report(scratch.path()).expect("fixture report");
    let downstream = report
        .agreement
        .downstream_atlas_coordinate_max_abs_error
        .expect("computed downstream error");
    assert!(
        downstream > 0.0 && downstream <= report.agreement.tolerance,
        "computed downstream error {downstream} should be within tolerance {}",
        report.agreement.tolerance
    );
}

#[test]
fn registration_report_threshold_includes_downstream_atlas_coordinate() {
    let scratch = Scratch::new("bad-downstream-threshold");
    write_generated_fixture(
        &scratch,
        r#"{
  "provenance": { "kind": "generated" },
  "paths": {
    "fixed_image": "fixed.npy",
    "moving_image": "moving.npy",
    "output_points": "outputpoints.txt",
    "log": "elastix.log"
  },
  "versions": {
    "elastix": "generated-test",
    "transformix": "generated-test"
  },
  "command_line": "generated native transform fixture",
  "parameter_files": ["align_affine.txt"],
  "transform_files": ["TransformParameters.0.txt"],
  "non_identity_probe": {
    "input_point": [3.2, 3.1, 2.0],
    "minimum_abs_displacement": 0.1
  },
  "downstream_atlas_coordinate": {
    "input_point": [3.2, 3.1, 2.0],
    "reference_point": [100.0, 100.0, 100.0]
  },
  "reference_runtime": {
    "wall_seconds": 1.25,
    "cpu_seconds": 1.10,
    "peak_rss_bytes": 123456
  },
  "agreement": { "tolerance": 0.08 }
}
"#,
    );

    let report = fixture_report(scratch.path()).expect("fixture report");
    let error = require_threshold(&report).unwrap_err().to_string();
    assert!(error.contains("downstream atlas coordinate"), "{error}");
    assert!(error.contains("max_abs_error"), "{error}");
    assert!(error.contains("tolerance"), "{error}");
}

#[test]
fn registration_report_uses_single_parameter_file_for_native_config() {
    let scratch = Scratch::new("parameter-driven-native-config");
    write_generated_fixture(
        &scratch,
        r#"{
  "provenance": { "kind": "generated" },
  "paths": {
    "fixed_image": "fixed.npy",
    "moving_image": "moving.npy",
    "output_points": "outputpoints.txt",
    "log": "elastix.log"
  },
  "versions": {
    "elastix": "generated-test",
    "transformix": "generated-test"
  },
  "command_line": "generated native transform fixture",
  "parameter_files": ["align_affine.txt"],
  "transform_files": ["TransformParameters.0.txt"],
  "non_identity_probe": {
    "input_point": [3.2, 3.1, 2.0],
    "minimum_abs_displacement": 0.1
  },
  "agreement": { "tolerance": 0.2 }
}
"#,
    );
    std::fs::write(
        scratch.path().join("align_affine.txt"),
        "(Transform \"TranslationTransform\")\n(Metric \"AdvancedNormalizedCorrelation\")\n",
    )
    .expect("parameter file");

    let report = fixture_report(scratch.path()).expect("fixture report");
    let parsed: serde_json::Value =
        serde_json::from_str(&render_json(&report)).expect("valid JSON");

    assert_eq!(parsed["native"]["model"], "translation");
    assert_eq!(parsed["native"]["metric"], "normalized_correlation");
    assert!(parsed["agreement"]["within_tolerance"].as_bool().unwrap());
}

#[test]
fn registration_report_refuses_half_specified_downstream_agreement() {
    let scratch = Scratch::new("bad-downstream");
    write_generated_fixture(
        &scratch,
        r#"{
  "provenance": { "kind": "generated" },
  "paths": {
    "fixed_image": "fixed.npy",
    "moving_image": "moving.npy",
    "output_points": "outputpoints.txt",
    "log": "elastix.log"
  },
  "versions": {
    "elastix": "generated-test",
    "transformix": "generated-test"
  },
  "command_line": "generated native transform fixture",
  "parameter_files": ["align_affine.txt"],
  "transform_files": ["TransformParameters.0.txt"],
  "non_identity_probe": {
    "input_point": [3.2, 3.1, 2.0],
    "minimum_abs_displacement": 0.1
  },
  "downstream_atlas_coordinate": {
    "native_point": [10.0, 20.25, 30.0]
  },
  "agreement": { "tolerance": 0.000000001 }
}
"#,
    );

    let error = fixture_report(scratch.path()).unwrap_err().to_string();
    assert!(error.contains("downstream_atlas_coordinate"), "{error}");
    assert!(error.contains("native_point"), "{error}");
    assert!(error.contains("reference_point"), "{error}");
}

#[test]
fn registration_report_refuses_ambiguous_downstream_native_forms() {
    let scratch = Scratch::new("ambiguous-downstream");
    write_generated_fixture(
        &scratch,
        r#"{
  "provenance": { "kind": "generated" },
  "paths": {
    "fixed_image": "fixed.npy",
    "moving_image": "moving.npy",
    "output_points": "outputpoints.txt",
    "log": "elastix.log"
  },
  "versions": {
    "elastix": "generated-test",
    "transformix": "generated-test"
  },
  "command_line": "generated native transform fixture",
  "parameter_files": ["align_affine.txt"],
  "transform_files": ["TransformParameters.0.txt"],
  "non_identity_probe": {
    "input_point": [3.2, 3.1, 2.0],
    "minimum_abs_displacement": 0.1
  },
  "downstream_atlas_coordinate": {
    "input_point": [3.2, 3.1, 2.0],
    "native_point": [3.95, 2.75, 2.4],
    "reference_point": [3.95, 2.75, 2.4]
  },
  "agreement": { "tolerance": 0.08 }
}
"#,
    );

    let error = fixture_report(scratch.path()).unwrap_err().to_string();
    assert!(error.contains("downstream_atlas_coordinate"), "{error}");
    assert!(error.contains("input_point"), "{error}");
    assert!(error.contains("native_point"), "{error}");
    assert!(error.contains("not both"), "{error}");
}

#[test]
fn registration_report_refuses_bad_fixture_paths_by_name() {
    let absolute = Scratch::new("absolute-path");
    write_generated_fixture(
        &absolute,
        r#"{
  "provenance": { "kind": "generated" },
  "paths": {
    "fixed_image": "/tmp/fixed.npy",
    "moving_image": "moving.npy",
    "output_points": "outputpoints.txt",
    "log": "elastix.log"
  },
  "versions": {
    "elastix": "generated-test",
    "transformix": "generated-test"
  },
  "command_line": "generated native transform fixture",
  "parameter_files": ["align_affine.txt"],
  "transform_files": ["TransformParameters.0.txt"],
  "non_identity_probe": {
    "input_point": [3.2, 3.1, 2.0],
    "minimum_abs_displacement": 0.1
  },
  "agreement": { "tolerance": 0.000000001 }
}
"#,
    );
    let error = fixture_report(absolute.path()).unwrap_err().to_string();
    assert!(error.contains("paths.fixed_image"), "{error}");
    assert!(error.contains("relative"), "{error}");

    let missing = Scratch::new("missing-side-file");
    write_generated_fixture(
        &missing,
        r#"{
  "provenance": { "kind": "generated" },
  "paths": {
    "fixed_image": "fixed.npy",
    "moving_image": "moving.npy",
    "output_points": "missing-outputpoints.txt",
    "log": "elastix.log"
  },
  "versions": {
    "elastix": "generated-test",
    "transformix": "generated-test"
  },
  "command_line": "generated native transform fixture",
  "parameter_files": ["align_affine.txt"],
  "transform_files": ["TransformParameters.0.txt"],
  "non_identity_probe": {
    "input_point": [3.2, 3.1, 2.0],
    "minimum_abs_displacement": 0.1
  },
  "agreement": { "tolerance": 0.000000001 }
}
"#,
    );
    let error = fixture_report(missing.path()).unwrap_err().to_string();
    assert!(error.contains("paths.output_points"), "{error}");
    assert!(error.contains("missing file"), "{error}");

    let missing_parameter = Scratch::new("missing-parameter-file");
    write_generated_fixture(
        &missing_parameter,
        r#"{
  "provenance": { "kind": "generated" },
  "paths": {
    "fixed_image": "fixed.npy",
    "moving_image": "moving.npy",
    "output_points": "outputpoints.txt",
    "log": "elastix.log"
  },
  "versions": {
    "elastix": "generated-test",
    "transformix": "generated-test"
  },
  "command_line": "generated native transform fixture",
  "parameter_files": ["missing-align.txt"],
  "transform_files": ["TransformParameters.0.txt"],
  "non_identity_probe": {
    "input_point": [3.2, 3.1, 2.0],
    "minimum_abs_displacement": 0.1
  },
  "agreement": { "tolerance": 0.000000001 }
}
"#,
    );
    let error = fixture_report(missing_parameter.path())
        .unwrap_err()
        .to_string();
    assert!(error.contains("parameter_files"), "{error}");
    assert!(error.contains("missing file"), "{error}");
}

#[test]
fn registration_report_refuses_output_points_that_disagree_with_transform_files() {
    let scratch = Scratch::new("bad-outputpoints");
    write_generated_fixture(
        &scratch,
        r#"{
  "provenance": { "kind": "generated" },
  "paths": {
    "fixed_image": "fixed.npy",
    "moving_image": "moving.npy",
    "output_points": "outputpoints.txt",
    "log": "elastix.log"
  },
  "versions": {
    "elastix": "generated-test",
    "transformix": "generated-test"
  },
  "command_line": "generated native transform fixture",
  "parameter_files": ["align_affine.txt"],
  "transform_files": ["TransformParameters.0.txt"],
  "non_identity_probe": {
    "input_point": [3.2, 3.1, 2.0],
    "minimum_abs_displacement": 0.1
  },
  "agreement": { "tolerance": 0.000000001 }
}
"#,
    );
    std::fs::write(
        scratch.path().join("outputpoints.txt"),
        "Point 0 ; InputPoint = [ 3.200000 3.100000 2.000000 ] ; OutputPoint = [ 100.000000 100.000000 100.000000 ]\n",
    )
    .expect("corrupt output points");

    let error = fixture_report(scratch.path()).unwrap_err().to_string();
    assert!(error.contains("paths.output_points"), "{error}");
    assert!(error.contains("transform_files"), "{error}");
    assert!(error.contains("disagrees"), "{error}");
}

#[test]
fn registration_report_refuses_output_points_for_the_wrong_probe() {
    let scratch = Scratch::new("wrong-outputpoints-input");
    write_generated_fixture(
        &scratch,
        r#"{
  "provenance": { "kind": "generated" },
  "paths": {
    "fixed_image": "fixed.npy",
    "moving_image": "moving.npy",
    "output_points": "outputpoints.txt",
    "log": "elastix.log"
  },
  "versions": {
    "elastix": "generated-test",
    "transformix": "generated-test"
  },
  "command_line": "generated native transform fixture",
  "parameter_files": ["align_affine.txt"],
  "transform_files": ["TransformParameters.0.txt"],
  "non_identity_probe": {
    "input_point": [3.2, 3.1, 2.0],
    "minimum_abs_displacement": 0.1
  },
  "agreement": { "tolerance": 0.000001 }
}
"#,
    );
    std::fs::write(
        scratch.path().join("outputpoints.txt"),
        "Point 0 ; InputPoint = [ 1.000000 1.000000 1.000000 ] ; OutputPoint = [ 3.979089 \
         2.768403 2.378348 ]\n",
    )
    .expect("wrong input point");

    let error = fixture_report(scratch.path()).unwrap_err().to_string();
    assert!(error.contains("paths.output_points"), "{error}");
    assert!(error.contains("InputPoint"), "{error}");
    assert!(error.contains("non_identity_probe.input_point"), "{error}");
}

#[test]
fn registration_report_refuses_parameter_files_that_native_compute_cannot_map() {
    let scratch = Scratch::new("unsupported-parameter-file");
    write_generated_fixture(
        &scratch,
        r#"{
  "provenance": { "kind": "generated" },
  "paths": {
    "fixed_image": "fixed.npy",
    "moving_image": "moving.npy",
    "output_points": "outputpoints.txt",
    "log": "elastix.log"
  },
  "versions": {
    "elastix": "generated-test",
    "transformix": "generated-test"
  },
  "command_line": "generated native transform fixture",
  "parameter_files": ["align_affine.txt"],
  "transform_files": ["TransformParameters.0.txt"],
  "non_identity_probe": {
    "input_point": [3.2, 3.1, 2.0],
    "minimum_abs_displacement": 0.1
  },
  "agreement": { "tolerance": 0.000001 }
}
"#,
    );
    std::fs::write(
        scratch.path().join("align_affine.txt"),
        "(Transform \"TranslationTransform\")\n(NumberOfSpatialSamples 1024)\n",
    )
    .expect("unsupported parameter file");

    let error = fixture_report(scratch.path()).unwrap_err().to_string();

    assert!(error.contains("parameter_files"), "{error}");
    assert!(error.contains("align_affine.txt"), "{error}");
    assert!(error.contains("NumberOfSpatialSamples"), "{error}");
    assert!(error.contains("unsupported"), "{error}");
}

#[test]
fn registration_report_refuses_impossible_fixture_measurements_by_name() {
    let negative_runtime = Scratch::new("negative-runtime");
    write_generated_fixture(
        &negative_runtime,
        r#"{
  "provenance": { "kind": "generated" },
  "paths": {
    "fixed_image": "fixed.npy",
    "moving_image": "moving.npy",
    "output_points": "outputpoints.txt",
    "log": "elastix.log"
  },
  "versions": {
    "elastix": "generated-test",
    "transformix": "generated-test"
  },
  "command_line": "generated native transform fixture",
  "parameter_files": ["align_affine.txt"],
  "transform_files": ["TransformParameters.0.txt"],
  "non_identity_probe": {
    "input_point": [3.2, 3.1, 2.0],
    "minimum_abs_displacement": 0.1
  },
  "reference_runtime": {
    "wall_seconds": -1.0,
    "peak_rss_bytes": 123456
  },
  "agreement": { "tolerance": 0.000001 }
}
"#,
    );
    let error = fixture_report(negative_runtime.path())
        .unwrap_err()
        .to_string();
    assert!(error.contains("reference_runtime.wall_seconds"), "{error}");
    assert!(error.contains("non-negative"), "{error}");

    let fractional_rss = Scratch::new("fractional-rss");
    write_generated_fixture(
        &fractional_rss,
        r#"{
  "provenance": { "kind": "generated" },
  "paths": {
    "fixed_image": "fixed.npy",
    "moving_image": "moving.npy",
    "output_points": "outputpoints.txt",
    "log": "elastix.log"
  },
  "versions": {
    "elastix": "generated-test",
    "transformix": "generated-test"
  },
  "command_line": "generated native transform fixture",
  "parameter_files": ["align_affine.txt"],
  "transform_files": ["TransformParameters.0.txt"],
  "non_identity_probe": {
    "input_point": [3.2, 3.1, 2.0],
    "minimum_abs_displacement": 0.1
  },
  "reference_runtime": {
    "wall_seconds": 1.0,
    "peak_rss_bytes": 12.5
  },
  "agreement": { "tolerance": 0.000001 }
}
"#,
    );
    let error = fixture_report(fractional_rss.path())
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("reference_runtime.peak_rss_bytes"),
        "{error}"
    );
    assert!(error.contains("positive integer byte count"), "{error}");
}

#[test]
fn registration_report_refuses_impossible_fixture_thresholds_by_name() {
    let bad_displacement = Scratch::new("bad-minimum-displacement");
    write_generated_fixture(
        &bad_displacement,
        r#"{
  "provenance": { "kind": "generated" },
  "paths": {
    "fixed_image": "fixed.npy",
    "moving_image": "moving.npy",
    "output_points": "outputpoints.txt",
    "log": "elastix.log"
  },
  "versions": {
    "elastix": "generated-test",
    "transformix": "generated-test"
  },
  "command_line": "generated native transform fixture",
  "parameter_files": ["align_affine.txt"],
  "transform_files": ["TransformParameters.0.txt"],
  "non_identity_probe": {
    "input_point": [3.2, 3.1, 2.0],
    "minimum_abs_displacement": 0.0
  },
  "agreement": { "tolerance": 0.000001 }
}
"#,
    );
    let error = fixture_report(bad_displacement.path())
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("non_identity_probe.minimum_abs_displacement"),
        "{error}"
    );
    assert!(error.contains("positive"), "{error}");

    let bad_tolerance = Scratch::new("bad-agreement-tolerance");
    write_generated_fixture(
        &bad_tolerance,
        r#"{
  "provenance": { "kind": "generated" },
  "paths": {
    "fixed_image": "fixed.npy",
    "moving_image": "moving.npy",
    "output_points": "outputpoints.txt",
    "log": "elastix.log"
  },
  "versions": {
    "elastix": "generated-test",
    "transformix": "generated-test"
  },
  "command_line": "generated native transform fixture",
  "parameter_files": ["align_affine.txt"],
  "transform_files": ["TransformParameters.0.txt"],
  "non_identity_probe": {
    "input_point": [3.2, 3.1, 2.0],
    "minimum_abs_displacement": 0.1
  },
  "agreement": { "tolerance": -0.01 }
}
"#,
    );
    let error = fixture_report(bad_tolerance.path())
        .unwrap_err()
        .to_string();
    assert!(error.contains("agreement.tolerance"), "{error}");
    assert!(error.contains("positive"), "{error}");
}

fn gaussian(center: [f64; 3]) -> Voxels {
    Array3::from_shape_fn((9, 7, 5), |(x, y, z)| {
        let dx = x as f64 - center[0];
        let dy = y as f64 - center[1];
        let dz = z as f64 - center[2];
        (-(dx * dx + dy * dy + dz * dz) / 3.0).exp()
    })
    .into()
}
