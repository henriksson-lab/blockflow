// SPDX-License-Identifier: MIT
// Original work for this crate.

// A recorded, non-identity elastix registration as the reference for the
// native fit. Needs real data, so it is ignored by default:
//
// ```text
// REGISTER_FIXTURE=/path/to/fixture \
//     cargo test --release -p blockflow-atlas-registration --test register_real_fixture -- --ignored
// ```
//
// The directory holds `registration-manifest.json` (see `report::fixture_report`)
// with the fixed and moving `.npy` volumes, the parameter files, the
// `TransformParameters` files, transformix's `outputpoints.txt`, the log and the
// tool versions.

use std::path::{Path, PathBuf};

use blockflow_atlas_registration::report::{fixture_report, require_threshold};
use blockflow_atlas_registration::{ElastixTransform, RegisterParams};
use serde_json::Value;

fn field<'a>(manifest: &'a Value, path: &str) -> &'a Value {
    path.split('.')
        .try_fold(manifest, |at, key| at.get(key))
        .unwrap_or_else(|| panic!("registration fixture manifest is missing `{path}`"))
}

fn text<'a>(manifest: &'a Value, path: &str) -> &'a str {
    field(manifest, path)
        .as_str()
        .unwrap_or_else(|| panic!("registration fixture manifest `{path}` must be a string"))
}

fn relative_files(root: &Path, manifest: &Value, path: &str) -> Vec<PathBuf> {
    let files = field(manifest, path)
        .as_array()
        .unwrap_or_else(|| panic!("registration fixture manifest `{path}` must be an array"));
    assert!(!files.is_empty(), "`{path}` names no file");
    files
        .iter()
        .map(|file| {
            let relative = file.as_str().expect("a file name");
            assert!(
                !Path::new(relative).is_absolute(),
                "`{path}` must be relative"
            );
            let file = root.join(relative);
            assert!(file.is_file(), "missing: {}", file.display());
            file
        })
        .collect()
}

#[test]
#[ignore = "needs a recorded elastix registration fixture: set REGISTER_FIXTURE to its directory"]
fn a_recorded_elastix_fixture_is_a_non_identity_transform_the_native_fit_agrees_with() {
    let root = PathBuf::from(
        std::env::var("REGISTER_FIXTURE").expect("REGISTER_FIXTURE names the fixture directory"),
    );
    let manifest: Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("registration-manifest.json"))
            .expect("the manifest reads"),
    )
    .expect("the manifest is JSON");
    for key in ["versions.elastix", "versions.transformix", "command_line"] {
        assert!(!text(&manifest, key).is_empty(), "`{key}` is empty");
    }
    relative_files(&root, &manifest, "parameter_files");
    let texts = relative_files(&root, &manifest, "transform_files")
        .iter()
        .map(|path| std::fs::read_to_string(path).expect("a transform file reads"))
        .collect::<Vec<_>>();
    let refs = texts.iter().map(String::as_str).collect::<Vec<_>>();
    let transform = ElastixTransform::from_texts(&refs).expect("the transform files parse");
    let probe = field(&manifest, "non_identity_probe.input_point")
        .as_array()
        .expect("a point")
        .iter()
        .map(|v| v.as_f64().expect("a number"))
        .collect::<Vec<_>>();
    let input = [probe[0], probe[1], probe[2]];
    let output = transform.apply(input, &RegisterParams::default());
    let displacement = (0..3)
        .map(|a| (output[a] - input[a]).abs())
        .fold(0.0, f64::max);
    let minimum = field(&manifest, "non_identity_probe.minimum_abs_displacement")
        .as_f64()
        .expect("a number");
    assert!(
        displacement >= minimum,
        "the recorded transform moves the probe by {displacement}, below {minimum}"
    );

    let report = fixture_report(&root).expect("the fixture feeds the report");
    assert_eq!(report.fixture.kind, "elastix_fixture");
    assert_eq!(
        report
            .reference
            .as_ref()
            .expect("a reference")
            .transformed_point,
        output
    );
    require_threshold(&report).expect("the native fit agrees within tolerance");
}
