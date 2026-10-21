use blockflow::zarr_env::AttachedImage;
use serde_json::Value;
use std::{path::Path, process::Command};
fn run(args: &[&str]) -> Value {
    let result = Command::new(env!("CARGO_BIN_EXE_blockflow-physical-plane-sampling"))
        .args(args)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    serde_json::from_slice(&result.stdout).unwrap()
}
fn values(path: &Path) -> Vec<f64> {
    AttachedImage::at(path)
        .read()
        .unwrap()
        .view::<f64>()
        .unwrap()
        .iter()
        .copied()
        .collect()
}
#[test]
fn prepared_zarr_matches_independent_analytic_values_and_block_grids() {
    let base = std::env::temp_dir().join(format!(
        "blockflow-plane-roundtrip-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&base).unwrap();
    let fixture = base.join("fixture");
    let multi = base.join("multi");
    let single = base.join("single");
    run(&["prepare", "--out", fixture.to_str().unwrap()]);
    let config = fixture.join("plane.json");
    let report = run(&[
        "sample",
        "--config",
        config.to_str().unwrap(),
        "--out",
        multi.to_str().unwrap(),
    ]);
    assert_eq!(report["outputs"][0]["blocks"], 16);
    assert_eq!(
        report["output_axes"],
        serde_json::json!(["singleton", "row", "column"])
    );
    assert!(
        report["outputs"][0]["max_source_voxels_per_block"]
            .as_u64()
            .unwrap()
            < 32 * 48 * 56 / 100
    );
    let actual = values(&multi.join("intensity/level1"));
    let support = values(&multi.join("support/level1"));
    let s = 0.5_f64.sqrt();
    let step = |p: f64, k: f64| (p - (k - 1.)).clamp(0., 1.);
    for row in 0..25 {
        for col in 0..31 {
            let dr = (row as f64 - 12.) * 0.8;
            let dc = (col as f64 - 15.) * 1.1;
            let q = [
                (40. + s * dr + s * 1.7 - 10.) / 2.,
                (9. + s * dr - s * 1.7 + 5.) / 0.7,
                (34. + dc - 8.) / 1.3,
            ];
            let expected = 2. * q[0] + 3. * q[1] + q[2] + 0.125;
            assert!((actual[row * 31 + col] - expected).abs() < 1e-10);
            assert!((support[row * 31 + col] - step(q[1], 15.) * step(q[2], 17.)).abs() < 1e-10);
        }
    }
    let mut settings: Value = serde_json::from_slice(&std::fs::read(&config).unwrap()).unwrap();
    settings["block_rc"] = serde_json::json!([25, 31]);
    let single_config = base.join("single.json");
    std::fs::write(&single_config, serde_json::to_vec(&settings).unwrap()).unwrap();
    let report = run(&[
        "sample",
        "--config",
        single_config.to_str().unwrap(),
        "--out",
        single.to_str().unwrap(),
    ]);
    assert_eq!(report["outputs"][0]["blocks"], 1);
    assert_eq!(values(&single.join("intensity/level1")), actual);
    assert_eq!(values(&single.join("support/level1")), support);
    // Refuse output replacement; this check must leave the existing data alone.
    assert!(
        !Command::new(env!("CARGO_BIN_EXE_blockflow-physical-plane-sampling"))
            .args([
                "sample",
                "--config",
                config.to_str().unwrap(),
                "--out",
                multi.to_str().unwrap()
            ])
            .output()
            .unwrap()
            .status
            .success()
    );
    assert_eq!(values(&multi.join("intensity/level1")), actual);
    std::fs::remove_dir_all(base).unwrap();
}
