// SPDX-License-Identifier: MIT

use serde_json::{json, Value};
use std::{
    fs,
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct Fixture {
    root: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "blockflow-tracking-suite-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        Self { root }
    }
    fn invoke(&self, input: &Value) -> std::process::Output {
        fs::write(
            self.root.join("input.json"),
            serde_json::to_vec(input).unwrap(),
        )
        .unwrap();
        Command::new(env!("CARGO_BIN_EXE_tracking-suite"))
            .arg(self.root.join("input.json"))
            .arg(self.root.join("output"))
            .arg("1")
            .output()
            .unwrap()
    }
    fn summary(&self) -> Value {
        serde_json::from_slice(&fs::read(self.root.join("output/summary.json")).unwrap()).unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn input() -> Value {
    json!({
        "schema": "blockflow.tracking-suite-input/v1", "extent": [1, 20, 20], "spacing": [1, 1, 1],
        "frames": [{"t_index": 0, "elapsed_seconds": 0}, {"t_index": 1, "elapsed_seconds": 7200}],
        "detections": [
            {"id": 1, "t_index": 0, "native": [0, 10, 10], "registered": [0, 10, 10], "size": 100},
            {"id": 2, "t_index": 1, "native": [0, 10, 9], "registered": [0, 10, 9], "size": 50},
            {"id": 3, "t_index": 1, "native": [0, 10, 11], "registered": [0, 10, 11], "size": 50}
        ],
        "max_distance": 5,
        "runs": [
            {"name": "frame", "solver": "frame_lap", "size_weight": 0},
            {"name": "segment", "solver": "segment_lap", "size_weight": 0, "splitting": 5}
        ],
        "provenance": {"fixture": "synthetic division"},
        "baseline_pairs": [[1, 2], [1, 3]],
        "reference": {"complete": true, "kind": "synthetic", "pairs": [[1, 2], [1, 3]], "provenance": {"fixture": true}}
    })
}

#[test]
fn cli_scores_joint_daughters_and_preserves_input() {
    let fixture = Fixture::new();
    let value = input();
    let output = fixture.invoke(&value);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result = fixture.summary();
    assert_eq!(
        result["runs"][0]["reference_evaluation"]["binary_divisions"]["false_negative"],
        1
    );
    assert_eq!(
        result["runs"][1]["reference_evaluation"]["binary_divisions"]["f1"],
        1.0
    );
    assert_eq!(result["runs"][1]["baseline_edge_difference"]["equal"], true);
    let copied: Value =
        serde_json::from_slice(&fs::read(fixture.root.join("output/input.json")).unwrap()).unwrap();
    assert_eq!(copied, value);
}

#[test]
fn baseline_agreement_is_not_reported_as_accuracy_and_existing_results_are_refused() {
    let fixture = Fixture::new();
    let mut value = input();
    value["reference"] = Value::Null;
    assert!(fixture.invoke(&value).status.success());
    let before = fixture.summary();
    assert_eq!(before["runs"][1]["reference_evaluation"], Value::Null);
    assert_eq!(before["runs"][1]["baseline_edge_difference"]["equal"], true);
    assert!(!fixture.invoke(&value).status.success());
    assert_eq!(fixture.summary(), before);
}

#[test]
fn partial_reference_and_unsafe_run_names_are_rejected_before_output_creation() {
    let fixture = Fixture::new();
    let mut value = input();
    value["reference"]["complete"] = json!(false);
    assert!(!fixture.invoke(&value).status.success());
    assert!(!fixture.root.join("output").exists());
    value["reference"]["complete"] = json!(true);
    value["runs"][0]["name"] = json!("../escape");
    assert!(!fixture.invoke(&value).status.success());
    assert!(!fixture.root.join("output").exists());
}

#[test]
fn imported_pair_scores_are_order_independent_and_shared_by_solvers() {
    let fixture = Fixture::new();
    let mut value = input();
    value["reference"] = Value::Null;
    value["score_tables"] = json!({"learned": {"costs": [
        {"from": 1, "to": 3, "cost": 0.4},
        {"from": 1, "to": 2, "cost": -0.4}
    ], "provenance": {"checkpoint_sha256": "synthetic"}}});
    value["runs"] = json!([
        {"name": "greedy", "solver": "frame_greedy", "size_weight": 0, "scorer": "learned", "alternative_cost": 0},
        {"name": "lap", "solver": "frame_lap", "size_weight": 0, "scorer": "learned", "alternative_cost": 0}
    ]);
    let output = fixture.invoke(&value);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    for name in ["greedy", "lap"] {
        let links: Value = serde_json::from_slice(
            &fs::read(fixture.root.join(format!("output/{name}/links.json"))).unwrap(),
        )
        .unwrap();
        assert_eq!(links.as_array().unwrap().len(), 1);
        assert_eq!(links[0]["from"], 1);
        assert_eq!(links[0]["to"], 2);
        assert_eq!(links[0]["cost"], -0.4);
    }
    assert_eq!(
        fixture.summary()["runs"][0]["scorer"]["parameters"]["checkpoint_sha256"],
        "synthetic"
    );
}

#[test]
fn duplicate_out_of_gate_or_incompatible_imported_scores_are_rejected() {
    let fixture = Fixture::new();
    let mut value = input();
    value["reference"] = Value::Null;
    value["score_tables"] = json!({"learned": {"costs": [
        {"from": 1, "to": 2, "cost": -0.4},
        {"from": 1, "to": 2, "cost": -0.3}
    ], "provenance": {}}});
    value["runs"] = json!([{ "name": "lap", "solver": "frame_lap", "size_weight": 0, "scorer": "learned", "alternative_cost": 0 }]);
    assert!(!fixture.invoke(&value).status.success());
    assert!(!fixture.root.join("output").exists());
    value["score_tables"]["learned"]["costs"] = json!([{ "from": 1, "to": 999, "cost": 0 }]);
    assert!(!fixture.invoke(&value).status.success());
    assert!(!fixture.root.join("output").exists());
    value["score_tables"]["learned"]["costs"] = json!([{ "from": 1, "to": 2, "cost": 0 }]);
    value["runs"][0]["solver"] = json!("segment_lap");
    assert!(!fixture.invoke(&value).status.success());
    assert!(!fixture.root.join("output").exists());
}

#[test]
fn spatial_matching_cli_keeps_exact_ids_and_reference_role() {
    let fixture = Fixture::new();
    let big = 1u64 << 60;
    let value = json!({"schema": "blockflow.frame-overlaps/v1", "minimum_iou": 0.5,
        "provenance": {"reference_role": "previous_prediction"},
        "frames": [{"t_index": 0,
            "reference_sizes": [{"id": big+1, "pixels": 10}, {"id": big+2, "pixels": 8}],
            "predicted_sizes": [{"id": 42, "pixels": 10}, {"id": 43, "pixels": 4}, {"id": 44, "pixels": 2}],
            "shared_pixels": [{"reference": big+1, "predicted": 42, "pixels": 10}, {"reference": big+2, "predicted": 43, "pixels": 4}]}]});
    let source = fixture.root.join("overlaps.json");
    let output = fixture.root.join("matching.json");
    fs::write(&source, serde_json::to_vec(&value).unwrap()).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_match-overlaps"))
        .arg(&source)
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let matching: Value = serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
    assert_eq!(
        matching["frames"][0]["matching"]["matches"][0]["reference"].as_u64(),
        Some(big + 1)
    );
    assert_eq!(
        matching["frames"][0]["matching"]["matches"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        matching["frames"][0]["matching"]["unmatched_predicted"],
        json!([44])
    );
    assert_eq!(
        matching["provenance"]["reference_role"],
        "previous_prediction"
    );
    assert!(matching.get("accuracy").is_none());
    let before = fs::read(&output).unwrap();
    assert!(!Command::new(env!("CARGO_BIN_EXE_match-overlaps"))
        .arg(source)
        .arg(&output)
        .status()
        .unwrap()
        .success());
    assert_eq!(fs::read(output).unwrap(), before);
}

#[test]
fn endpoint_cli_preserves_original_rows_and_refuses_overwrite() {
    let fixture = Fixture::new();
    let mut value = input();
    value["frames"]
        .as_array_mut()
        .unwrap()
        .push(json!({"t_index":2,"elapsed_seconds":14400}));
    value["detections"]
        .as_array_mut()
        .unwrap()
        .push(json!({"id":4,"t_index":2,"native":[0,10,9],"registered":[0,10,9],"size":50}));
    let big = 1_u64 << 60;
    for row in value["detections"].as_array_mut().unwrap() {
        if matches!(row["id"].as_u64(), Some(2 | 4)) {
            row["size"] = json!(0);
        }
        row["id"] = json!(big + row["id"].as_u64().unwrap());
    }
    let base = json!([{"from":big+1,"to":big+2,"dt":1,"weight":0.91,"kind":"continue"}]);
    let scores = json!({"schema":"blockflow.endpoint-scores/v1","scores":[{"from":big+2,"to":big+4,"cost":-0.2}],"provenance":{"fixture":"synthetic incremental costs"}});
    for (name, data) in [
        ("input.json", &value),
        ("links.json", &base),
        ("scores.json", &scores),
    ] {
        fs::write(fixture.root.join(name), serde_json::to_vec(data).unwrap()).unwrap();
    }
    let invoke = || {
        Command::new(env!("CARGO_BIN_EXE_join-endpoints"))
            .arg(fixture.root.join("input.json"))
            .arg(fixture.root.join("links.json"))
            .arg(fixture.root.join("scores.json"))
            .arg(fixture.root.join("joined"))
            .output()
            .unwrap()
    };
    let result = invoke();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let path = fixture.root.join("joined/links.json");
    let bytes = fs::read(&path).unwrap();
    let links: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(links[0], base[0]);
    assert_eq!(links[1]["from"].as_u64(), Some(big + 2));
    assert_eq!(links[1]["to"].as_u64(), Some(big + 4));
    assert_eq!(links[1]["kind"], "continue");
    let summary: Value =
        serde_json::from_slice(&fs::read(fixture.root.join("joined/summary.json")).unwrap())
            .unwrap();
    assert_eq!(summary["reference_evaluation"], Value::Null);
    assert_eq!(summary["added_continuations"], 1);
    assert!(!invoke().status.success());
    assert_eq!(fs::read(path).unwrap(), bytes);
    let preserved: Value =
        serde_json::from_slice(&fs::read(fixture.root.join("joined/base-links.json")).unwrap())
            .unwrap();
    assert_eq!(preserved, base);
}
