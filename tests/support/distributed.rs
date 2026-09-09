use std::path::{Path, PathBuf};
use std::time::Duration;

use blockflow::decomposition::Decomposition;
use blockflow::distributed::local::{Binaries, LocalOptions};
use blockflow::distributed::shared_volume::SharedVolumes;
use blockflow::distributed::spec::JobSpec;
use serde_json::Value;

pub use super::volume::flat_ramp_f64 as ramp;

pub fn binaries() -> Binaries {
    Binaries {
        coordinator: PathBuf::from(env!("CARGO_BIN_EXE_blockflow-coordinator")),
        worker: PathBuf::from(env!("CARGO_BIN_EXE_blockflow-worker")),
    }
}

pub fn scratch(prefix: &str, name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("blockflow-{prefix}-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

pub fn local_options(dir: &Path, workers: usize) -> LocalOptions {
    let mut options = LocalOptions::new(dir, workers).expect("local options");
    options.binaries = binaries();
    options.timeout = Duration::from_secs(120);
    options
}

pub fn write_ramp_input(dir: &Path, spec: &JobSpec, decomposition: &Decomposition) {
    let volumes = dir.join("volumes");
    let store = SharedVolumes::create(
        &volumes,
        spec.workflow.shape,
        spec.workflow.chunk,
        decomposition.n_phases(),
    )
    .expect("image files");
    store
        .write_image(0, &ramp(spec.workflow.shape))
        .expect("an input");
}

pub fn output_bytes(dir: &Path, spec: &JobSpec, decomposition: &Decomposition) -> Vec<u8> {
    SharedVolumes::open(
        &dir.join("volumes"),
        spec.workflow.shape,
        spec.workflow.chunk,
        decomposition.n_phases(),
    )
    .expect("the volumes")
    .image_bytes(decomposition.n_phases())
    .expect("the output image")
}

/// Where two images first differ, and by how much.
pub fn first_difference(left: &[u8], right: &[u8]) -> Option<String> {
    if left.len() != right.len() {
        return Some(format!(
            "different sizes: {} byte(s) against {}",
            left.len(),
            right.len()
        ));
    }
    let at = left.iter().zip(right).position(|(a, b)| a != b)?;
    let voxel = at / std::mem::size_of::<f64>();
    let word = |bytes: &[u8]| {
        let start = voxel * 8;
        bytes
            .get(start..start + 8)
            .and_then(|slice| slice.try_into().ok())
            .map(f64::from_le_bytes)
    };
    let differing = left.iter().zip(right).filter(|(a, b)| a != b).count();
    Some(format!(
        "first differ at byte {at} (voxel {voxel}): {:?} against {:?}; {differing} of {} \
         byte(s) differ",
        word(left),
        word(right),
        left.len()
    ))
}

pub fn worker_field(reports: &[Value], field: &str) -> Vec<u64> {
    reports
        .iter()
        .map(|report| report.get(field).and_then(Value::as_u64).unwrap_or(0))
        .collect()
}
