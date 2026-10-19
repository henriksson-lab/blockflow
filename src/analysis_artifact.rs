// SPDX-License-Identifier: MIT

//! Storage-neutral planning for cross-dataset analysis artifacts.
//!
//! Applications own their observation schema and encoder. Blockflow owns the batch boundary:
//! deterministic partition names, path validation, and publishing the manifest last. The same
//! plan can therefore run in-process, in a local worker, or in a future SLURM worker.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Component, Path, PathBuf};

use serde::Serialize;

use crate::{Error, Result};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnalysisPartition {
    pub dataset_id: String,
    pub relative_path: PathBuf,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnalysisArtifactPlan {
    root: PathBuf,
    pub partitions: Vec<AnalysisPartition>,
}

impl AnalysisArtifactPlan {
    pub fn new(
        root: impl Into<PathBuf>,
        dataset_ids: impl IntoIterator<Item = String>,
    ) -> Result<Self> {
        let root = root.into();
        let mut seen = BTreeSet::new();
        let mut partitions = Vec::new();
        for dataset_id in dataset_ids {
            validate_id("dataset ID", &dataset_id)?;
            if !seen.insert(dataset_id.clone()) {
                return Err(Error::invalid(format!(
                    "analysis collection repeats dataset {dataset_id:?}"
                )));
            }
            partitions.push(AnalysisPartition {
                relative_path: PathBuf::from("observations")
                    .join(format!("dataset_id={dataset_id}"))
                    .join("part-000.parquet"),
                dataset_id,
            });
        }
        if partitions.is_empty() {
            return Err(Error::invalid("analysis collection has no source datasets"));
        }
        Ok(Self { root, partitions })
    }

    pub fn partition_path(&self, index: usize) -> Result<PathBuf> {
        let partition = self
            .partitions
            .get(index)
            .ok_or_else(|| Error::invalid("analysis partition index is out of bounds"))?;
        Ok(self.root.join(&partition.relative_path))
    }

    pub fn prepare(&self) -> Result<()> {
        for index in 0..self.partitions.len() {
            let path = self.partition_path(index)?;
            fs::create_dir_all(
                path.parent()
                    .ok_or_else(|| Error::invalid("analysis partition has no parent directory"))?,
            )
            .map_err(Error::backend)?;
        }
        Ok(())
    }

    /// Publish last, after every partition has been closed and validated.
    pub fn publish_manifest<T: Serialize>(&self, manifest: &T) -> Result<()> {
        fs::create_dir_all(&self.root).map_err(Error::backend)?;
        let temporary = self.root.join(".collection.json.tmp");
        let final_path = self.root.join("collection.json");
        let bytes = serde_json::to_vec_pretty(manifest)
            .map_err(|error| Error::invalid(format!("serialize collection manifest: {error}")))?;
        fs::write(&temporary, bytes).map_err(Error::backend)?;
        fs::rename(temporary, final_path).map_err(Error::backend)?;
        Ok(())
    }
}

fn validate_id(label: &str, value: &str) -> Result<()> {
    let path = Path::new(value);
    if value.is_empty()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
        || value.contains('/')
        || value.contains('\\')
        || value.chars().any(char::is_control)
    {
        return Err(Error::invalid(format!("invalid {label}")));
    }
    Ok(())
}
