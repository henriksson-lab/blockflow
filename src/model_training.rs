//! Reproducible native segmentation training inputs and outputs.
//!
//! Training consumes an immutable annotation snapshot rather than a live ROZT database. The
//! snapshot embeds the exact user-named class schema and content hashes for every accepted mask.
//! A recipe selects a native Rust trainer with typed options, and the resulting model bundle
//! keeps both the snapshot identity and schema. Python reference programs may compare exported
//! fixtures in regression work, but are not a production runtime represented by this contract.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::{
    model_registry::{LabelSchema, TrainingScope},
    Error, Result,
};

pub const ANNOTATION_SNAPSHOT_SCHEMA_VERSION: u32 = 2;
pub const TRAINING_RECIPE_SCHEMA_VERSION: u32 = 1;
pub const EVALUATION_REPORT_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "camelCase", deny_unknown_fields)]
pub enum TrainingDevice {
    Cpu,
    Cuda { index: u32 },
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum DataSplit {
    Train,
    Validation,
    Test,
}

/// Identity of input pixels. `dataset_revision_id` is a manifest/content revision, not a path.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ImageSourceRevision {
    pub dataset_id: String,
    pub dataset_revision_id: String,
    #[serde(default)]
    pub timepoint: u32,
    #[serde(default)]
    pub channel_indices: Vec<u32>,
}

/// One final corrected instance or semantic region included in a training snapshot.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AnnotationSample {
    pub sample_id: String,
    pub image_source: ImageSourceRevision,
    pub result_id: String,
    pub result_revision_id: String,
    pub object_id: u64,
    pub class_id: u32,
    /// SHA-256 of the final composed mask after model proposal and user corrections.
    pub mask_sha256: String,
    /// SHA-256 of the exact model RGB input crop after the frozen image recipe.
    pub image_sha256: String,
    pub split: DataSplit,
    /// Half-open XYZ bounds in level-zero pixels.
    pub bounds_xyz: [u64; 6],
}

/// Frozen data selection. Changing a name, mask, split or sampling rule creates another snapshot.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AnnotationSnapshotManifest {
    pub schema_version: u32,
    pub snapshot_id: String,
    pub label_schema: LabelSchema,
    pub samples: Vec<AnnotationSample>,
    pub split_seed: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selection_query: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_by: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
}

impl AnnotationSnapshotManifest {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != ANNOTATION_SNAPSHOT_SCHEMA_VERSION {
            return Err(Error::invalid(
                "unsupported annotation snapshot schema version",
            ));
        }
        required("annotation snapshot id", &self.snapshot_id)?;
        self.label_schema.validate()?;
        if self.samples.is_empty() {
            return Err(Error::invalid("annotation snapshot has no samples"));
        }
        let class_ids = self
            .label_schema
            .classes
            .iter()
            .map(|class| class.class_id)
            .collect::<HashSet<_>>();
        let mut sample_ids = HashSet::new();
        for sample in &self.samples {
            for (what, value) in [
                ("sample id", &sample.sample_id),
                ("dataset id", &sample.image_source.dataset_id),
                ("dataset revision", &sample.image_source.dataset_revision_id),
                ("annotation result id", &sample.result_id),
                ("annotation result revision", &sample.result_revision_id),
            ] {
                required(what, value)?;
            }
            if !sample_ids.insert(&sample.sample_id) {
                return Err(Error::invalid("annotation sample IDs must be unique"));
            }
            if !class_ids.contains(&sample.class_id) {
                return Err(Error::invalid(format!(
                    "annotation sample {} uses unknown class {}",
                    sample.sample_id, sample.class_id
                )));
            }
            validate_sha256("annotation mask", &sample.mask_sha256)?;
            validate_sha256("annotation image crop", &sample.image_sha256)?;
            let [x0, y0, z0, x1, y1, z1] = sample.bounds_xyz;
            if x0 >= x1 || y0 >= y1 || z0 >= z1 {
                return Err(Error::invalid(format!(
                    "annotation sample {} has empty bounds",
                    sample.sample_id
                )));
            }
        }
        if !self
            .samples
            .iter()
            .any(|sample| sample.split == DataSplit::Train)
        {
            return Err(Error::invalid("annotation snapshot has no training split"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "camelCase", deny_unknown_fields)]
pub enum NativeTrainerOptions {
    SamMaskDecoder {
        learning_rate: f64,
        #[serde(default)]
        weight_decay: f64,
        epochs: u32,
        batch_size: u32,
        #[serde(default)]
        train_iou_head: bool,
    },
    SemanticHead {
        learning_rate: f64,
        epochs: u32,
        batch_size: u32,
        hidden_channels: u32,
    },
    LowRankAdapter {
        learning_rate: f64,
        epochs: u32,
        batch_size: u32,
        rank: u32,
        alpha: f64,
    },
}

impl NativeTrainerOptions {
    fn validate(&self) -> Result<()> {
        let (learning_rate, weight_decay, epochs, batch_size) = match self {
            Self::SamMaskDecoder {
                learning_rate,
                weight_decay,
                epochs,
                batch_size,
                ..
            } => (*learning_rate, *weight_decay, *epochs, *batch_size),
            Self::SemanticHead {
                learning_rate,
                epochs,
                batch_size,
                ..
            }
            | Self::LowRankAdapter {
                learning_rate,
                epochs,
                batch_size,
                ..
            } => (*learning_rate, 0.0, *epochs, *batch_size),
        };
        if !learning_rate.is_finite()
            || learning_rate <= 0.0
            || !weight_decay.is_finite()
            || weight_decay < 0.0
            || epochs == 0
            || batch_size == 0
        {
            return Err(Error::invalid(
                "native trainer hyperparameters must be positive",
            ));
        }
        match self {
            Self::SemanticHead {
                hidden_channels, ..
            } if *hidden_channels == 0 => {
                Err(Error::invalid("semantic head needs hidden channels"))
            }
            Self::LowRankAdapter { rank, alpha, .. }
                if *rank == 0 || !alpha.is_finite() || *alpha <= 0.0 =>
            {
                Err(Error::invalid(
                    "low-rank adapter rank and alpha must be positive",
                ))
            }
            _ => Ok(()),
        }
    }

    fn supports(&self, scope: TrainingScope) -> bool {
        matches!(
            (self, scope),
            (Self::SamMaskDecoder { .. }, TrainingScope::MaskDecoder)
                | (Self::SemanticHead { .. }, TrainingScope::SemanticHead)
                | (Self::LowRankAdapter { .. }, TrainingScope::Adapter)
        )
    }
}

/// Fully typed JSON job payload for a native Blockflow training job.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TrainingRecipe {
    pub schema_version: u32,
    pub recipe_id: String,
    pub base_model_revision_id: String,
    pub snapshot_id: String,
    pub output_model_id: String,
    pub output_display_name: String,
    pub scope: TrainingScope,
    pub random_seed: u64,
    pub maximum_memory_bytes: u64,
    pub device: TrainingDevice,
    pub trainer: NativeTrainerOptions,
}

impl TrainingRecipe {
    pub fn validate(&self, snapshot: &AnnotationSnapshotManifest) -> Result<()> {
        if self.schema_version != TRAINING_RECIPE_SCHEMA_VERSION {
            return Err(Error::invalid("unsupported training recipe schema version"));
        }
        snapshot.validate()?;
        for (what, value) in [
            ("training recipe id", &self.recipe_id),
            ("base model revision", &self.base_model_revision_id),
            ("snapshot id", &self.snapshot_id),
            ("output model id", &self.output_model_id),
            ("output model display name", &self.output_display_name),
        ] {
            required(what, value)?;
        }
        if self.snapshot_id != snapshot.snapshot_id {
            return Err(Error::invalid("training recipe refers to another snapshot"));
        }
        if self.maximum_memory_bytes == 0 {
            return Err(Error::invalid("training recipe memory budget is zero"));
        }
        self.trainer.validate()?;
        if !self.trainer.supports(self.scope) {
            return Err(Error::invalid(
                "native trainer does not implement the requested scope",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ClassEvaluationMetrics {
    pub class_id: u32,
    pub object_count: u64,
    pub mean_dice: f64,
    pub mean_iou: f64,
    pub boundary_f_score: f64,
    pub splits: u64,
    pub merges: u64,
    pub missed: u64,
    pub spurious: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct EvaluationReport {
    pub schema_version: u32,
    pub candidate_model_revision_id: String,
    pub parent_model_revision_id: String,
    pub snapshot_id: String,
    pub split: DataSplit,
    pub per_class: Vec<ClassEvaluationMetrics>,
    /// Parent revision evaluated on the identical frozen split before optimization.
    #[serde(default)]
    pub parent_per_class: Vec<ClassEvaluationMetrics>,
    pub median_latency_micros: u64,
    #[serde(default)]
    pub parent_median_latency_micros: u64,
    pub peak_memory_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preview_sample_id: Option<String>,
}

impl EvaluationReport {
    pub fn validate(&self, snapshot: &AnnotationSnapshotManifest) -> Result<()> {
        if self.schema_version != EVALUATION_REPORT_SCHEMA_VERSION {
            return Err(Error::invalid(
                "unsupported evaluation report schema version",
            ));
        }
        for (what, value) in [
            (
                "candidate model revision",
                &self.candidate_model_revision_id,
            ),
            ("parent model revision", &self.parent_model_revision_id),
            ("evaluation snapshot", &self.snapshot_id),
        ] {
            required(what, value)?;
        }
        if self.snapshot_id != snapshot.snapshot_id {
            return Err(Error::invalid(
                "evaluation report refers to another snapshot",
            ));
        }
        let classes = snapshot
            .label_schema
            .classes
            .iter()
            .map(|class| class.class_id)
            .collect::<HashSet<_>>();
        for group in [&self.per_class, &self.parent_per_class] {
            let mut seen = HashSet::new();
            for metrics in group {
                if !classes.contains(&metrics.class_id) || !seen.insert(metrics.class_id) {
                    return Err(Error::invalid(
                        "evaluation contains an unknown or duplicate class",
                    ));
                }
                for metric in [
                    metrics.mean_dice,
                    metrics.mean_iou,
                    metrics.boundary_f_score,
                ] {
                    if !metric.is_finite() || !(0.0..=1.0).contains(&metric) {
                        return Err(Error::invalid(
                            "evaluation scores must be between zero and one",
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

fn required(what: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        Err(Error::invalid(format!("{what} is required")))
    } else {
        Ok(())
    }
}

fn validate_sha256(what: &str, value: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Err(Error::invalid(format!("{what} sha256 is invalid")))
    } else {
        Ok(())
    }
}
