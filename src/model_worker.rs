//! Typed protocol for isolated native segmentation workers.
//!
//! A server chooses and launches a preinstalled executable. The executable receives only this
//! protocol: browser supplied commands, paths, and environment variables never cross the model
//! boundary. Batch runners exchange one JSON request manifest plus content-addressed binary
//! artifacts in a job directory. Interactive runners use the same messages in a length-delimited
//! stream. Keeping both forms on one Rust contract avoids handwritten camelCase mappings.

use std::{
    collections::HashSet,
    io::{Read, Write},
    path::{Component, Path},
};

use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};

use crate::{Error, Result};

pub const MODEL_WORKER_PROTOCOL_VERSION: u32 = 1;
pub const MAX_MODEL_WORKER_FRAME_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum WorkerArrayDtype {
    U8,
    U16,
    U32,
    I32,
    F16,
    F32,
}

/// Binary data is always an artifact beneath the job directory, never an arbitrary host path.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WorkerArtifact {
    pub relative_path: String,
    pub sha256: String,
    pub size_bytes: u64,
    pub media_type: String,
    pub role: String,
}

impl WorkerArtifact {
    pub fn validate(&self) -> Result<()> {
        safe_relative_path(&self.relative_path)?;
        required("artifact media type", &self.media_type)?;
        required("artifact role", &self.role)?;
        validate_sha256("artifact", &self.sha256)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WorkerArray {
    pub artifact: WorkerArtifact,
    pub dtype: WorkerArrayDtype,
    pub shape: Vec<u64>,
    /// Axis names in the same order as `shape`, for example `["y", "x", "c"]`.
    pub axes: Vec<String>,
}

impl WorkerArray {
    pub fn validate(&self) -> Result<()> {
        self.artifact.validate()?;
        if self.shape.is_empty() || self.shape.len() != self.axes.len() || self.shape.contains(&0) {
            return Err(Error::invalid(
                "worker array shape and axes are inconsistent",
            ));
        }
        let mut axes = HashSet::new();
        if self
            .axes
            .iter()
            .any(|axis| axis.trim().is_empty() || !axes.insert(axis))
        {
            return Err(Error::invalid(
                "worker array axes must be nonempty and unique",
            ));
        }
        let elements = self
            .shape
            .iter()
            .try_fold(1_u64, |n, &extent| n.checked_mul(extent))
            .ok_or_else(|| Error::invalid("worker array shape overflows"))?;
        let bytes = elements
            .checked_mul(dtype_bytes(self.dtype))
            .ok_or_else(|| Error::invalid("worker array byte size overflows"))?;
        if bytes != self.artifact.size_bytes {
            return Err(Error::invalid(
                "worker array size does not match dtype and shape",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WorkerPointPrompt {
    pub xy: [f64; 2],
    pub positive: bool,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WorkerPromptSet {
    #[serde(default)]
    pub points: Vec<WorkerPointPrompt>,
    #[serde(default)]
    pub boxes_xyxy: Vec<[f64; 4]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dense_mask: Option<WorkerArray>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prior_logits: Option<WorkerArray>,
}

impl WorkerPromptSet {
    fn validate(&self) -> Result<()> {
        if self
            .points
            .iter()
            .any(|point| point.xy.iter().any(|v| !v.is_finite()))
            || self.boxes_xyxy.iter().flatten().any(|v| !v.is_finite())
        {
            return Err(Error::invalid("worker prompt coordinates must be finite"));
        }
        if let Some(mask) = &self.dense_mask {
            mask.validate()?;
        }
        if let Some(prior) = &self.prior_logits {
            prior.validate()?;
        }
        Ok(())
    }
}

/// Operations are model neutral. Family-specific settings live in validated recipe artifacts.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "camelCase", deny_unknown_fields)]
pub enum ModelWorkerOperation {
    Encode {
        image: WorkerArray,
    },
    Predict {
        embedding_id: String,
        prompts: WorkerPromptSet,
        #[serde(default)]
        multimask: bool,
    },
    AutomaticInfer {
        image: WorkerArray,
    },
    Train {
        recipe: WorkerArtifact,
        annotation_snapshot: WorkerArtifact,
        #[serde(default)]
        parent_model_artifacts: Vec<WorkerArtifact>,
    },
    Evaluate {
        recipe: WorkerArtifact,
        annotation_snapshot: WorkerArtifact,
        #[serde(default)]
        model_artifacts: Vec<WorkerArtifact>,
    },
}

impl ModelWorkerOperation {
    fn validate(&self) -> Result<()> {
        match self {
            Self::Encode { image } | Self::AutomaticInfer { image } => image.validate(),
            Self::Predict {
                embedding_id,
                prompts,
                ..
            } => {
                required("embedding id", embedding_id)?;
                prompts.validate()
            }
            Self::Train {
                recipe,
                annotation_snapshot,
                parent_model_artifacts,
            } => {
                recipe.validate()?;
                annotation_snapshot.validate()?;
                for artifact in parent_model_artifacts {
                    artifact.validate()?;
                }
                Ok(())
            }
            Self::Evaluate {
                recipe,
                annotation_snapshot,
                model_artifacts,
            } => {
                recipe.validate()?;
                annotation_snapshot.validate()?;
                for artifact in model_artifacts {
                    artifact.validate()?;
                }
                Ok(())
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelWorkerRequest {
    pub protocol_version: u32,
    pub request_id: String,
    pub runtime_id: String,
    pub model_revision_id: String,
    pub maximum_memory_bytes: u64,
    pub operation: ModelWorkerOperation,
}

impl ModelWorkerRequest {
    pub fn validate(&self) -> Result<()> {
        if self.protocol_version != MODEL_WORKER_PROTOCOL_VERSION {
            return Err(Error::invalid("unsupported model worker protocol version"));
        }
        required("worker request id", &self.request_id)?;
        required("worker runtime id", &self.runtime_id)?;
        required("worker model revision", &self.model_revision_id)?;
        if self.maximum_memory_bytes == 0 {
            return Err(Error::invalid("worker memory budget is zero"));
        }
        self.operation.validate()
    }
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "camelCase", deny_unknown_fields)]
pub enum ModelWorkerCommand {
    Request { request: Box<ModelWorkerRequest> },
    Cancel { request_id: String },
    Health { request_id: String },
}

#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "camelCase", deny_unknown_fields)]
pub enum ModelWorkerEvent {
    Ready {
        protocol_version: u32,
        runtime_id: String,
    },
    Progress {
        request_id: String,
        completed: u64,
        total: u64,
        message: String,
    },
    Completed {
        request_id: String,
        #[serde(default)]
        artifacts: Vec<WorkerArtifact>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        embedding_id: Option<String>,
    },
    Cancelled {
        request_id: String,
    },
    Failed {
        request_id: String,
        code: String,
        message: String,
        retryable: bool,
    },
    Healthy {
        request_id: String,
    },
}

/// JSON Schema is generated from the same Rust contract used for serialization.
pub fn command_json_schema() -> serde_json::Value {
    serde_json::to_value(schema_for!(ModelWorkerCommand)).expect("JSON schema is serializable")
}

pub fn event_json_schema() -> serde_json::Value {
    serde_json::to_value(schema_for!(ModelWorkerEvent)).expect("JSON schema is serializable")
}

/// Read one little-endian u32 length-prefixed JSON command from a supervised worker stream.
pub fn read_command(mut reader: impl Read) -> Result<ModelWorkerCommand> {
    read_frame(&mut reader)
}

pub fn write_command(mut writer: impl Write, value: &ModelWorkerCommand) -> Result<()> {
    write_frame(&mut writer, value)
}

pub fn read_event(mut reader: impl Read) -> Result<ModelWorkerEvent> {
    read_frame(&mut reader)
}

pub fn write_event(mut writer: impl Write, value: &ModelWorkerEvent) -> Result<()> {
    write_frame(&mut writer, value)
}

fn read_frame<T: for<'de> Deserialize<'de>>(reader: &mut impl Read) -> Result<T> {
    let mut length = [0_u8; 4];
    reader
        .read_exact(&mut length)
        .map_err(|e| Error::backend(format!("read worker frame length: {e}")))?;
    let length = u32::from_le_bytes(length) as usize;
    if length == 0 || length > MAX_MODEL_WORKER_FRAME_BYTES {
        return Err(Error::invalid("model worker frame length is invalid"));
    }
    let mut bytes = vec![0; length];
    reader
        .read_exact(&mut bytes)
        .map_err(|e| Error::backend(format!("read worker frame: {e}")))?;
    serde_json::from_slice(&bytes)
        .map_err(|e| Error::invalid(format!("decode model worker frame: {e}")))
}

fn write_frame<T: Serialize>(writer: &mut impl Write, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec(value)
        .map_err(|e| Error::invalid(format!("encode model worker frame: {e}")))?;
    if bytes.is_empty() || bytes.len() > MAX_MODEL_WORKER_FRAME_BYTES {
        return Err(Error::invalid("model worker frame length is invalid"));
    }
    writer
        .write_all(&(bytes.len() as u32).to_le_bytes())
        .and_then(|_| writer.write_all(&bytes))
        .map_err(|e| Error::backend(format!("write model worker frame: {e}")))
}

fn dtype_bytes(dtype: WorkerArrayDtype) -> u64 {
    match dtype {
        WorkerArrayDtype::U8 => 1,
        WorkerArrayDtype::U16 | WorkerArrayDtype::F16 => 2,
        WorkerArrayDtype::U32 | WorkerArrayDtype::I32 | WorkerArrayDtype::F32 => 4,
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
    if value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit()) {
        Ok(())
    } else {
        Err(Error::invalid(format!("{what} sha256 is invalid")))
    }
}

fn safe_relative_path(value: &str) -> Result<()> {
    let path = Path::new(value);
    if value.is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        Err(Error::invalid(
            "worker artifact path must be a safe relative path",
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn framed_contract_round_trips_camel_case_and_rejects_unknown_fields() {
        let command = ModelWorkerCommand::Health {
            request_id: "health-1".into(),
        };
        let mut bytes = Vec::new();
        write_command(&mut bytes, &command).unwrap();
        assert_eq!(read_command(bytes.as_slice()).unwrap(), command);
        assert!(serde_json::from_str::<ModelWorkerCommand>(
            r#"{"kind":"health","requestId":"x","request_id":"wrong"}"#
        )
        .is_err());
        assert!(command_json_schema().get("$schema").is_some());
    }
}
