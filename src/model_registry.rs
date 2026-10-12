//! Immutable segmentation model bundles and their installed catalog.
//!
//! A browser or workflow selects a content revision, never a checkpoint path. The registry owns
//! paths, verifies every artifact before installation, and atomically moves a complete staged
//! bundle into place. Model-family adapters remain separate modules; this file describes what
//! they can do and the data that gives their outputs meaning.

use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    fs::{self, File},
    io::Read,
    path::{Component, Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{Error, Result};

pub const MODEL_BUNDLE_SCHEMA_VERSION: u32 = 1;
pub const MODEL_REGISTRY_STATE_SCHEMA_VERSION: u32 = 1;

/// Mutable catalog metadata is deliberately separate from immutable model bundles. Aliases and
/// archive visibility can change without changing a revision digest or its scientific lineage.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ModelRegistryState {
    schema_version: u32,
    #[serde(default)]
    aliases: BTreeMap<String, String>,
    #[serde(default)]
    archived_revisions: BTreeSet<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum TaskKind {
    InteractiveBinaryInstance,
    AutomaticInstance,
    SemanticSegmentation,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum PromptCapability {
    PositivePoints,
    NegativePoints,
    Boxes,
    DenseMask,
    PriorLogits,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Dimensionality {
    TwoD,
    NativeThreeD,
    OrderedSequence,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum TrainingScope {
    MaskDecoder,
    SemanticHead,
    InstanceDecoder,
    Adapter,
    FullModel,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum ClassRole {
    ModelOutput,
    SamplingMetadata,
    ClassifierTarget,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AnnotationClass {
    pub class_id: u32,
    /// Supplied by the user and copied into every trained child model.
    pub name: String,
    pub color_rgba: [u8; 4],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_class_id: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ontology_term: Option<String>,
    #[serde(default)]
    pub disabled: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LabelSchema {
    pub schema_id: String,
    pub revision_id: String,
    pub name: String,
    pub classes: Vec<AnnotationClass>,
}

impl LabelSchema {
    pub fn validate(&self) -> Result<()> {
        required("label schema id", &self.schema_id)?;
        required("label schema revision", &self.revision_id)?;
        required("label schema name", &self.name)?;
        let mut ids = HashSet::new();
        let mut names = HashSet::new();
        for class in &self.classes {
            if class.class_id == 0 {
                return Err(Error::invalid(
                    "annotation class 0 is reserved for background",
                ));
            }
            required("annotation class name", &class.name)?;
            if !ids.insert(class.class_id) {
                return Err(Error::invalid("annotation class IDs must be unique"));
            }
            if !names.insert(class.name.trim().to_lowercase()) {
                return Err(Error::invalid("annotation class names must be unique"));
            }
        }
        if self.classes.iter().any(|class| {
            class
                .parent_class_id
                .is_some_and(|parent| parent == class.class_id || !ids.contains(&parent))
        }) {
            return Err(Error::invalid(
                "annotation class parent refers to an unknown or identical class",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelCapabilities {
    pub tasks: Vec<TaskKind>,
    pub prompts: Vec<PromptCapability>,
    pub dimensionality: Vec<Dimensionality>,
    pub training_scopes: Vec<TrainingScope>,
}

impl ModelCapabilities {
    fn validate(&self) -> Result<()> {
        if self.tasks.is_empty() || self.dimensionality.is_empty() {
            return Err(Error::invalid(
                "a model needs at least one task and dimensionality",
            ));
        }
        unique("model tasks", &self.tasks)?;
        unique("prompt capabilities", &self.prompts)?;
        unique("model dimensionality", &self.dimensionality)?;
        unique("training scopes", &self.training_scopes)?;
        if !self.tasks.contains(&TaskKind::InteractiveBinaryInstance) && !self.prompts.is_empty() {
            return Err(Error::invalid(
                "only an interactive model may declare prompt capabilities",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelArtifact {
    /// Portable path relative to the bundle directory.
    pub path: PathBuf,
    pub sha256: String,
    pub size: u64,
    pub media_type: String,
    pub role: String,
}

/// Reproducible origin of a trained revision. The full manifests are artifacts in the bundle;
/// these identities and hashes make lineage available without opening arbitrary files.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TrainingLineage {
    pub annotation_snapshot_id: String,
    pub training_recipe_id: String,
    pub training_recipe_sha256: String,
    pub evaluation_report_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelBundleManifest {
    pub schema_version: u32,
    pub model_id: String,
    pub revision_id: String,
    pub display_name: String,
    pub family: String,
    pub runtime_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_revision_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_lineage: Option<TrainingLineage>,
    pub capabilities: ModelCapabilities,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Complete user-named annotation type schema. Required for every trained revision and kept
    /// in the immutable model bundle rather than inferred from numeric output indices.
    pub label_schema: Option<LabelSchema>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub class_role: Option<ClassRole>,
    pub artifacts: Vec<ModelArtifact>,
    #[serde(default)]
    pub pretrained: bool,
    #[serde(default)]
    pub archived: bool,
}

impl ModelBundleManifest {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != MODEL_BUNDLE_SCHEMA_VERSION {
            return Err(Error::invalid(format!(
                "unsupported model bundle schema version {}",
                self.schema_version
            )));
        }
        for (what, value) in [
            ("model id", &self.model_id),
            ("model revision", &self.revision_id),
            ("model display name", &self.display_name),
            ("model family", &self.family),
            ("model runtime", &self.runtime_id),
        ] {
            required(what, value)?;
        }
        safe_component("model id", &self.model_id)?;
        safe_component("model revision", &self.revision_id)?;
        self.capabilities.validate()?;
        if let Some(training) = &self.training_lineage {
            if self.parent_revision_id.is_none() {
                return Err(Error::invalid(
                    "a trained model revision must identify its parent revision",
                ));
            }
            for (what, value) in [
                ("annotation snapshot", &training.annotation_snapshot_id),
                ("training recipe", &training.training_recipe_id),
            ] {
                required(what, value)?;
            }
            validate_sha256("training recipe", &training.training_recipe_sha256)?;
            validate_sha256("evaluation report", &training.evaluation_report_sha256)?;
            if self.label_schema.is_none() {
                return Err(Error::invalid(
                    "a trained model revision must embed its user-named label schema",
                ));
            }
        }
        if let Some(schema) = &self.label_schema {
            schema.validate()?;
            if self.class_role.is_none() {
                return Err(Error::invalid(
                    "a model with annotation classes must declare their role",
                ));
            }
        } else if self.class_role.is_some() {
            return Err(Error::invalid(
                "model class role requires an embedded label schema",
            ));
        }
        if self.artifacts.is_empty() {
            return Err(Error::invalid("a model bundle contains no artifacts"));
        }
        let mut paths = HashSet::new();
        for artifact in &self.artifacts {
            safe_relative_path(&artifact.path)?;
            required("artifact digest", &artifact.sha256)?;
            required("artifact media type", &artifact.media_type)?;
            required("artifact role", &artifact.role)?;
            if artifact.sha256.len() != 64
                || !artifact.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
            {
                return Err(Error::invalid(
                    "artifact sha256 is not 64 hexadecimal digits",
                ));
            }
            if !paths.insert(artifact.path.clone()) {
                return Err(Error::invalid("model artifact paths must be unique"));
            }
        }
        Ok(())
    }

    pub fn read(bundle: &Path) -> Result<Self> {
        let path = bundle.join("manifest.json");
        let bytes = fs::read(&path).map_err(|error| {
            Error::backend(format!("read model manifest {}: {error}", path.display()))
        })?;
        let manifest: Self = serde_json::from_slice(&bytes).map_err(|error| {
            Error::invalid(format!("parse model manifest {}: {error}", path.display()))
        })?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn verify_artifacts(&self, bundle: &Path) -> Result<()> {
        for artifact in &self.artifacts {
            let path = bundle.join(&artifact.path);
            let metadata = fs::metadata(&path).map_err(|error| {
                Error::backend(format!(
                    "inspect model artifact {}: {error}",
                    path.display()
                ))
            })?;
            if !metadata.is_file() || metadata.len() != artifact.size {
                return Err(Error::invalid(format!(
                    "model artifact {} has the wrong type or size",
                    artifact.path.display()
                )));
            }
            let actual = sha256_file(&path)?;
            if !actual.eq_ignore_ascii_case(&artifact.sha256) {
                return Err(Error::invalid(format!(
                    "model artifact {} failed its sha256 check",
                    artifact.path.display()
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct ModelRegistry {
    root: PathBuf,
}

impl ModelRegistry {
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        fs::create_dir_all(root.join("models")).map_err(|error| {
            Error::backend(format!("create model registry {}: {error}", root.display()))
        })?;
        fs::create_dir_all(root.join(".staging")).map_err(|error| {
            Error::backend(format!("create model staging {}: {error}", root.display()))
        })?;
        Ok(Self { root })
    }

    pub fn staging_path(&self, id: &str) -> Result<PathBuf> {
        safe_component("model staging id", id)?;
        Ok(self.root.join(".staging").join(id))
    }

    pub fn revision_path(&self, model_id: &str, revision_id: &str) -> Result<PathBuf> {
        safe_component("model id", model_id)?;
        safe_component("model revision", revision_id)?;
        Ok(self.root.join("models").join(model_id).join(revision_id))
    }

    /// Install a complete bundle staged beneath this registry. Existing revisions are immutable.
    pub fn install_staged(&self, staged: &Path) -> Result<ModelBundleManifest> {
        let staging = self.root.join(".staging");
        if staged.parent() != Some(staging.as_path()) {
            return Err(Error::invalid(
                "model bundle must be staged inside this registry",
            ));
        }
        let manifest = ModelBundleManifest::read(staged)?;
        manifest.verify_artifacts(staged)?;
        let destination = self.revision_path(&manifest.model_id, &manifest.revision_id)?;
        if destination.exists() {
            return Err(Error::invalid("model revision is already installed"));
        }
        let parent = destination
            .parent()
            .ok_or_else(|| Error::invalid("model revision has no parent directory"))?;
        fs::create_dir_all(parent).map_err(|error| {
            Error::backend(format!(
                "create model lineage {}: {error}",
                parent.display()
            ))
        })?;
        fs::rename(staged, &destination).map_err(|error| {
            Error::backend(format!(
                "install model bundle {} at {}: {error}",
                staged.display(),
                destination.display()
            ))
        })?;
        Ok(manifest)
    }

    pub fn list(&self) -> Result<Vec<ModelBundleManifest>> {
        let state = self.read_state()?;
        let mut manifests = Vec::new();
        let models = self.root.join("models");
        for model in read_directories(&models)? {
            for revision in read_directories(&model)? {
                let mut manifest = ModelBundleManifest::read(&revision)?;
                manifest.verify_artifacts(&revision)?;
                manifest.archived = state.archived_revisions.contains(&manifest.revision_id);
                manifests.push(manifest);
            }
        }
        manifests.sort_by(|left, right| {
            left.display_name
                .cmp(&right.display_name)
                .then_with(|| left.revision_id.cmp(&right.revision_id))
        });
        Ok(manifests)
    }

    pub fn set_archived(&self, model_id: &str, revision_id: &str, archived: bool) -> Result<()> {
        let path = self.revision_path(model_id, revision_id)?;
        let manifest = ModelBundleManifest::read(&path)?;
        manifest.verify_artifacts(&path)?;
        let mut state = self.read_state()?;
        if archived {
            state.archived_revisions.insert(revision_id.to_owned());
            state.aliases.retain(|_, target| target != revision_id);
        } else {
            state.archived_revisions.remove(revision_id);
        }
        self.write_state(&state)
    }

    pub fn set_alias(&self, alias: &str, revision_id: &str) -> Result<()> {
        safe_component("model deployment alias", alias)?;
        let installed = self
            .list()?
            .into_iter()
            .find(|model| model.revision_id == revision_id)
            .ok_or_else(|| Error::invalid("alias target model revision is not installed"))?;
        if installed.archived {
            return Err(Error::invalid(
                "an archived model revision cannot receive an alias",
            ));
        }
        self.set_alias_to_known_revision(alias, revision_id)
    }

    /// Set an alias after a catalog containing external pretrained revisions has validated the
    /// target. Registry callers should normally use `set_alias`, which validates local bundles.
    pub fn set_alias_to_known_revision(&self, alias: &str, revision_id: &str) -> Result<()> {
        safe_component("model deployment alias", alias)?;
        required("model alias target revision", revision_id)?;
        let mut state = self.read_state()?;
        state
            .aliases
            .insert(alias.to_owned(), revision_id.to_owned());
        self.write_state(&state)
    }

    pub fn remove_alias(&self, alias: &str) -> Result<()> {
        safe_component("model deployment alias", alias)?;
        let mut state = self.read_state()?;
        state.aliases.remove(alias);
        self.write_state(&state)
    }

    pub fn resolve_alias(&self, alias_or_revision: &str) -> Result<String> {
        let state = self.read_state()?;
        Ok(state
            .aliases
            .get(alias_or_revision)
            .cloned()
            .unwrap_or_else(|| alias_or_revision.to_owned()))
    }

    pub fn aliases_for(&self, revision_id: &str) -> Result<Vec<String>> {
        let state = self.read_state()?;
        Ok(state
            .aliases
            .into_iter()
            .filter_map(|(alias, target)| (target == revision_id).then_some(alias))
            .collect())
    }

    fn read_state(&self) -> Result<ModelRegistryState> {
        let path = self.root.join("registry-state.json");
        if !path.exists() {
            return Ok(ModelRegistryState {
                schema_version: MODEL_REGISTRY_STATE_SCHEMA_VERSION,
                ..Default::default()
            });
        }
        let bytes = fs::read(&path).map_err(|error| {
            Error::backend(format!(
                "read model registry state {}: {error}",
                path.display()
            ))
        })?;
        let state: ModelRegistryState = serde_json::from_slice(&bytes).map_err(|error| {
            Error::invalid(format!(
                "parse model registry state {}: {error}",
                path.display()
            ))
        })?;
        if state.schema_version != MODEL_REGISTRY_STATE_SCHEMA_VERSION {
            return Err(Error::invalid("unsupported model registry state version"));
        }
        Ok(state)
    }

    fn write_state(&self, state: &ModelRegistryState) -> Result<()> {
        let path = self.root.join("registry-state.json");
        let temporary = self
            .root
            .join(format!(".registry-state-{}.tmp", std::process::id()));
        let bytes = serde_json::to_vec_pretty(state)
            .map_err(|error| Error::backend(format!("serialize model registry state: {error}")))?;
        fs::write(&temporary, bytes).map_err(|error| {
            Error::backend(format!(
                "write model registry state {}: {error}",
                temporary.display()
            ))
        })?;
        fs::rename(&temporary, &path).map_err(|error| {
            Error::backend(format!(
                "publish model registry state {}: {error}",
                path.display()
            ))
        })
    }
}

fn read_directories(root: &Path) -> Result<Vec<PathBuf>> {
    let entries = fs::read_dir(root).map_err(|error| {
        Error::backend(format!("read model registry {}: {error}", root.display()))
    })?;
    let mut directories = entries
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| entry.file_type().ok()?.is_dir().then(|| entry.path()))
        .collect::<Vec<_>>();
    directories.sort();
    Ok(directories)
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).map_err(|error| {
        Error::backend(format!("open model artifact {}: {error}", path.display()))
    })?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(|error| {
            Error::backend(format!("hash model artifact {}: {error}", path.display()))
        })?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
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

fn safe_component(what: &str, value: &str) -> Result<()> {
    let path = Path::new(value);
    if path.components().count() != 1
        || !matches!(path.components().next(), Some(Component::Normal(_)))
    {
        return Err(Error::invalid(format!(
            "{what} is not a safe path component"
        )));
    }
    Ok(())
}

fn safe_relative_path(path: &Path) -> Result<()> {
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(Error::invalid(
            "model artifact path is not a safe relative path",
        ));
    }
    Ok(())
}

fn unique<T: Eq + std::hash::Hash>(what: &str, values: &[T]) -> Result<()> {
    let mut unique = HashSet::with_capacity(values.len());
    if values.iter().all(|value| unique.insert(value)) {
        Ok(())
    } else {
        Err(Error::invalid(format!("{what} contain duplicates")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trained_schema_rejects_unnamed_annotation_types() {
        let mut schema = LabelSchema {
            schema_id: "cells".into(),
            revision_id: "r1".into(),
            name: "Cell types".into(),
            classes: vec![AnnotationClass {
                class_id: 1,
                name: "Macrophage".into(),
                color_rgba: [255, 0, 0, 255],
                parent_class_id: None,
                description: None,
                ontology_term: None,
                disabled: false,
            }],
        };
        assert!(schema.validate().is_ok());
        schema.classes[0].name.clear();
        assert!(schema.validate().is_err());
    }
}
