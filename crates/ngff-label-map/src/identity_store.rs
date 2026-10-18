//! Atomic publication, validation, and legacy migration of identity graph metadata.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::{
    DomainId, Error, IdScope, IdentityDomain, IdentityGraph, IdentityGraphMetadata,
    RelationDescriptor, RelationKind, Result, UnmappedBehavior,
};

fn metadata_path(group: &Path) -> Result<PathBuf> {
    for name in ["zarr.json", ".zattrs"] {
        let path = group.join(name);
        if path.is_file() {
            return Ok(path);
        }
    }
    Err(Error::Invalid(format!(
        "{} is not a Zarr group with zarr.json or .zattrs",
        group.display()
    )))
}

fn read_document(group: &Path) -> Result<(PathBuf, Value)> {
    let path = metadata_path(group)?;
    let value = serde_json::from_slice(&fs::read(&path)?)
        .map_err(|error| Error::Corrupt(format!("{}: {error}", path.display())))?;
    Ok((path, value))
}

fn attributes<'a>(document: &'a Value, path: &Path) -> Result<&'a serde_json::Map<String, Value>> {
    if path.file_name().and_then(|name| name.to_str()) == Some(".zattrs") {
        document
            .as_object()
            .ok_or_else(|| Error::Corrupt(".zattrs is not an object".into()))
    } else {
        document
            .get("attributes")
            .and_then(Value::as_object)
            .ok_or_else(|| Error::Corrupt("zarr.json attributes are missing".into()))
    }
}

pub fn read_identity_graph(group: &Path) -> Result<Option<IdentityGraphMetadata>> {
    let (path, document) = read_document(group)?;
    attributes(&document, &path)?
        .get("identity_graph")
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .map_err(|error| Error::Corrupt(format!("identity_graph: {error}")))
}

pub fn write_identity_graph(group: &Path, metadata: &IdentityGraphMetadata) -> Result<()> {
    IdentityGraph::new(metadata.clone())?;
    let (path, mut document) = read_document(group)?;
    let target = if path.file_name().and_then(|name| name.to_str()) == Some(".zattrs") {
        document
            .as_object_mut()
            .ok_or_else(|| Error::Corrupt(".zattrs is not an object".into()))?
    } else {
        document
            .get_mut("attributes")
            .and_then(Value::as_object_mut)
            .ok_or_else(|| Error::Corrupt("zarr.json attributes are missing".into()))?
    };
    target.insert(
        "identity_graph".into(),
        serde_json::to_value(metadata)
            .map_err(|error| Error::Invalid(format!("identity graph cannot serialize: {error}")))?,
    );
    let bytes = serde_json::to_vec_pretty(&document)
        .map_err(|error| Error::Invalid(format!("metadata cannot serialize: {error}")))?;
    let temporary = path.with_extension("identity.tmp");
    {
        let mut file = fs::File::create(&temporary)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
    }
    fs::rename(temporary, path)?;
    Ok(())
}

pub fn validate_identity_group(group: &Path) -> Result<IdentityGraphMetadata> {
    let metadata = read_identity_graph(group)?
        .ok_or_else(|| Error::Invalid("the group has no identity_graph metadata".into()))?;
    let graph = IdentityGraph::new(metadata.clone())?;
    for relation in &metadata.relations {
        if relation.kind == RelationKind::Partition && relation.editable {
            let scope = metadata.domains[&relation.from].scope;
            if scope == IdScope::Timepoint {
                let storage = relation
                    .parameters
                    .get("storage")
                    .and_then(Value::as_str)
                    .unwrap_or("global");
                if !matches!(storage, "compound_key" | "per_timepoint") {
                    return Err(Error::Invalid(format!(
                        "editable relation {:?} has a timepoint-scoped source but no time-aware storage",
                        relation.name
                    )));
                }
            }
        }
    }
    graph.path(&metadata.stored_domain, &metadata.default_display_domain)?;
    Ok(metadata)
}

/// Synthesize a version-1 graph from the legacy `segment_map` declaration of one label group.
/// Tracking-aware writers should publish their full graph directly; this command intentionally
/// does not guess table relationships from column names.
pub fn migrate_legacy_identity_graph(group: &Path, layer: &str) -> Result<IdentityGraphMetadata> {
    if let Some(metadata) = read_identity_graph(group)? {
        IdentityGraph::new(metadata.clone())?;
        return Ok(metadata);
    }
    let (path, document) = read_document(group)?;
    let attrs = attributes(&document, &path)?;
    let has_partition = attrs.get("segment_map").is_some();
    let stored = DomainId::new(format!(
        "{layer}.{}",
        if has_partition {
            "fragments"
        } else {
            "detections"
        }
    ))?;
    let display = DomainId::new(format!("{layer}.detections"))?;
    let mut domains = BTreeMap::from([(
        stored.clone(),
        IdentityDomain {
            role: if has_partition {
                "fragment"
            } else {
                "detection"
            }
            .into(),
            scope: IdScope::Dataset,
        },
    )]);
    let relations = if has_partition {
        domains.insert(
            display.clone(),
            IdentityDomain {
                role: "detection".into(),
                scope: IdScope::Dataset,
            },
        );
        vec![RelationDescriptor {
            name: "segmentation".into(),
            kind: RelationKind::Partition,
            from: stored.clone(),
            to: display.clone(),
            editable: true,
            unmapped: UnmappedBehavior::Identity,
            parameters: serde_json::Map::from_iter([("path".into(), json!("segment-map"))]),
        }]
    } else {
        Vec::new()
    };
    let metadata = IdentityGraphMetadata {
        version: IdentityGraph::VERSION,
        stored_domain: stored,
        default_display_domain: display,
        domains,
        relations,
    };
    write_identity_graph(group, &metadata)?;
    Ok(metadata)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_preserves_the_zarr_document_and_is_idempotent() {
        let root = tempfile::tempdir().unwrap();
        fs::write(
            root.path().join("zarr.json"),
            r#"{"zarr_format":3,"node_type":"group","attributes":{"segment_map":{"version":1,"path":"segment-map"},"keep":7}}"#,
        )
        .unwrap();
        let first = migrate_legacy_identity_graph(root.path(), "cells").unwrap();
        let second = migrate_legacy_identity_graph(root.path(), "ignored").unwrap();
        assert_eq!(first, second);
        assert_eq!(validate_identity_group(root.path()).unwrap(), first);
        let value: Value =
            serde_json::from_slice(&fs::read(root.path().join("zarr.json")).unwrap()).unwrap();
        assert_eq!(value["attributes"]["keep"], 7);
    }
}
