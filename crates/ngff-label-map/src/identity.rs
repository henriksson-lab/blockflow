//! Typed identity domains and relation paths over label IDs.
//!
//! A numeric label alone cannot say whether it names a fragment, detection, track, lineage, or
//! class. This module attaches IDs to declared domains and validates the graph of transforms
//! between them. The canonical stores remain specialised: [`crate::SegmentMap`] owns editable
//! fragment partitions, while a tracking graph owns temporal links. Renderers can nevertheless
//! use one validated relation path for both.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// Stable dataset-local name of an identity domain, for example `cells.detections`.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DomainId(String);

impl DomainId {
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        let valid = !value.is_empty()
            && value.len() <= 255
            && value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'));
        if !valid {
            return Err(Error::Invalid(format!(
                "identity domain {value:?} must contain 1..=255 ASCII letters, digits, '.', '_' or '-'"
            )));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for DomainId {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        out.write_str(&self.0)
    }
}

/// Whether numeric IDs are unique throughout a dataset or only inside one timepoint.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IdScope {
    Dataset,
    Timepoint,
}

/// One declared identity domain.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct IdentityDomain {
    /// Semantic role such as `fragment`, `detection`, `track`, `lineage`, `class`, or `region`.
    pub role: String,
    pub scope: IdScope,
}

/// A numeric ID with the scope required by its domain.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScopedId {
    pub local_id: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timepoint: Option<u32>,
}

impl ScopedId {
    pub fn dataset(local_id: u64) -> Self {
        Self {
            local_id,
            timepoint: None,
        }
    }

    pub fn at_time(timepoint: u32, local_id: u64) -> Self {
        Self {
            local_id,
            timepoint: Some(timepoint),
        }
    }
}

/// An unambiguous entity key used by selection, inspection, review, gates, and rendering.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EntityKey {
    pub domain: DomainId,
    #[serde(flatten)]
    pub id: ScopedId,
}

/// How a relation is produced and therefore which subsystem owns its edits.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationKind {
    Partition,
    Column,
    GraphComponent,
    Lookup,
    Identity,
}

/// Result for a source ID absent from a relation's explicit storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnmappedBehavior {
    Identity,
    Zero,
    Missing,
}

fn default_unmapped() -> UnmappedBehavior {
    UnmappedBehavior::Missing
}

/// Metadata for one directed identity relation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct RelationDescriptor {
    pub name: String,
    pub kind: RelationKind,
    pub from: DomainId,
    pub to: DomainId,
    #[serde(default)]
    pub editable: bool,
    #[serde(default = "default_unmapped")]
    pub unmapped: UnmappedBehavior,
    /// Relation-specific metadata: a partition path, table and columns, or graph references.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub parameters: serde_json::Map<String, serde_json::Value>,
}

/// Versioned metadata stored on a label or table group.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct IdentityGraphMetadata {
    pub version: u32,
    pub stored_domain: DomainId,
    pub default_display_domain: DomainId,
    pub domains: BTreeMap<DomainId, IdentityDomain>,
    pub relations: Vec<RelationDescriptor>,
}

/// The unique ordered relation path between two domains.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RelationPath {
    pub source: DomainId,
    pub target: DomainId,
    /// Indices into [`IdentityGraphMetadata::relations`].
    pub relations: Vec<usize>,
}

/// One canonical relation revision contributing to a composite generation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RelationGeneration {
    pub relation: String,
    pub revision: u64,
}

/// Exact cache identity of a resolved path. The ordered components avoid relying on a lossy hash.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct Generation(pub Vec<RelationGeneration>);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedBatch {
    pub source: DomainId,
    pub target: DomainId,
    pub generation: Generation,
    /// One output per input. `None` is a missing or background mapping.
    pub entities: Vec<Option<EntityKey>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemberSet {
    pub source: DomainId,
    pub target: EntityKey,
    pub generation: Generation,
    pub members: Vec<EntityKey>,
}

/// Storage-independent identity resolution. Implementors own canonical relation storage and
/// mutation; this trait gives renderers, selection, review, and gates one read contract.
pub trait IdentityResolver {
    fn graph(&self) -> &IdentityGraph;

    fn relation_revision(&self, relation: &RelationDescriptor) -> Result<u64>;

    fn apply_relation(
        &self,
        relation: &RelationDescriptor,
        source: ScopedId,
    ) -> Result<Option<ScopedId>>;

    /// Direct members of `target` in one relation's source domain.
    fn relation_members(
        &self,
        relation: &RelationDescriptor,
        target: ScopedId,
    ) -> Result<Vec<ScopedId>>;

    fn resolve(&self, source: EntityKey, target: &DomainId) -> Result<Option<EntityKey>> {
        let path = self.graph().path(&source.domain, target)?;
        self.graph().resolve(&path, source, |relation, id| {
            self.apply_relation(relation, id)
        })
    }

    fn resolve_many(
        &self,
        source: DomainId,
        ids: &[ScopedId],
        target: &DomainId,
    ) -> Result<ResolvedBatch> {
        let path = self.graph().path(&source, target)?;
        let generation = self
            .graph()
            .generation(&path, |relation| self.relation_revision(relation))?;
        let entities = ids
            .iter()
            .map(|id| {
                self.graph().resolve(
                    &path,
                    EntityKey {
                        domain: source.clone(),
                        id: *id,
                    },
                    |relation, id| self.apply_relation(relation, id),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(ResolvedBatch {
            source,
            target: target.clone(),
            generation,
            entities,
        })
    }

    fn members(&self, target: EntityKey, source: &DomainId) -> Result<MemberSet> {
        self.graph().validate_key(&target)?;
        let path = self.graph().path(source, &target.domain)?;
        let generation = self
            .graph()
            .generation(&path, |relation| self.relation_revision(relation))?;
        let mut ids = vec![target.id];
        for &index in path.relations.iter().rev() {
            let relation = &self.graph().metadata().relations[index];
            let mut expanded = Vec::new();
            for id in ids {
                expanded.extend(self.relation_members(relation, id)?);
            }
            expanded.sort_unstable();
            expanded.dedup();
            ids = expanded;
        }
        Ok(MemberSet {
            source: source.clone(),
            target,
            generation,
            members: ids
                .into_iter()
                .map(|id| EntityKey {
                    domain: source.clone(),
                    id,
                })
                .collect(),
        })
    }
}

/// Validated identity graph and path resolver.
#[derive(Clone, Debug)]
pub struct IdentityGraph {
    metadata: IdentityGraphMetadata,
    outgoing: BTreeMap<DomainId, Vec<usize>>,
}

impl IdentityGraph {
    pub const VERSION: u32 = 1;

    pub fn new(metadata: IdentityGraphMetadata) -> Result<Self> {
        if metadata.version != Self::VERSION {
            return Err(Error::Invalid(format!(
                "identity graph version {} is unsupported; expected {}",
                metadata.version,
                Self::VERSION
            )));
        }
        for domain in metadata.domains.keys() {
            DomainId::new(domain.as_str())?;
        }
        for required in [&metadata.stored_domain, &metadata.default_display_domain] {
            if !metadata.domains.contains_key(required) {
                return Err(Error::Invalid(format!(
                    "identity graph refers to undeclared domain {required}"
                )));
            }
        }
        let mut names = BTreeSet::new();
        let mut outgoing: BTreeMap<DomainId, Vec<usize>> = BTreeMap::new();
        for (index, relation) in metadata.relations.iter().enumerate() {
            if relation.name.is_empty() || !names.insert(relation.name.as_str()) {
                return Err(Error::Invalid(format!(
                    "identity relation name {:?} is empty or duplicated",
                    relation.name
                )));
            }
            for domain in [&relation.from, &relation.to] {
                if !metadata.domains.contains_key(domain) {
                    return Err(Error::Invalid(format!(
                        "identity relation {:?} refers to undeclared domain {domain}",
                        relation.name
                    )));
                }
            }
            if relation.from == relation.to && relation.kind != RelationKind::Identity {
                return Err(Error::Invalid(format!(
                    "identity relation {:?} maps a domain to itself but is not identity",
                    relation.name
                )));
            }
            if relation.unmapped == UnmappedBehavior::Identity
                && metadata.domains[&relation.from].scope != metadata.domains[&relation.to].scope
            {
                return Err(Error::Invalid(format!(
                    "identity relation {:?} cannot preserve unmapped IDs across different scopes",
                    relation.name
                )));
            }
            outgoing
                .entry(relation.from.clone())
                .or_default()
                .push(index);
        }
        let graph = Self { metadata, outgoing };
        graph.reject_cycles()?;
        // The default render path must be valid and unique at discovery time, not first paint.
        graph.path(
            &graph.metadata.stored_domain,
            &graph.metadata.default_display_domain,
        )?;
        Ok(graph)
    }

    pub fn metadata(&self) -> &IdentityGraphMetadata {
        &self.metadata
    }

    pub fn validate_key(&self, key: &EntityKey) -> Result<()> {
        let domain = self.metadata.domains.get(&key.domain).ok_or_else(|| {
            Error::Invalid(format!("entity key uses undeclared domain {}", key.domain))
        })?;
        if key.id.local_id == 0 {
            return Err(Error::Invalid("entity id 0 is background".into()));
        }
        match (domain.scope, key.id.timepoint) {
            (IdScope::Dataset, None) | (IdScope::Timepoint, Some(_)) => Ok(()),
            (IdScope::Dataset, Some(_)) => Err(Error::Invalid(format!(
                "dataset-scoped domain {} must not carry a timepoint",
                key.domain
            ))),
            (IdScope::Timepoint, None) => Err(Error::Invalid(format!(
                "timepoint-scoped domain {} requires a timepoint",
                key.domain
            ))),
        }
    }

    /// Return the sole path from `source` to `target`; refuse no path or ambiguous paths.
    pub fn path(&self, source: &DomainId, target: &DomainId) -> Result<RelationPath> {
        if !self.metadata.domains.contains_key(source) {
            return Err(Error::Invalid(format!(
                "unknown source identity domain {source}"
            )));
        }
        if !self.metadata.domains.contains_key(target) {
            return Err(Error::Invalid(format!(
                "unknown target identity domain {target}"
            )));
        }
        if source == target {
            return Ok(RelationPath {
                source: source.clone(),
                target: target.clone(),
                relations: Vec::new(),
            });
        }
        let mut found: Option<Vec<usize>> = None;
        let mut queue = VecDeque::from([(source.clone(), Vec::new())]);
        // The graph is acyclic. A path cannot contain more relations than the graph has.
        while let Some((domain, path)) = queue.pop_front() {
            for &relation_index in self.outgoing.get(&domain).into_iter().flatten() {
                let relation = &self.metadata.relations[relation_index];
                let mut next = path.clone();
                next.push(relation_index);
                if &relation.to == target {
                    if found.is_some() {
                        return Err(Error::Invalid(format!(
                            "identity path from {source} to {target} is ambiguous"
                        )));
                    }
                    found = Some(next);
                } else {
                    queue.push_back((relation.to.clone(), next));
                }
            }
        }
        found
            .map(|relations| RelationPath {
                source: source.clone(),
                target: target.clone(),
                relations,
            })
            .ok_or_else(|| Error::Invalid(format!("no identity path from {source} to {target}")))
    }

    /// Validate and apply a path. `apply` owns storage-specific lookup and returns the mapped ID.
    pub fn resolve(
        &self,
        path: &RelationPath,
        source: EntityKey,
        mut apply: impl FnMut(&RelationDescriptor, ScopedId) -> Result<Option<ScopedId>>,
    ) -> Result<Option<EntityKey>> {
        self.validate_key(&source)?;
        if source.domain != path.source {
            return Err(Error::Invalid(format!(
                "entity domain {} does not match path source {}",
                source.domain, path.source
            )));
        }
        let mut id = source.id;
        let mut domain = source.domain;
        for &index in &path.relations {
            let relation = self.metadata.relations.get(index).ok_or_else(|| {
                Error::Invalid("identity path contains an invalid relation index".into())
            })?;
            if relation.from != domain {
                return Err(Error::Invalid("identity path is not contiguous".into()));
            }
            id = match apply(relation, id)? {
                Some(mapped) if mapped.local_id != 0 => mapped,
                Some(_) | None if relation.unmapped == UnmappedBehavior::Zero => return Ok(None),
                None if relation.unmapped == UnmappedBehavior::Missing => return Ok(None),
                None => id,
                Some(mapped) => mapped,
            };
            domain = relation.to.clone();
            let key = EntityKey {
                domain: domain.clone(),
                id,
            };
            self.validate_key(&key)?;
        }
        if domain != path.target {
            return Err(Error::Invalid(
                "identity path did not reach its target".into(),
            ));
        }
        Ok(Some(EntityKey { domain, id }))
    }

    pub fn generation(
        &self,
        path: &RelationPath,
        mut revision: impl FnMut(&RelationDescriptor) -> Result<u64>,
    ) -> Result<Generation> {
        path.relations
            .iter()
            .map(|&index| {
                let relation = self.metadata.relations.get(index).ok_or_else(|| {
                    Error::Invalid("identity path contains an invalid relation index".into())
                })?;
                Ok(RelationGeneration {
                    relation: relation.name.clone(),
                    revision: revision(relation)?,
                })
            })
            .collect::<Result<Vec<_>>>()
            .map(Generation)
    }

    fn reject_cycles(&self) -> Result<()> {
        fn visit(
            graph: &IdentityGraph,
            domain: &DomainId,
            visiting: &mut BTreeSet<DomainId>,
            visited: &mut BTreeSet<DomainId>,
        ) -> Result<()> {
            if visited.contains(domain) {
                return Ok(());
            }
            if !visiting.insert(domain.clone()) {
                return Err(Error::Invalid(format!(
                    "identity relations contain a cycle through {domain}"
                )));
            }
            for &index in graph.outgoing.get(domain).into_iter().flatten() {
                visit(
                    graph,
                    &graph.metadata.relations[index].to,
                    visiting,
                    visited,
                )?;
            }
            visiting.remove(domain);
            visited.insert(domain.clone());
            Ok(())
        }

        let mut visiting = BTreeSet::new();
        let mut visited = BTreeSet::new();
        for domain in self.metadata.domains.keys() {
            visit(self, domain, &mut visiting, &mut visited)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn domain(name: &str) -> DomainId {
        DomainId::new(name).unwrap()
    }

    fn graph() -> IdentityGraph {
        let domains = [
            ("cells.fragments", "fragment"),
            ("cells.detections", "detection"),
            ("cells.tracks", "track"),
            ("cells.lineages", "lineage"),
        ]
        .into_iter()
        .map(|(name, role)| {
            (
                domain(name),
                IdentityDomain {
                    role: role.into(),
                    scope: IdScope::Dataset,
                },
            )
        })
        .collect();
        let relation = |name: &str, from: &str, to: &str, kind| RelationDescriptor {
            name: name.into(),
            kind,
            from: domain(from),
            to: domain(to),
            editable: kind == RelationKind::Partition,
            unmapped: UnmappedBehavior::Missing,
            parameters: serde_json::Map::new(),
        };
        IdentityGraph::new(IdentityGraphMetadata {
            version: 1,
            stored_domain: domain("cells.fragments"),
            default_display_domain: domain("cells.detections"),
            domains,
            relations: vec![
                relation(
                    "segmentation",
                    "cells.fragments",
                    "cells.detections",
                    RelationKind::Partition,
                ),
                relation(
                    "tracking",
                    "cells.detections",
                    "cells.tracks",
                    RelationKind::Column,
                ),
                relation(
                    "lineage",
                    "cells.tracks",
                    "cells.lineages",
                    RelationKind::GraphComponent,
                ),
            ],
        })
        .unwrap()
    }

    #[test]
    fn resolves_a_typed_relation_chain_and_builds_its_generation() {
        let graph = graph();
        let path = graph
            .path(&domain("cells.fragments"), &domain("cells.tracks"))
            .unwrap();
        let resolved = graph
            .resolve(
                &path,
                EntityKey {
                    domain: domain("cells.fragments"),
                    id: ScopedId::dataset(7),
                },
                |relation, id| {
                    Ok(Some(ScopedId::dataset(match relation.name.as_str() {
                        "segmentation" => id.local_id + 100,
                        "tracking" => 9,
                        _ => unreachable!(),
                    })))
                },
            )
            .unwrap()
            .unwrap();
        assert_eq!(resolved.domain, domain("cells.tracks"));
        assert_eq!(resolved.id, ScopedId::dataset(9));
        assert_eq!(
            graph
                .generation(&path, |relation| Ok(relation.name.len() as u64))
                .unwrap(),
            Generation(vec![
                RelationGeneration {
                    relation: "segmentation".into(),
                    revision: 12,
                },
                RelationGeneration {
                    relation: "tracking".into(),
                    revision: 8,
                },
            ])
        );
    }

    #[test]
    fn rejects_ambiguous_paths_cycles_and_wrong_scope() {
        let graph = graph();
        let mut ambiguous = graph.metadata().clone();
        ambiguous.relations.push(RelationDescriptor {
            name: "direct-track".into(),
            kind: RelationKind::Lookup,
            from: domain("cells.fragments"),
            to: domain("cells.tracks"),
            editable: false,
            unmapped: UnmappedBehavior::Missing,
            parameters: serde_json::Map::new(),
        });
        let ambiguous = IdentityGraph::new(ambiguous).unwrap();
        assert!(ambiguous
            .path(&domain("cells.fragments"), &domain("cells.tracks"))
            .unwrap_err()
            .to_string()
            .contains("ambiguous"));

        let mut cyclic = graph.metadata().clone();
        cyclic.relations.push(RelationDescriptor {
            name: "bad-cycle".into(),
            kind: RelationKind::Lookup,
            from: domain("cells.tracks"),
            to: domain("cells.fragments"),
            editable: false,
            unmapped: UnmappedBehavior::Missing,
            parameters: serde_json::Map::new(),
        });
        assert!(IdentityGraph::new(cyclic)
            .unwrap_err()
            .to_string()
            .contains("cycle"));

        assert!(graph
            .validate_key(&EntityKey {
                domain: domain("cells.tracks"),
                id: ScopedId::at_time(2, 9),
            })
            .unwrap_err()
            .to_string()
            .contains("must not carry"));
    }
}
