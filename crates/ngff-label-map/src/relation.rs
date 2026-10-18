//! Scope aware relation storage for external producers that reuse label IDs per timepoint.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{Error, IdScope, Result, ScopedId};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RelationEntry {
    pub source: ScopedId,
    pub target: ScopedId,
}

/// A sorted compound-key relation. Dataset-scoped producers omit time; timepoint-scoped
/// producers must provide it. This is the canonical in-memory form for both one-map-per-timepoint
/// and `(t, source_id, target_id)` disk layouts.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScopedRelation {
    pub version: u32,
    pub revision: u64,
    pub source_scope: IdScope,
    pub target_scope: IdScope,
    entries: Vec<RelationEntry>,
}

impl ScopedRelation {
    pub const VERSION: u32 = 1;

    pub fn new(
        revision: u64,
        source_scope: IdScope,
        target_scope: IdScope,
        entries: impl IntoIterator<Item = RelationEntry>,
    ) -> Result<Self> {
        let mut entries = entries.into_iter().collect::<Vec<_>>();
        entries.sort_unstable_by_key(|entry| entry.source);
        let mut seen = BTreeSet::new();
        for entry in &entries {
            validate_scope(source_scope, entry.source, "source")?;
            validate_scope(target_scope, entry.target, "target")?;
            if entry.source.local_id == 0 || entry.target.local_id == 0 {
                return Err(Error::Invalid(
                    "identity relation entries cannot use background id 0".into(),
                ));
            }
            if !seen.insert(entry.source) {
                return Err(Error::Invalid(format!(
                    "identity relation has duplicate source {:?}",
                    entry.source
                )));
            }
        }
        Ok(Self {
            version: Self::VERSION,
            revision,
            source_scope,
            target_scope,
            entries,
        })
    }

    pub fn entries(&self) -> &[RelationEntry] {
        &self.entries
    }

    pub fn resolve(&self, source: ScopedId) -> Result<Option<ScopedId>> {
        validate_scope(self.source_scope, source, "source")?;
        Ok(self
            .entries
            .binary_search_by_key(&source, |entry| entry.source)
            .ok()
            .map(|index| self.entries[index].target))
    }

    pub fn members(&self, target: ScopedId) -> Result<Vec<ScopedId>> {
        validate_scope(self.target_scope, target, "target")?;
        Ok(self
            .entries
            .iter()
            .filter_map(|entry| (entry.target == target).then_some(entry.source))
            .collect())
    }

    /// Compact one-timepoint maps into the same relation while preserving the time coordinate.
    pub fn from_timepoint_maps(
        revision: u64,
        maps: BTreeMap<u32, BTreeMap<u64, u64>>,
        target_scope: IdScope,
    ) -> Result<Self> {
        let entries = maps.into_iter().flat_map(|(timepoint, map)| {
            map.into_iter().map(move |(source, target)| RelationEntry {
                source: ScopedId::at_time(timepoint, source),
                target: match target_scope {
                    IdScope::Dataset => ScopedId::dataset(target),
                    IdScope::Timepoint => ScopedId::at_time(timepoint, target),
                },
            })
        });
        Self::new(revision, IdScope::Timepoint, target_scope, entries)
    }

    /// A writable global `u64 -> u64` map cannot safely own a timepoint-scoped source.
    pub fn validate_global_partition_source(scope: IdScope) -> Result<()> {
        if scope == IdScope::Timepoint {
            return Err(Error::Invalid(
                "a timepoint-scoped source requires a compound-key or per-timepoint relation"
                    .into(),
            ));
        }
        Ok(())
    }
}

fn validate_scope(scope: IdScope, id: ScopedId, side: &str) -> Result<()> {
    match (scope, id.timepoint) {
        (IdScope::Dataset, None) | (IdScope::Timepoint, Some(_)) => Ok(()),
        (IdScope::Dataset, Some(_)) => Err(Error::Invalid(format!(
            "dataset-scoped relation {side} must not carry a timepoint"
        ))),
        (IdScope::Timepoint, None) => Err(Error::Invalid(format!(
            "timepoint-scoped relation {side} requires a timepoint"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_ids_are_distinct_at_each_timepoint() {
        let relation = ScopedRelation::from_timepoint_maps(
            4,
            BTreeMap::from([
                (0, BTreeMap::from([(7, 100)])),
                (1, BTreeMap::from([(7, 200)])),
            ]),
            IdScope::Dataset,
        )
        .unwrap();
        assert_eq!(
            relation.resolve(ScopedId::at_time(0, 7)).unwrap(),
            Some(ScopedId::dataset(100))
        );
        assert_eq!(
            relation.resolve(ScopedId::at_time(1, 7)).unwrap(),
            Some(ScopedId::dataset(200))
        );
        assert!(ScopedRelation::validate_global_partition_source(IdScope::Timepoint).is_err());
    }
}
