//! Durable identities and revisioned edits for temporal tracking graphs.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{Error, Result, ScopedId};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DurableIdAllocator {
    next: u64,
}

impl DurableIdAllocator {
    pub fn after_existing(ids: impl IntoIterator<Item = u64>) -> Result<Self> {
        let maximum = ids.into_iter().max().unwrap_or(0);
        Ok(Self {
            next: maximum
                .checked_add(1)
                .ok_or_else(|| Error::Invalid("durable identity space is exhausted".into()))?,
        })
    }

    pub fn allocate(&mut self) -> Result<u64> {
        let id = self.next;
        if id == 0 {
            return Err(Error::Invalid("durable identity space is exhausted".into()));
        }
        self.next = self
            .next
            .checked_add(1)
            .ok_or_else(|| Error::Invalid("durable identity space is exhausted".into()))?;
        Ok(id)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "op")]
pub enum TrackingEditOperation {
    Link {
        from: ScopedId,
        to: ScopedId,
        kind: u8,
    },
    Unlink {
        from: ScopedId,
        to: ScopedId,
    },
    AssignTrack {
        detection: ScopedId,
        track_id: u64,
    },
    AssignLineage {
        track_id: u64,
        lineage_id: u64,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrackingEdit {
    pub revision: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transaction_id: Option<String>,
    pub operation: TrackingEditOperation,
}

/// Canonical editable tracking identity state. IDs are allocated once and assignments are
/// explicit, so relinking cannot silently recolor a track by changing its first detection.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrackingIdentityState {
    pub revision: u64,
    #[serde(with = "scoped_track_map")]
    pub track_of_detection: BTreeMap<ScopedId, u64>,
    pub lineage_of_track: BTreeMap<u64, u64>,
    #[serde(with = "link_map")]
    pub links: BTreeMap<(ScopedId, ScopedId), u8>,
    pub track_ids: DurableIdAllocator,
    pub lineage_ids: DurableIdAllocator,
}

mod scoped_track_map {
    use super::*;
    use serde::{Deserializer, Serializer};

    #[derive(Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct Entry {
        detection: ScopedId,
        track_id: u64,
    }

    pub fn serialize<S>(
        value: &BTreeMap<ScopedId, u64>,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        value
            .iter()
            .map(|(detection, track_id)| Entry {
                detection: *detection,
                track_id: *track_id,
            })
            .collect::<Vec<_>>()
            .serialize(serializer)
    }

    pub fn deserialize<'de, D>(
        deserializer: D,
    ) -> std::result::Result<BTreeMap<ScopedId, u64>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let entries = Vec::<Entry>::deserialize(deserializer)?;
        let mut value = BTreeMap::new();
        for entry in entries {
            if value.insert(entry.detection, entry.track_id).is_some() {
                return Err(serde::de::Error::custom("duplicate detection identity"));
            }
        }
        Ok(value)
    }
}

mod link_map {
    use super::*;
    use serde::{Deserializer, Serializer};

    #[derive(Serialize, Deserialize)]
    struct Entry {
        from: ScopedId,
        to: ScopedId,
        kind: u8,
    }

    pub fn serialize<S>(
        value: &BTreeMap<(ScopedId, ScopedId), u8>,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        value
            .iter()
            .map(|((from, to), kind)| Entry {
                from: *from,
                to: *to,
                kind: *kind,
            })
            .collect::<Vec<_>>()
            .serialize(serializer)
    }

    pub fn deserialize<'de, D>(
        deserializer: D,
    ) -> std::result::Result<BTreeMap<(ScopedId, ScopedId), u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let entries = Vec::<Entry>::deserialize(deserializer)?;
        let mut value = BTreeMap::new();
        for entry in entries {
            if value.insert((entry.from, entry.to), entry.kind).is_some() {
                return Err(serde::de::Error::custom("duplicate tracking link"));
            }
        }
        Ok(value)
    }
}

impl TrackingIdentityState {
    pub fn new(
        track_of_detection: BTreeMap<ScopedId, u64>,
        lineage_of_track: BTreeMap<u64, u64>,
        links: BTreeMap<(ScopedId, ScopedId), u8>,
    ) -> Result<Self> {
        validate_links(&links)?;
        if track_of_detection.values().any(|id| *id == 0)
            || lineage_of_track
                .iter()
                .any(|(track, lineage)| *track == 0 || *lineage == 0)
        {
            return Err(Error::Invalid("track and lineage id 0 are reserved".into()));
        }
        let track_ids = DurableIdAllocator::after_existing(
            track_of_detection
                .values()
                .copied()
                .chain(lineage_of_track.keys().copied()),
        )?;
        let lineage_ids = DurableIdAllocator::after_existing(lineage_of_track.values().copied())?;
        Ok(Self {
            revision: 0,
            track_of_detection,
            lineage_of_track,
            links,
            track_ids,
            lineage_ids,
        })
    }

    pub fn apply(
        &mut self,
        expected_revision: u64,
        edit: TrackingEditOperation,
    ) -> Result<TrackingEdit> {
        if expected_revision != self.revision {
            return Err(Error::Conflict {
                expected: expected_revision,
                current: self.revision,
            });
        }
        match &edit {
            TrackingEditOperation::Link { from, to, kind } => {
                validate_link(*from, *to)?;
                self.links.insert((*from, *to), *kind);
                self.rebuild_stable_tracks_and_lineages()?;
            }
            TrackingEditOperation::Unlink { from, to } => {
                if self.links.remove(&(*from, *to)).is_none() {
                    return Err(Error::Invalid("tracking link does not exist".into()));
                }
                self.rebuild_stable_tracks_and_lineages()?;
            }
            TrackingEditOperation::AssignTrack {
                detection,
                track_id,
            } => {
                if *track_id == 0 {
                    return Err(Error::Invalid("track id 0 is reserved".into()));
                }
                self.track_of_detection.insert(*detection, *track_id);
            }
            TrackingEditOperation::AssignLineage {
                track_id,
                lineage_id,
            } => {
                if *track_id == 0 || *lineage_id == 0 {
                    return Err(Error::Invalid("track and lineage id 0 are reserved".into()));
                }
                self.lineage_of_track.insert(*track_id, *lineage_id);
            }
        }
        self.revision += 1;
        Ok(TrackingEdit {
            revision: self.revision,
            transaction_id: None,
            operation: edit,
        })
    }

    pub fn allocate_track(&mut self) -> Result<u64> {
        self.track_ids.allocate()
    }

    pub fn allocate_lineage(&mut self) -> Result<u64> {
        self.lineage_ids.allocate()
    }

    /// Rebuild maximal unbranched tracks after a link edit. Existing IDs are assigned by maximum
    /// detection overlap, one old ID to one new path; only unmatched paths allocate a new ID.
    /// Lineages use the same rule over their detections. This makes the preservation decision
    /// deterministic and records its result explicitly in the state saved with the edit.
    fn rebuild_stable_tracks_and_lineages(&mut self) -> Result<()> {
        let old_tracks = self.track_of_detection.clone();
        let old_lineages = self.lineage_of_track.clone();
        let nodes = old_tracks.keys().copied().collect::<BTreeSet<_>>();
        let mut incoming: BTreeMap<ScopedId, Vec<ScopedId>> = BTreeMap::new();
        let mut outgoing: BTreeMap<ScopedId, Vec<ScopedId>> = BTreeMap::new();
        for &(from, to) in self.links.keys() {
            if !nodes.contains(&from) || !nodes.contains(&to) {
                return Err(Error::Invalid(
                    "tracking link names an unknown detection".into(),
                ));
            }
            outgoing.entry(from).or_default().push(to);
            incoming.entry(to).or_default().push(from);
        }
        let is_start = |node: ScopedId| {
            let parents = incoming.get(&node).map_or(&[][..], Vec::as_slice);
            if parents.len() != 1 {
                return true;
            }
            outgoing.get(&parents[0]).map_or(0, Vec::len) != 1
        };
        let mut visited = BTreeSet::new();
        let mut paths = Vec::<Vec<ScopedId>>::new();
        for node in nodes.iter().copied().filter(|node| is_start(*node)) {
            if visited.contains(&node) {
                continue;
            }
            let mut path = vec![node];
            visited.insert(node);
            let mut cursor = node;
            loop {
                let children = outgoing.get(&cursor).map_or(&[][..], Vec::as_slice);
                if children.len() != 1 {
                    break;
                }
                let next = children[0];
                if incoming.get(&next).map_or(0, Vec::len) != 1 || !visited.insert(next) {
                    break;
                }
                path.push(next);
                cursor = next;
            }
            paths.push(path);
        }
        for node in nodes.iter().copied() {
            if visited.insert(node) {
                paths.push(vec![node]);
            }
        }
        paths.sort_by_key(|path| path[0]);
        let mut candidates = Vec::new();
        for (path_index, path) in paths.iter().enumerate() {
            let mut counts = BTreeMap::<u64, usize>::new();
            for detection in path {
                *counts.entry(old_tracks[detection]).or_default() += 1;
            }
            candidates.extend(
                counts
                    .into_iter()
                    .map(|(track, count)| (count, track, path_index)),
            );
        }
        candidates.sort_by(|a, b| b.cmp(a));
        let mut used_tracks = BTreeSet::new();
        let mut assigned = vec![None; paths.len()];
        for (_, track, path_index) in candidates {
            if assigned[path_index].is_none() && used_tracks.insert(track) {
                assigned[path_index] = Some(track);
            }
        }
        for slot in &mut assigned {
            if slot.is_none() {
                *slot = Some(self.track_ids.allocate()?);
            }
        }
        self.track_of_detection.clear();
        for (path, track) in paths.iter().zip(assigned.iter().copied()) {
            let track = track.expect("every path was assigned");
            self.track_of_detection
                .extend(path.iter().copied().map(|detection| (detection, track)));
        }

        let mut track_edges = BTreeMap::<u64, BTreeSet<u64>>::new();
        for &(from, to) in self.links.keys() {
            let a = self.track_of_detection[&from];
            let b = self.track_of_detection[&to];
            if a != b {
                track_edges.entry(a).or_default().insert(b);
                track_edges.entry(b).or_default().insert(a);
            }
        }
        let tracks = self
            .track_of_detection
            .values()
            .copied()
            .collect::<BTreeSet<_>>();
        let mut components = Vec::<BTreeSet<u64>>::new();
        let mut seen = BTreeSet::new();
        for seed in tracks {
            if !seen.insert(seed) {
                continue;
            }
            let mut component = BTreeSet::from([seed]);
            let mut pending = vec![seed];
            while let Some(track) = pending.pop() {
                for &other in track_edges.get(&track).into_iter().flatten() {
                    if seen.insert(other) {
                        component.insert(other);
                        pending.push(other);
                    }
                }
            }
            components.push(component);
        }
        let mut lineage_candidates = Vec::new();
        for (component_index, component) in components.iter().enumerate() {
            let mut counts = BTreeMap::<u64, usize>::new();
            for (detection, &new_track) in &self.track_of_detection {
                if component.contains(&new_track) {
                    let old_track = old_tracks[detection];
                    if let Some(&lineage) = old_lineages.get(&old_track) {
                        *counts.entry(lineage).or_default() += 1;
                    }
                }
            }
            lineage_candidates.extend(
                counts
                    .into_iter()
                    .map(|(lineage, count)| (count, lineage, component_index)),
            );
        }
        lineage_candidates.sort_by(|a, b| b.cmp(a));
        let mut used_lineages = BTreeSet::new();
        let mut lineage_assignment = vec![None; components.len()];
        for (_, lineage, component_index) in lineage_candidates {
            if lineage_assignment[component_index].is_none() && used_lineages.insert(lineage) {
                lineage_assignment[component_index] = Some(lineage);
            }
        }
        self.lineage_of_track.clear();
        for (component, lineage) in components.into_iter().zip(lineage_assignment) {
            let lineage = match lineage {
                Some(lineage) => lineage,
                None => self.lineage_ids.allocate()?,
            };
            self.lineage_of_track
                .extend(component.into_iter().map(|track| (track, lineage)));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TrackingEditLog(pub Vec<TrackingEdit>);

impl TrackingEditLog {
    pub fn replay(&self, state: &mut TrackingIdentityState) -> Result<()> {
        let mut revisions = BTreeSet::new();
        for record in &self.0 {
            if !revisions.insert(record.revision) || record.revision != state.revision + 1 {
                return Err(Error::Corrupt(
                    "tracking edit revisions are not contiguous".into(),
                ));
            }
            state.apply(state.revision, record.operation.clone())?;
        }
        Ok(())
    }
}

fn validate_links(links: &BTreeMap<(ScopedId, ScopedId), u8>) -> Result<()> {
    for &(from, to) in links.keys() {
        validate_link(from, to)?;
    }
    Ok(())
}

fn validate_link(from: ScopedId, to: ScopedId) -> Result<()> {
    if from.local_id == 0 || to.local_id == 0 {
        return Err(Error::Invalid(
            "tracking links cannot use background id 0".into(),
        ));
    }
    match (from.timepoint, to.timepoint) {
        (Some(a), Some(b)) if a < b => Ok(()),
        (Some(_), Some(_)) => Err(Error::Invalid(
            "tracking links must move forward in time".into(),
        )),
        _ => Err(Error::Invalid(
            "tracking link detections require timepoints".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durable_ids_and_edits_are_revisioned() {
        let a = ScopedId::at_time(0, 7);
        let b = ScopedId::at_time(1, 7);
        let mut state = TrackingIdentityState::new(
            BTreeMap::from([(a, 12), (b, 12)]),
            BTreeMap::from([(12, 3)]),
            BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(state.allocate_track().unwrap(), 13);
        assert_eq!(state.allocate_lineage().unwrap(), 4);
        let edit = state
            .apply(
                0,
                TrackingEditOperation::Link {
                    from: a,
                    to: b,
                    kind: 0,
                },
            )
            .unwrap();
        assert_eq!(edit.revision, 1);
        let json = serde_json::to_vec(&state).unwrap();
        assert_eq!(
            serde_json::from_slice::<TrackingIdentityState>(&json).unwrap(),
            state
        );
        assert!(matches!(
            state.apply(0, edit.operation),
            Err(Error::Conflict { .. })
        ));
    }

    #[test]
    fn unlink_keeps_the_old_id_for_the_strongest_path_and_allocates_the_other() {
        let a = ScopedId::at_time(0, 1);
        let b = ScopedId::at_time(1, 2);
        let c = ScopedId::at_time(2, 3);
        let mut state = TrackingIdentityState::new(
            BTreeMap::from([(a, 10), (b, 10), (c, 10)]),
            BTreeMap::from([(10, 5)]),
            BTreeMap::from([((a, b), 0), ((b, c), 0)]),
        )
        .unwrap();
        state
            .apply(0, TrackingEditOperation::Unlink { from: a, to: b })
            .unwrap();
        assert_eq!(state.track_of_detection[&b], 10);
        assert_eq!(state.track_of_detection[&c], 10);
        assert_ne!(state.track_of_detection[&a], 10);
        assert_eq!(
            state
                .lineage_of_track
                .values()
                .copied()
                .collect::<BTreeSet<_>>()
                .len(),
            2
        );
    }
}
