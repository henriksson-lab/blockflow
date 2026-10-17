//! The append-only edit log and the map state it produces.
//!
//! Revision `n` is the state after the first `n` records. Undo never removes a record: it
//! appends the inverse of the most recent edit that has not been undone yet, so the log is a
//! complete history and every revision can be reconstructed.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::map::{Move, SegmentMap};
use crate::{Error, Result};

/// What an edit record did.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EditOp {
    /// Two segments became one; `segments = [survivor, absorbed]`.
    Merge,
    /// Fragments were put into one segment; `segments = [target]`. A split is an assign of one
    /// side of a cut to a fresh id.
    Assign,
    /// The record `undoes` was reverted; `moves` are its inverse.
    Undo,
}

/// One line of `edits.jsonl`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EditRecord {
    /// The revision this record produces: its 1-based position in the log.
    pub revision: u64,
    pub op: EditOp,
    /// Segment ids involved (see [`EditOp`]).
    #[serde(default)]
    pub segments: Vec<u64>,
    /// For [`EditOp::Undo`]: the revision of the record it reverts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub undoes: Option<u64>,
    /// `[fragment, from, to]` triples, applied in order.
    pub moves: Vec<Move>,
    /// Milliseconds since the Unix epoch, when the writer knows it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_ms: Option<u64>,
    /// Who made the edit, when the writer knows it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
}

/// A [`SegmentMap`] together with the edit log that produced it.
#[derive(Clone, Debug, Default)]
pub struct SegmentMapState {
    map: SegmentMap,
    records: Vec<EditRecord>,
    undone: HashSet<u64>,
}

impl SegmentMapState {
    /// A state at revision 0 over `map`.
    pub fn new(map: SegmentMap) -> Self {
        Self {
            map,
            records: Vec::new(),
            undone: HashSet::new(),
        }
    }

    /// Rebuild a state from a snapshot taken at `snapshot_revision` and the complete log.
    /// Records up to the snapshot revision are kept as history; later ones are replayed.
    pub fn from_snapshot(
        map: SegmentMap,
        snapshot_revision: u64,
        records: Vec<EditRecord>,
    ) -> Result<Self> {
        if (records.len() as u64) < snapshot_revision {
            return Err(Error::Corrupt(format!(
                "snapshot is at revision {snapshot_revision} but the log has only {} records",
                records.len()
            )));
        }
        let mut state = Self::new(map);
        for (index, record) in records.into_iter().enumerate() {
            if record.revision != index as u64 + 1 {
                return Err(Error::Corrupt(format!(
                    "edit log record {} carries revision {}",
                    index + 1,
                    record.revision
                )));
            }
            if record.revision <= snapshot_revision {
                state.note(&record);
                state.records.push(record);
            } else {
                state.commit(record)?;
            }
        }
        Ok(state)
    }

    pub fn map(&self) -> &SegmentMap {
        &self.map
    }

    /// The number of applied records.
    pub fn revision(&self) -> u64 {
        self.records.len() as u64
    }

    pub fn records(&self) -> &[EditRecord] {
        &self.records
    }

    /// The revision [`Self::plan_undo`] would revert, if any.
    pub fn undo_target(&self) -> Option<u64> {
        self.records
            .iter()
            .rev()
            .find(|r| r.op != EditOp::Undo && !self.undone.contains(&r.revision))
            .map(|r| r.revision)
    }

    pub fn can_undo(&self) -> bool {
        self.undo_target().is_some()
    }

    /// The record merging segments `a` and `b` (small-to-large; see
    /// [`SegmentMap::plan_merge`]). Nothing is applied until [`Self::commit`].
    pub fn plan_merge(&self, a: u64, b: u64) -> Result<EditRecord> {
        let plan = self.map.plan_merge(a, b)?;
        Ok(self.record(
            EditOp::Merge,
            vec![plan.survivor, plan.absorbed],
            None,
            plan.moves,
        ))
    }

    /// The record putting `fragments` into `segment` (see [`SegmentMap::plan_assign`]).
    pub fn plan_assign(&self, fragments: &[u64], segment: u64) -> Result<EditRecord> {
        let moves = self.map.plan_assign(fragments, segment)?;
        if moves.is_empty() {
            return Err(Error::Invalid(format!(
                "every fragment is already in segment {segment}"
            )));
        }
        Ok(self.record(EditOp::Assign, vec![segment], None, moves))
    }

    /// The record reverting the most recent edit that is not yet undone. Undoing in this stack
    /// order means the state is exactly the one the reverted edit left, so its inverse is exact.
    pub fn plan_undo(&self) -> Result<EditRecord> {
        let target = self
            .undo_target()
            .ok_or_else(|| Error::Invalid("there is no edit to undo".into()))?;
        let record = &self.records[target as usize - 1];
        let moves = record.moves.iter().rev().map(|m| m.inverse()).collect();
        Ok(self.record(EditOp::Undo, record.segments.clone(), Some(target), moves))
    }

    /// Apply a planned (or replayed) record. It must carry the next revision, and its moves must
    /// start from the current segments.
    pub fn commit(&mut self, record: EditRecord) -> Result<()> {
        if record.revision != self.revision() + 1 {
            return Err(Error::Conflict {
                expected: record.revision - 1,
                current: self.revision(),
            });
        }
        if record.op == EditOp::Undo {
            let target = record
                .undoes
                .ok_or_else(|| Error::Corrupt("undo record names no revision".into()))?;
            if Some(target) != self.undo_target() {
                return Err(Error::Corrupt(format!(
                    "undo record reverts revision {target}, which is not the latest live edit"
                )));
            }
        }
        self.map.apply_moves(&record.moves)?;
        self.note(&record);
        self.records.push(record);
        Ok(())
    }

    /// Merge `a` and `b` in memory; returns the record.
    pub fn merge(&mut self, a: u64, b: u64) -> Result<&EditRecord> {
        let record = self.plan_merge(a, b)?;
        self.commit(record)?;
        Ok(self.records.last().expect("just pushed"))
    }

    /// Assign `fragments` to `segment` in memory; returns the record.
    pub fn assign(&mut self, fragments: &[u64], segment: u64) -> Result<&EditRecord> {
        let record = self.plan_assign(fragments, segment)?;
        self.commit(record)?;
        Ok(self.records.last().expect("just pushed"))
    }

    /// Undo the most recent live edit in memory; returns the undo record.
    pub fn undo(&mut self) -> Result<&EditRecord> {
        let record = self.plan_undo()?;
        self.commit(record)?;
        Ok(self.records.last().expect("just pushed"))
    }

    fn note(&mut self, record: &EditRecord) {
        if let Some(target) = record.undoes {
            self.undone.insert(target);
        }
    }

    fn record(
        &self,
        op: EditOp,
        segments: Vec<u64>,
        undoes: Option<u64>,
        moves: Vec<Move>,
    ) -> EditRecord {
        EditRecord {
            revision: self.revision() + 1,
            op,
            segments,
            undoes,
            moves,
            time_ms: None,
            author: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn start() -> SegmentMapState {
        SegmentMapState::new(
            SegmentMap::from_pairs([(1, 10), (2, 10), (3, 20), (4, 30), (5, 30), (6, 30)])
                .unwrap()
                .with_complete(true),
        )
    }

    #[test]
    fn undo_reverts_in_stack_order_and_appends() {
        let mut state = start();
        let initial = state.map().clone();
        state.merge(10, 20).unwrap();
        let after_first = state.map().clone();
        state.merge(10, 30).unwrap();
        assert_eq!(state.map().fragments_of(10), vec![1, 2, 3, 4, 5, 6]);
        state.undo().unwrap();
        assert_eq!(state.revision(), 3);
        assert_eq!(*state.map(), after_first);
        state.assign(&[6], 99).unwrap();
        state.undo().unwrap();
        assert_eq!(*state.map(), after_first);
        state.undo().unwrap();
        assert_eq!(*state.map(), initial);
        assert!(!state.can_undo());
        assert!(state.undo().is_err());
        assert_eq!(state.revision(), 6);
        let ops: Vec<_> = state.records().iter().map(|r| r.op).collect();
        use EditOp::*;
        assert_eq!(ops, [Merge, Merge, Undo, Assign, Undo, Undo]);
    }

    #[test]
    fn replay_from_any_snapshot_matches() {
        let mut state = start();
        let initial = state.map().clone();
        let mut snapshots = vec![(0, initial.clone())];
        state.merge(10, 20).unwrap();
        snapshots.push((1, state.map().clone()));
        state.assign(&[4, 5], 77).unwrap();
        snapshots.push((2, state.map().clone()));
        state.undo().unwrap();
        snapshots.push((3, state.map().clone()));
        state.merge(30, 10).unwrap();
        let records = state.records().to_vec();
        for (revision, map) in snapshots {
            let replayed = SegmentMapState::from_snapshot(map, revision, records.clone()).unwrap();
            assert_eq!(replayed.map(), state.map(), "from snapshot {revision}");
            assert_eq!(replayed.undo_target(), state.undo_target());
        }
    }

    #[test]
    fn commit_refuses_a_stale_record() {
        let mut state = start();
        let stale = state.plan_merge(10, 20).unwrap();
        state.merge(10, 30).unwrap();
        assert!(matches!(
            state.commit(stale),
            Err(Error::Conflict {
                expected: 0,
                current: 1
            })
        ));
    }
}
