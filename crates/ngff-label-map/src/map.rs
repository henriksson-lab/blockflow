//! [`SegmentMap`]: fragment id -> segment id, with a lazily built reverse index.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::{Error, IdTransform, Result};

/// Internal marker for "no explicit entry" in the dense table. It is also the one id a map can
/// never hold, as fragment or as segment.
pub(crate) const UNMAPPED: u64 = u64::MAX;

/// A dense table is chosen when it is at most this many times longer than the number of
/// entries it holds (plus [`DENSE_SLACK`]), so a compact id range costs 8 bytes per fragment and
/// a scattered one falls back to a hash map.
const DENSE_FACTOR: u64 = 2;
const DENSE_SLACK: u64 = 1024;

/// How a map stores its entries. Both layouts answer every query identically; the choice only
/// affects memory and lookup speed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// Dense when the fragment ids are compact, sparse otherwise.
    Auto,
    /// One slot per id between the smallest and the largest fragment.
    Dense,
    /// A hash map holding only the explicit entries.
    Sparse,
}

/// One fragment changing segment: the unit an edit is made of.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(from = "(u64, u64, u64)", into = "(u64, u64, u64)")]
pub struct Move {
    pub fragment: u64,
    pub from: u64,
    pub to: u64,
}

impl From<(u64, u64, u64)> for Move {
    fn from((fragment, from, to): (u64, u64, u64)) -> Self {
        Self { fragment, from, to }
    }
}

impl From<Move> for (u64, u64, u64) {
    fn from(value: Move) -> Self {
        (value.fragment, value.from, value.to)
    }
}

impl Move {
    /// The move that undoes this one.
    pub fn inverse(self) -> Self {
        Self {
            fragment: self.fragment,
            from: self.to,
            to: self.from,
        }
    }
}

/// The result of planning a merge: which segment survives and which fragments move into it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MergePlan {
    /// The segment id that remains: the larger of the two (by fragment count; `a` on a tie).
    pub survivor: u64,
    /// The segment id that disappears.
    pub absorbed: u64,
    pub moves: Vec<Move>,
}

/// Fragment id -> segment id.
///
/// * Background: fragment `0` always maps to `0`.
/// * Identity default: a fragment without an explicit entry maps to itself, so an empty map is
///   the identity and a map need only list the fragments a multicut or an edit has grouped.
/// * `u64::MAX` is reserved and can be neither fragment nor segment.
///
/// Entries live in a dense table indexed by `fragment - base` when the explicit fragment ids
/// are compact, with a hash map for anything outside that range. The reverse index (segment ->
/// fragments) is only needed for edits and is built on first use, so a read-only renderer never
/// pays for it.
#[derive(Debug, Default)]
pub struct SegmentMap {
    base: u64,
    dense: Vec<u64>,
    dense_explicit: usize,
    sparse: HashMap<u64, u64>,
    reverse: OnceLock<HashMap<u64, Vec<u64>>>,
    complete: bool,
}

impl Clone for SegmentMap {
    fn clone(&self) -> Self {
        let reverse = OnceLock::new();
        if let Some(built) = self.reverse.get() {
            let _ = reverse.set(built.clone());
        }
        Self {
            base: self.base,
            dense: self.dense.clone(),
            dense_explicit: self.dense_explicit,
            complete: self.complete,
            sparse: self.sparse.clone(),
            reverse,
        }
    }
}

impl PartialEq for SegmentMap {
    /// Two maps are equal when they map every id alike, whatever their layout. In a map that is
    /// not complete an explicit `f -> f` entry equals no entry; in a complete one it does not
    /// (it says `f` is a fragment).
    fn eq(&self, other: &Self) -> bool {
        let covers = |a: &Self, b: &Self| {
            a.entries().all(|(fragment, segment)| {
                if a.complete {
                    b.explicit(fragment) == Some(segment)
                } else {
                    b.get(fragment) == segment
                }
            })
        };
        self.complete == other.complete && covers(self, other) && covers(other, self)
    }
}

impl Eq for SegmentMap {}

impl SegmentMap {
    /// The identity map.
    pub fn new() -> Self {
        Self::default()
    }

    /// A map from explicit `(fragment, segment)` pairs, laid out by [`Layout::Auto`].
    pub fn from_pairs(pairs: impl IntoIterator<Item = (u64, u64)>) -> Result<Self> {
        Self::from_pairs_with(pairs, Layout::Auto)
    }

    /// A map from explicit pairs in the requested layout. A later pair for the same fragment
    /// replaces an earlier one.
    pub fn from_pairs_with(
        pairs: impl IntoIterator<Item = (u64, u64)>,
        layout: Layout,
    ) -> Result<Self> {
        let pairs: Vec<(u64, u64)> = pairs.into_iter().collect();
        for &(fragment, segment) in &pairs {
            check_entry(fragment, segment)?;
        }
        let mut map = Self::new();
        let dense = match layout {
            Layout::Dense => true,
            Layout::Sparse => false,
            Layout::Auto => {
                let (min, max) = pairs
                    .iter()
                    .fold((u64::MAX, 0), |(lo, hi), &(f, _)| (lo.min(f), hi.max(f)));
                !pairs.is_empty()
                    && max - min < (pairs.len() as u64).saturating_mul(DENSE_FACTOR) + DENSE_SLACK
            }
        };
        if dense && !pairs.is_empty() {
            let min = pairs.iter().map(|p| p.0).min().unwrap_or(1);
            let max = pairs.iter().map(|p| p.0).max().unwrap_or(1);
            let span = usize::try_from(max - min + 1)
                .map_err(|_| Error::Invalid("fragment id range too wide for a dense map".into()))?;
            map.base = min;
            map.dense = vec![UNMAPPED; span];
        }
        for (fragment, segment) in pairs {
            map.store(fragment, segment, dense);
        }
        Ok(map)
    }

    /// A dense map where fragment `base + i` maps to `segments[i]`. A slot holding `u64::MAX`
    /// has no explicit entry.
    pub fn from_dense(base: u64, segments: Vec<u64>) -> Result<Self> {
        if base == 0 && segments.first().is_some_and(|s| *s != 0 && *s != UNMAPPED) {
            return Err(Error::Invalid(
                "fragment 0 is background and must map to 0".into(),
            ));
        }
        let mut dense_explicit = 0;
        for (index, &segment) in segments.iter().enumerate() {
            if segment == UNMAPPED {
                continue;
            }
            let fragment = base
                .checked_add(index as u64)
                .filter(|f| *f != UNMAPPED)
                .ok_or_else(|| Error::Invalid("dense map overflows the id range".into()))?;
            if fragment != 0 {
                dense_explicit += 1;
            }
        }
        let mut segments = segments;
        if base == 0 {
            if let Some(first) = segments.first_mut() {
                *first = UNMAPPED;
            }
        }
        Ok(Self {
            base,
            dense: segments,
            dense_explicit,
            sparse: HashMap::new(),
            reverse: OnceLock::new(),
            complete: false,
        })
    }

    /// Reassemble a map from a dense table and sparse entries outside it (the snapshot
    /// layout), validating every entry.
    pub(crate) fn from_parts(base: u64, dense: Vec<u64>, sparse: Vec<(u64, u64)>) -> Result<Self> {
        let mut map = if dense.is_empty() {
            Self::new()
        } else {
            Self::from_dense(base, dense)?
        };
        for (fragment, segment) in map.entries() {
            check_entry(fragment, segment)?;
        }
        for (fragment, segment) in sparse {
            check_entry(fragment, segment)?;
            let inside = fragment
                .checked_sub(map.base)
                .is_some_and(|offset| offset < map.dense.len() as u64);
            if inside {
                return Err(Error::Corrupt(format!(
                    "sparse entry for fragment {fragment} lies inside the dense table"
                )));
            }
            if map.sparse.insert(fragment, segment).is_some() {
                return Err(Error::Corrupt(format!(
                    "fragment {fragment} is listed twice"
                )));
            }
        }
        Ok(map)
    }

    /// The explicit entries held outside the dense table.
    pub(crate) fn sparse_entries(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        self.sparse.iter().map(|(f, s)| (*f, *s))
    }

    /// The segment of `fragment`.
    #[inline]
    pub fn get(&self, fragment: u64) -> u64 {
        self.explicit(fragment).unwrap_or(fragment)
    }

    /// The explicit entry of `fragment`, if it has one.
    #[inline]
    pub fn explicit(&self, fragment: u64) -> Option<u64> {
        if fragment == 0 {
            return None;
        }
        if let Some(offset) = fragment.checked_sub(self.base) {
            if offset < self.dense.len() as u64 {
                let segment = self.dense[offset as usize];
                return (segment != UNMAPPED).then_some(segment);
            }
        }
        self.sparse.get(&fragment).copied()
    }

    /// Number of explicit entries.
    pub fn len(&self) -> usize {
        self.dense_explicit + self.sparse.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether the map holds a dense table (possibly alongside sparse entries).
    pub fn is_dense(&self) -> bool {
        !self.dense.is_empty()
    }

    /// The dense table, as `(base, segments)`; `u64::MAX` marks a slot without an entry.
    pub fn dense_table(&self) -> Option<(u64, &[u64])> {
        self.is_dense()
            .then_some((self.base, self.dense.as_slice()))
    }

    /// Every explicit `(fragment, segment)` entry: dense entries in fragment order, then sparse
    /// entries in arbitrary order.
    pub fn entries(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        let base = self.base;
        self.dense
            .iter()
            .enumerate()
            .filter(|(_, s)| **s != UNMAPPED)
            .map(move |(i, s)| (base + i as u64, *s))
            .chain(self.sparse.iter().map(|(f, s)| (*f, *s)))
    }

    /// The segments of `fragments`, in order.
    pub fn segments_of(&self, fragments: &[u64]) -> Vec<u64> {
        fragments.iter().map(|f| self.get(*f)).collect()
    }

    /// Every fragment of `segment`, sorted: the fragments explicitly mapped to it, plus the id
    /// itself when it has no explicit entry and the map is not [complete](Self::with_complete)
    /// (the identity default). Segment 0 has none (background is not editable).
    pub fn fragments_of(&self, segment: u64) -> Vec<u64> {
        if segment == 0 || segment == UNMAPPED {
            return Vec::new();
        }
        let mut out = self.reverse().get(&segment).cloned().unwrap_or_default();
        if self.implicit_member(segment) {
            out.push(segment);
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Number of fragments of `segment` (see [`Self::fragments_of`]).
    pub fn segment_size(&self, segment: u64) -> usize {
        if segment == 0 || segment == UNMAPPED {
            return 0;
        }
        self.reverse().get(&segment).map_or(0, Vec::len)
            + usize::from(self.implicit_member(segment))
    }

    /// Whether `segment`'s own id counts as one of its fragments without an explicit entry.
    fn implicit_member(&self, segment: u64) -> bool {
        !self.complete && self.explicit(segment).is_none()
    }

    /// Declare that every fragment of the label image has an explicit entry (what a pipeline
    /// writing a full multicut grouping produces). Then an id without an entry is not a
    /// fragment, so a segment id that is not also a fragment id never counts itself as one.
    /// Without it, an unmapped id is assumed to be its own fragment, which is right for a map
    /// started empty over a plain instance label, and for segment ids chosen as one of their
    /// fragments (union-find roots).
    pub fn with_complete(mut self, complete: bool) -> Self {
        self.complete = complete;
        self
    }

    pub fn is_complete(&self) -> bool {
        self.complete
    }

    fn reverse(&self) -> &HashMap<u64, Vec<u64>> {
        self.reverse.get_or_init(|| {
            let mut reverse: HashMap<u64, Vec<u64>> = HashMap::new();
            for (fragment, segment) in self.entries() {
                if segment != 0 {
                    reverse.entry(segment).or_default().push(fragment);
                }
            }
            reverse
        })
    }

    /// Plan merging segments `a` and `b` into one. Small-to-large: the segment with fewer
    /// fragments is relabelled to the other's id, so a merge costs the smaller side only.
    pub fn plan_merge(&self, a: u64, b: u64) -> Result<MergePlan> {
        for id in [a, b] {
            if id == 0 {
                return Err(Error::Invalid("segment 0 is background".into()));
            }
            if id == UNMAPPED {
                return Err(Error::Invalid("u64::MAX is not a valid segment id".into()));
            }
        }
        if a == b {
            return Err(Error::Invalid(format!(
                "segment {a} cannot merge with itself"
            )));
        }
        let fragments_a = self.fragments_of(a);
        let fragments_b = self.fragments_of(b);
        for (id, fragments) in [(a, &fragments_a), (b, &fragments_b)] {
            if fragments.is_empty() {
                return Err(Error::Invalid(format!("segment {id} has no fragments")));
            }
        }
        let (survivor, absorbed, moving) = if fragments_b.len() > fragments_a.len() {
            (b, a, fragments_a)
        } else {
            (a, b, fragments_b)
        };
        let moves = moving
            .into_iter()
            .map(|fragment| Move {
                fragment,
                from: absorbed,
                to: survivor,
            })
            .collect();
        Ok(MergePlan {
            survivor,
            absorbed,
            moves,
        })
    }

    /// Plan putting `fragments` into `segment`, which may be a new id. This is the primitive a
    /// split uses: a min cut picks the fragments of one side and assigns them a fresh id.
    /// Segment 0 removes fragments to background. Fragments already in `segment` and repeated
    /// fragments produce no move.
    pub fn plan_assign(&self, fragments: &[u64], segment: u64) -> Result<Vec<Move>> {
        if segment == UNMAPPED {
            return Err(Error::Invalid("u64::MAX is not a valid segment id".into()));
        }
        let mut seen = HashSet::new();
        let mut moves = Vec::new();
        for &fragment in fragments {
            if fragment == 0 {
                return Err(Error::Invalid("fragment 0 is background".into()));
            }
            if fragment == UNMAPPED {
                return Err(Error::Invalid("u64::MAX is not a valid fragment id".into()));
            }
            if !seen.insert(fragment) {
                continue;
            }
            let from = self.get(fragment);
            if from != segment {
                moves.push(Move {
                    fragment,
                    from,
                    to: segment,
                });
            }
        }
        Ok(moves)
    }

    /// Merge segments `a` and `b` (see [`Self::plan_merge`]); returns the surviving id.
    pub fn merge(&mut self, a: u64, b: u64) -> Result<u64> {
        let plan = self.plan_merge(a, b)?;
        self.apply_moves(&plan.moves)?;
        Ok(plan.survivor)
    }

    /// Put `fragments` into `segment` (see [`Self::plan_assign`]); returns the moves made.
    pub fn assign(&mut self, fragments: &[u64], segment: u64) -> Result<Vec<Move>> {
        let moves = self.plan_assign(fragments, segment)?;
        self.apply_moves(&moves)?;
        Ok(moves)
    }

    /// Apply moves in order. Every move's `from` must be the fragment's current segment, which
    /// is what makes a replayed or undone log exact; on a mismatch nothing is changed.
    pub fn apply_moves(&mut self, moves: &[Move]) -> Result<()> {
        // Validate against the state each move will see (a fragment may move twice).
        let mut pending: HashMap<u64, u64> = HashMap::new();
        for m in moves {
            check_entry(m.fragment, m.to)?;
            let current = pending
                .get(&m.fragment)
                .copied()
                .unwrap_or_else(|| self.get(m.fragment));
            if current != m.from {
                return Err(Error::Corrupt(format!(
                    "fragment {} is in segment {current}, but the edit expects {}",
                    m.fragment, m.from
                )));
            }
            pending.insert(m.fragment, m.to);
        }
        for m in moves {
            self.set(m.fragment, m.to);
        }
        Ok(())
    }

    /// Set one explicit entry, keeping the reverse index current if it has been built.
    ///
    /// In a map that is not complete, an entry mapping a fragment to itself is the identity
    /// default and is removed instead of stored, so undoing an edit restores the map exactly.
    fn set(&mut self, fragment: u64, segment: u64) {
        let previous = self.get(fragment);
        let had_entry = self.explicit(fragment).is_some();
        let remove = !self.complete && segment == fragment;
        if remove {
            self.unstore(fragment);
        } else {
            let dense = self.is_dense();
            self.store(fragment, segment, dense);
        }
        if let Some(reverse) = self.reverse.get_mut() {
            if had_entry && previous != 0 {
                if let Some(list) = reverse.get_mut(&previous) {
                    if let Some(at) = list.iter().position(|f| *f == fragment) {
                        list.swap_remove(at);
                    }
                    if list.is_empty() {
                        reverse.remove(&previous);
                    }
                }
            }
            if segment != 0 && !remove {
                reverse.entry(segment).or_default().push(fragment);
            }
        }
    }

    fn unstore(&mut self, fragment: u64) {
        if let Some(offset) = fragment.checked_sub(self.base) {
            if offset < self.dense.len() as u64 {
                let slot = &mut self.dense[offset as usize];
                if *slot != UNMAPPED {
                    self.dense_explicit -= 1;
                }
                *slot = UNMAPPED;
                return;
            }
        }
        self.sparse.remove(&fragment);
    }

    fn store(&mut self, fragment: u64, segment: u64, prefer_dense: bool) {
        if let Some(offset) = fragment.checked_sub(self.base) {
            if offset < self.dense.len() as u64 {
                let slot = &mut self.dense[offset as usize];
                if *slot == UNMAPPED {
                    self.dense_explicit += 1;
                }
                *slot = segment;
                return;
            }
            // Grow a dense table upwards while it stays dense.
            if prefer_dense
                && !self.dense.is_empty()
                && !self.sparse.contains_key(&fragment)
                && self.dense.len() < usize::MAX / 2
                && offset
                    < (self.dense_explicit as u64 + 1).saturating_mul(DENSE_FACTOR) + DENSE_SLACK
            {
                // Grow geometrically so a run of fresh ids is amortised, and pull any sparse
                // entries the new range covers into it (a slot must never shadow one).
                let old_len = self.dense.len() as u64;
                let new_len = (offset + 1).max(old_len + old_len / 2);
                self.dense.resize(new_len as usize, UNMAPPED);
                let base = self.base;
                let covered: Vec<u64> = self
                    .sparse
                    .keys()
                    .copied()
                    .filter(|f| *f >= base + old_len && *f < base + new_len)
                    .collect();
                for f in covered {
                    let s = self.sparse.remove(&f).expect("key was just listed");
                    self.dense[(f - base) as usize] = s;
                    self.dense_explicit += 1;
                }
                self.dense[offset as usize] = segment;
                self.dense_explicit += 1;
                return;
            }
        }
        self.sparse.insert(fragment, segment);
    }
}

fn check_entry(fragment: u64, segment: u64) -> Result<()> {
    if fragment == 0 {
        return Err(Error::Invalid(
            "fragment 0 is background and always maps to 0".into(),
        ));
    }
    if fragment == UNMAPPED || segment == UNMAPPED {
        return Err(Error::Invalid(
            "u64::MAX is reserved and cannot be mapped".into(),
        ));
    }
    Ok(())
}

impl IdTransform for SegmentMap {
    #[inline]
    fn apply(&self, id: u64) -> u64 {
        self.get(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn both(pairs: &[(u64, u64)]) -> [SegmentMap; 2] {
        [
            SegmentMap::from_pairs_with(pairs.iter().copied(), Layout::Dense).unwrap(),
            SegmentMap::from_pairs_with(pairs.iter().copied(), Layout::Sparse).unwrap(),
        ]
    }

    #[test]
    fn identity_default_and_background() {
        let map = SegmentMap::from_pairs([(2, 1), (3, 1)]).unwrap();
        assert_eq!(map.get(0), 0);
        assert_eq!(map.get(2), 1);
        assert_eq!(map.get(7), 7);
        assert_eq!(map.fragments_of(1), vec![1, 2, 3]);
        assert_eq!(map.fragments_of(7), vec![7]);
        assert!(map.fragments_of(0).is_empty());
        assert!(SegmentMap::from_pairs([(0, 4)]).is_err());
    }

    #[test]
    fn auto_layout_picks_dense_for_compact_ids() {
        let compact = SegmentMap::from_pairs((1..5000).map(|f| (f, f / 10 + 1))).unwrap();
        assert!(compact.is_dense());
        let scattered = SegmentMap::from_pairs((1..100).map(|f| (f * 1_000_000_007, 1))).unwrap();
        assert!(!scattered.is_dense());
    }

    #[test]
    fn dense_and_sparse_answer_identically_through_edits() {
        let pairs: Vec<(u64, u64)> = (1..200).map(|f| (f, (f - 1) / 5 + 1000)).collect();
        let [mut dense, mut sparse] = both(&pairs);
        assert!(dense.is_dense() && !sparse.is_dense());
        for map in [&mut dense, &mut sparse] {
            // Build the reverse index first on one, lazily on the other path below.
            let _ = map.fragments_of(1000);
            map.merge(1000, 1001).unwrap();
            map.assign(&[3, 4, 500, 9_000_000], 77).unwrap();
            map.merge(77, 1010).unwrap();
            map.assign(&[6], 0).unwrap();
        }
        assert_eq!(dense, sparse);
        for id in 0..600 {
            assert_eq!(dense.get(id), sparse.get(id), "fragment {id}");
        }
        for segment in [0, 77, 1000, 1001, 1010, 1011, 500, 9_000_000] {
            assert_eq!(dense.fragments_of(segment), sparse.fragments_of(segment));
        }
        // A freshly built reverse index agrees with the incrementally maintained one.
        let rebuilt = SegmentMap::from_pairs_with(dense.entries(), Layout::Sparse).unwrap();
        for segment in [77, 1000, 1001, 1010, 1011] {
            assert_eq!(dense.fragments_of(segment), rebuilt.fragments_of(segment));
        }
    }

    #[test]
    fn merge_relabels_the_smaller_side() {
        let mut map = SegmentMap::from_pairs([(1, 10), (2, 10), (3, 10), (4, 20)])
            .unwrap()
            .with_complete(true);
        let plan = map.plan_merge(20, 10).unwrap();
        assert_eq!((plan.survivor, plan.absorbed), (10, 20));
        assert_eq!(
            plan.moves,
            vec![Move {
                fragment: 4,
                from: 20,
                to: 10
            }]
        );
        assert_eq!(map.merge(20, 10).unwrap(), 10);
        assert_eq!(map.fragments_of(10), vec![1, 2, 3, 4]);
        assert!(map.fragments_of(20).is_empty());
        assert!(map.plan_merge(10, 10).is_err());
        assert!(map.plan_merge(10, 0).is_err());
        assert!(
            map.plan_merge(10, 20).is_err(),
            "20 no longer has fragments"
        );
    }

    #[test]
    fn incomplete_maps_count_a_bare_id_as_its_own_fragment() {
        // Plain instance labels 7 and 9, no map yet: merging them, then merging the result
        // elsewhere, must carry both original ids along.
        let mut map = SegmentMap::new();
        let survivor = map.merge(7, 9).unwrap();
        assert_eq!(survivor, 7);
        assert_eq!(map.fragments_of(7), vec![7, 9]);
        map.merge(50, 7).unwrap();
        assert_eq!((map.get(7), map.get(9)), (map.get(50), map.get(50)));
        // A complete map knows 20 is not a fragment.
        let complete = SegmentMap::from_pairs([(1, 20)])
            .unwrap()
            .with_complete(true);
        assert_eq!(complete.fragments_of(20), vec![1]);
        assert!(complete.fragments_of(5).is_empty());
        let incomplete = SegmentMap::from_pairs([(1, 20)]).unwrap();
        assert_eq!(incomplete.fragments_of(20), vec![1, 20]);
    }

    #[test]
    fn stale_moves_are_refused_without_change() {
        let mut map = SegmentMap::from_pairs([(1, 10)]).unwrap();
        let before = map.clone();
        let stale = [
            Move {
                fragment: 2,
                from: 2,
                to: 10,
            },
            Move {
                fragment: 1,
                from: 99,
                to: 5,
            },
        ];
        assert!(matches!(map.apply_moves(&stale), Err(Error::Corrupt(_))));
        assert_eq!(map, before);
    }
}
