//! Id transforms and the per-tile applier.

use crate::map::UNMAPPED;

/// A function on label ids, applied to the ids a label image stores before they are drawn.
///
/// A [`crate::SegmentMap`] is one (fragment -> segment); a track-id relabelling, an atlas
/// hierarchy cut at some depth or an id -> class table are others. Transforms compose in a
/// chain, first to last. Background `0` is never passed to a transform.
pub trait IdTransform {
    fn apply(&self, id: u64) -> u64;
}

impl<T: IdTransform + ?Sized> IdTransform for &T {
    fn apply(&self, id: u64) -> u64 {
        (**self).apply(id)
    }
}

impl<T: IdTransform + ?Sized> IdTransform for std::sync::Arc<T> {
    fn apply(&self, id: u64) -> u64 {
        (**self).apply(id)
    }
}

/// A closure as a transform.
pub struct FnTransform<F>(pub F);

impl<F: Fn(u64) -> u64> IdTransform for FnTransform<F> {
    fn apply(&self, id: u64) -> u64 {
        (self.0)(id)
    }
}

/// Transforms applied in order.
pub struct Chain<'a>(pub &'a [&'a dyn IdTransform]);

impl IdTransform for Chain<'_> {
    fn apply(&self, id: u64) -> u64 {
        apply_chain(self.0, id)
    }
}

#[inline]
fn apply_chain(transforms: &[&dyn IdTransform], id: u64) -> u64 {
    let mut id = id;
    for transform in transforms {
        if id == 0 {
            break;
        }
        id = transform.apply(id);
    }
    id
}

/// What one remap call did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RemapStats {
    /// Distinct non-zero ids in the tile: the number of times the chain was evaluated.
    pub unique: usize,
    /// Ids the chain changed.
    pub changed: usize,
}

/// Remap a tile's ids through `transforms` in place.
///
/// The chain is evaluated once per distinct id, not once per pixel: pixels go through a
/// one-entry run cache (label images are mostly runs) and then a small open-addressed table
/// local to the call, so the per-pixel cost is a compare or a probe into a table that fits in
/// cache, whatever the transforms cost.
pub fn remap_ids_in_place(ids: &mut [u64], transforms: &[&dyn IdTransform]) -> RemapStats {
    let mut stats = RemapStats::default();
    if transforms.is_empty() || ids.is_empty() {
        return stats;
    }
    let mut table = LocalTable::with_capacity(64);
    let (mut last_in, mut last_out) = (0u64, 0u64);
    for id in ids.iter_mut() {
        let value = *id;
        if value == 0 {
            continue;
        }
        if value == last_in {
            *id = last_out;
            continue;
        }
        let mapped = match table.get(value) {
            Some(mapped) => mapped,
            None => {
                let mapped = apply_chain(transforms, value);
                table.insert(value, mapped);
                stats.unique += 1;
                stats.changed += usize::from(mapped != value);
                mapped
            }
        };
        last_in = value;
        last_out = mapped;
        *id = mapped;
    }
    stats
}

/// [`remap_ids_in_place`] into a new buffer.
pub fn remap_ids(ids: &[u64], transforms: &[&dyn IdTransform]) -> Vec<u64> {
    let mut out = ids.to_vec();
    remap_ids_in_place(&mut out, transforms);
    out
}

/// Open addressing keyed by non-zero ids (`0` marks an empty slot), linear probing, kept at most
/// half full.
struct LocalTable {
    keys: Vec<u64>,
    values: Vec<u64>,
    len: usize,
    shift: u32,
}

impl LocalTable {
    fn with_capacity(slots: usize) -> Self {
        let slots = slots.next_power_of_two().max(16);
        Self {
            keys: vec![0; slots],
            values: vec![UNMAPPED; slots],
            len: 0,
            shift: 64 - slots.trailing_zeros(),
        }
    }

    #[inline]
    fn slot(&self, key: u64) -> usize {
        (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> self.shift) as usize
    }

    #[inline]
    fn get(&self, key: u64) -> Option<u64> {
        let mask = self.keys.len() - 1;
        let mut at = self.slot(key);
        loop {
            let stored = self.keys[at];
            if stored == key {
                return Some(self.values[at]);
            }
            if stored == 0 {
                return None;
            }
            at = (at + 1) & mask;
        }
    }

    fn insert(&mut self, key: u64, value: u64) {
        if (self.len + 1) * 2 > self.keys.len() {
            let mut grown = Self::with_capacity(self.keys.len() * 2);
            for (k, v) in self.keys.iter().zip(&self.values) {
                if *k != 0 {
                    grown.insert(*k, *v);
                }
            }
            *self = grown;
        }
        let mask = self.keys.len() - 1;
        let mut at = self.slot(key);
        while self.keys[at] != 0 && self.keys[at] != key {
            at = (at + 1) & mask;
        }
        if self.keys[at] == 0 {
            self.len += 1;
        }
        self.keys[at] = key;
        self.values[at] = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Layout, SegmentMap};

    /// A deterministic pseudo-random tile with runs, scattered ids and background.
    fn tile(len: usize, seed: u64) -> Vec<u64> {
        let mut state = seed;
        let mut out = Vec::with_capacity(len);
        let mut current = 0;
        for _ in 0..len {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            if state.is_multiple_of(5) {
                current = match state % 7 {
                    0 => 0,
                    1 => state >> 20,
                    _ => state % 3000,
                };
            }
            out.push(current);
        }
        out
    }

    #[test]
    fn applier_equals_naive_per_pixel_mapping() {
        let map =
            SegmentMap::from_pairs_with((1..2500).map(|f| (f, (f % 400) + 10_000)), Layout::Dense)
                .unwrap();
        let sparse = SegmentMap::from_pairs_with(map.entries(), Layout::Sparse).unwrap();
        let classes = FnTransform(|id: u64| if id.is_multiple_of(2) { id / 2 } else { 0 });
        let chains: [&[&dyn IdTransform]; 4] = [
            &[&map],
            &[&sparse],
            &[&map, &classes],
            &[&classes, &map, &sparse],
        ];
        for (seed, chain) in [1u64, 2, 3, 99].into_iter().zip(chains) {
            let ids = tile(256 * 256 + 3, seed);
            let naive: Vec<u64> = ids
                .iter()
                .map(|&id| {
                    let mut id = id;
                    for t in chain {
                        if id != 0 {
                            id = t.apply(id);
                        }
                    }
                    id
                })
                .collect();
            assert_eq!(remap_ids(&ids, chain), naive, "seed {seed}");
            let mut in_place = ids.clone();
            let stats = remap_ids_in_place(&mut in_place, chain);
            assert_eq!(in_place, naive);
            assert!(stats.unique > 100 && stats.unique < ids.len() / 4);
        }
    }

    #[test]
    fn empty_chain_is_the_identity() {
        let ids = tile(1000, 5);
        assert_eq!(remap_ids(&ids, &[]), ids);
    }
}
