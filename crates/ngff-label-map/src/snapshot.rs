//! The binary snapshot codec (`snapshot.bin`). Pure bytes in and out, so it also runs in wasm.
//!
//! All integers are little-endian `u64` unless noted:
//!
//! | offset | field |
//! |---|---|
//! | 0 | magic `b"NGFFSMAP"` (8 bytes) |
//! | 8 | format version, `u32` = 1 |
//! | 12 | flags, `u32`: bit 0 = complete (every fragment has an entry; see `SegmentMap::with_complete`) |
//! | 16 | `revision`: the number of `edits.jsonl` records already folded in |
//! | 24 | `generation`: identity of the map's lineage (kept by compaction, new on create) |
//! | 32 | `base`: fragment id of dense slot 0 |
//! | 40 | `dense_count`: number of dense slots |
//! | 48 | `pair_count`: number of sparse pairs |
//! | 56 | `dense_count` segment ids; slot `i` is fragment `base + i`; `u64::MAX` = no entry |
//! | ... | `pair_count` `(fragment, segment)` pairs, sorted by fragment |
//!
//! Fragments without an entry map to themselves, so a pipeline writing a multicut result can
//! use the dense form with `base = 1` and one segment id per fragment.

use crate::map::SegmentMap;
use crate::{Error, Result};

pub const MAGIC: &[u8; 8] = b"NGFFSMAP";
pub const VERSION: u32 = 1;
pub const HEADER_LEN: usize = 56;

/// The fixed header of a snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SnapshotHeader {
    pub complete: bool,
    pub revision: u64,
    pub generation: u64,
    pub base: u64,
    pub dense_count: u64,
    pub pair_count: u64,
}

/// Encode `map` as a snapshot at `revision`.
pub fn encode(map: &SegmentMap, revision: u64, generation: u64) -> Vec<u8> {
    let (base, dense) = map.dense_table().unwrap_or((0, &[]));
    let mut pairs: Vec<(u64, u64)> = map.sparse_entries().collect();
    pairs.sort_unstable();
    let mut out = Vec::with_capacity(HEADER_LEN + dense.len() * 8 + pairs.len() * 16);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&u32::from(map.is_complete()).to_le_bytes());
    for value in [
        revision,
        generation,
        base,
        dense.len() as u64,
        pairs.len() as u64,
    ] {
        out.extend_from_slice(&value.to_le_bytes());
    }
    for segment in dense {
        out.extend_from_slice(&segment.to_le_bytes());
    }
    for (fragment, segment) in pairs {
        out.extend_from_slice(&fragment.to_le_bytes());
        out.extend_from_slice(&segment.to_le_bytes());
    }
    out
}

/// Read only the header (the first [`HEADER_LEN`] bytes suffice).
pub fn decode_header(bytes: &[u8]) -> Result<SnapshotHeader> {
    if bytes.len() < HEADER_LEN || &bytes[..8] != MAGIC {
        return Err(Error::Corrupt("not a segment map snapshot".into()));
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().expect("4 bytes"));
    if version != VERSION {
        return Err(Error::Corrupt(format!(
            "segment map snapshot version {version} is not supported (expected {VERSION})"
        )));
    }
    let word = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().expect("8 bytes"));
    let flags = u32::from_le_bytes(bytes[12..16].try_into().expect("4 bytes"));
    if flags & !1 != 0 {
        return Err(Error::Corrupt(format!(
            "unknown segment map snapshot flags {flags:#x}"
        )));
    }
    Ok(SnapshotHeader {
        complete: flags & 1 == 1,
        revision: word(16),
        generation: word(24),
        base: word(32),
        dense_count: word(40),
        pair_count: word(48),
    })
}

/// Decode a snapshot into its header and map.
pub fn decode(bytes: &[u8]) -> Result<(SnapshotHeader, SegmentMap)> {
    let header = decode_header(bytes)?;
    let expected = header
        .dense_count
        .checked_mul(8)
        .and_then(|d| {
            header
                .pair_count
                .checked_mul(16)
                .and_then(|p| d.checked_add(p))
        })
        .and_then(|n| n.checked_add(HEADER_LEN as u64))
        .ok_or_else(|| Error::Corrupt("snapshot sizes overflow".into()))?;
    if bytes.len() as u64 != expected {
        return Err(Error::Corrupt(format!(
            "snapshot is {} bytes but its header describes {expected}",
            bytes.len()
        )));
    }
    let words = bytes[HEADER_LEN..]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|w| u64::from_le_bytes(*w));
    let dense: Vec<u64> = words.clone().take(header.dense_count as usize).collect();
    let pairs: Vec<u64> = words.skip(header.dense_count as usize).collect();
    let sparse = pairs
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| (p[0], p[1]))
        .collect();
    let map = SegmentMap::from_parts(header.base, dense, sparse)?.with_complete(header.complete);
    Ok((header, map))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Layout;

    #[test]
    fn round_trips_dense_sparse_and_mixed() {
        let mut mixed =
            SegmentMap::from_pairs_with((1..50).map(|f| (f, f / 7 + 100)), Layout::Dense).unwrap();
        mixed.assign(&[5_000_000_000, 3], 4).unwrap();
        assert!(mixed.is_dense() && mixed.len() == 50);
        for map in [
            SegmentMap::new(),
            SegmentMap::from_pairs_with((1..50).map(|f| (f, f / 7 + 100)), Layout::Dense).unwrap(),
            SegmentMap::from_pairs_with((1..50).map(|f| (f * 31, 1)), Layout::Sparse).unwrap(),
            mixed,
            SegmentMap::from_pairs([(3, 4)])
                .unwrap()
                .with_complete(true),
        ] {
            let bytes = encode(&map, 12, 99);
            let (header, decoded) = decode(&bytes).unwrap();
            assert_eq!((header.revision, header.generation), (12, 99));
            assert_eq!(decoded, map);
            assert_eq!(decoded.is_dense(), map.is_dense());
            assert_eq!(decoded.is_complete(), map.is_complete());
        }
    }

    #[test]
    fn decodes_a_hand_written_dense_multicut() {
        // The layout the README tells pipelines to write directly.
        let segments: [u64; 4] = [7, 7, 9, 7];
        let mut bytes = b"NGFFSMAP".to_vec();
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        for word in [0u64, 42, 1, segments.len() as u64, 0] {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        for s in segments {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let (header, map) = decode(&bytes).unwrap();
        assert!(header.complete && map.is_complete());
        assert_eq!((header.revision, header.generation), (0, 42));
        assert_eq!(map.segments_of(&[0, 1, 2, 3, 4, 5]), vec![0, 7, 7, 9, 7, 5]);
        assert_eq!(map.fragments_of(7), vec![1, 2, 4]);
        assert_eq!(map.fragments_of(9), vec![3]);
    }

    #[test]
    fn rejects_truncated_and_foreign_bytes() {
        let bytes = encode(&SegmentMap::from_pairs([(1, 2)]).unwrap(), 0, 1);
        assert!(decode(&bytes[..bytes.len() - 1]).is_err());
        assert!(
            decode(b"definitely not a snapshot, but long enough to have a header....").is_err()
        );
    }
}
