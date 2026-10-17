//! Editable id transform layers for NGFF label images.
//!
//! A label image stores ids; what a viewer draws, measures or exports may be a function of
//! them. The motivating case is EM proofreading: an automatic segmentation writes a label image
//! of *fragment* ids (supervoxels) and an initial fragment -> *segment* grouping (a multicut);
//! proofreading edits only the grouping. Merging two segments then rewrites a handful of map
//! entries instead of millions of voxels, and every edit can be undone. The same layer serves
//! track-id relabelling, atlas hierarchy depth, or id -> class tables: anything expressible as
//! an [`IdTransform`].
//!
//! * [`SegmentMap`]: fragment -> segment (dense table for compact ids, hash map otherwise;
//!   unmapped ids map to themselves, `0` to `0`), with a lazily built reverse index,
//!   small-to-large [`SegmentMap::merge`] and [`SegmentMap::assign`] (the split primitive).
//! * [`SegmentMapState`]: the map plus its append-only edit log. Revision = number of records;
//!   undo appends the inverse of the latest live edit, so history is never rewritten.
//! * [`remap_ids_in_place`]: apply a chain of transforms to a tile, evaluating the chain once
//!   per distinct id.
//! * [`snapshot`]: the binary snapshot codec.
//! * `store` (feature `fs`, on by default): the on-disk store, writer lock and discovery.
//!
//! # On-disk format (version 1)
//!
//! ```text
//! labels/<layer>/zarr.json      attributes.segment_map = {"version": 1, "path": "segment-map"}
//! labels/<layer>/segment-map/
//!     snapshot.bin              the map at some revision R (see `snapshot` for the layout)
//!     edits.jsonl               every edit record, one JSON object per line, revisions 1..N
//! labels/<layer>/.segment-map.lock   writer lock (created on first edit)
//! ```
//!
//! The map at revision N is `snapshot.bin` (revision R <= N) with records R+1..N of
//! `edits.jsonl` replayed. A record is
//!
//! ```json
//! {"revision":3,"op":"merge","segments":[12,40],"moves":[[40,40,12],[41,40,12]],"time_ms":1760000000000,"author":"jo"}
//! ```
//!
//! with `op` one of `merge` (`segments = [survivor, absorbed]`), `assign`
//! (`segments = [target]`) or `undo` (with `"undoes": <revision>`), and `moves` the
//! `[fragment, from, to]` changes in order. A pipeline publishing a fresh segmentation writes
//! `snapshot.bin` at revision 0 and an empty `edits.jsonl` (see `store::create_for_label`).
//!
//! Writers take the lock, re-read the log, refuse with [`Error::Conflict`] unless the store is
//! at the revision the caller last saw (an HTTP server answers 409), append one line and
//! `fsync`. Every 64 edits the snapshot is rewritten by temp file + rename. A torn final line
//! (a crash mid-append) is ignored by readers and dropped by the next writer. Readers need no
//! lock.

mod apply;
mod history;
mod map;
pub mod snapshot;
#[cfg(feature = "fs")]
pub mod store;

use std::fmt;

pub use apply::{remap_ids, remap_ids_in_place, Chain, FnTransform, IdTransform, RemapStats};
pub use history::{EditOp, EditRecord, SegmentMapState};
pub use map::{Layout, MergePlan, Move, SegmentMap};
#[cfg(feature = "fs")]
pub use store::{EditMeta, LoadedSegmentMap, SegmentMapStore};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    /// The store moved on since the caller read it: re-read and retry (HTTP 409).
    Conflict {
        expected: u64,
        current: u64,
    },
    /// The request cannot be applied (background id, unknown segment, nothing to undo, ...).
    Invalid(String),
    /// Stored data is inconsistent or unreadable.
    Corrupt(String),
    Io(std::io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Conflict { expected, current } => write!(
                out,
                "segment map revision conflict: expected {expected}, current {current}"
            ),
            Self::Invalid(message) => out.write_str(message),
            Self::Corrupt(message) => write!(out, "corrupt segment map: {message}"),
            Self::Io(error) => error.fmt(out),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}
