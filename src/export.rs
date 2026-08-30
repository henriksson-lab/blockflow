// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The order log, as JSON, for consumers outside Rust.
//
// This is a **cross-language contract**, not a debug dump: `tools/
// animate_block_progress.py` reads it, and anything else that wants to analyse
// a schedule offline will read it too. So the schema is written down here, in
// the module that produces it, and the producer is the only thing allowed to
// change it. A consumer that finds a field missing has found a bug, not a
// version drift.
//
// ============================== SCHEMA v1 ==============================
//
// ```json
// {
//   "schema":  "clearmap-rs.block_ops.order_log",   // exact string, identifies the format
//   "version": 1,                                    // integer, bumped only on a breaking change
//   "strategy": "greedy",                            // which Strategy produced the run
//   "volume":  [512, 512, 512],                      // voxels, axis order as below
//   "grid":    [8, 4, 4],                            // blocks per axis, = max block index + 1
//   "phases":  2,                                    // number of phases in the decomposition
//   "ops":     [{"slot": 0, "name": "median"}, ...], // chain slots, in chain order
//   "blocks":  [ ...block table... ],
//   "events":  [ ...event stream... ]
// }
// ```
//
// ## Conventions that apply everywhere in the document
//
// * **Axis order** is the volume's own — axis 0, 1, 2 of the `Workflow`'s
//   shape, in that order. No transposition is applied anywhere. If the caller
//   thinks of axis 0 as Z, then index 0 of every triple in this document is Z.
// * **Two different coordinate systems, never mixed.**
//   * `index` is a **block index**: a position in the block grid, `0 <= index[a]
//     < grid[a]`. It is not a voxel coordinate and must never be multiplied by
//     anything to become one — blocks at the volume's far edge are shorter, so
//     `index * block_shape` is wrong there.
//   * `start` / `shape` are **voxel** coordinates. `start` is inclusive, and the
//     region covers `start[a] .. start[a] + shape[a]`, upper bound exclusive.
//   * To go from a block index to voxels, use the **`blocks` table**, which
//     records the actual regions. Do not compute them.
// * **Units.** `voxels` is a count of voxels. `bytes` is uncompressed bytes in
//   memory (`voxels * sizeof(dtype)`), never on-disk size. `chunks` is a count
//   of storage chunks the region touches. `duration_ns` is wall-clock
//   nanoseconds, measured around the call, including any time the thread spent
//   descheduled — it is not CPU time.
//
// ## The block table
//
// One entry per block index that appeared in the run:
//
// ```json
// {"index": [0,0,0], "phases": [0,1],
//  "read":  {"start": [0,0,0], "shape": [72,72,72]},
//  "valid": {"start": [0,0,0], "shape": [64,64,64]}}
// ```
//
// `read` and `valid` are from the block's **first** phase: the read extent
// includes the halo, the valid extent is what the block is trusted for and what
// it wrote. The valid regions of one phase tile the volume exactly — the
// executor asserts this — so a consumer may rely on them to draw a partition
// with no gaps and no overlaps. Read extents *do* overlap; that is the halo.
//
// ## The event stream
//
// `events` is a JSON array in **emission order**. Every event has `seq` (its
// 0-based position, so a consumer may sort or slice on it without re-deriving
// it) and `type`. What a consumer may rely on:
//
// * **Per-block op order is deterministic** and equals the chain's slot order.
//   This is the framework's acceptance criterion and is asserted in Rust.
// * **Global order is one valid linearisation, not the only one.** Concurrent
//   tasks interleave, and the interleaving is a scheduling artefact that may
//   differ run to run at the same settings. Anything derived from the global
//   order is a picture of *this* run.
// * `phase` is non-decreasing per block, never per run.
// * A `block_short_circuited` event means those slots were **not computed** but
//   the block holds exactly what computing them would have produced. For
//   coverage or progress purposes, treat the listed slots as applied.
//
// Event types and their fields, beyond `seq` and `type`:
//
// | `type` | fields |
// |---|---|
// | `phase_started` | `phase` |
// | `task_admitted` | `phase`, `index` — the scheduler's choice, before any work |
// | `region_read` | `source`, `level`, `index` (may be `null`), `start`, `shape`, `voxels`, `bytes`, `chunks`, `duration_ns` |
// | `region_written` | `sink`, `level`, `index` (may be `null`), `start`, `shape`, `voxels`, `bytes`, `chunks`, `duration_ns` |
// | `materialised` | `phase`, `level`, `bytes`, `intermediate` |
// | `block_read` | `phase`, `index`, `start`, `shape`, `voxels`, `chunks` |
// | `op_applied` | `phase`, `index`, `slot`, `op`, `start`, `shape`, `duration_ns` — `start`/`shape` are the extent computed **over**, i.e. the read extent including halo |
// | `block_short_circuited` | `phase`, `index`, `from`, `to`, `slots`, `ops` |
// | `block_written` | `phase`, `index`, `start`, `shape`, `materialised` |
// | `sidecar_written` | `stream`, `phase`, `index`, `bytes`, `duration_ns` — a block's non-pixel fragment |
// | `sidecar_read` | `stream`, `phase`, `index`, `bytes`, `found`, `duration_ns` |
// | `sidecar_discarded` | `stream`, `fragments`, `bytes` — a whole stream removed |
// | `side_output_written` | `output`, `phase`, `index`, `start`, `shape`, `bytes` — a block's slice of an array the op writes beside its primary result. `start`/`shape` are in **that output's own** space and may be of a different rank from the volume's, so they must not be read as voxel coordinates of it |
//
// `level` is a storage image: 0 is the workflow input, image `p` is the
// intermediate that phase `p-1` wrote. `index` is `null` when the emitter is a
// bare `RegionSource`/`RegionSink`, which has no notion of blocks.
//
// ## What a consumer may *not* rely on
//
// * That the block grid is the same in every phase. The design leaves per-phase
//   block sizes open. Today every strategy uses one grid, and `grid` is
//   therefore a single triple; if that ever changes, `version` goes to 2 and
//   `grid` becomes per-phase. A consumer should not silently assume phase 0's
//   geometry applies to phase 3 — the `blocks` table records which phases each
//   index appeared in, so a mismatch is detectable.
// * Wall-clock timestamps. There are none, deliberately: `seq` orders the
//   stream and `duration_ns` prices individual calls, and adding absolute times
//   would invite consumers to subtract them across threads.
//
// =======================================================================

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::{json, Map, Value};

use crate::error::{Error, Result};
use crate::region::Region;

use super::log::{Event, ExecutionLog, PrefetchWaste, Tier};

/// The parts of a run the log itself does not carry.
///
/// Taken as an argument rather than inferred so that the export never *guesses*
/// something the caller knows: the volume shape and the chain's op names come
/// from the workflow, not from the events.
#[derive(Debug, Clone)]
pub struct ExportMeta {
    pub strategy: String,
    pub volume: [usize; 3],
    pub phases: usize,
    /// `(slot, name)` in chain order.
    pub ops: Vec<(usize, String)>,
}

impl ExportMeta {
    pub fn new(strategy: impl Into<String>, volume: [usize; 3], phases: usize) -> Self {
        Self {
            strategy: strategy.into(),
            volume,
            phases,
            ops: Vec::new(),
        }
    }

    pub fn with_ops(mut self, ops: Vec<(usize, String)>) -> Self {
        self.ops = ops;
        self
    }
}

fn region_fields(map: &mut Map<String, Value>, region: &Region) {
    map.insert("start".to_string(), json!(region.start));
    map.insert("shape".to_string(), json!(region.shape));
}

fn object(pairs: Vec<(&str, Value)>) -> Value {
    let mut map = Map::new();
    for (key, value) in pairs {
        map.insert(key.to_string(), value);
    }
    Value::Object(map)
}

/// One event, as schema v1, with its position in the stream.
fn event_to_json(seq: usize, event: &Event) -> Value {
    let mut value = event_json(event);
    if let Value::Object(map) = &mut value {
        map.insert("seq".to_string(), json!(seq));
    }
    value
}

/// One event, as schema v1, **without** `seq`.
///
/// Split out from the document writer because an event now has a second
/// destination: a distributed run sends events to its coordinator one at a
/// time, as they happen, and the coordinator merges streams from several
/// workers. A sender has no idea what its event's position in the merged stream
/// will be, so `seq` belongs to whoever assembles the stream — and the encoding
/// of an event does not otherwise change because it travelled.
///
/// Keeping this in the module that documents the schema is deliberate: the
/// header calls the format a cross-language contract and says the producer is
/// the only thing allowed to change it. A second encoder living next to the
/// wire protocol would be a second producer.
pub fn event_json(event: &Event) -> Value {
    let mut map = Map::new();
    match event {
        Event::PhaseStarted { phase } => {
            map.insert("type".to_string(), json!("phase_started"));
            map.insert("phase".to_string(), json!(phase));
        }
        Event::TaskAdmitted { phase, index } => {
            map.insert("type".to_string(), json!("task_admitted"));
            map.insert("phase".to_string(), json!(phase));
            map.insert("index".to_string(), json!(index));
        }
        Event::RegionRead {
            source,
            image,
            index,
            region,
            voxels,
            bytes,
            chunks,
            duration_ns,
        } => {
            map.insert("type".to_string(), json!("region_read"));
            map.insert("source".to_string(), json!(source));
            map.insert("level".to_string(), json!(image));
            map.insert("index".to_string(), json!(index));
            region_fields(&mut map, region);
            map.insert("voxels".to_string(), json!(voxels));
            map.insert("bytes".to_string(), json!(bytes));
            map.insert("chunks".to_string(), json!(chunks));
            map.insert("duration_ns".to_string(), json!(duration_ns));
        }
        Event::RegionWritten {
            sink,
            image,
            index,
            region,
            voxels,
            bytes,
            chunks,
            duration_ns,
        } => {
            map.insert("type".to_string(), json!("region_written"));
            map.insert("sink".to_string(), json!(sink));
            map.insert("level".to_string(), json!(image));
            map.insert("index".to_string(), json!(index));
            region_fields(&mut map, region);
            map.insert("voxels".to_string(), json!(voxels));
            map.insert("bytes".to_string(), json!(bytes));
            map.insert("chunks".to_string(), json!(chunks));
            map.insert("duration_ns".to_string(), json!(duration_ns));
        }
        Event::Materialised {
            phase,
            image,
            bytes,
            intermediate,
        } => {
            map.insert("type".to_string(), json!("materialised"));
            map.insert("phase".to_string(), json!(phase));
            map.insert("level".to_string(), json!(image));
            map.insert("bytes".to_string(), json!(bytes));
            map.insert("intermediate".to_string(), json!(intermediate));
        }
        Event::BlockRead {
            phase,
            index,
            region,
            voxels,
            chunks,
        } => {
            map.insert("type".to_string(), json!("block_read"));
            map.insert("phase".to_string(), json!(phase));
            map.insert("index".to_string(), json!(index));
            region_fields(&mut map, region);
            map.insert("voxels".to_string(), json!(voxels));
            map.insert("chunks".to_string(), json!(chunks));
        }
        Event::OpApplied {
            phase,
            index,
            slot,
            op,
            over,
            duration_ns,
        } => {
            map.insert("type".to_string(), json!("op_applied"));
            map.insert("phase".to_string(), json!(phase));
            map.insert("index".to_string(), json!(index));
            map.insert("slot".to_string(), json!(slot));
            map.insert("op".to_string(), json!(op));
            region_fields(&mut map, over);
            map.insert("duration_ns".to_string(), json!(duration_ns));
        }
        Event::BlockShortCircuited {
            phase,
            index,
            from,
            to,
            slots,
            names,
        } => {
            map.insert("type".to_string(), json!("block_short_circuited"));
            map.insert("phase".to_string(), json!(phase));
            map.insert("index".to_string(), json!(index));
            map.insert("from".to_string(), json!(from));
            map.insert("to".to_string(), json!(to));
            map.insert("slots".to_string(), json!(slots));
            map.insert("ops".to_string(), json!(names));
        }
        Event::BlockWritten {
            phase,
            index,
            valid,
            materialised,
        } => {
            map.insert("type".to_string(), json!("block_written"));
            map.insert("phase".to_string(), json!(phase));
            map.insert("index".to_string(), json!(index));
            region_fields(&mut map, valid);
            map.insert("materialised".to_string(), json!(materialised));
        }

        // Sidecar events. Keyed by `(phase, index)` like the block events, so a
        // consumer that draws blocks can use them directly — a block that
        // produced a fragment is a block that did something.
        Event::SidecarWritten {
            stream,
            phase,
            index,
            bytes,
            duration_ns,
        } => {
            map.insert("type".to_string(), json!("sidecar_written"));
            map.insert("stream".to_string(), json!(stream));
            map.insert("phase".to_string(), json!(phase));
            map.insert("index".to_string(), json!(index));
            map.insert("bytes".to_string(), json!(bytes));
            map.insert("duration_ns".to_string(), json!(duration_ns));
        }
        Event::SidecarRead {
            stream,
            phase,
            index,
            bytes,
            found,
            duration_ns,
        } => {
            map.insert("type".to_string(), json!("sidecar_read"));
            map.insert("stream".to_string(), json!(stream));
            map.insert("phase".to_string(), json!(phase));
            map.insert("index".to_string(), json!(index));
            map.insert("bytes".to_string(), json!(bytes));
            map.insert("found".to_string(), json!(found));
            map.insert("duration_ns".to_string(), json!(duration_ns));
        }
        Event::SidecarDiscarded {
            stream,
            fragments,
            bytes,
        } => {
            map.insert("type".to_string(), json!("sidecar_discarded"));
            map.insert("stream".to_string(), json!(stream));
            map.insert("fragments".to_string(), json!(fragments));
            map.insert("bytes".to_string(), json!(bytes));
        }

        // Cache and prefetch events. Keyed by `(array, chunk)` rather than by
        // block, so a consumer that only draws blocks skips them on `type` —
        // which is why the type strings are prefixed, and why the schema
        // version did not need to change to carry them.
        Event::SideOutputWritten {
            output,
            phase,
            index,
            region,
            bytes,
        } => {
            map.insert("type".to_string(), json!("side_output_written"));
            map.insert("output".to_string(), json!(output));
            map.insert("phase".to_string(), json!(phase));
            map.insert("index".to_string(), json!(index));
            region_fields(&mut map, region);
            map.insert("bytes".to_string(), json!(bytes));
        }
        Event::CacheHit {
            array,
            chunk,
            tier,
            bytes,
            decode_ns,
        } => {
            map.insert("type".to_string(), json!("cache_hit"));
            map.insert("array".to_string(), json!(array));
            map.insert("chunk".to_string(), json!(chunk));
            map.insert("tier".to_string(), json!(tier.as_str()));
            map.insert("bytes".to_string(), json!(bytes));
            map.insert("decode_ns".to_string(), json!(decode_ns));
        }
        Event::CacheMiss {
            array,
            chunk,
            bytes,
            duration_ns,
        } => {
            map.insert("type".to_string(), json!("cache_miss"));
            map.insert("array".to_string(), json!(array));
            map.insert("chunk".to_string(), json!(chunk));
            map.insert("bytes".to_string(), json!(bytes));
            map.insert("duration_ns".to_string(), json!(duration_ns));
        }
        Event::CacheEvicted {
            array,
            chunk,
            tier,
            bytes,
            for_array,
            prefetched_unused,
        } => {
            map.insert("type".to_string(), json!("cache_evicted"));
            map.insert("array".to_string(), json!(array));
            map.insert("chunk".to_string(), json!(chunk));
            map.insert("tier".to_string(), json!(tier.as_str()));
            map.insert("bytes".to_string(), json!(bytes));
            map.insert("for_array".to_string(), json!(for_array));
            map.insert("prefetched_unused".to_string(), json!(prefetched_unused));
        }
        Event::CacheRefused {
            array,
            chunk,
            bytes,
        } => {
            map.insert("type".to_string(), json!("cache_refused"));
            map.insert("array".to_string(), json!(array));
            map.insert("chunk".to_string(), json!(chunk));
            map.insert("bytes".to_string(), json!(bytes));
        }
        Event::CacheKnownEmpty {
            array,
            chunk,
            bytes,
        } => {
            map.insert("type".to_string(), json!("cache_known_empty"));
            map.insert("array".to_string(), json!(array));
            map.insert("chunk".to_string(), json!(chunk));
            map.insert("bytes".to_string(), json!(bytes));
        }
        Event::PrefetchIssued { array, chunk, rank } => {
            map.insert("type".to_string(), json!("prefetch_issued"));
            map.insert("array".to_string(), json!(array));
            map.insert("chunk".to_string(), json!(chunk));
            map.insert("rank".to_string(), json!(rank));
        }
        Event::PrefetchUsed {
            array,
            chunk,
            waited_ns,
        } => {
            map.insert("type".to_string(), json!("prefetch_used"));
            map.insert("array".to_string(), json!(array));
            map.insert("chunk".to_string(), json!(chunk));
            map.insert("waited_ns".to_string(), json!(waited_ns));
        }
        Event::PrefetchWasted {
            array,
            chunk,
            reason,
        } => {
            map.insert("type".to_string(), json!("prefetch_wasted"));
            map.insert("array".to_string(), json!(array));
            map.insert("chunk".to_string(), json!(chunk));
            map.insert(
                "reason".to_string(),
                json!(match reason {
                    crate::log::PrefetchWaste::Evicted => "evicted",
                    crate::log::PrefetchWaste::Cancelled => "cancelled",
                    crate::log::PrefetchWaste::Refused => "refused",
                }),
            );
        }
    }
    Value::Object(map)
}

// -------------------------------------------------------------- decoding --
//
// The other direction of the same contract. It lives here, beside the encoder,
// because a schema with two owners has no owner: the module header says the
// producer is the only thing allowed to change the format, and a decoder
// maintained somewhere else would agree with it on the day it was written and
// drift at the first new variant. Both consumers of this — the replay view and
// a coordinator merging workers' streams — get the same reader.
//
// Strict about the fields an event declares, forgiving about the event types it
// does not know: an unfamiliar `type` is `None` rather than an error, because a
// viewer that refuses to open a log produced by a newer executor is worse than
// one that draws the rest and says how many lines it skipped.

fn field<'a>(object: &'a Value, name: &str, at: usize) -> Result<&'a Value> {
    object
        .get(name)
        .ok_or_else(|| Error::invalid(format!("event {at}: no {name:?} field")))
}

fn usize_at(object: &Value, name: &str, at: usize) -> Result<usize> {
    field(object, name, at)?
        .as_u64()
        .map(|value| value as usize)
        .ok_or_else(|| Error::invalid(format!("event {at}: {name:?} is not a whole number")))
}

fn u64_at(object: &Value, name: &str, at: usize) -> Result<u64> {
    field(object, name, at)?
        .as_u64()
        .ok_or_else(|| Error::invalid(format!("event {at}: {name:?} is not a whole number")))
}

fn u32_at(object: &Value, name: &str, at: usize) -> Result<u32> {
    Ok(u64_at(object, name, at)? as u32)
}

fn f64_at(object: &Value, name: &str, at: usize) -> Result<f64> {
    field(object, name, at)?
        .as_f64()
        .ok_or_else(|| Error::invalid(format!("event {at}: {name:?} is not a number")))
}

fn bool_at(object: &Value, name: &str, at: usize) -> Result<bool> {
    field(object, name, at)?
        .as_bool()
        .ok_or_else(|| Error::invalid(format!("event {at}: {name:?} is not a boolean")))
}

fn string_at(object: &Value, name: &str, at: usize) -> Result<String> {
    field(object, name, at)?
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| Error::invalid(format!("event {at}: {name:?} is not a string")))
}

fn triple(value: &Value, name: &str, at: usize) -> Result<[usize; 3]> {
    let array = value
        .as_array()
        .ok_or_else(|| Error::invalid(format!("event {at}: {name:?} is not an array")))?;
    if array.len() != 3 {
        return Err(Error::invalid(format!(
            "event {at}: {name:?} has {} entries, and this crate is three-dimensional",
            array.len()
        )));
    }
    let mut out = [0usize; 3];
    for (axis, entry) in array.iter().enumerate() {
        out[axis] = entry.as_u64().ok_or_else(|| {
            Error::invalid(format!(
                "event {at}: {name:?}[{axis}] is not a whole number"
            ))
        })? as usize;
    }
    Ok(out)
}

fn triple_at(object: &Value, name: &str, at: usize) -> Result<[usize; 3]> {
    triple(field(object, name, at)?, name, at)
}

/// `index` is `null` for IO emitted by a bare source or sink, which the schema
/// documents and which is not an error.
fn optional_index(object: &Value, at: usize) -> Result<Option<[usize; 3]>> {
    match object.get("index") {
        None | Some(Value::Null) => Ok(None),
        Some(value) => triple(value, "index", at).map(Some),
    }
}

/// `start` and `shape` are voxel coordinates and belong together; the schema
/// never carries one without the other.
fn region_at(object: &Value, at: usize) -> Result<Region> {
    let start = triple_at(object, "start", at)?;
    let shape = triple_at(object, "shape", at)?;
    Ok(Region::new(&start, &shape))
}

/// The same pair, at whatever rank the producer wrote.
///
/// Separate from [`region_at`] rather than replacing it, deliberately: every
/// region in the volume's own space **is** three-dimensional, and a decoder that
/// silently accepted four entries there would turn a producer bug into a region
/// nobody can index. A side output's region is in a space of its own, where any
/// rank is the honest answer, so it is the one place the triple check is
/// dropped — and the two are then different functions with different contracts
/// rather than one function with a weakened one.
fn any_rank_region_at(object: &Value, at: usize) -> Result<Region> {
    let read = |name: &str| -> Result<Vec<usize>> {
        field(object, name, at)?
            .as_array()
            .ok_or_else(|| Error::invalid(format!("event {at}: {name:?} is not an array")))?
            .iter()
            .map(|entry| {
                entry.as_u64().map(|value| value as usize).ok_or_else(|| {
                    Error::invalid(format!("event {at}: {name:?} holds a non-number"))
                })
            })
            .collect()
    };
    let start = read("start")?;
    let shape = read("shape")?;
    if start.len() != shape.len() {
        return Err(Error::invalid(format!(
            "event {at}: \"start\" has {} entries and \"shape\" has {}",
            start.len(),
            shape.len()
        )));
    }
    Ok(Region::new(&start, &shape))
}

fn slots_at(object: &Value, name: &str, at: usize) -> Result<Vec<usize>> {
    field(object, name, at)?
        .as_array()
        .ok_or_else(|| Error::invalid(format!("event {at}: {name:?} is not an array")))?
        .iter()
        .map(|entry| {
            entry
                .as_u64()
                .map(|value| value as usize)
                .ok_or_else(|| Error::invalid(format!("event {at}: {name:?} holds a non-number")))
        })
        .collect()
}

fn names_at(object: &Value, name: &str, at: usize) -> Result<Vec<String>> {
    field(object, name, at)?
        .as_array()
        .ok_or_else(|| Error::invalid(format!("event {at}: {name:?} is not an array")))?
        .iter()
        .map(|entry| {
            entry
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| Error::invalid(format!("event {at}: {name:?} holds a non-string")))
        })
        .collect()
}

fn tier_at(object: &Value, at: usize) -> Result<Tier> {
    match string_at(object, "tier", at)?.as_str() {
        "decoded" => Ok(Tier::Decoded),
        "encoded" => Ok(Tier::Encoded),
        other => Err(Error::invalid(format!(
            "event {at}: {other:?} is not a cache tier"
        ))),
    }
}

/// One event object, as the encoder wrote it. `None` for a `type` this build
/// does not know — see the section header.
///
/// `at` is only ever used in messages, and is the caller's idea of where the
/// event came from: a position in a document, or a sequence number on a wire.
pub fn event_from_json(object: &Value, at: usize) -> Result<Option<Event>> {
    let kind = string_at(object, "type", at)?;
    Ok(Some(match kind.as_str() {
        "phase_started" => Event::PhaseStarted {
            phase: usize_at(object, "phase", at)?,
        },
        "task_admitted" => Event::TaskAdmitted {
            phase: usize_at(object, "phase", at)?,
            index: triple_at(object, "index", at)?,
        },
        "region_read" => Event::RegionRead {
            source: string_at(object, "source", at)?,
            image: usize_at(object, "level", at)?,
            index: optional_index(object, at)?,
            region: region_at(object, at)?,
            voxels: usize_at(object, "voxels", at)?,
            bytes: u64_at(object, "bytes", at)?,
            chunks: u64_at(object, "chunks", at)?,
            duration_ns: u64_at(object, "duration_ns", at)?,
        },
        "region_written" => Event::RegionWritten {
            sink: string_at(object, "sink", at)?,
            image: usize_at(object, "level", at)?,
            index: optional_index(object, at)?,
            region: region_at(object, at)?,
            voxels: usize_at(object, "voxels", at)?,
            bytes: u64_at(object, "bytes", at)?,
            chunks: u64_at(object, "chunks", at)?,
            duration_ns: u64_at(object, "duration_ns", at)?,
        },
        "materialised" => Event::Materialised {
            phase: usize_at(object, "phase", at)?,
            image: usize_at(object, "level", at)?,
            bytes: u64_at(object, "bytes", at)?,
            intermediate: bool_at(object, "intermediate", at)?,
        },
        "block_read" => Event::BlockRead {
            phase: usize_at(object, "phase", at)?,
            index: triple_at(object, "index", at)?,
            region: region_at(object, at)?,
            voxels: usize_at(object, "voxels", at)?,
            chunks: u64_at(object, "chunks", at)?,
        },
        "op_applied" => Event::OpApplied {
            phase: usize_at(object, "phase", at)?,
            index: triple_at(object, "index", at)?,
            slot: usize_at(object, "slot", at)?,
            op: string_at(object, "op", at)?,
            over: region_at(object, at)?,
            duration_ns: u64_at(object, "duration_ns", at)?,
        },
        // The encoder writes the op names under `ops`, not `names`; the field
        // is named for the consumer, and the decoder is the consumer.
        "block_short_circuited" => Event::BlockShortCircuited {
            phase: usize_at(object, "phase", at)?,
            index: triple_at(object, "index", at)?,
            from: f64_at(object, "from", at)?,
            to: f64_at(object, "to", at)?,
            slots: slots_at(object, "slots", at)?,
            names: names_at(object, "ops", at)?,
        },
        "block_written" => Event::BlockWritten {
            phase: usize_at(object, "phase", at)?,
            index: triple_at(object, "index", at)?,
            valid: region_at(object, at)?,
            materialised: bool_at(object, "materialised", at)?,
        },
        "sidecar_written" => Event::SidecarWritten {
            stream: string_at(object, "stream", at)?,
            phase: usize_at(object, "phase", at)?,
            index: triple_at(object, "index", at)?,
            bytes: u64_at(object, "bytes", at)?,
            duration_ns: u64_at(object, "duration_ns", at)?,
        },
        "sidecar_read" => Event::SidecarRead {
            stream: string_at(object, "stream", at)?,
            phase: usize_at(object, "phase", at)?,
            index: triple_at(object, "index", at)?,
            bytes: u64_at(object, "bytes", at)?,
            found: bool_at(object, "found", at)?,
            duration_ns: u64_at(object, "duration_ns", at)?,
        },
        "sidecar_discarded" => Event::SidecarDiscarded {
            stream: string_at(object, "stream", at)?,
            fragments: usize_at(object, "fragments", at)?,
            bytes: u64_at(object, "bytes", at)?,
        },
        "side_output_written" => Event::SideOutputWritten {
            output: string_at(object, "output", at)?,
            phase: usize_at(object, "phase", at)?,
            index: triple_at(object, "index", at)?,
            region: any_rank_region_at(object, at)?,
            bytes: u64_at(object, "bytes", at)?,
        },
        "cache_hit" => Event::CacheHit {
            array: string_at(object, "array", at)?,
            chunk: u64_at(object, "chunk", at)?,
            tier: tier_at(object, at)?,
            bytes: u64_at(object, "bytes", at)?,
            decode_ns: u64_at(object, "decode_ns", at)?,
        },
        "cache_miss" => Event::CacheMiss {
            array: string_at(object, "array", at)?,
            chunk: u64_at(object, "chunk", at)?,
            bytes: u64_at(object, "bytes", at)?,
            duration_ns: u64_at(object, "duration_ns", at)?,
        },
        "cache_evicted" => Event::CacheEvicted {
            array: string_at(object, "array", at)?,
            chunk: u64_at(object, "chunk", at)?,
            tier: tier_at(object, at)?,
            bytes: u64_at(object, "bytes", at)?,
            for_array: string_at(object, "for_array", at)?,
            prefetched_unused: bool_at(object, "prefetched_unused", at)?,
        },
        "cache_refused" => Event::CacheRefused {
            array: string_at(object, "array", at)?,
            chunk: u64_at(object, "chunk", at)?,
            bytes: u64_at(object, "bytes", at)?,
        },
        "cache_known_empty" => Event::CacheKnownEmpty {
            array: string_at(object, "array", at)?,
            chunk: u64_at(object, "chunk", at)?,
            bytes: u64_at(object, "bytes", at)?,
        },
        "prefetch_issued" => Event::PrefetchIssued {
            array: string_at(object, "array", at)?,
            chunk: u64_at(object, "chunk", at)?,
            rank: u32_at(object, "rank", at)?,
        },
        "prefetch_used" => Event::PrefetchUsed {
            array: string_at(object, "array", at)?,
            chunk: u64_at(object, "chunk", at)?,
            waited_ns: u64_at(object, "waited_ns", at)?,
        },
        "prefetch_wasted" => Event::PrefetchWasted {
            array: string_at(object, "array", at)?,
            chunk: u64_at(object, "chunk", at)?,
            reason: match string_at(object, "reason", at)?.as_str() {
                "evicted" => PrefetchWaste::Evicted,
                "cancelled" => PrefetchWaste::Cancelled,
                "refused" => PrefetchWaste::Refused,
                other => {
                    return Err(Error::invalid(format!(
                        "event {at}: {other:?} is not a prefetch waste reason"
                    )))
                }
            },
        },
        _ => return Ok(None),
    }))
}

/// The whole document. See the module header for the schema.
pub fn order_log_to_json(log: &ExecutionLog, meta: &ExportMeta) -> Value {
    let events = log.events();

    // The block table, built from the events rather than from the
    // decomposition: the point of the export is to describe what *happened*,
    // and a table taken from the plan would agree with the plan by
    // construction even when the run did not.
    struct BlockEntry {
        read: Option<Region>,
        valid: Option<Region>,
        phases: Vec<usize>,
    }
    let mut blocks: BTreeMap<[usize; 3], BlockEntry> = BTreeMap::new();
    fn note(
        blocks: &mut BTreeMap<[usize; 3], BlockEntry>,
        index: [usize; 3],
        phase: usize,
    ) -> &mut BlockEntry {
        let entry = blocks.entry(index).or_insert_with(|| BlockEntry {
            read: None,
            valid: None,
            phases: Vec::new(),
        });
        if !entry.phases.contains(&phase) {
            entry.phases.push(phase);
        }
        entry
    }
    for event in &events {
        match event {
            Event::BlockRead {
                phase,
                index,
                region,
                ..
            } => {
                let entry = note(&mut blocks, *index, *phase);
                if entry.read.is_none() {
                    entry.read = Some(region.clone());
                }
            }
            Event::BlockWritten {
                phase,
                index,
                valid,
                ..
            } => {
                let entry = note(&mut blocks, *index, *phase);
                if entry.valid.is_none() {
                    entry.valid = Some(valid.clone());
                }
            }
            Event::OpApplied { phase, index, .. }
            | Event::BlockShortCircuited { phase, index, .. }
            | Event::TaskAdmitted { phase, index } => {
                note(&mut blocks, *index, *phase);
            }
            _ => {}
        }
    }

    let mut grid = [0usize; 3];
    for index in blocks.keys() {
        for axis in 0..3 {
            grid[axis] = grid[axis].max(index[axis] + 1);
        }
    }

    let block_table: Vec<Value> = blocks
        .iter()
        .map(|(index, entry)| {
            let mut map = Map::new();
            map.insert("index".to_string(), json!(index));
            let mut phases = entry.phases.clone();
            phases.sort_unstable();
            map.insert("phases".to_string(), json!(phases));
            map.insert(
                "read".to_string(),
                match &entry.read {
                    Some(region) => object(vec![
                        ("start", json!(region.start)),
                        ("shape", json!(region.shape)),
                    ]),
                    None => Value::Null,
                },
            );
            map.insert(
                "valid".to_string(),
                match &entry.valid {
                    Some(region) => object(vec![
                        ("start", json!(region.start)),
                        ("shape", json!(region.shape)),
                    ]),
                    None => Value::Null,
                },
            );
            Value::Object(map)
        })
        .collect();

    let ops: Vec<Value> = meta
        .ops
        .iter()
        .map(|(slot, name)| object(vec![("slot", json!(slot)), ("name", json!(name))]))
        .collect();

    json!({
        "schema": "clearmap-rs.block_ops.order_log",
        "version": 1,
        "strategy": meta.strategy,
        "volume": meta.volume,
        "grid": grid,
        "phases": meta.phases,
        "ops": ops,
        "blocks": block_table,
        "events": events
            .iter()
            .enumerate()
            .map(|(seq, event)| event_to_json(seq, event))
            .collect::<Vec<_>>(),
    })
}

/// Write the document to `path`, pretty-printed.
///
/// Pretty rather than compact because these are read by people as often as by
/// programs, and a 60 000-task log is a few tens of megabytes either way.
pub fn write_order_log_json(
    log: &ExecutionLog,
    meta: &ExportMeta,
    path: impl AsRef<Path>,
) -> Result<()> {
    let document = order_log_to_json(log, meta);
    let text = serde_json::to_string_pretty(&document)
        .map_err(|err| Error::InvalidArgument(format!("serialising order log: {err}")))?;
    std::fs::write(path.as_ref(), text).map_err(|err| {
        Error::InvalidArgument(format!(
            "writing order log to {}: {err}",
            path.as_ref().display()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::listener::EventListener;

    fn sample_log() -> ExecutionLog {
        let log = ExecutionLog::new();
        log.on_event(&Event::PhaseStarted { phase: 0 });
        for block in 0..2usize {
            let index = [block, 0, 0];
            let read = Region::new(&[block * 8, 0, 0], &[10, 4, 4]);
            let valid = Region::new(&[block * 8, 0, 0], &[8, 4, 4]);
            log.on_event(&Event::TaskAdmitted { phase: 0, index });
            log.on_event(&Event::RegionRead {
                source: "level 0".to_string(),
                image: 0,
                index: Some(index),
                region: read.clone(),
                voxels: read.voxels(),
                bytes: read.voxels() as u64 * 8,
                chunks: 2,
                duration_ns: 42,
            });
            log.on_event(&Event::BlockRead {
                phase: 0,
                index,
                region: read.clone(),
                voxels: read.voxels(),
                chunks: 2,
            });
            for (slot, name) in [(0usize, "median"), (1, "threshold")] {
                log.on_event(&Event::OpApplied {
                    phase: 0,
                    index,
                    slot,
                    op: name.to_string(),
                    over: read.clone(),
                    duration_ns: 100,
                });
            }
            log.on_event(&Event::BlockWritten {
                phase: 0,
                index,
                valid,
                materialised: false,
            });
        }
        log.on_event(&Event::Materialised {
            phase: 0,
            image: 1,
            bytes: 1024,
            intermediate: false,
        });
        log
    }

    #[test]
    fn the_document_carries_the_contract_the_header_documents() {
        let log = sample_log();
        let meta = ExportMeta::new("trivial", [16, 4, 4], 1).with_ops(vec![
            (0, "median".to_string()),
            (1, "threshold".to_string()),
        ]);
        let document = order_log_to_json(&log, &meta);

        assert_eq!(document["schema"], "clearmap-rs.block_ops.order_log");
        assert_eq!(document["version"], 1);
        assert_eq!(document["grid"], json!([2, 1, 1]));
        assert_eq!(document["volume"], json!([16, 4, 4]));
        assert_eq!(document["blocks"].as_array().unwrap().len(), 2);
        assert_eq!(document["blocks"][1]["valid"]["start"], json!([8, 0, 0]));
        assert_eq!(document["blocks"][1]["read"]["shape"], json!([10, 4, 4]));
        assert_eq!(document["blocks"][1]["phases"], json!([0]));

        // seq is dense and matches position, which the consumer relies on
        let events = document["events"].as_array().unwrap();
        assert_eq!(events.len(), log.len());
        for (position, event) in events.iter().enumerate() {
            assert_eq!(event["seq"], position);
            assert!(event["type"].is_string());
        }
        assert_eq!(events[0]["type"], "phase_started");
        assert_eq!(events[1]["type"], "task_admitted");
        assert_eq!(events[2]["type"], "region_read");
        assert_eq!(events[2]["duration_ns"], 42);
        assert_eq!(events.last().unwrap()["type"], "materialised");
    }

    /// Every variant must serialise, or a consumer meets an event it cannot
    /// name. A `match` on the enum here would compile whatever fields were
    /// added; asserting the type strings is what catches a silently dropped
    /// variant.
    #[test]
    fn every_event_variant_has_a_type_string() {
        let region = Region::new(&[0, 0, 0], &[2, 2, 2]);
        let all = vec![
            Event::PhaseStarted { phase: 0 },
            Event::TaskAdmitted {
                phase: 0,
                index: [0, 0, 0],
            },
            Event::RegionRead {
                source: "s".to_string(),
                image: 0,
                index: None,
                region: region.clone(),
                voxels: 8,
                bytes: 64,
                chunks: 1,
                duration_ns: 1,
            },
            Event::RegionWritten {
                sink: "s".to_string(),
                image: 1,
                index: None,
                region: region.clone(),
                voxels: 8,
                bytes: 64,
                chunks: 1,
                duration_ns: 1,
            },
            Event::Materialised {
                phase: 0,
                image: 1,
                bytes: 64,
                intermediate: true,
            },
            Event::BlockRead {
                phase: 0,
                index: [0, 0, 0],
                region: region.clone(),
                voxels: 8,
                chunks: 1,
            },
            Event::OpApplied {
                phase: 0,
                index: [0, 0, 0],
                slot: 0,
                op: "op".to_string(),
                over: region.clone(),
                duration_ns: 1,
            },
            Event::BlockShortCircuited {
                phase: 0,
                index: [0, 0, 0],
                from: 0.0,
                to: 1.0,
                slots: vec![0],
                names: vec!["op".to_string()],
            },
            Event::BlockWritten {
                phase: 0,
                index: [0, 0, 0],
                valid: region,
                materialised: false,
            },
        ];
        let names: Vec<String> = all
            .iter()
            .enumerate()
            .map(|(seq, event)| {
                event_to_json(seq, event)["type"]
                    .as_str()
                    .unwrap()
                    .to_string()
            })
            .collect();
        assert_eq!(
            names,
            vec![
                "phase_started",
                "task_admitted",
                "region_read",
                "region_written",
                "materialised",
                "block_read",
                "op_applied",
                "block_short_circuited",
                "block_written",
            ]
        );
        // `index: null` is part of the contract, not an accident
        assert!(event_to_json(0, &all[2])["index"].is_null());
    }

    /// The sidecar events go over a wire — a distributed worker sends them to
    /// its coordinator one at a time — so the encoder and the decoder have to
    /// agree about them, not merely each be self-consistent.
    #[test]
    fn the_sidecar_events_survive_the_round_trip_that_carries_them_between_processes() {
        let all = vec![
            Event::SidecarWritten {
                stream: "fragments".to_string(),
                phase: 1,
                index: [3, 0, 2],
                bytes: 32,
                duration_ns: 17,
            },
            Event::SidecarRead {
                stream: "fragments".to_string(),
                phase: 1,
                index: [3, 0, 2],
                bytes: 0,
                found: false,
                duration_ns: 4,
            },
            Event::SidecarDiscarded {
                stream: "fragments".to_string(),
                fragments: 16,
                bytes: 512,
            },
        ];
        let names: Vec<String> = all
            .iter()
            .map(|event| event_json(event)["type"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(
            names,
            vec!["sidecar_written", "sidecar_read", "sidecar_discarded"]
        );
        for event in &all {
            let encoded = event_json(event);
            assert_eq!(
                event_from_json(&encoded, 0).unwrap().as_ref(),
                Some(event),
                "{encoded}"
            );
        }
        // `found` is carried rather than inferred from `bytes`, because a
        // zero-length fragment and an absent one are different answers.
        assert_eq!(event_json(&all[1])["found"], serde_json::json!(false));
    }

    /// A side output's region is in **its own** space, which may be of a rank
    /// the volume does not have. The encoder writes `start`/`shape` of whatever
    /// length the region has and the decoder reads them back, so a rank-4 slot
    /// survives a crossing that a triple would have truncated.
    #[test]
    fn a_side_output_write_survives_the_round_trip_at_a_rank_the_volume_lacks() {
        let event = Event::SideOutputWritten {
            output: "labels".to_string(),
            phase: 0,
            index: [2, 1, 0],
            region: Region::new(&[8, 0, 0, 0], &[4, 6, 6, 3]),
            bytes: 4 * 6 * 6 * 3 * 4,
        };
        let encoded = event_json(&event);
        assert_eq!(encoded["type"], serde_json::json!("side_output_written"));
        assert_eq!(encoded["shape"], serde_json::json!([4, 6, 6, 3]));
        assert_eq!(
            event_from_json(&encoded, 0).unwrap().as_ref(),
            Some(&event),
            "{encoded}"
        );
    }

    #[test]
    fn a_written_document_round_trips_through_the_file_system() {
        let dir = std::env::temp_dir().join(format!("block_ops_export_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("order_log.json");
        let meta = ExportMeta::new("trivial", [16, 4, 4], 1);
        write_order_log_json(&sample_log(), &meta, &path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let parsed: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed["events"].as_array().unwrap().len(), 14);
        std::fs::remove_dir_all(&dir).ok();
    }
}
