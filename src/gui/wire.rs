// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The payloads, as JSON.
//
// One encoder for both modes, which is the mechanical half of the claim that
// the browser cannot tell a replay from a live run: the two sources produce the
// same Rust types and those types have exactly one way of becoming bytes. A
// second encoder is how "same API shape" would quietly stop being true.
//
// This is **not** the order log's schema. That one — see `export`'s header — is
// a cross-language file format with a version and a compatibility promise, and
// it is the thing to read if you are analysing a run. What is here is a view
// model for one browser: it exists to be small over a poll, it names a version
// so a stale page in a browser cache can say so rather than misdraw, and it is
// free to change whenever the page it feeds does.
//
// Field names are chosen to match the order log's wherever they mean the same
// thing (`index`, `phase`, `slot`, `op`, `seq`), so that somebody reading both
// documents is not made to learn two vocabularies for one idea.

use serde_json::{json, Value};

use super::source::{Meta, State, TimelinePage};

/// Bumped when a field changes meaning or disappears. The page checks it and
/// says "reload" rather than drawing something wrong.
pub const WIRE_VERSION: u64 = 1;

pub fn meta_to_json(meta: &Meta) -> Value {
    json!({
        "wire": WIRE_VERSION,
        "mode": meta.mode.as_str(),
        "controllable": meta.controllable,
        "strategy": meta.strategy,
        "volume": meta.volume,
        "grid": meta.grid,
        "phases": meta.phases,
        "ops": meta.ops
            .iter()
            .map(|(slot, name)| json!({"slot": slot, "name": name}))
            .collect::<Vec<_>>(),
        "total_events": meta.total_events,
    })
}

/// The per-block state, as **parallel arrays**.
///
/// One array per field rather than an array of objects. For a 60 000-block grid
/// the object form is around 6 MB per poll and this is around 1 MB, almost all
/// of the saving being repeated key names; at 5 Hz that is the difference
/// between a poll a port forward carries comfortably and one it does not. The
/// cost is that the page must zip the arrays, which is four lines and is worth
/// it. `op` is sent once in `meta.ops` and referred to here by slot, for the
/// same reason.
///
/// `index` is flattened — three entries per block, axis 0 first — rather than
/// an array of triples, which is the same argument applied once more.
pub fn state_to_json(state: &State) -> Value {
    let mut index = Vec::with_capacity(state.blocks.len() * 3);
    let mut phase = Vec::with_capacity(state.blocks.len());
    let mut kind = Vec::with_capacity(state.blocks.len());
    let mut slot = Vec::with_capacity(state.blocks.len());
    let mut seq = Vec::with_capacity(state.blocks.len());
    for block in &state.blocks {
        index.extend_from_slice(&block.index);
        phase.push(block.phase as u64);
        kind.push(block.kind.as_str());
        // -1 for "no op has landed yet", which is a real state a block spends
        // its first moments in and must be drawn differently from slot 0.
        slot.push(match block.slot {
            Some(slot) => slot as i64,
            None => -1,
        });
        seq.push(block.seq);
    }
    json!({
        "wire": WIRE_VERSION,
        "cursor": state.cursor,
        "total": state.total,
        "seq": state.seq,
        "running": state.running,
        "blocks": state.blocks.len(),
        "index": index,
        "phase": phase,
        "kind": kind,
        "slot": slot,
        "block_seq": seq,
    })
}

pub fn timeline_to_json(page: &TimelinePage) -> Value {
    json!({
        "wire": WIRE_VERSION,
        "since": page.since,
        "next": page.next,
        "available": page.available,
        "events": page.events
            .iter()
            .map(|event| json!({
                "seq": event.seq,
                "type": event.kind,
                "phase": event.phase,
                "index": event.index,
                "slot": event.slot,
                "op": event.op,
                "duration_ns": event.duration_ns,
            }))
            .collect::<Vec<_>>(),
    })
}
