// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// A small HTTP server and a browser view for watching the block scheduler.
//
// Two things a person wants to see, and they are the same thing
// -------------------------------------------------------------
// * **Replay.** A run finished, its order log was exported, and somebody wants
//   to step through what the scheduler did.
// * **Live.** A run is in flight on a compute node and somebody wants to know
//   how far it has got.
//
// These are usually built as two programs, and they diverge — the live view
// grows a feature the replay view never gets, the replay view learns a file
// format the live view cannot produce, and eventually they disagree about what
// a block's state even means. This module refuses the split at the source: a
// replay is *a recorded stream being re-emitted*, so it is fed through the same
// listener the live run feeds ([`LatestOpPerChunk`]), answers the same
// [`ProgressSource`] trait, and is served over the same endpoints. The browser
// asks the same four URLs either way and does not know which it is looking at.
//
// The only thing the two do not share is *who moves the playhead*. A live run
// moves it by running; a replay moves it because the server advances it with
// the clock. One boolean in `/api/meta` — `controllable` — says whether the
// transport controls do anything, and that is a statement about a capability,
// not about a mode.
//
// [`LatestOpPerChunk`]: crate::listener::LatestOpPerChunk
//
// Why polling, and why not WebSockets
// -----------------------------------
// A view of a run on a compute node is normally reached through an SSH port
// forward, sometimes through a second hop or a proxy. Plain HTTP request /
// response survives all of that; WebSocket upgrades are the classic thing that
// works on a laptop and fails on the cluster, and the failure is opaque when it
// happens. A progress view needs a few updates a second, which polling supplies
// with room to spare — the per-block state is one atomic word per block and a
// poll of a 60 000-block run is well under a millisecond of work. There is
// nothing here that would benefit from a persistent socket except elegance, and
// elegance that fails behind a jump host is not elegance.
//
// If polling is ever shown to be inadequate — a grid so large the JSON is the
// bottleneck, say — the answer is server-sent events, which are still ordinary
// HTTP responses and still survive a forward. That would be a reason. "Sockets
// are more modern" is not one.
//
// What this must never do
// -----------------------
// **Attaching a viewer must not change the run.** The same rule `listener`
// states for observation, extended to observation over a wire:
//
// * the server runs on its own threads and never touches the executor;
// * a poll reads `LatestOpPerChunk::snapshot`, which takes the same *shared*
//   guard the write path takes and therefore excludes nobody;
// * the timeline, when one is attached, is an ordinary `OrderLog` listener —
//   the same cost the executor's own built-in log already pays;
// * nothing in this module can return an error into the executor, because
//   nothing in this module is called by it.
//
// The measurement behind that claim is in `gui::tests`: a run with a client
// polling as hard as it can produces the same statistics and the same wall time
// as a run with no server at all.
//
// Layout
// ------
// | module | what it owns |
// |---|---|
// | `source` | [`ProgressSource`] — the one shape both modes answer — and the payload types. |
// | `live` | [`LiveSource`]: a running execution, seen through its listeners. |
// | `replay` | [`ReplaySource`]: an exported order log, decoded back into `Event`s and re-emitted against the clock. |
// | `wire` | The payloads as JSON. One encoder, both modes. |
// | `server` | The HTTP surface, the asset directory, and the bind policy. |
//
// The browser half is a separate crate, `webui/`, compiled to WebAssembly. It
// is not a dependency of this one and is not built by `cargo`; see
// [`server::AssetDir`] for how the built files are found and what happens when
// they are not there.

pub mod live;
pub mod replay;
pub mod server;
pub mod source;
pub mod wire;

pub use live::{LiveSource, TimelineListener, DEFAULT_MIN_SNAPSHOT_INTERVAL};
pub use replay::{decode_order_log, read_order_log, DecodedLog, ReplaySource};
pub use server::{check_bind, find_assets, serve, Options, ServerHandle, DEFAULT_PORT};
pub use source::{
    project, Control, Meta, Mode, ProgressSource, State, TimelineEvent, TimelinePage, Window,
};

#[cfg(test)]
mod tests;
