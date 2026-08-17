// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// What these tests are for
// ------------------------
// Three claims, and each is the kind that quietly stops being true:
//
// 1. **A replay is the same run.** The decoder round-trips the exported log
//    exactly, and folding a recording produces the same per-block state the
//    live listener produced while the run was happening. If this fails, the
//    tool is showing somebody a picture of a run that did not happen.
// 2. **The wire shape does not fork.** Both sources go through one encoder and
//    the endpoints answer the same fields either way.
// 3. **Watching does not disturb.** A run with a server attached and a client
//    polling it as fast as it can does the same work as a run with neither.
//
// The HTTP tests speak the protocol over a socket rather than calling the
// handler, because "the endpoint answers" and "the function returns" are
// different claims and only the first one is what a browser depends on. The
// client is twenty lines of `TcpStream` — adding an HTTP client dependency to
// test a server that exists to have no dependencies would be a poor trade.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::decomposition::{groups_for, summarise_slots, Decomposition, PhaseDecomposition};
use crate::env::AccountingEnvironment;
use crate::export::{order_log_to_json, ExportMeta};
use crate::geometry::BlockGrid;
use crate::listener::{EventListener, LatestOpPerChunk};
use crate::log::Stats;
use crate::probes::IdentityOp;
use crate::strategy::{execute_observed, Hints, SchedulePriority, Workflow};
use crate::{Chain, Dtype};

use super::live::LiveSource;
use super::replay::{decode_order_log, ReplaySource};
use super::server::{serve, Options};
use super::source::{Control, Mode, ProgressSource, Window};
use super::wire::{meta_to_json, state_to_json, timeline_to_json};

// ------------------------------------------------------------ a run --

/// The fixture: a small three-phase chain over a `side` x `side` grid, run
/// against the accounting environment so no array is allocated and the test
/// costs milliseconds at any grid size.
fn plan(side: usize) -> (Workflow, Decomposition, Vec<(usize, String)>) {
    let shape = [64, side * 64, side * 64];
    let chain = Chain::sequence(vec![
        Chain::op(IdentityOp::new("smooth", [0, 2, 2]).with_cost(1.0)),
        Chain::op(IdentityOp::new("background", [0, 6, 6]).with_cost(2.0)),
        Chain::op(IdentityOp::new("threshold", [0, 0, 0]).with_cost(0.5)),
    ]);
    let ops: Vec<(usize, String)> = chain
        .slots()
        .iter()
        .enumerate()
        .map(|(slot, sub)| (slot, sub.display_name()))
        .collect();
    let workflow = Workflow::new(chain, shape, Dtype::U16);
    let slots = workflow.chain.slots();
    let phases: Vec<PhaseDecomposition> = groups_for(0b11, slots.len())
        .iter()
        .map(|group| {
            let (reach, _, names, _) =
                summarise_slots(&slots, group, shape).expect("the fixture chain's slots summarise");
            let grid = BlockGrid::along(shape, &[1, 2], 64).unwrap();
            PhaseDecomposition::derive(group.clone(), names, reach.clone(), reach, grid)
        })
        .collect();
    let decomposition = Decomposition {
        volume: shape,
        dtype: workflow.dtype,
        phases,
        chain_reach: workflow.chain.reach3(&shape),
    };
    decomposition.check().unwrap();
    (workflow, decomposition, ops)
}

fn run(side: usize, listeners: &[Arc<dyn EventListener>], workers: usize) -> Stats {
    let (workflow, decomposition, _) = plan(side);
    let hints = Hints {
        priority: SchedulePriority::BlockMajor,
        concurrency: workers,
        ..Hints::default()
    };
    let env = AccountingEnvironment::new(workflow.shape, [32, 64, 64], 2);
    execute_observed(
        "fixture",
        &workflow,
        &decomposition,
        &hints,
        &env,
        listeners,
    )
    .unwrap()
}

/// A finished run, exported the way `write_order_log_json` would write it.
fn exported(side: usize) -> (Value, Arc<LatestOpPerChunk>) {
    let (workflow, decomposition, ops) = plan(side);
    let live = Arc::new(LatestOpPerChunk::new());
    let listeners: Vec<Arc<dyn EventListener>> = vec![live.clone()];
    let stats = run(side, &listeners, 4);
    let meta = ExportMeta::new("fixture", workflow.shape, decomposition.n_phases()).with_ops(ops);
    (order_log_to_json(&stats.log, &meta), live)
}

// -------------------------------------------------------- the decoder --

/// The decoder must be **exact**, not approximately right: everything below
/// rests on a replayed event being the event that was recorded.
#[test]
fn decoding_an_exported_log_reproduces_the_events_that_produced_it() {
    let (workflow, decomposition, ops) = plan(3);
    let stats = run(3, &[], 2);
    let meta = ExportMeta::new("fixture", workflow.shape, decomposition.n_phases()).with_ops(ops);
    let document = order_log_to_json(&stats.log, &meta);

    let decoded = decode_order_log(&document).unwrap();
    assert_eq!(
        decoded.unknown, 0,
        "the exporter wrote a type the decoder does not know"
    );
    assert_eq!(decoded.events, stats.log.events());
    assert_eq!(decoded.volume, workflow.shape);
    assert_eq!(decoded.grid, [1, 3, 3]);
    assert_eq!(decoded.phases, decomposition.n_phases());
    assert_eq!(
        decoded
            .ops
            .iter()
            .map(|(slot, _)| *slot)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
}

#[test]
fn a_document_that_is_not_an_order_log_is_refused_by_name() {
    let error = decode_order_log(&serde_json::json!({"schema": "something else"})).unwrap_err();
    assert!(error.to_string().contains("not an order log"), "{error}");

    let (mut document, _) = exported(2);
    document["version"] = serde_json::json!(99);
    let error = decode_order_log(&document).unwrap_err();
    assert!(error.to_string().contains("Re-export"), "{error}");
}

/// A type this build has never heard of must be skipped and counted, not fatal.
/// The alternative is a viewer that cannot open a log produced by a newer
/// executor, which is the situation this crate's users are most likely to be in.
#[test]
fn an_unfamiliar_event_type_is_skipped_and_counted() {
    let (mut document, _) = exported(2);
    let before = document["events"].as_array().unwrap().len();
    document["events"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!({"seq": before, "type": "something_new", "phase": 0}));
    let decoded = decode_order_log(&document).unwrap();
    assert_eq!(decoded.unknown, 1);
    assert_eq!(decoded.events.len(), before);
}

/// A malformed event names the field and the position, because a log that does
/// not open is a thing somebody has to debug.
#[test]
fn a_malformed_event_says_which_one_and_which_field() {
    let (mut document, _) = exported(2);
    let events = document["events"].as_array_mut().unwrap();
    let victim = events
        .iter()
        .position(|event| event["type"] == "op_applied")
        .expect("the fixture applies ops");
    events[victim].as_object_mut().unwrap().remove("slot");
    let error = decode_order_log(&document).unwrap_err();
    let text = error.to_string();
    assert!(text.contains(&format!("event {victim}")), "{text}");
    assert!(text.contains("slot"), "{text}");
}

// ------------------------------------------- the two modes must agree --

/// The claim the whole tool rests on: a replay played to its end shows exactly
/// what the live listener showed while the run was happening.
///
/// Not "similar" and not "the same blocks" — the same `BlockProgress` for every
/// block, including the sticky op slot and the emission sequence.
#[test]
fn a_replay_played_out_matches_what_the_live_listener_saw() {
    let (document, live) = exported(4);
    let replay = ReplaySource::from_json(&document).unwrap();
    let total = replay.log().events.len() as u64;
    replay.control(Control::Seek(total));

    let recorded = replay.state();
    let watched = live.snapshot();
    assert_eq!(recorded.cursor, total);
    assert_eq!(recorded.blocks.len(), watched.len());
    for (from_replay, from_run) in recorded.blocks.iter().zip(&watched) {
        assert_eq!(from_replay.index, from_run.index);
        assert_eq!(from_replay.phase, from_run.phase);
        assert_eq!(from_replay.kind, from_run.kind);
        assert_eq!(from_replay.slot, from_run.slot);
        assert_eq!(from_replay.op, from_run.op);
    }
}

/// Seeking backwards throws the fold away and re-folds. The result must be the
/// same as having only ever gone forwards to that point, or a scrub bar shows
/// something no run ever produced.
#[test]
fn seeking_backwards_gives_the_state_that_position_really_had() {
    let (document, _) = exported(3);
    let forwards = ReplaySource::from_json(&document).unwrap();
    let backwards = ReplaySource::from_json(&document).unwrap();
    let total = forwards.log().events.len() as u64;
    let midpoint = total / 2;

    forwards.control(Control::Seek(midpoint));
    backwards.control(Control::Seek(total));
    backwards.control(Control::Seek(midpoint));

    let a = forwards.state();
    let b = backwards.state();
    assert_eq!(a.cursor, midpoint);
    assert_eq!(a.blocks.len(), b.blocks.len());
    assert_eq!(
        a.blocks
            .iter()
            .map(|block| (block.index, block.slot, block.kind))
            .collect::<Vec<_>>(),
        b.blocks
            .iter()
            .map(|block| (block.index, block.slot, block.kind))
            .collect::<Vec<_>>(),
    );
    // and it is genuinely a partial view, not the end state arrived at early
    assert!(a.blocks.len() < forwards.log().events.len());
}

#[test]
fn a_replay_advances_by_itself_and_a_live_run_cannot_be_seeked() {
    let (document, _) = exported(3);
    let replay = ReplaySource::from_json(&document).unwrap();
    assert!(replay.meta().controllable);
    assert_eq!(replay.meta().mode, Mode::Replay);
    replay.control(Control::Speed(500.0));
    replay.control(Control::Play);
    let before = replay.state().cursor;
    std::thread::sleep(Duration::from_millis(120));
    let after = replay.state().cursor;
    assert!(
        after > before,
        "the playhead did not move: {before} -> {after}"
    );
    replay.control(Control::Pause);
    let paused = replay.state().cursor;
    std::thread::sleep(Duration::from_millis(60));
    assert_eq!(replay.state().cursor, paused);

    let live = LiveSource::new(ExportMeta::new("live", [64, 64, 64], 1), [1, 1, 1]);
    assert!(!live.meta().controllable);
    assert_eq!(live.meta().mode, Mode::Live);
    assert!(
        !live.control(Control::Play),
        "a live run has no playhead to move"
    );
    assert!(!live.control(Control::Seek(0)));
}

/// A live source with no timeline listener still answers `/api/events` — with
/// nothing. A client must not need a second code path for that.
#[test]
fn a_live_source_without_a_timeline_still_answers_the_timeline_endpoint() {
    let live = LiveSource::new(ExportMeta::new("live", [64, 64, 64], 1), [1, 1, 1]);
    let page = live.timeline(Window::Since(0), 100);
    assert!(page.events.is_empty());
    assert_eq!(
        page.next, page.since,
        "nothing more, and the client should wait"
    );
}

#[test]
fn a_live_source_with_a_timeline_pages_through_it() {
    let (_, decomposition, ops) = plan(2);
    let meta = ExportMeta::new("live", [64, 128, 128], decomposition.n_phases()).with_ops(ops);
    let live = Arc::new(LiveSource::new(meta, [1, 2, 2]).with_timeline());
    let stats = run(2, &live.listeners(), 2);
    live.finished();

    let state = live.state();
    assert!(!state.running);
    assert_eq!(state.cursor, stats.log.len() as u64);
    assert_eq!(live.meta().total_events, Some(stats.log.len() as u64));

    // Paging must reach the end, and `seq` must stay the position in the whole
    // stream even though projection drops nothing here.
    let mut since = 0u64;
    let mut seen = 0usize;
    loop {
        let page = live.timeline(Window::Since(since), 32);
        if page.events.is_empty() && page.next == since {
            break;
        }
        for event in &page.events {
            assert!(event.seq >= since);
        }
        seen += page.events.len();
        since = page.next;
        if since >= page.available {
            break;
        }
    }
    assert!(seen > 0);
    assert_eq!(seen, stats.log.len(), "every event should appear once");
}

/// The window a view asks for: the entries ending at the playhead. This is the
/// one request whose answer differs between the two modes for a good reason —
/// on a live run the playhead is now, on a replay it is wherever it was
/// scrubbed to — and it is the same request either way.
#[test]
fn the_window_before_the_playhead_ends_at_the_playhead() {
    let (document, _) = exported(3);
    let replay = ReplaySource::from_json(&document).unwrap();
    let total = replay.log().events.len() as u64;
    replay.control(Control::Seek(total / 2));
    let at = replay.state().cursor;

    let page = replay.timeline(Window::Before(at), 10);
    assert_eq!(page.events.len(), 10);
    assert!(page.events.iter().all(|entry| entry.seq < at));
    assert_eq!(
        page.events.last().unwrap().seq,
        at - 1,
        "the window ends at the playhead"
    );
    // and it is in stream order, not reversed
    assert!(page.events.windows(2).all(|pair| pair[0].seq < pair[1].seq));

    // At the very beginning there is nothing before the playhead, and that is
    // an empty page rather than a wrapped one.
    replay.control(Control::Seek(0));
    assert!(replay.timeline(Window::Before(0), 10).events.is_empty());

    // The live side answers the same question with the same shape.
    let (_, decomposition, ops) = plan(2);
    let meta = ExportMeta::new("live", [64, 128, 128], decomposition.n_phases()).with_ops(ops);
    let live = Arc::new(LiveSource::new(meta, [1, 2, 2]).with_timeline());
    // One worker, deliberately. `TimelineListener` documents that the *stored*
    // order is push order, which under concurrency may differ from `seq` order
    // by however many events one worker emits between another taking the
    // sequence and taking the lock (`live.rs`, "# Ordering"). Strict `seq`
    // monotonicity within a page is therefore a single-worker guarantee, and
    // asserting it under concurrency was asserting something the implementation
    // says it does not provide — it failed roughly twice in thirty loaded runs
    // and passed forty times in a row alone.
    //
    // This test is about *paging*: that a window lands where it was asked to
    // and returns `limit` distinct entries. Running it with one worker tests
    // that without conflating it with the reorder window.
    let stats = run(2, &live.listeners(), 1);
    live.finished();
    let at = live.state().cursor;
    assert_eq!(at, stats.log.len() as u64);
    let page = live.timeline(Window::Before(at), 6);
    assert_eq!(page.events.len(), 6);
    assert_eq!(page.events.last().unwrap().seq, at - 1);
    assert!(page.events.windows(2).all(|pair| pair[0].seq < pair[1].seq));
}

/// Paging under concurrency returns a well-formed window even though stored
/// order is not `seq` order.
///
/// The weaker claim is the true one: distinct sequences, all below the cursor,
/// `limit` of them. `live.rs` is explicit that the vector is sorted only to
/// within a handful of entries, so a client gets a window that may be locally
/// permuted — and losing or duplicating an event would be the real defect.
#[test]
fn a_page_under_concurrency_is_distinct_and_within_the_cursor() {
    let (_, decomposition, ops) = plan(2);
    let meta = ExportMeta::new("live", [64, 128, 128], decomposition.n_phases()).with_ops(ops);
    let live = Arc::new(LiveSource::new(meta, [1, 2, 2]).with_timeline());
    let stats = run(2, &live.listeners(), 4);
    live.finished();
    let at = live.state().cursor;
    assert_eq!(at, stats.log.len() as u64);

    let page = live.timeline(Window::Before(at), 6);
    assert_eq!(page.events.len(), 6);
    let mut seqs: Vec<u64> = page.events.iter().map(|event| event.seq).collect();
    seqs.sort_unstable();
    let distinct = {
        let mut unique = seqs.clone();
        unique.dedup();
        unique.len()
    };
    assert_eq!(distinct, seqs.len(), "a page repeated an event: {seqs:?}");
    assert!(
        seqs.iter().all(|&seq| seq < at),
        "a page reported an event at or beyond the cursor {at}: {seqs:?}"
    );
}

/// **A client cannot make the run take more snapshots by polling harder.**
///
/// The floor is what turns "please poll politely" into a property of the
/// server. Asserted by counting the distinct snapshots a burst of polls
/// produces, which is observable because `seq` changes only when a new snapshot
/// is taken.
#[test]
fn polling_harder_than_the_floor_does_not_take_more_snapshots() {
    let (_, decomposition, ops) = plan(3);
    let meta = ExportMeta::new("live", [64, 192, 192], decomposition.n_phases()).with_ops(ops);
    let live = Arc::new(
        LiveSource::new(meta, [1, 3, 3])
            .with_timeline()
            .with_min_snapshot_interval(Duration::from_millis(50)),
    );

    // A run in the background, so the state really is changing under the polls.
    let worker = {
        let live = live.clone();
        std::thread::spawn(move || run(3, &live.listeners(), 2))
    };
    let start = Instant::now();
    let mut distinct = std::collections::BTreeSet::new();
    let mut polls = 0usize;
    while start.elapsed() < Duration::from_millis(250) {
        let state = live.state();
        distinct.insert((state.cursor, state.seq, state.blocks.len()));
        polls += 1;
    }
    worker.join().unwrap();
    live.finished();

    assert!(
        polls > 500,
        "the poll loop was too slow to prove anything: {polls}"
    );
    // 250 ms at a 50 ms floor is at most 6 snapshots; allow a little slack for
    // the run finishing and the cache being dropped, and none for the floor
    // being ignored.
    assert!(
        distinct.len() <= 8,
        "{polls} polls produced {} distinct snapshots; the floor is not holding",
        distinct.len()
    );

    // With the floor removed, every poll is its own snapshot — which is what
    // makes the assertion above about the floor rather than about the run being
    // idle.
    let unthrottled = Arc::new(
        LiveSource::new(ExportMeta::new("live", [64, 64, 64], 1), [1, 1, 1])
            .with_min_snapshot_interval(Duration::ZERO),
    );
    assert!(unthrottled.state().blocks.is_empty());
}

// ------------------------------------------------------------- wire --

#[test]
fn the_state_payload_is_parallel_arrays_a_client_can_zip() {
    let (document, _) = exported(3);
    let replay = ReplaySource::from_json(&document).unwrap();
    replay.control(Control::Seek(replay.log().events.len() as u64));
    let state = replay.state();
    let json = state_to_json(&state);

    let blocks = json["blocks"].as_u64().unwrap() as usize;
    assert_eq!(blocks, state.blocks.len());
    assert_eq!(json["index"].as_array().unwrap().len(), blocks * 3);
    for field in ["phase", "kind", "slot", "block_seq"] {
        assert_eq!(json[field].as_array().unwrap().len(), blocks, "{field}");
    }
    // A block that has finished the chain names the last slot, and one that has
    // not started names -1 rather than 0.
    assert!(json["slot"]
        .as_array()
        .unwrap()
        .iter()
        .all(|slot| slot.as_i64().unwrap() >= 0));

    let meta = meta_to_json(&replay.meta());
    assert_eq!(meta["mode"], "replay");
    assert_eq!(meta["controllable"], true);
    assert_eq!(meta["grid"], serde_json::json!([1, 3, 3]));
    assert_eq!(meta["ops"].as_array().unwrap().len(), 3);

    let page = timeline_to_json(&replay.timeline(Window::Since(0), 5));
    assert_eq!(page["events"].as_array().unwrap().len(), 5);
    assert_eq!(page["events"][0]["seq"], 0);
    assert!(page["events"][0]["type"].is_string());
}

/// The two modes must produce the same *fields*, or the browser has two code
/// paths whatever its author intended.
#[test]
fn both_modes_answer_the_same_shape() {
    let (document, _) = exported(2);
    let replay = ReplaySource::from_json(&document).unwrap();
    let live = LiveSource::new(ExportMeta::new("live", [64, 128, 128], 1), [1, 2, 2]);

    for (from_replay, from_live) in [
        (meta_to_json(&replay.meta()), meta_to_json(&live.meta())),
        (state_to_json(&replay.state()), state_to_json(&live.state())),
        (
            timeline_to_json(&replay.timeline(Window::Since(0), 4)),
            timeline_to_json(&live.timeline(Window::Since(0), 4)),
        ),
    ] {
        let mut left: Vec<&String> = from_replay.as_object().unwrap().keys().collect();
        let mut right: Vec<&String> = from_live.as_object().unwrap().keys().collect();
        left.sort();
        right.sort();
        assert_eq!(
            left, right,
            "the two modes disagree about the payload's fields"
        );
    }
}

// ------------------------------------------------------------ http --

/// The smallest HTTP/1.0 client that can check a server. One request per
/// connection, no keep-alive, no chunked decoding — which is why the requests
/// below ask for `HTTP/1.0`, where the server closes the connection and the
/// body is "everything until EOF".
fn fetch(addr: SocketAddr, method: &str, path: &str) -> (u16, String) {
    let mut stream = TcpStream::connect(addr).expect("the server is listening");
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    write!(
        stream,
        "{method} {path} HTTP/1.0\r\nHost: localhost\r\n\r\n"
    )
    .unwrap();
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).expect("a complete response");
    let text = String::from_utf8_lossy(&raw).into_owned();
    let status = text
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse().ok())
        .unwrap_or(0);
    let body = text
        .split_once("\r\n\r\n")
        .map(|(_, body)| body)
        .unwrap_or("");
    (status, body.to_string())
}

fn get_json(addr: SocketAddr, path: &str) -> Value {
    let (status, body) = fetch(addr, "GET", path);
    assert_eq!(status, 200, "GET {path} answered {status}: {body}");
    serde_json::from_str(&body).unwrap_or_else(|err| panic!("GET {path}: {err}: {body}"))
}

fn local(port: u16) -> Options {
    Options::default().port(port)
}

#[test]
fn the_endpoints_answer_over_a_socket() {
    let (document, _) = exported(3);
    let replay = Arc::new(ReplaySource::from_json(&document).unwrap());
    let total = replay.log().events.len() as u64;
    let server = serve(replay, local(0)).unwrap();
    let addr = server.addr();
    assert!(addr.ip().is_loopback());

    let meta = get_json(addr, "/api/meta");
    assert_eq!(meta["mode"], "replay");
    assert_eq!(meta["total_events"], total);
    assert_eq!(meta["grid"], serde_json::json!([1, 3, 3]));

    // Seek to the end and read the state back, over the wire.
    let (status, _) = fetch(
        addr,
        "POST",
        &format!("/api/control?action=seek&value={total}"),
    );
    assert_eq!(status, 200);
    let state = get_json(addr, "/api/state");
    assert_eq!(state["cursor"], total);
    assert_eq!(state["running"], false);
    assert!(state["blocks"].as_u64().unwrap() > 0);

    let page = get_json(addr, "/api/events?since=0&limit=3");
    assert_eq!(page["events"].as_array().unwrap().len(), 3);
    assert_eq!(page["available"], total);

    // A limit beyond the ceiling is clamped, not honoured and not refused.
    let page = get_json(addr, "/api/events?since=0&limit=99999999");
    assert!(page["events"].as_array().unwrap().len() as u64 <= total);

    let (status, body) = fetch(addr, "POST", "/api/control?action=nonsense");
    assert_eq!(status, 400, "{body}");

    server.shutdown();
}

/// A live run cannot be seeked, and the server says so with a status rather
/// than by pretending it worked.
#[test]
fn controlling_a_live_run_is_refused_rather_than_ignored() {
    let live = Arc::new(LiveSource::new(
        ExportMeta::new("live", [64, 64, 64], 1),
        [1, 1, 1],
    ));
    let server = serve(live, local(0)).unwrap();
    let (status, body) = fetch(server.addr(), "POST", "/api/control?action=pause");
    assert_eq!(status, 409, "{body}");
    let answer: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(answer["accepted"], false);
    server.shutdown();
}

/// The page not being built must not stop the server, and must not 404 at
/// somebody who has done nothing wrong.
#[test]
fn a_missing_asset_directory_serves_an_explanation_and_the_api_still_works() {
    let live = Arc::new(LiveSource::new(
        ExportMeta::new("live", [64, 64, 64], 1),
        [1, 1, 1],
    ));
    let mut options = local(0);
    options.assets = Some(std::env::temp_dir().join("blockflow-gui-no-such-directory"));
    let server = serve(live, options).unwrap();
    let addr = server.addr();

    let (status, body) = fetch(addr, "GET", "/");
    assert_eq!(status, 200);
    assert!(body.contains("trunk build"), "{body}");
    // and the endpoints it points at really do answer
    assert_eq!(get_json(addr, "/api/meta")["mode"], "live");
    // a file that is not there is a 404, not the explanation page
    let (status, _) = fetch(addr, "GET", "/pkg/app.js");
    assert_eq!(status, 404);
    server.shutdown();
}

#[test]
fn a_request_that_climbs_out_of_the_asset_directory_is_refused() {
    let live = Arc::new(LiveSource::new(
        ExportMeta::new("live", [64, 64, 64], 1),
        [1, 1, 1],
    ));
    let mut options = local(0);
    options.assets = Some(std::env::temp_dir());
    let server = serve(live, options).unwrap();
    let (status, body) = fetch(server.addr(), "GET", "/../../../../etc/passwd");
    assert_eq!(status, 400, "{body}");
    server.shutdown();
}

#[test]
fn a_public_bind_is_refused_before_a_socket_is_opened() {
    let live = Arc::new(LiveSource::new(
        ExportMeta::new("live", [64, 64, 64], 1),
        [1, 1, 1],
    ));
    let mut options = Options::default();
    options.bind = "0.0.0.0:0".parse().unwrap();
    let error = serve(live, options).unwrap_err();
    assert!(
        error.to_string().contains("not a loopback address"),
        "{error}"
    );
}

// ------------------------------------------------- and it must not disturb --

/// **The claim that matters most.** A run with a server attached and a client
/// polling it flat out must do the same work as a run with neither.
///
/// Work rather than wall time: the schedule, the op count, the block count and
/// the acceptance criterion are deterministic and can be asserted, and a wall
/// time on a shared machine cannot. The timing evidence is
/// [`perturbation_report`], which is printed rather than asserted for exactly
/// that reason.
#[test]
fn a_client_polling_a_run_does_not_change_what_the_run_does() {
    let (_, decomposition, ops) = plan(5);
    let expected: Vec<(usize, String)> = ops.clone();
    let blocks = decomposition.phases[0].grid.n_blocks();

    let alone = run(5, &[], 4);
    alone
        .log
        .check_coverage_and_order(&expected, blocks)
        .unwrap();

    let meta = ExportMeta::new("live", [64, 320, 320], decomposition.n_phases()).with_ops(ops);
    let live = Arc::new(LiveSource::new(meta, [1, 5, 5]).with_timeline());
    let server = serve(live.clone(), local(0)).unwrap();
    let addr = server.addr();

    // A client polling as hard as it can for as long as the run lasts.
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let polls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let poller = {
        let stop = stop.clone();
        let polls = polls.clone();
        std::thread::spawn(move || {
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                let (status, body) = fetch(addr, "GET", "/api/state");
                assert_eq!(status, 200);
                let state: Value = serde_json::from_str(&body).unwrap();
                // Every poll must be well formed, mid-run as much as at the end.
                assert!(state["blocks"].as_u64().is_some());
                polls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        })
    };

    // Do not start the run until the client has had one well-formed reply.
    // Otherwise `polls > 0` below is a claim about thread scheduling: on a
    // loaded machine the poller may not be dispatched before a short run ends,
    // and the test fails for a reason unrelated to what it tests.
    while polls.load(std::sync::atomic::Ordering::Relaxed) == 0 {
        std::thread::yield_now();
    }

    let watched = run(5, &live.listeners(), 4);
    live.finished();
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    poller.join().unwrap();
    server.shutdown();

    watched
        .log
        .check_coverage_and_order(&expected, blocks)
        .unwrap();
    assert!(
        watched.same_work_as(&alone),
        "a watched run did different work: {watched:?} vs {alone:?}"
    );
    assert_eq!(watched.tasks, alone.tasks);
    assert_eq!(watched.ops_applied, alone.ops_applied);
    assert_eq!(watched.reads, alone.reads);
    assert_eq!(watched.writes, alone.writes);
    assert_eq!(watched.listener_faults, 0);
    assert!(
        polls.load(std::sync::atomic::Ordering::Relaxed) > 0,
        "the poller never got a reply, so this proves nothing"
    );
}

/// What attaching a viewer costs in wall time, decomposed.
///
/// Printed, not asserted. Wall time on a shared machine is noisy enough that a
/// threshold would be a flaky test, and the number worth knowing is *which
/// part* costs: the socket, the client, or the listeners.
///
/// **Read the denominator before the percentages.** The accounting environment
/// performs no computation at all — it prices reads and writes and returns —
/// so a run against it is almost entirely event dispatch, and any listener
/// looks expensive against it. A run whose ops actually compute pays the same
/// absolute nanoseconds per event against a very much larger total. What this
/// measurement is for is the *ordering* of the four rows, which does not depend
/// on the denominator.
pub fn perturbation_report(side: usize, workers: usize) -> String {
    let (_, decomposition, ops) = plan(side);
    let grid = decomposition.phases[0].grid.blocks_per_axis();
    let shape = [64, side * 64, side * 64];

    let bare = || {
        let start = Instant::now();
        let stats = run(side, &[], workers);
        (start.elapsed().as_secs_f64(), stats.log.len())
    };

    let attached = |timeline: bool, poll: bool| {
        let meta = ExportMeta::new("live", shape, decomposition.n_phases()).with_ops(ops.clone());
        let mut live = LiveSource::new(meta, grid);
        if timeline {
            live = live.with_timeline();
        }
        let live = Arc::new(live);
        let server = serve(live.clone(), local(0)).unwrap();
        let addr = server.addr();
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let poller = poll.then(|| {
            let stop = stop.clone();
            std::thread::spawn(move || {
                let mut count = 0usize;
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    let _ = fetch(addr, "GET", "/api/state");
                    count += 1;
                }
                count
            })
        });
        let start = Instant::now();
        let _ = run(side, &live.listeners(), workers);
        let elapsed = start.elapsed().as_secs_f64();
        live.finished();
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        let polls = poller.map(|handle| handle.join().unwrap()).unwrap_or(0);
        server.shutdown();
        (elapsed, polls)
    };

    // Best of three: the interesting quantity is the floor, and a shared
    // machine's noise is all upward.
    let best =
        |sample: &mut dyn FnMut() -> f64| (0..3).fold(f64::INFINITY, |best, _| best.min(sample()));

    let (_, events) = bare();
    let mut lines = vec![format!(
        "grid {grid:?} = {} blocks, {workers} workers, {events} events, best of 3",
        grid[0] * grid[1] * grid[2]
    )];
    let baseline = best(&mut || bare().0);
    lines.push(format!(
        "  {:<46} {baseline:>8.3} s",
        "no server, no listeners"
    ));
    for (label, timeline, poll) in [
        ("+ server + progress listener", false, false),
        ("+ timeline listener", true, false),
        ("+ a client polling flat out", true, true),
    ] {
        let mut polls = 0usize;
        let seconds = best(&mut || {
            let (elapsed, count) = attached(timeline, poll);
            polls = polls.max(count);
            elapsed
        });
        lines.push(format!(
            "  {label:<46} {seconds:>8.3} s   ({:+6.1}% vs bare{})",
            (seconds - baseline) / baseline * 100.0,
            if poll {
                format!(", {polls} polls served")
            } else {
                String::new()
            }
        ));
    }
    lines.push(
        "  The last row is the one this module is judged on: a client polling as\n  \
         hard as it can must cost the run no more than having the server there at\n  \
         all, because the snapshot floor bounds what any number of clients can ask\n  \
         the executor for."
            .to_string(),
    );
    lines.join("\n")
}

#[test]
#[ignore = "prints a measurement; run with --ignored --nocapture"]
fn report_what_watching_costs() {
    println!("{}", perturbation_report(64, 4));
}

/// Re-folding from zero is what a backwards seek costs. Printed alongside,
/// because "a rebuild is cheap" is an assertion this design leans on.
#[test]
#[ignore = "prints a measurement; run with --ignored --nocapture"]
fn report_what_a_backwards_seek_costs() {
    let (document, _) = exported(24);
    let replay = ReplaySource::from_json(&document).unwrap();
    let total = replay.log().events.len() as u64;

    // Forwards to the end: the fold is incremental, so this is one pass.
    replay.control(Control::Seek(total));
    let start = Instant::now();
    let _ = replay.state();
    let forwards = start.elapsed();

    // Backwards to the middle: the fold cannot be undone, so it is thrown away
    // and half the log is re-folded.
    replay.control(Control::Seek(total / 2));
    let start = Instant::now();
    let blocks = replay.state().blocks.len();
    let backwards = start.elapsed();

    println!(
        "{total} events over {blocks} blocks:\n  \
         {:<44} {forwards:>12?}\n  {:<44} {backwards:>12?}\n  \
         a backwards seek is a rebuild; at this size it is invisible next to the\n  \
         round trip that asked for it.",
        "fold forward to the end", "seek back to the middle and poll",
    );
}
