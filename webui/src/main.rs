// SPDX-License-Identifier: MIT
//
// Original work.
//
// The browser half of the block-schedule view.
//
// **It does not know whether it is watching a run or a recording.** It fetches
// `/api/meta` once, polls `/api/state`, and draws what comes back. The only
// thing it branches on is `meta.controllable`, which says whether the transport
// buttons would do anything — a statement about a capability, not about a mode.
// The word "replay" appears in this file exactly once, in a badge, and nothing
// is conditional on it. If that ever stops being true, the two views have begun
// to diverge and the server's `ProgressSource` is the thing to fix.
//
// Polling, not sockets: see the server's module header. Two intervals, because
// the two things being fetched have different natural rates — the grid wants to
// be smooth, and the timeline is read, not watched.
//
// What is drawn, and what is deliberately not
// -------------------------------------------
// A block's fill is **how far through the op chain it has got**, and nothing
// else. Not its duration, not its worker, not its phase. A cell that carried
// three encodings would be a puzzle rather than a picture, and the thing
// somebody opens this for is "how far has it got, and where is it working".
// Phase, kind, sequence and op name are all in the cell's tooltip and in the
// timeline, which is where a second question belongs.
//
// This is not the presentation animation. That is `tools/`, it renders a movie,
// and it is allowed to be beautiful. This is the thing you leave open on a
// second monitor while a job runs.

use std::collections::HashMap;

use gloo_net::http::Request;
use gloo_timers::callback::Interval;
use serde::Deserialize;
use wasm_bindgen_futures::spawn_local;
use yew::prelude::*;

/// The payload version this page understands. The server sends its own in every
/// response; a mismatch means the page in the browser is older than the server
/// and is drawing fields that may have moved, so it says so instead.
const WIRE_VERSION: u64 = 1;

/// How often the grid is refreshed. Fast enough to look continuous, slow enough
/// that a poll over a port forward from another continent still keeps up.
const STATE_POLL_MS: u32 = 200;
/// The timeline is read rather than watched.
const EVENT_POLL_MS: u32 = 700;
/// How many timeline entries to keep on screen.
const TIMELINE_ROWS: usize = 14;
/// The steps in the chain-progress ramp. Five is the most that keep a visible
/// lightness gap between neighbours on both surfaces; see `index.html`.
const RAMP_STEPS: usize = 5;
/// Beyond this many cells the browser spends its time in layout rather than in
/// drawing, and the picture stops being legible anyway. The counts above the
/// grid stay correct, so the view still answers "how far has it got".
const MAX_CELLS: usize = 20_000;

// ------------------------------------------------------------ payloads --

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Op {
    slot: usize,
    name: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Meta {
    wire: u64,
    mode: String,
    controllable: bool,
    strategy: String,
    volume: [usize; 3],
    grid: [usize; 3],
    phases: usize,
    ops: Vec<Op>,
}

/// Parallel arrays, one entry per touched block. See the server's `wire` module
/// for why it is shaped this way; zipping them is the four lines below.
#[derive(Debug, Clone, PartialEq, Deserialize)]
struct StateMsg {
    wire: u64,
    cursor: u64,
    total: Option<u64>,
    running: bool,
    blocks: usize,
    index: Vec<usize>,
    phase: Vec<usize>,
    kind: Vec<String>,
    slot: Vec<i64>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Entry {
    seq: u64,
    #[serde(rename = "type")]
    kind: String,
    phase: Option<usize>,
    index: Option<[usize; 3]>,
    op: Option<String>,
    duration_ns: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Page {
    events: Vec<Entry>,
}

/// One block, as the page needs it.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Cell {
    phase: usize,
    slot: i64,
    /// The block has been touched but not written: the scheduler still has work
    /// on it. Drawn as a ring, which is a shape rather than a hue and so does
    /// not compete with the progress fill.
    open: bool,
    written: bool,
    short_circuited: bool,
}

fn cells_of(state: &StateMsg) -> HashMap<[usize; 3], Cell> {
    let mut out = HashMap::with_capacity(state.blocks);
    for position in 0..state.blocks {
        let index = [
            state.index[position * 3],
            state.index[position * 3 + 1],
            state.index[position * 3 + 2],
        ];
        let kind = state.kind[position].as_str();
        out.insert(
            index,
            Cell {
                phase: state.phase[position],
                slot: state.slot[position],
                open: kind != "written",
                written: kind == "written",
                short_circuited: kind == "short_circuited",
            },
        );
    }
    out
}

/// Which ramp step a slot gets. Slots are spread across the whole ramp rather
/// than taking its first *n* steps, so a three-op chain uses the ends and the
/// middle instead of three near-identical blues. A chain longer than the ramp
/// bins — neighbouring ops share a step — which the legend states and the
/// tooltip escapes.
fn ramp_step(slot: i64, ops: usize) -> Option<usize> {
    let slot = usize::try_from(slot).ok()?;
    if ops <= 1 {
        return Some(RAMP_STEPS - 1);
    }
    let last = RAMP_STEPS - 1;
    Some((slot.min(ops - 1) * last + (ops - 1) / 2) / (ops - 1))
}

// ------------------------------------------------------------ fetching --

async fn get<T: for<'a> Deserialize<'a>>(path: &str) -> Result<T, String> {
    let response = Request::get(path)
        .send()
        .await
        .map_err(|err| format!("{err}"))?;
    if !response.ok() {
        return Err(format!("{} {}", response.status(), response.status_text()));
    }
    response.json::<T>().await.map_err(|err| format!("{err}"))
}

async fn post(path: &str) {
    let _ = Request::post(path).send().await;
}

/// Whether the last poll arrived, and what went wrong if it did not.
#[derive(Debug, Clone, PartialEq)]
struct Link {
    up: bool,
    message: String,
}

impl Default for Link {
    fn default() -> Self {
        Self {
            up: false,
            message: "connecting".to_string(),
        }
    }
}

// ---------------------------------------------------------------- app --

#[function_component(App)]
fn app() -> Html {
    let meta = use_state(|| None::<Meta>);
    let state = use_state(|| None::<StateMsg>);
    let events = use_state(Vec::<Entry>::new);
    let link = use_state(Link::default);
    // Shared between the render and the polling closures. A `use_state` value
    // captured by an interval would be the value at the time the effect ran and
    // would never change; these cells are the same allocation for the life of
    // the component, which is what the pollers need.
    let cursor = use_mut_ref(|| 0u64);
    let have_meta = use_mut_ref(|| false);

    {
        let meta = meta.clone();
        let link = link.clone();
        let have_meta = have_meta.clone();
        let state = state.clone();
        let cursor = cursor.clone();
        use_effect_with((), move |_| {
            let interval = Interval::new(STATE_POLL_MS, {
                let meta = meta.clone();
                let link = link.clone();
                let have_meta = have_meta.clone();
                let state = state.clone();
                let cursor = cursor.clone();
                move || {
                    let meta = meta.clone();
                    let link = link.clone();
                    let have_meta = have_meta.clone();
                    let state = state.clone();
                    let cursor = cursor.clone();
                    spawn_local(async move {
                        // The metadata is fetched here rather than once on mount
                        // so that a page opened before the server was ready —
                        // or reloaded across a restart — recovers by itself.
                        if !*have_meta.borrow() {
                            if let Ok(fetched) = get::<Meta>("api/meta").await {
                                *have_meta.borrow_mut() = true;
                                meta.set(Some(fetched));
                            }
                        }
                        match get::<StateMsg>("api/state").await {
                            Ok(fetched) => {
                                *cursor.borrow_mut() = fetched.cursor;
                                state.set(Some(fetched));
                                link.set(Link {
                                    up: true,
                                    message: "connected".to_string(),
                                });
                            }
                            Err(message) => link.set(Link { up: false, message }),
                        }
                    });
                }
            });
            move || drop(interval)
        });
    }

    {
        let events = events.clone();
        let cursor = cursor.clone();
        use_effect_with((), move |_| {
            let interval = Interval::new(EVENT_POLL_MS, move || {
                let events = events.clone();
                let cursor = cursor.clone();
                spawn_local(async move {
                    // The window ends at the playhead. That one parameter is
                    // what makes this identical for a live run and a scrubbed
                    // replay: in both cases "recent" means "just before where
                    // we are", and the server answers it without either side
                    // knowing which it is.
                    let at = *cursor.borrow();
                    if let Ok(page) =
                        get::<Page>(&format!("api/events?before={at}&limit={TIMELINE_ROWS}")).await
                    {
                        let mut rows = page.events;
                        rows.reverse();
                        events.set(rows);
                    }
                });
            });
            move || drop(interval)
        });
    }

    let control = {
        let state = state.clone();
        Callback::from(move |query: String| {
            let state = state.clone();
            spawn_local(async move {
                post(&format!("api/control?{query}")).await;
                // Answer the click immediately rather than at the next tick.
                if let Ok(fetched) = get::<StateMsg>("api/state").await {
                    state.set(Some(fetched));
                }
            });
        })
    };

    let Some(meta) = (*meta).clone() else {
        return html! {
            <main>
                <h1>{ "blockflow" }</h1>
                <p class="note">{ (*link).message.clone() }</p>
            </main>
        };
    };

    html! {
        <main>
            <div class="head">
                <h1>{ "blockflow — schedule" }</h1>
                <span class="badge">{ meta.mode.clone() }</span>
                <span class="note">{ format!("strategy {}", meta.strategy) }</span>
                <span class="note">{ format!(
                    "grid {}x{}x{} over {}x{}x{} voxels, {} phase{}",
                    meta.grid[0], meta.grid[1], meta.grid[2],
                    meta.volume[0], meta.volume[1], meta.volume[2],
                    meta.phases, if meta.phases == 1 { "" } else { "s" },
                ) }</span>
                <span class="spacer" />
                { connection(&link) }
            </div>
            { wire_warning(&meta, state.as_ref()) }
            { tiles(&meta, state.as_ref()) }
            { transport(&meta, state.as_ref(), &control) }
            <div class="panel">
                { grid(&meta, state.as_ref()) }
                <div class="legend" style="margin-top:.9rem">{ legend(&meta) }</div>
            </div>
            { timeline(&events) }
        </main>
    }
}

fn connection(link: &Link) -> Html {
    html! {
        <span class="link">
            <span class={classes!("dot", (!link.up).then_some("down"))} />
            { link.message.clone() }
        </span>
    }
}

fn wire_warning(meta: &Meta, state: Option<&StateMsg>) -> Html {
    let server = state.map(|state| state.wire).unwrap_or(meta.wire);
    if meta.wire == WIRE_VERSION && server == WIRE_VERSION {
        return html! {};
    }
    html! {
        <div class="panel">
            <span class="note">{ format!(
                "This page speaks payload version {WIRE_VERSION} and the server speaks \
                 {server}. Reload; if that does not help, the page and the server are \
                 from different builds."
            ) }</span>
        </div>
    }
}

fn tile(value: String, label: &str) -> Html {
    html! {
        <div class="tile">
            <div class="value">{ value }</div>
            <div class="label">{ label.to_string() }</div>
        </div>
    }
}

fn tiles(meta: &Meta, state: Option<&StateMsg>) -> Html {
    let blocks = meta.grid[0] * meta.grid[1] * meta.grid[2];
    let Some(state) = state else {
        return html! { <div class="tiles">{ tile("—".to_string(), "waiting") }</div> };
    };
    let cells = cells_of(state);
    let written = cells.values().filter(|cell| cell.written).count();
    let open = cells.values().filter(|cell| cell.open).count();
    let phase = cells.values().map(|cell| cell.phase).max().unwrap_or(0);
    let position = match state.total {
        Some(total) if total > 0 => format!("{} / {}", state.cursor, total),
        _ => format!("{}", state.cursor),
    };
    html! {
        <div class="tiles">
            { tile(format!("{} / {}", cells.len(), blocks), "blocks touched") }
            { tile(format!("{written}"), "written this phase") }
            { tile(format!("{open}"), "open") }
            { tile(format!("{}", phase), "furthest phase") }
            { tile(position, "events") }
            { tile(
                if state.running { "running".to_string() } else { "idle".to_string() },
                "playhead",
            ) }
        </div>
    }
}

fn transport(meta: &Meta, state: Option<&StateMsg>, control: &Callback<String>) -> Html {
    if !meta.controllable {
        return html! {};
    }
    let cursor = state.map(|state| state.cursor).unwrap_or(0);
    let total = state.and_then(|state| state.total).unwrap_or(0);
    let running = state.map(|state| state.running).unwrap_or(false);
    let send = |query: String| {
        let control = control.clone();
        Callback::from(move |_: MouseEvent| control.emit(query.clone()))
    };
    let scrub = {
        let control = control.clone();
        Callback::from(move |event: InputEvent| {
            use wasm_bindgen::JsCast;
            if let Some(input) = event
                .target()
                .and_then(|target| target.dyn_into::<web_sys::HtmlInputElement>().ok())
            {
                control.emit(format!("action=seek&value={}", input.value()));
            }
        })
    };
    html! {
        <div class="panel transport">
            <button onclick={send("action=seek&value=0".to_string())}>{ "|<" }</button>
            <button onclick={send("action=step&value=-50".to_string())}>{ "<<" }</button>
            <button onclick={send(
                if running { "action=pause".to_string() } else { "action=play".to_string() }
            )}>{ if running { "pause" } else { "play" } }</button>
            <button onclick={send("action=step&value=50".to_string())}>{ ">>" }</button>
            <input type="range" min="0" max={total.to_string()} value={cursor.to_string()}
                   oninput={scrub} />
            <span class="scrub-label">{ format!("{cursor} / {total}") }</span>
            <button onclick={send("action=speed&value=200".to_string())}>{ "slow" }</button>
            <button onclick={send("action=speed&value=2000".to_string())}>{ "fast" }</button>
        </div>
    }
}

fn grid(meta: &Meta, state: Option<&StateMsg>) -> Html {
    let cells = state.map(cells_of).unwrap_or_default();
    let ops = meta.ops.len().max(1);
    let total_cells = meta.grid[0] * meta.grid[1] * meta.grid[2];
    if total_cells > MAX_CELLS {
        return html! {
            <p class="note">{ format!(
                "{total_cells} blocks is more than this view draws ({MAX_CELLS}); the counts \
                 above are still exact. Export the order log and render it if you want the \
                 whole lattice."
            ) }</p>
        };
    }
    // Wide enough to be looked at rather than squinted at, and bounded so a
    // hundred-column grid still fits beside its neighbours.
    let width = (meta.grid[2] as f64 * 44.0).clamp(200.0, 560.0);
    let layers = (0..meta.grid[0])
        .map(|plane| {
            let rows = (0..meta.grid[1])
                .flat_map(|row| (0..meta.grid[2]).map(move |column| (row, column)));
            let drawn = rows
                .map(|(row, column)| {
                    let index = [plane, row, column];
                    let cell = cells.get(&index).copied();
                    let step = cell.and_then(|cell| ramp_step(cell.slot, ops));
                    let op = cell
                        .and_then(|cell| usize::try_from(cell.slot).ok())
                        .and_then(|slot| meta.ops.iter().find(|op| op.slot == slot))
                        .map(|op| op.name.clone());
                    let title = match cell {
                        None => format!("[{plane}, {row}, {column}] — not yet touched"),
                        Some(cell) => format!(
                            "[{plane}, {row}, {column}] — phase {}, {}{}{}",
                            cell.phase,
                            match &op {
                                Some(name) => format!("through {name}"),
                                None => "no op yet".to_string(),
                            },
                            if cell.written { ", written" } else { ", open" },
                            if cell.short_circuited {
                                ", short-circuited"
                            } else {
                                ""
                            },
                        ),
                    };
                    let classes = classes!(
                        "cell",
                        step.map(|step| format!("s{step}")),
                        cell.map(|cell| cell.open)
                            .unwrap_or(false)
                            .then_some("open"),
                    );
                    html! { <div class={classes} title={title} /> }
                })
                .collect::<Html>();
            html! {
                <div class="layer">
                    if meta.grid[0] > 1 {
                        <div class="caption">{ format!("axis 0 = {plane}") }</div>
                    }
                    <div class="cells"
                         style={format!(
                             "grid-template-columns: repeat({}, 1fr); width: {width}px",
                             meta.grid[2]
                         )}>
                        { drawn }
                    </div>
                </div>
            }
        })
        .collect::<Html>();
    html! { <><h2>{ "blocks, coloured by progress through the chain" }</h2>
    <div class="layers">{ layers }</div></> }
}

fn legend(meta: &Meta) -> Html {
    let ops = meta.ops.len().max(1);
    let keys = meta
        .ops
        .iter()
        .map(|op| {
            let step = ramp_step(op.slot as i64, ops).unwrap_or(0);
            html! {
                <span class="key">
                    <span class={classes!("swatch", format!("s{step}"))} />
                    { format!("{}. {}", op.slot, op.name) }
                </span>
            }
        })
        .collect::<Html>();
    html! {
        <>
            <span class="key"><span class="swatch" />{ "not yet touched" }</span>
            { keys }
            <span class="key"><span class="swatch open" />{ "open — still being worked on" }</span>
            if meta.ops.len() > RAMP_STEPS {
                <span class="note">{ format!(
                    "{} ops share {RAMP_STEPS} shades; hover a block for its exact op",
                    meta.ops.len()
                ) }</span>
            }
        </>
    }
}

fn timeline(events: &[Entry]) -> Html {
    if events.is_empty() {
        return html! {
            <div class="panel">
                <h2>{ "recent events" }</h2>
                <p class="note">{
                    "Nothing to show. A live run needs a timeline listener attached; a \
                     replay has one always."
                }</p>
            </div>
        };
    }
    let rows = events.iter().map(|entry| {
        html! {
            <tr>
                <td>{ entry.seq }</td>
                <td>{ entry.kind.clone() }</td>
                <td>{ entry.phase.map(|phase| phase.to_string()).unwrap_or_default() }</td>
                <td>{ entry.index.map(|index| format!("[{}, {}, {}]", index[0], index[1], index[2])).unwrap_or_default() }</td>
                <td>{ entry.op.clone().unwrap_or_default() }</td>
                <td>{ entry.duration_ns.map(|ns| format!("{:.3} ms", ns as f64 / 1e6)).unwrap_or_default() }</td>
            </tr>
        }
    }).collect::<Html>();
    html! {
        <div class="panel">
            <h2>{ "recent events" }</h2>
            <div class="scroll">
                <table>
                    <thead><tr>
                        <th>{ "seq" }</th><th>{ "event" }</th><th>{ "phase" }</th>
                        <th>{ "block" }</th><th>{ "op" }</th><th>{ "took" }</th>
                    </tr></thead>
                    <tbody>{ rows }</tbody>
                </table>
            </div>
        </div>
    }
}

fn main() {
    yew::Renderer::<App>::new().render();
}
