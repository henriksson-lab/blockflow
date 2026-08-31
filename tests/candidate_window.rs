// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **The candidate window: how much of the ready set a scheduler is shown.**
//
// `simulate` hands a `Scheduler` every task that could start, and every
// scheduler in the crate walks the whole slice. One dispatch is therefore
// `O(ready)` and a run of `T` tasks is `O(T x R)`.
// `planner_arena::print_the_cost_of_the_dispatch_loop` measures what that comes
// to: at 98 304 tasks, `ExecutorOrder` takes 36.3 s against 0.57 s for an
// `O(1)` control, so **98% of a large simulation is the scheduler scanning the
// ready set**. It is the second of the two quadratic terms in the event loop;
// the first — the readiness scan — was made incremental, which is what exposed
// this one as the larger.
//
// `Machine::candidate_window` caps `R`. This file is what says the cap is
// sound, and what it costs.
//
// What is asserted, and in what order
// -----------------------------------
//
// | claim | why it comes first |
// |---|---|
// | an unbounded window is the machine this crate has always simulated | every recorded figure was taken unbounded; if `0` moved anything the whole corpus would be unreadable and nothing below would be worth measuring |
// | the window is the first `n` ready tasks in ascending id order | a window whose membership depended on admission order would make two runs of one plan schedule differently, which is the one thing the simulator may not do |
// | a window changes the order and never the work | `tasks_run` and `written_bytes` are properties of the plan; a window that moved either would be dropping or duplicating work rather than reordering it |
// | a small window changes *some* schedule | the vacuity control for all three above — a field that is not wired in passes every one of them |
//
// What this file is not: a claim that a window is free. It is not, the last
// test prints the price, and the recommendation lives in
// `print_what_a_window_costs`'s own doc.

use std::collections::BTreeSet;

use blockflow::assemble::{Assembly, PlanBuilder};
use blockflow::distributed::handout::HandoutPolicy;
use blockflow::geometry::BlockGrid;
use blockflow::op::Chain;
use blockflow::probes::IdentityOp;
use blockflow::simulate::{
    simulate, Decision, ExecutorOrder, Handout, Machine, Outcome, PerPhase, Rates, ReleaseAware,
    Scheduler,
};
use blockflow::Dtype;

/// Big enough that the ready set outgrows every window swept below — at block
/// edge 8 a phase is 512 blocks, so the scheduler is regularly shown hundreds
/// of candidates and a window of 16 is a real truncation. A `64^3` volume would
/// make the whole sweep a comparison of numbers below the window.
const VOLUME: [usize; 3] = [64, 64, 64];

/// Chunks small enough that a block's read extent spans several and neighbours
/// share some — which is what makes "which candidate is warm" a question at
/// all. The same chunk `tests/multiple_computers.rs` states its figures in.
const CHUNK: [usize; 3] = [16, 16, 16];

/// The volume the 98% figure was measured on, and the only place the saving can
/// be quoted against it.
///
/// `planner_arena::print_the_cost_of_the_dispatch_loop` took its
/// `36.3 s against 0.57 s` at 98 304 tasks, which is this volume cut at edge 4.
/// A saving quoted at [`VOLUME`]'s scale would be a saving on a run that takes
/// under half a second, where nobody has a problem.
const LARGE_VOLUME: [usize; 3] = [128, 128, 128];

/// A three-phase pixel chain with a reach of one, so blocks overlap and the
/// halo chunks are the shared ones. The same fixture the dispatch-loop
/// measurement uses, so the two files' figures are about one plan.
fn plan(volume: [usize; 3], edge: usize) -> Assembly {
    let grid = BlockGrid::new(volume, [edge, edge, edge]).expect("a grid");
    let mut builder = PlanBuilder::new(volume, Dtype::F64, grid);
    for name in ["first", "second", "third"] {
        builder
            .pixels(Chain::op(IdentityOp::new(name, [1, 1, 1])))
            .expect("a pixel phase");
    }
    builder.finish().expect("an assembly")
}

fn rates() -> Rates {
    Rates {
        chunk: CHUNK,
        chunk_bytes: (CHUNK.iter().product::<usize>() * 8) as u64,
        ..Rates::default()
    }
}

/// Four computers, one worker each, and a window.
///
/// **One worker per node on purpose**, and it is what makes
/// `Outcome::duplicated_fetches` a number rather than a zero: a chunk two
/// machines both read is fetched twice, which is the quantity a handout policy
/// exists to reduce and therefore the quantity a window has to be shown not to
/// wreck. On one node with a shared pool it is zero by construction and the
/// quality column would be blank.
///
/// The cache is a quarter of the volume's chunks, so eviction is real and an
/// ordering can still hit — a cache that holds everything makes every schedule
/// equally good and a window free by construction.
fn machine(candidate_window: usize) -> Machine {
    Machine {
        nodes: 4,
        workers: 4,
        cache_bytes: 1 << 20,
        candidate_window,
        ..Machine::default()
    }
}

fn run(assembly: &Assembly, machine: Machine, scheduler: &mut dyn Scheduler) -> Outcome {
    simulate(
        &assembly.decomposition,
        &assembly.work(),
        &machine,
        &rates(),
        &BTreeSet::new(),
        &BTreeSet::new(),
        PerPhase::default(),
        scheduler,
    )
    .expect("a simulable plan")
}

/// The windows every sweep here uses, and the reason for each.
///
/// Not a round-number ladder: each rung is a claim about the fixture.
///
/// * `0` — unbounded, the recorded machine, the thing everything is compared
///   against;
/// * `256` — half a phase of the edge-8 fixture (512 blocks), so the scheduler
///   still sees most of what is runnable and any degradation here would be a
///   surprise;
/// * `64` — an eighth of a phase, and above the worker count by a wide margin,
///   so a worker is still choosing between many candidates;
/// * `16` — four candidates per worker on this machine, which is the smallest
///   window at which a policy is choosing rather than being told.
const WINDOWS: [usize; 4] = [0, 256, 64, 16];

/// The ladder the **large** sweep uses, with two rungs above [`WINDOWS`].
///
/// At 98 304 tasks a phase holds 32 768 blocks and the ready set reaches
/// thousands, so 256 is a far harsher truncation there than the same number is
/// on the assertion fixture — and the first run of this measurement found
/// `nearest-first` already 19% worse at that rung. `4096` and `1024` are added
/// so the sweep can say **where** the degradation starts rather than only that
/// it has started by 256: they are an eighth and a thirty-second of that
/// phase's 32 768 blocks, which is the range `256` and `64` cover on the edge-8
/// fixture.
const LARGE_WINDOWS: [usize; 6] = [0, 4096, 1024, 256, 64, 16];

/// The block edge the assertions run at. 512 blocks a phase, 1 536 tasks: large
/// enough that the ready set exceeds `WINDOWS`'s every rung, small enough that
/// the debug build's ready-set oracle — which re-scans the whole graph at every
/// dispatch — leaves the suite fast.
const ASSERTION_EDGE: usize = 8;

// ------------------- claim 1: an unbounded window is what was always run --

/// Wraps a scheduler and records the widest ready set it was ever handed.
///
/// The widest is what makes the claim below testable: a window at or above it
/// truncates nothing, so it must reproduce the unbounded run exactly — and if
/// it does, the ready sets it saw were the unbounded ones, which is the fixed
/// point that makes the argument close.
struct Watching<'a> {
    inner: &'a mut dyn Scheduler,
    widest: usize,
    dispatches: usize,
}

impl Scheduler for Watching<'_> {
    fn name(&self) -> &'static str {
        "watching"
    }

    fn pick(&mut self, decision: &Decision<'_>) -> usize {
        self.widest = self.widest.max(decision.ready.len());
        self.dispatches += 1;
        self.inner.pick(decision)
    }
}

/// **`candidate_window: 0` is the simulator this crate has recorded every
/// figure on**, and a window wide enough to truncate nothing is the same run.
///
/// This is the load-bearing one. Every number in `docs/`, in `costs/` and in
/// the doc comment of every measurement in the suite was taken before this
/// field existed, which is to say unbounded. A default that moved any of them
/// would not be a regression in one place — it would make the record
/// unreadable, because nothing says which machine a recorded figure was taken
/// on.
///
/// Three things are checked and the third is the one that stops this being a
/// tautology:
///
/// * the shipped default is `0`;
/// * a window at or above the widest ready set the run ever reaches gives a
///   **bit-identical** `Outcome`, as does `usize::MAX`;
/// * that widest set is genuinely far above the windows swept elsewhere here —
///   without which "wide enough not to truncate" would be true of every window
///   and the equality above would say nothing.
#[test]
fn an_unbounded_window_is_the_machine_this_crate_has_always_simulated() {
    assert_eq!(
        Machine::default().candidate_window,
        0,
        "the shipped default must be unbounded, or every recorded figure in this crate is on a \
         machine nobody can name"
    );

    let assembly = plan(VOLUME, ASSERTION_EDGE);
    let mut watching = Watching {
        inner: &mut ExecutorOrder::phase_major(),
        widest: 0,
        dispatches: 0,
    };
    let unbounded = run(&assembly, machine(0), &mut watching);
    let widest = watching.widest;
    let dispatches = watching.dispatches;
    println!(
        "{} tasks, {dispatches} dispatches, widest ready set {widest}",
        unbounded.tasks_run
    );

    for window in [widest, widest + 1, usize::MAX] {
        let bounded = run(
            &assembly,
            machine(window),
            &mut ExecutorOrder::phase_major(),
        );
        assert_eq!(
            bounded, unbounded,
            "a window of {window} truncates nothing on a run whose ready set never exceeds \
             {widest}, so it must be the unbounded run in every field"
        );
    }

    // and the equality above must be a statement about a wide set, not about a
    // fixture whose ready set is always shorter than any window anyone would
    // set. The bar is the widest *bounded* rung of the ladder: below it, "wide
    // enough not to truncate" would be true of every window in this file and
    // the equality would be a comparison of the unbounded run against itself.
    let widest_rung = WINDOWS
        .into_iter()
        .filter(|&window| window != 0)
        .max()
        .expect("a bounded rung");
    assert!(
        widest > widest_rung,
        "the widest ready set is only {widest}, at or below the {widest_rung} this file's \
         ladder tops out at, so 'wide enough not to truncate' says nothing here"
    );
}

// ---------------- claim 2: the window is the first n, in ascending order --

/// Records the ready slice it was handed at every dispatch, then takes the
/// first.
///
/// **Taking the first is what makes an exact prefix comparison possible.** Index
/// `0` is the lowest ready task id whether or not the slice was truncated, so a
/// windowed run and an unbounded one dispatch the same task at every step and
/// their recordings line up index for index. Any scheduler that actually looked
/// at the slice would diverge at the first truncation and the two recordings
/// would be of two different runs.
struct Recording {
    seen: Vec<Vec<usize>>,
}

impl Scheduler for Recording {
    fn name(&self) -> &'static str {
        "recording"
    }

    fn pick(&mut self, decision: &Decision<'_>) -> usize {
        self.seen.push(decision.ready.to_vec());
        0
    }
}

/// **A window is the first `n` ready task ids, in ascending order** — the same
/// prefix, at every dispatch, of the slice the unbounded run saw.
///
/// Two properties, and the crate needs both:
///
/// * **ascending id**, because `Decision::ready`'s own doc says so and several
///   schedulers break a tie by the first entry they see. A window that reordered
///   would silently change what `PlanOrder` means.
/// * **a prefix, not a sample**, because the simulator's entire use is comparing
///   one run against another. A window drawn by admission time, by a hash or by
///   anything else that is not a function of the ready set would make two runs
///   of one plan schedule differently, and every figure this crate reports would
///   become a figure plus a coin toss.
///
/// The vacuity control is counted rather than assumed: the run must contain
/// dispatches where the truncation actually bit, and the assertion prints how
/// many.
#[test]
fn the_window_is_the_first_n_ready_tasks_in_ascending_id_order() {
    let assembly = plan(VOLUME, ASSERTION_EDGE);
    let mut unbounded = Recording { seen: Vec::new() };
    run(&assembly, machine(0), &mut unbounded);

    for window in WINDOWS.into_iter().filter(|&w| w != 0) {
        let mut bounded = Recording { seen: Vec::new() };
        run(&assembly, machine(window), &mut bounded);
        assert_eq!(
            bounded.seen.len(),
            unbounded.seen.len(),
            "taking the first candidate is the same choice windowed or not, so the two runs \
             must have the same number of dispatches; they do not, so the window has changed \
             something other than what the scheduler was shown"
        );
        let mut truncated = 0usize;
        for (at, (shown, whole)) in bounded.seen.iter().zip(&unbounded.seen).enumerate() {
            assert_eq!(
                shown.as_slice(),
                &whole[..window.min(whole.len())],
                "window {window}, dispatch {at}: the scheduler was shown something other than \
                 the first {window} of the {} ready tasks",
                whole.len()
            );
            assert!(
                shown.windows(2).all(|pair| pair[0] < pair[1]),
                "window {window}, dispatch {at}: the candidates are not in ascending task id"
            );
            if whole.len() > window {
                truncated += 1;
            }
        }
        assert!(
            truncated * 4 > bounded.seen.len(),
            "window {window} truncated only {truncated} of {} dispatches, so this comparison is \
             mostly of a slice against itself",
            bounded.seen.len()
        );
        println!(
            "window {window}: {truncated} of {} dispatches truncated",
            bounded.seen.len()
        );
    }
}

// ------------------------ claim 3: the order moves, the work does not --

/// The schedulers every sweep here runs, named for the table.
///
/// `ExecutorOrder::phase_major` and `Handout::new(NearestFirst)` because they
/// are the two the cost measurement is required to cover — the executor's own
/// dispatch order and the real coordinator's policy. `block_major` and
/// `ReleaseAware` because they are the arms whose ranking a window can actually
/// disturb: both prefer something the ready set does *not* already offer first,
/// so a truncation takes candidates away from them.
fn schedulers() -> Vec<(&'static str, Box<dyn Scheduler>)> {
    vec![
        (
            "executor:phase-major",
            Box::new(ExecutorOrder::phase_major()) as Box<dyn Scheduler>,
        ),
        (
            "executor:block-major",
            Box::new(ExecutorOrder::block_major()),
        ),
        ("release-aware", Box::new(ReleaseAware)),
        (
            "nearest-first",
            Box::new(Handout::new(HandoutPolicy::NearestFirst)),
        ),
    ]
}

/// **A window changes the order and never the work.**
///
/// `Outcome::tasks_run` and `Outcome::written_bytes` document themselves as
/// conservation laws — properties of the plan rather than of the schedule, so
/// any two schedulers on one plan must agree on them. A candidate window is a
/// change to *which* ready task runs next and to nothing else, so it is under
/// the same law: a window that moved either would be dropping a task, running
/// one twice, or writing a block it should not have.
///
/// `materialised_bytes` and `tasks_short_circuited` are held to the same
/// standard for the same reason. `cache_misses` and `makespan_ns` deliberately
/// are **not**: those are what a schedule is for, and asserting them equal would
/// be asserting that scheduling does nothing.
#[test]
fn a_window_changes_the_order_and_never_the_work() {
    let assembly = plan(VOLUME, ASSERTION_EDGE);
    let tasks = assembly.decomposition.n_tasks() as u64;
    for (name, mut scheduler) in schedulers() {
        let unbounded = run(&assembly, machine(0), scheduler.as_mut());
        assert_eq!(
            unbounded.tasks_run, tasks,
            "{name}: the unbounded run did not run the plan's own task count, so the baseline \
             below is not a baseline"
        );
        for window in WINDOWS.into_iter().filter(|&w| w != 0) {
            let bounded = run(&assembly, machine(window), scheduler.as_mut());
            for (what, a, b) in [
                ("tasks_run", unbounded.tasks_run, bounded.tasks_run),
                (
                    "written_bytes",
                    unbounded.written_bytes,
                    bounded.written_bytes,
                ),
                (
                    "materialised_bytes",
                    unbounded.materialised_bytes,
                    bounded.materialised_bytes,
                ),
                (
                    "tasks_short_circuited",
                    unbounded.tasks_short_circuited,
                    bounded.tasks_short_circuited,
                ),
            ] {
                assert_eq!(
                    a, b,
                    "{name} at window {window}: {what} moved from {a} to {b}. A window changes \
                     the order tasks run in; it cannot change which tasks there are."
                );
            }
        }
    }
}

/// **A small window changes some schedule** — the vacuity control for every
/// claim above.
///
/// Each of the three is a statement of the form "the window did not break X",
/// and a field that no code reads passes all three. This is the one that fails
/// if `Machine::candidate_window` is never consulted: at a window of 16 on a
/// fixture whose ready set runs to hundreds, at least one scheduler must reach a
/// different `Outcome`.
///
/// It is deliberately **not** asserted of every scheduler.
/// `ExecutorOrder::phase_major`'s key is `[phase, x, y, z, block]` and a
/// `TaskGraph` is built phase by phase, block by block, so its argmin is the
/// lowest ready id — which a prefix always contains. That it is *unmoved* by a
/// window is a finding and is recorded in `print_what_a_window_costs`; making it
/// a requirement here would be requiring the simulator to be worse.
#[test]
fn a_small_window_moves_a_schedule_that_looks_past_the_first_candidate() {
    let assembly = plan(VOLUME, ASSERTION_EDGE);
    let mut moved: Vec<&'static str> = Vec::new();
    for (name, mut scheduler) in schedulers() {
        let unbounded = run(&assembly, machine(0), scheduler.as_mut());
        let narrow = run(
            &assembly,
            machine(*WINDOWS.last().expect("a ladder")),
            scheduler.as_mut(),
        );
        if narrow != unbounded {
            moved.push(name);
        }
        println!(
            "{name:>22}  makespan {:>12} -> {:>12}   misses {:>7} -> {:>7}",
            unbounded.makespan_ns, narrow.makespan_ns, unbounded.cache_misses, narrow.cache_misses
        );
    }
    assert!(
        !moved.is_empty(),
        "a window of {} changed no scheduler's outcome on a fixture whose ready set reaches the \
         hundreds. The field is not being read, and every other test in this file is passing on \
         a run the window never touched.",
        WINDOWS.last().expect("a ladder")
    );
    println!("moved by the narrowest window: {moved:?}");
}

// ------------------------------------------- what the window actually costs --

/// One block of the table: every window, against one plan, for the named
/// schedulers.
///
/// Prints `identical` where the `Outcome` is bit-for-bit the unbounded one,
/// because that is a stronger and shorter statement than four percentages of
/// zero — and it is the statement `ExecutorOrder::phase_major` earns at every
/// rung.
fn sweep(
    volume: [usize; 3],
    edge: usize,
    windows: &[usize],
    schedulers: Vec<(&'static str, Box<dyn Scheduler>)>,
) {
    use std::time::Instant;

    let assembly = plan(volume, edge);
    let tasks = assembly.decomposition.n_tasks();
    println!("\n{volume:?} at edge {edge}, {tasks} tasks, 4 workers on 4 computers");
    println!(
        "{:>22} {:>8} {:>12} {:>12} {:>14} {:>10} {:>12}",
        "scheduler", "window", "wall (ms)", "us/dispatch", "makespan", "misses", "duplicated"
    );
    for (name, mut scheduler) in schedulers {
        let mut baseline: Option<Outcome> = None;
        for &window in windows {
            let mut best = f64::INFINITY;
            let mut outcome = Outcome::default();
            // Best of three: the fastest run is the one least disturbed by
            // whatever else the machine was doing, and the quantity wanted here
            // is a floor rather than an average.
            for _ in 0..3 {
                let started = Instant::now();
                outcome = run(&assembly, machine(window), scheduler.as_mut());
                best = best.min(started.elapsed().as_secs_f64());
            }
            assert_eq!(outcome.tasks_run as usize, tasks);
            let base = *baseline.get_or_insert(outcome);
            println!(
                "{name:>22} {:>8} {:>12.1} {:>12.2} {:>14} {:>10} {:>12}   {}",
                if window == 0 {
                    "none".to_string()
                } else {
                    window.to_string()
                },
                best * 1e3,
                best * 1e6 / tasks as f64,
                outcome.makespan_ns,
                outcome.cache_misses,
                outcome.duplicated_fetches,
                if outcome == base {
                    "identical".to_string()
                } else {
                    format!(
                        "makespan {:+.2}%, misses {:+.2}%",
                        100.0 * (outcome.makespan_ns as f64 / base.makespan_ns as f64 - 1.0),
                        100.0 * (outcome.cache_misses as f64 / base.cache_misses as f64 - 1.0),
                    )
                }
            );
        }
    }
}

/// **What a candidate window buys and what it costs**, and the answer to the
/// question it was added for: *how small can the window get before the schedule
/// degrades, and how much time does that save.*
///
/// Run with `--release`. `simulate` carries a `debug_assertions` oracle that
/// re-scans the whole task graph at every dispatch, so a debug build times the
/// term the window removes *plus* a full scan and every ratio below is wrong.
///
/// ```text
/// cargo test --release --test candidate_window -- --ignored --nocapture
/// ```
///
/// # Measured
///
/// This file's fixture — a three-phase pixel chain, `16^3` chunks, four
/// computers of one worker each, a `1 MiB` pool per computer — best of three,
/// release, on this machine. `identical` means the `Outcome` matched the
/// unbounded run in **every** field; the percentages are against that run.
///
/// ```text
/// [64, 64, 64] at edge 8, 1 536 tasks, ready set peaks at 512
///            scheduler   window   wall (ms)   us/dispatch     makespan   misses   duplicated
/// executor:phase-major     none         9.6          6.27     43211927      808          610   identical
/// executor:phase-major      256         7.0          4.55     43211927      808          610   identical
/// executor:phase-major       64         4.6          3.02     43211927      808          610   identical
/// executor:phase-major       16         4.6          3.00     43211927      808          610   identical
/// executor:block-major     none         7.1          4.64     67274029     3746         2788   identical
/// executor:block-major      256         6.3          4.09     60921882     2969         2226   makespan  -9.44%, misses -20.74%
/// executor:block-major       64         4.6          2.99     46884716     1257          951   makespan -30.31%, misses -66.44%
/// executor:block-major       16         3.9          2.52     44604602      975          736   makespan -33.70%, misses -73.97%
///        release-aware     none       148.4         96.60     65901095     3577         2662   identical
///        release-aware      256        94.7         61.63     60056611     2864         2140   makespan  -8.87%, misses -19.93%
///        release-aware       64        16.8         10.93     46566393     1217          915   makespan -29.34%, misses -65.98%
///        release-aware       16         6.1          3.97     44416280      955          721   makespan -32.60%, misses -73.30%
///        nearest-first     none         9.6          6.27     42722668      741          515   identical
///        nearest-first      256         7.6          4.92     42951079      773          555   makespan  +0.53%, misses  +4.32%
///        nearest-first       64         4.8          3.15     42987864      778          568   makespan  +0.62%, misses  +4.99%
///        nearest-first       16         4.0          2.62     43289083      818          602   makespan  +1.33%, misses +10.39%
///
/// [64, 64, 64] at edge 4, 12 288 tasks
/// executor:phase-major     none       456.5         37.15     69121067      768          576   identical
/// executor:phase-major      256        57.3          4.66     69121067      768          576   identical
/// executor:phase-major       64        35.3          2.87     69121067      768          576   identical
/// executor:phase-major       16        29.4          2.39     69121067      768          576   identical
/// executor:block-major     none       234.2         19.06     99486885     4475         3365   identical
/// executor:block-major      256        56.7          4.62     70692949      960          720   makespan -28.94%, misses -78.55%
/// executor:block-major       64        34.3          2.79     69715518      840          630   makespan -29.92%, misses -81.23%
/// executor:block-major       16        29.8          2.43     69379750      799          599   makespan -30.26%, misses -82.15%
///        release-aware     none     72546.8       5903.87     99147846     4433         3334   identical
///        release-aware      256      1166.5         94.93     70696843      960          720   makespan -28.70%, misses -78.34%
///        release-aware       64       152.9         12.44     69738507      843          632   makespan -29.66%, misses -80.98%
///        release-aware       16        48.8          3.97     69383684      800          600   makespan -30.02%, misses -81.95%
///        nearest-first     none       385.4         31.37     71880096     1104          781   identical
///        nearest-first      256        61.7          5.02     69033092      757          564   makespan  -3.96%, misses -31.43%
///        nearest-first       64        36.1          2.94     69121931      768          576   makespan  -3.84%, misses -30.43%
///        nearest-first       16        32.7          2.66     69132396      769          575   makespan  -3.82%, misses -30.34%
///
/// [128, 128, 128] at edge 4, 98 304 tasks — the scale the 98% was measured at
/// executor:phase-major     none     39293.2        399.71    807820792    35328        26496   identical
/// executor:phase-major     4096      5234.9         53.25    807820792    35328        26496   identical
/// executor:phase-major     1024      1543.0         15.70    807820792    35328        26496   identical
/// executor:phase-major      256       797.7          8.11    807820792    35328        26496   identical
/// executor:phase-major       64       637.0          6.48    807820792    35328        26496   identical
/// executor:phase-major       16       606.3          6.17    807820792    35328        26496   identical
///        nearest-first     none     35935.8        365.56    621438131    12569         8010   identical
///        nearest-first     4096      8000.5         81.38    616873687    12016         8164   makespan  -0.73%, misses  -4.40%
///        nearest-first     1024      1608.3         16.36    648578620    15889        11153   makespan  +4.37%, misses +26.41%
///        nearest-first      256       812.0          8.26    737693332    26765        19633   makespan +18.71%, misses +112.94%
///        nearest-first       64       625.0          6.36    746648950    27861        20594   makespan +20.15%, misses +121.66%
///        nearest-first       16       587.0          5.97    795473933    33821        25263   makespan +28.01%, misses +169.08%
/// ```
///
/// # What the numbers say
///
/// **1. The window removes the term it was aimed at, and the saving is what the
/// 98% figure promised.** At 98 304 tasks `ExecutorOrder::phase_major` goes from
/// **39.3 s to 0.80 s at a window of 256 — 49x — and to 0.61 s at 16, 65x**,
/// with a **bit-identical schedule at every rung**. `us/dispatch` is the shape
/// of it: 399.71 unbounded, 6.17 windowed, and the windowed figure barely moves
/// across an 64x range of task counts where the unbounded one moves by the same
/// 64x.
///
/// **2. `ExecutorOrder::phase_major` costs nothing at all, and that is a theorem
/// rather than luck.** Its key is `[phase, x, y, z, block]`; a `TaskGraph` is
/// built phase by phase and block by block; so its argmin over the ready set is
/// always the lowest ready id, which every prefix contains. The executor's own
/// dispatch order — the baseline every other scheduler is judged against — is
/// free to window.
///
/// **3. The safe window is not a constant, and this is the finding that decides
/// the default.** The *same* window of 256 is an improvement at 12 288 tasks
/// (`nearest-first`, -3.96% makespan) and a 19% regression at 98 304
/// (+18.71%, and cache misses more than doubled). A window is a fraction of the
/// ready set whether or not it is written as one, and the ready set grows with
/// the plan: 256 is half a phase of the small fixture and a hundred-and-
/// twenty-eighth of the large one. Any constant default is therefore a
/// different policy at every plan size.
///
/// **4. Where the knee is, for the policy that has one.** `nearest-first` at
/// 98 304 tasks: **4096 is free** (-0.73% makespan, and it still saves 4.5x),
/// 1024 costs 4.4% for 22x, 256 costs 18.7% for 44x, 16 costs 28% for 61x. So
/// the knee sits between **4096 and 1024** — an eighth and a thirty-second of a
/// phase — and the ninefold difference in saving between them (4.5x against
/// 22x) is what makes the choice a judgement rather than an obvious one.
///
/// **5. A window *improves* `block_major` and `ReleaseAware` by ~30% of
/// makespan and ~80% of misses. That is a warning, not a bonus.** Both prefer
/// something the front of the ready set does not offer, so unbounded they range
/// over the whole runnable set and scatter the traversal; a prefix confines them
/// to tasks near each other in plan order, which is near each other in the
/// volume, which is the locality the cache wants. A window is therefore **not
/// neutral** — it is a locality prior laid on top of whatever the policy does,
/// and a policy measured under one is measured with that prior included. Two
/// policies may be compared only at the same window.
///
/// **6. `ReleaseAware` is the outlier and its cost is its own.** 72.5 s at
/// 12 288 tasks against 0.39 s for `ExecutorOrder`, because it walks the live
/// images *and* re-counts the ready set for every candidate — `O(ready^2)` per
/// dispatch, not `O(ready)`. A window fixes it by force (48.8 ms at 16, 1 500x),
/// which is worth knowing but is not an argument for windows: it is an argument
/// for that scheduler keeping a count instead of recomputing one.
///
/// # What is left, and it is not the scheduler
///
/// A windowed dispatch is `O(window)`, but the loop around it is still not
/// linear: `us/dispatch` at a window of 16 is 2.39 at 12 288 tasks and 6.17 at
/// 98 304. The likeliest remaining `O(T x R)` term is the `Vec::remove` that
/// takes the chosen task out of the ready set — a memmove of up to a phase's
/// worth of `usize`, whose constant is two orders below a policy evaluation,
/// which is why it did not show until the policy stopped dominating. Not
/// measured here, and recorded as the next thing to look at rather than as a
/// claim.
///
/// # The recommendation: leave the default at `0`
///
/// Finding 3 is decisive on its own. A non-zero default is a constant, a
/// constant is a different fraction of the ready set at every plan size, and the
/// same constant that improves the 12 288-task run by 4% costs the 98 304-task
/// run 19%. There is no number that is right at both, so the honest default is
/// the one that says nothing.
///
/// Findings 2 and 5 close it. A default would silently re-rank the arena's whole
/// table — flattering `block_major` and `ReleaseAware` by 30% of makespan,
/// penalising `nearest-first` by up to 28%, leaving `phase_major` alone — and
/// every figure this crate has recorded was taken unbounded, with nothing in the
/// record saying so because there was nothing else it could have been.
///
/// **What to do instead**: an arena sweep that needs the speed sets the window
/// explicitly, states it beside its figures, and holds it fixed across the arms
/// it compares. **4096 for the 98 304-task shape** if the schedule must be
/// undisturbed — 4.5x for -0.73% — and **1024** if 22x is worth 4.4%. Below
/// that the window is choosing the schedule rather than bounding the search for
/// one.
#[test]
#[ignore = "a measurement, not an assertion"]
fn print_what_a_window_costs() {
    sweep(VOLUME, 8, &WINDOWS, schedulers());
    sweep(VOLUME, 4, &WINDOWS, schedulers());
    // **The scale the 98% was measured at**, and only the two schedulers the
    // question was asked about. `ReleaseAware` unbounded is already 74 s at
    // 12 288 tasks — it walks the live images *and* the ready set per candidate,
    // so it grows faster than the others — and at 98 304 it would be the whole
    // measurement rather than a row of it.
    sweep(
        LARGE_VOLUME,
        4,
        &LARGE_WINDOWS,
        vec![
            (
                "executor:phase-major",
                Box::new(ExecutorOrder::phase_major()) as Box<dyn Scheduler>,
            ),
            (
                "nearest-first",
                Box::new(Handout::new(HandoutPolicy::NearestFirst)),
            ),
        ],
    );
}
