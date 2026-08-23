// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **CPU-seconds, so that a run can be told whether it is efficient.**
//
// Wall time says how long a phase took. It cannot say whether forty threads did
// forty threads' worth of work, and that is the question a caller deciding how
// many threads to hand out is actually asking. CPU-seconds against wall-seconds
// answers it: the ratio *is* the mean number of cores kept busy, and comparing
// it against the concurrency the plan assumed is the acceptance bar for every
// threading change this crate makes.
//
// What this reads, and why that one
// ---------------------------------
// `/proc/self/stat`, fields 14 and 15 — `utime` and `stime`, in clock ticks.
// **Process-wide, and including threads that have already exited**, which is the
// property that decides it. The obvious alternative, summing
// `/proc/self/task/*/schedstat`, loses a thread's CPU the moment it is joined —
// so an instrument built on it reports less work the more work finishes, and a
// pool that retires its workers at a phase boundary would appear to free the
// very thing being measured.
//
// **Per-task attribution is a different instrument and is not this one.**
// `CLOCK_THREAD_CPUTIME_ID` read from inside each worker is what answers "which
// thread burned it"; a process-wide counter cannot, and is useless under
// concurrency for that question. This is deliberately the coarse one, because
// the coarse one is what the phase boundary can afford.
//
// What it costs, and where it may be called
// -----------------------------------------
// One `read_to_string` of a small pseudo-file and two integer parses. That is
// far too much for a block, and it is nothing at a **phase boundary** — which is
// the only place `crate::strategy` calls it.
//
// **Nothing here allocates or formats inside a region somebody is timing**, and
// that rule is written down because breaking it has already cost this project a
// measurement: a `String` shape key built while block buffers were live moved
// the figure by an amount that varied with the length of the chain's *names*.
// A caller wanting a per-phase figure takes two [`CpuTime`] readings and
// subtracts them; the subtraction is integer arithmetic on a `Copy` struct and
// the formatting happens wherever the caller reports, which is not inside the
// timed region.
//
// Not portable, and it says so
// ----------------------------
// `/proc` is Linux. Everything here returns `None` where it is not there, and a
// caller that gets `None` has learned that this machine does not offer the
// figure — which is a different thing from a figure of zero and is why the
// return type is an `Option` rather than a saturating count.

use std::sync::atomic::{AtomicU64, Ordering};

/// Ticks of CPU this process has consumed, user and system.
///
/// The unit is the kernel's clock tick, which is what `/proc/self/stat` reports
/// and which this does not convert: the conversion needs `sysconf(_SC_CLK_TCK)`,
/// which needs libc, and this crate's dependency list is deliberately short. The
/// tick is 100 Hz on every Linux this has run on and [`CpuTime::HERTZ`] says so
/// with the caveat attached, so a caller that wants seconds can have them and
/// can see what it is trusting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CpuTime {
    pub user_ticks: u64,
    pub system_ticks: u64,
}

impl CpuTime {
    /// The tick rate this crate assumes when converting to seconds.
    ///
    /// `CONFIG_HZ` is 100 on every mainstream Linux distribution and
    /// `sysconf(_SC_CLK_TCK)` has returned 100 on every machine this has been
    /// run on; the kernel's userspace ABI fixes `USER_HZ` at 100 independently
    /// of the internal tick. It is a constant here rather than a syscall because
    /// the syscall needs a dependency this crate does not carry, and the failure
    /// mode of being wrong is a *scaled* figure rather than a wrong comparison —
    /// every ratio in this module is a ratio of two of these and is unaffected.
    pub const HERTZ: f64 = 100.0;

    /// Read the process's CPU counters, or `None` where `/proc` is not there.
    ///
    /// **Both fields or neither.** A `/proc/self/stat` that parsed one and not
    /// the other would give a figure that is quietly half of what it claims, so
    /// a partial parse is an absence.
    pub fn now() -> Option<Self> {
        Self::parse(&std::fs::read_to_string("/proc/self/stat").ok()?)
    }

    /// The parse, split out so it can be tested against a line rather than
    /// against a machine.
    ///
    /// **The command name is skipped by its closing parenthesis and not by
    /// whitespace**, because a process may be named `weird ) name` and the
    /// kernel does not escape it — a field-splitting parse of `/proc/self/stat`
    /// is the classic way to read the wrong number here. Everything after the
    /// last `)` is fixed-width in fields, and `utime` and `stime` are the
    /// twelfth and thirteenth of them.
    pub fn parse(line: &str) -> Option<Self> {
        let after_name = &line[line.rfind(')')? + 1..];
        let mut fields = after_name.split_ascii_whitespace();
        // `state` is the first field after the name, so `utime` — field 14 of
        // the file, one-based — is the twelfth here.
        let user_ticks = fields.nth(11)?.parse().ok()?;
        let system_ticks = fields.next()?.parse().ok()?;
        Some(Self {
            user_ticks,
            system_ticks,
        })
    }

    /// User plus system.
    pub fn total_ticks(self) -> u64 {
        self.user_ticks.saturating_add(self.system_ticks)
    }

    /// Ticks consumed since `earlier`, saturating.
    ///
    /// Saturating rather than wrapping because these counters only rise, so a
    /// negative difference is a reading taken out of order and zero is the
    /// honest answer to it.
    pub fn since(self, earlier: Self) -> u64 {
        self.total_ticks().saturating_sub(earlier.total_ticks())
    }

    /// Seconds, at [`Self::HERTZ`].
    pub fn seconds(ticks: u64) -> f64 {
        ticks as f64 / Self::HERTZ
    }
}

/// A phase's CPU, accumulated across a run.
///
/// **Two atomics and no allocation**, so a phase boundary can add to it without
/// the accounting becoming part of what is being accounted. There is no name, no
/// map and no key: a run's phases are numbered and the caller that reads this
/// knows which phase it asked about.
#[derive(Debug, Default)]
pub struct CpuLedger {
    ticks: AtomicU64,
    wall_nanos: AtomicU64,
    phases: AtomicU64,
}

impl CpuLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one phase's CPU ticks and wall nanoseconds.
    pub fn record(&self, ticks: u64, wall_nanos: u64) {
        self.ticks.fetch_add(ticks, Ordering::Relaxed);
        self.wall_nanos.fetch_add(wall_nanos, Ordering::Relaxed);
        self.phases.fetch_add(1, Ordering::Relaxed);
    }

    pub fn ticks(&self) -> u64 {
        self.ticks.load(Ordering::Relaxed)
    }

    pub fn wall_nanos(&self) -> u64 {
        self.wall_nanos.load(Ordering::Relaxed)
    }

    pub fn phases(&self) -> u64 {
        self.phases.load(Ordering::Relaxed)
    }

    /// **The number the whole module exists for: how many cores the run kept
    /// busy, on average.**
    ///
    /// CPU-seconds divided by wall-seconds. Against the concurrency the plan
    /// assumed, this is the answer to "is it efficient, and how many threads
    /// should I give out" — a figure of 3 under a pool of 40 says the threads
    /// were parked, which is exactly the condition intra-block slicing exists
    /// for. `None` when no wall time was recorded, because a ratio over zero is
    /// not zero.
    pub fn mean_cores_busy(&self) -> Option<f64> {
        let wall = self.wall_nanos();
        if wall == 0 {
            return None;
        }
        Some(CpuTime::seconds(self.ticks()) / (wall as f64 / 1e9))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A command name containing a space and a parenthesis**, which is the
    /// case a whitespace-splitting parse gets wrong and the reason this parses
    /// from the last `)` instead.
    #[test]
    fn the_parse_survives_a_process_named_like_a_stat_line() {
        // Fields after the name: state, ppid, pgrp, session, tty, tpgid, flags,
        // minflt, cminflt, majflt, cmajflt, utime, stime, ...
        let line = "42 (we ) i (rd) S 1 42 42 0 -1 4194304 100 0 0 0 777 333 5 6 20 0 1 0 99";
        let read = CpuTime::parse(line).expect("a stat line");
        assert_eq!(read.user_ticks, 777);
        assert_eq!(read.system_ticks, 333);
        assert_eq!(read.total_ticks(), 1110);
        // **Liveness.** A parse that split on whitespace from the start would
        // read the pid and the first word of the name and get neither of these,
        // so the fixture has to be one that distinguishes them — asserted rather
        // than assumed.
        let naive: Vec<&str> = line.split_ascii_whitespace().collect();
        assert_ne!(
            naive[13], "777",
            "the fixture must defeat a naive field split"
        );
    }

    #[test]
    fn a_difference_is_saturating_and_a_reading_only_rises() {
        let early = CpuTime {
            user_ticks: 10,
            system_ticks: 5,
        };
        let late = CpuTime {
            user_ticks: 30,
            system_ticks: 7,
        };
        assert_eq!(late.since(early), 22);
        assert_eq!(
            early.since(late),
            0,
            "a reading out of order is zero, not negative"
        );
    }

    /// The ledger's arithmetic, and the number the module exists for.
    #[test]
    fn the_ledger_reports_how_many_cores_were_busy() {
        let ledger = CpuLedger::new();
        assert_eq!(
            ledger.mean_cores_busy(),
            None,
            "no wall time is not zero cores"
        );
        // 400 ticks is four CPU-seconds at 100 Hz; over one wall second that is
        // four cores busy.
        ledger.record(400, 1_000_000_000);
        let busy = ledger.mean_cores_busy().expect("a ratio");
        assert!((busy - 4.0).abs() < 1e-9, "got {busy}");
        assert_eq!(ledger.phases(), 1);
        // A second phase accumulates rather than replacing.
        ledger.record(400, 1_000_000_000);
        let busy = ledger.mean_cores_busy().expect("a ratio");
        assert!((busy - 4.0).abs() < 1e-9, "got {busy}");
        assert_eq!(ledger.phases(), 2);
    }

    /// On a machine that has `/proc`, the counters must actually move when work
    /// is done — otherwise this module is a well-tested way to read zero.
    #[test]
    fn the_counters_rise_when_the_process_burns_cpu() {
        let Some(before) = CpuTime::now() else {
            // Not Linux. The absence is the honest answer and not a failure.
            return;
        };
        // **Two defences, because this test needed both and got them one at a
        // time.**
        //
        // *The first version counted iterations* — forty million of them — and
        // reported that the counters do not move. They had not: the loop finished
        // inside a single 10 ms tick. So the condition became the **wall clock**,
        // which cannot be optimised away because it is read rather than counted.
        //
        // *That was still not enough, and the second defect is the deeper one.*
        // `black_box` on the **result** does not keep the loop: LLVM replaces a
        // closed-form accumulation with its closed form and hands the answer to
        // `black_box` directly. Measured with `rustc -O`: **one billion
        // iterations in 0.00 ms**, against 70.95 ms for forty million with
        // `black_box` *inside* the loop. A test that spins on the clock still
        // burns its 250 ms either way — so it passes — but it burns them calling
        // `Instant::now`, not doing the arithmetic the comment claims. **A
        // liveness test that was itself not alive**, one level below the first
        // one.
        //
        // Both are now shut: the clock bounds the duration and the `black_box`
        // **inside** the inner loop keeps the work. Measured in this shape:
        // 250.20 ms and **80.1 million real iterations**.
        let started = std::time::Instant::now();
        let mut sink = 0u64;
        let mut value = 0u64;
        while started.elapsed() < std::time::Duration::from_millis(250) {
            for _ in 0..100_000 {
                value = value.wrapping_add(1);
                sink = sink.wrapping_add(value.wrapping_mul(2_654_435_761));
                std::hint::black_box(&sink);
            }
        }
        std::hint::black_box(sink);
        let after = CpuTime::now().expect("the second reading must work if the first did");
        assert!(
            after.since(before) > 0,
            "burning 250 ms of wall time on this thread recorded no CPU at all, which is more \
             than {} ticks' worth",
            (0.25 * CpuTime::HERTZ) as u64
        );
    }
}
