// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The messages, and what each is for.
//
// The whole protocol is metadata. **A task never needs a peer's in-memory
// output** — a `(block, phase)` task depends on the previous phase's tasks
// covering its read extent, and satisfies that by *reading storage*, not by
// receiving from another node. So no message here carries a byte of image data,
// and none of them names an element type.
//
// The shape, and why it is a pull
// -------------------------------
// A worker only ever does *connect, pull a task, execute, report*. It plans
// nothing. Pull rather than static assignment because 37-41 % of blocks are
// empty and which ones is data-dependent, so a `block % N` split hands one
// worker a dense region and another a sparse one — and because pull *can* give
// fault tolerance for free: a claim unhonoured past its lease is reissued,
// which would cover preemption and reclamation without a second mechanism.
//
// That second half is **available and switched off**. A job has no lease unless
// it sets one, so a claim is held until it completes; the `lease_ms` a handout
// carries is `null` by default and is informational either way. See the module
// header for the decision and its date. The first half — which blocks go where
// — is untouched by that and is the reason the handout is a pull.
//
// Granularity: one block per handout
// ----------------------------------
// At 256³ that is 5 597 handouts across a whole run — **0.6 requests per second
// on eight nodes**, and under five per second even at 128³. There is no round
// trip worth amortising, and a batch is a thing to get wrong: partial failure,
// uneven batches, a worker holding blocks it will not reach.
//
// The caveat that is *not* batching, and it is in `Pull` rather than in a batch
// size: a worker asks for its next task **before finishing the current one**, so
// handout latency never sits on the critical path. That matters precisely for
// the cheap blocks — an empty block short-circuits in milliseconds, and a worker
// that only asked when idle would spend most of its time waiting on a round
// trip.
//
// Events: sent as they happen, unbatched
// --------------------------------------
// One `Report` per event. A whole run is on the order of 101 k events, which
// works out at **4-11 per second across all workers combined**; at that rate a
// batching window is machinery with nothing to buy, and an unbatched event is
// simpler to reason about, simpler to test and arrives immediately. Add a
// window only if a measurement demands one.

use std::time::Duration;

use serde_json::{json, Value};

use crate::error::Result;
use crate::region::Region;

use super::wire::{
    count, get, millis_json, millis_or_none, number, number_or, region_at, region_json, text,
    text_or, triple,
};

/// Bumped only when a message changes shape. A worker and a coordinator that
/// disagree refuse each other at `Join` rather than at some later message whose
/// absence looks like a hang.
pub const PROTOCOL_VERSION: u64 = 1;

/// Endpoint paths, named once so the client and the server cannot drift.
pub mod path {
    pub const JOIN: &str = "/join";
    pub const PULL: &str = "/pull";
    pub const REPORT: &str = "/report";
    pub const COMPLETED: &str = "/completed";
    pub const FAILED: &str = "/failed";
    pub const STATUS: &str = "/status";
    pub const SUBMIT: &str = "/job";
    pub const JOBS: &str = "/jobs";
    pub const LOG: &str = "/log";
    /// A worker is gone and the job must stop. Named `/lost` rather than
    /// `/shutdown` because it says *why*: the coordinator does not detect a
    /// node loss itself — it has no signal for a process it did not start —
    /// so whoever started the worker tells it, and the message it prints has
    /// to be able to name the worker.
    pub const LOST: &str = "/lost";
}

/// One unit of work, as handed to a worker.
///
/// Carries the regions as well as the ids. Not a necessity — every worker has
/// the deterministic decomposition and could resolve the block index itself —
/// but a few dozen bytes against 0.6 requests per second buys skipping the
/// lookup, and it makes a handout readable in a packet capture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Assignment {
    pub job: String,
    /// Task id in the DAG both ends build from the same decomposition.
    pub task: usize,
    pub phase: usize,
    pub index: [usize; 3],
    pub core: Region,
    pub read: Region,
    pub valid: Region,
    /// 1 the first time this task is handed out, 2 after one reissue, and so
    /// on. A worker does nothing with it; it is in the message so that a
    /// reissued task is visible in the log rather than inferred from a
    /// duplicate.
    pub attempt: u32,
    /// How long the coordinator will wait before taking this task back and
    /// giving it to somebody else — or `None`, the default, meaning it never
    /// will.
    ///
    /// Informational: no worker reads it, and none ever has. It is on the wire
    /// so that a handout says, in the packet, which contract it was made under
    /// — and `null` is a statement a reader cannot misjudge, where a very large
    /// millisecond count would be a deadline that merely looks unreachable.
    /// Travels as `lease_ms`: a number when there is a lease, `null` when there
    /// is not.
    pub lease: Option<Duration>,
}

impl Assignment {
    pub fn to_json(&self) -> Value {
        json!({
            "job": self.job,
            "task": self.task,
            "phase": self.phase,
            "index": self.index,
            "core": region_json(&self.core),
            "read": region_json(&self.read),
            "valid": region_json(&self.valid),
            "attempt": self.attempt,
            "lease_ms": millis_json(self.lease),
        })
    }

    pub fn from_json(value: &Value) -> Result<Self> {
        Ok(Self {
            job: text(value, "job")?,
            task: count(value, "task")?,
            phase: count(value, "phase")?,
            index: triple(value, "index")?,
            core: region_at(value, "core")?,
            read: region_at(value, "read")?,
            valid: region_at(value, "valid")?,
            attempt: number_or(value, "attempt", 1) as u32,
            lease: millis_or_none(value, "lease_ms"),
        })
    }
}

/// What a pull returns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Handout {
    /// Work. The common answer, and the only one on the hot path.
    Task(Box<Assignment>),
    /// Nothing *yet*: tasks remain but every one of them is either claimed or
    /// waiting on a dependency in the previous phase.
    ///
    /// Distinct from `Finished` because the two mean opposite things to a
    /// worker — retry, or stop — and a worker that confused them would either
    /// spin at the end of a run or exit at a phase boundary.
    Wait { after_ms: u64, remaining: usize },
    /// Every task in the job is done. Go home.
    Finished,
}

impl Handout {
    pub fn to_json(&self) -> Value {
        match self {
            Self::Task(assignment) => json!({"state": "task", "task": assignment.to_json()}),
            Self::Wait {
                after_ms,
                remaining,
            } => json!({"state": "wait", "after_ms": after_ms, "remaining": remaining}),
            Self::Finished => json!({"state": "finished"}),
        }
    }

    pub fn from_json(value: &Value) -> Result<Self> {
        Ok(match text(value, "state")?.as_str() {
            "task" => Self::Task(Box::new(Assignment::from_json(get(value, "task")?)?)),
            "wait" => Self::Wait {
                after_ms: number_or(value, "after_ms", 25),
                remaining: number_or(value, "remaining", 0) as usize,
            },
            _ => Self::Finished,
        })
    }
}

/// What `Join` answers with: who you are, and everything you need to run.
#[derive(Debug, Clone)]
pub struct Joined {
    pub protocol: u64,
    pub job: String,
    pub worker: String,
    /// The workflow, as a spec the worker's factory resolves locally, and the
    /// decomposition, which is binding and therefore given rather than derived.
    pub spec: Value,
}

impl Joined {
    pub fn to_json(&self) -> Value {
        json!({
            "protocol": self.protocol,
            "job": self.job,
            "worker": self.worker,
            "spec": self.spec,
        })
    }

    pub fn from_json(value: &Value) -> Result<Self> {
        Ok(Self {
            protocol: number(value, "protocol")?,
            job: text(value, "job")?,
            worker: text(value, "worker")?,
            spec: get(value, "spec")?.clone(),
        })
    }
}

/// A job's progress, for a runner, an operator or a progress view.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct JobStatus {
    pub job: String,
    pub tasks: usize,
    pub done: usize,
    pub claimed: usize,
    /// Ready or blocked, but not claimed and not done.
    pub pending: usize,
    pub workers: usize,
    /// Claims that outlived their lease and were handed to somebody else. The
    /// number that says fault tolerance actually fired rather than being
    /// merely available.
    pub reissued: usize,
    pub failed: usize,
    pub events: usize,
    pub finished: bool,
}

impl JobStatus {
    pub fn to_json(&self) -> Value {
        json!({
            "job": self.job,
            "tasks": self.tasks,
            "done": self.done,
            "claimed": self.claimed,
            "pending": self.pending,
            "workers": self.workers,
            "reissued": self.reissued,
            "failed": self.failed,
            "events": self.events,
            "finished": self.finished,
        })
    }

    pub fn from_json(value: &Value) -> Result<Self> {
        Ok(Self {
            job: text_or(value, "job", ""),
            tasks: count(value, "tasks")?,
            done: count(value, "done")?,
            claimed: count(value, "claimed")?,
            pending: count(value, "pending")?,
            workers: count(value, "workers")?,
            reissued: count(value, "reissued")?,
            failed: number_or(value, "failed", 0) as usize,
            events: number_or(value, "events", 0) as usize,
            finished: value
                .get("finished")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assignment() -> Assignment {
        Assignment {
            job: "j".to_string(),
            task: 17,
            phase: 1,
            index: [2, 0, 3],
            core: Region::new(&[16, 0, 0], &[16, 8, 8]),
            read: Region::new(&[12, 0, 0], &[24, 8, 8]),
            valid: Region::new(&[16, 0, 0], &[16, 8, 8]),
            attempt: 2,
            lease: Some(Duration::from_millis(5_000)),
        }
    }

    #[test]
    fn an_assignment_survives_the_wire() {
        let original = assignment();
        assert_eq!(
            Assignment::from_json(&original.to_json()).unwrap(),
            original
        );
    }

    #[test]
    fn every_handout_answer_survives_the_wire_as_itself() {
        // The three answers mean go, retry and stop; confusing any two of them
        // is a hang or an early exit, so the round trip is asserted for each.
        for handout in [
            Handout::Task(Box::new(assignment())),
            Handout::Wait {
                after_ms: 25,
                remaining: 400,
            },
            Handout::Finished,
        ] {
            assert_eq!(Handout::from_json(&handout.to_json()).unwrap(), handout);
        }
    }

    #[test]
    fn a_status_survives_the_wire() {
        let status = JobStatus {
            job: "j".to_string(),
            tasks: 100,
            done: 40,
            claimed: 4,
            pending: 56,
            workers: 4,
            reissued: 1,
            failed: 0,
            events: 900,
            finished: false,
        };
        assert_eq!(JobStatus::from_json(&status.to_json()).unwrap(), status);
    }
}
