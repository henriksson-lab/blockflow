// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// Multi-node execution: a coordinator, workers that pull, and four ways for
// them to find each other.
//
// The property that decides the architecture
// ------------------------------------------
// **A task never needs a peer's in-memory output.** A `(block, phase)` task
// depends on the previous phase's tasks covering its read extent, and satisfies
// that by *reading storage* — halos come out of the source array, phase
// boundaries materialise. So the workload is embarrassingly parallel at block
// granularity given shared storage, and **coordination traffic is metadata
// only**: a task descriptor per block, never a byte of image data.
//
// Which is why this is not MPI. MPI's strengths are collectives and RDMA
// between ranks, and there is no rank-to-rank transfer here to use them on.
// What it would cost is a C dependency, a **static rank model** that fights the
// greedy adaptive scheduler the design deliberately chose, and **fault
// intolerance** — one rank dies, the job dies — which is bad on a long batch
// run and disqualifying on reclaimable instances.
//
// Nodes do not die (decided 2026-08-17)
// -------------------------------------
// That last clause is now **narrower than it reads**, by decision rather than
// by discovery, and the narrowing is worth stating precisely because the two
// remaining arguments against MPI are untouched by it.
//
// The deployment this crate is built for is **10-20 cooperating nodes on AWS
// and SLURM**. At that size a machine going down is not a routine event to be
// absorbed; it is a major one. And absorbing it is not the small thing the
// claim table makes it look like: a node that goes has been holding blocks in
// memory and in its own chunk cache, so re-running the tasks it had *claimed*
// restores the claim table and not the position. Real recovery is a large
// piece of design. It is deliberately not being attempted yet, because a
// stable, fast baseline is worth more right now than a partial answer to a
// rare event.
//
// So, in this order:
//
// 1. **A claim has no expiry.** `JobSpec::lease` is `None` unless a caller sets
//    it, and a claim handed out is held until it completes. `Option<Duration>`
//    rather than a very large number of milliseconds: a big number is still a
//    deadline — comparable, addable, overflowable, and above all *meetable* —
//    whereas `None` is not a deadline at all, so there is no arithmetic to get
//    wrong because there is no operand.
// 2. **A lost node aborts the whole job**, loudly, naming the worker and every
//    task it was holding — see `coordinator::Job::worker_lost`. Not a stall,
//    not a reassignment. What to do about it is decided *above* this crate, by
//    the batch script or the orchestrator that started the run, which is also
//    the only layer that knows whether re-running the whole thing is cheaper
//    than the alternative.
// 3. **The coordinator does not detect the loss itself, on purpose.** It has no
//    signal for a process it did not start: its HTTP server hands it requests,
//    not connections, so a peer's socket closing is invisible to it, and the
//    only thing left to watch would be *silence* — which is a timeout, which is
//    exactly the mechanism this decision removes. Whoever launched the worker
//    has the real signal — a process exit, with a status — and relays it to
//    `/lost`. `local::run` does this for a local run; `srun` and an
//    orchestrator already do the equivalent on a cluster by killing the step.
// 4. **The reissue machinery stays compiled and tested**, and a job opts in by
//    setting a lease. `tests/local_multi_node.rs` has one test that does, and
//    it is the documentation as much as the coverage.
//
// Why the lease is not merely set high, which is the tempting version: a
// claim's deadline is stamped at handout while a worker keeps `ahead` tasks in
// hand, so the real contract is `lease > (ahead + 1) x task duration` — implicit,
// unenforceable, and violated silently. Measured on the local multi-node
// fixture with **nobody killed**, a 400 ms lease reissued 13 of 16 tasks and
// recomputed 11 of 16 blocks: 69 % of a job duplicated with no fault at all.
// Output stayed byte-identical, so it was waste and not corruption — but waste
// is what this decision is optimising against, and no number makes the contract
// explicit. Removing the deadline removes the contract with it, which is also
// what frees `ahead` to be chosen for pipelining alone.
//
// What is *not* claimed by any of this: that node loss is impossible, or that
// MPI would now be fine. The other two objections — the C dependency and the
// static rank model against a greedy adaptive scheduler — stand on their own,
// and the mechanism here still exists, still runs and is one field away.
//
// The shape
// ---------
// The coordinator is its own program, always — not "rank 0 behaves
// differently". Three benefits fall out of that and they are the reason it is
// worth a separate binary: local testing is trivial (a coordinator and two
// workers on a laptop, same binaries, no scheduler); a cloud deployment stops
// being a special case, becoming a placement choice rather than a code path;
// and there is **one place to look** — the coordinator already holds the DAG,
// every claim and every completion, and a progress view over it has the same
// shape whether it is fed by one executor or by N workers.
//
// | module | what it owns |
// |---|---|
// | `protocol` | The messages. Metadata only; no message names an element type. |
// | `wire` | JSON for the messages and for a `Decomposition`, which is the one binding thing that travels. |
// | `coordinator` | The job registry, the claim table, the handout, the merged event stream. Both lifetimes, one implementation. |
// | `handout` | Which unclaimed task goes to which worker: seeded far apart, then nearest-first. |
// | `placement` | Which *worker* gets a task when there are fewer tasks than workers. A barrier phase is one block, so that is the only placement decision it has. |
// | `cache_model` | What the coordinator *believes* each worker holds, derived from assignments and never reported. |
// | `spec` | What a job is on the wire, and the factory seam that turns one back into a chain and an environment. |
// | `rendezvous` | Four ways to find the coordinator, behind one trait. |
// | `client` | A JSON-over-HTTP client, no dependency. |
// | `server` | *(feature `distributed`)* The coordinator's HTTP surface. |
// | `worker` | Pull, execute through the crate's own executor, report. Plans nothing. |
// | `shared_volume` | An `Environment` over files several processes can open, so a distributed run can be exercised for real. |
// | `local` | A coordinator and N workers as separate processes on one machine. The primary verification vehicle. |
//
// The feature flag, and what it does and does not gate
// ----------------------------------------------------
// `distributed` pulls in the HTTP **server** and the three binaries, because the
// server needs `tiny_http`. Everything else — the protocol, the coordinator's
// whole state machine, the handout policy, the cache model, the rendezvous
// backends, the client and the worker loop — compiles and is tested with no
// feature at all. That split is deliberate: the parts that can be wrong in an
// interesting way are the parts that need no socket to test.

pub mod cache_model;
pub mod client;
pub mod coordinator;
pub mod handout;
pub mod local;
pub mod placement;
pub mod protocol;
pub mod rendezvous;
#[cfg(feature = "distributed")]
pub mod server;
pub mod shared_volume;
pub mod spec;
pub mod wire;
pub mod worker;

#[cfg(test)]
mod tests;

/// The coordinator's default port, where an environment supplies a host and not
/// a port. One above the progress view's, so a node running both needs neither
/// moved.
pub const DEFAULT_COORDINATOR_PORT: u16 = 8732;

pub use cache_model::{ChunkGrid, ModelledCache};
pub use client::Client;
pub use coordinator::{
    Aborted, Coordinator, HeldClaim, Job, DEFAULT_LINGER_MS, SUGGESTED_LEASE_MS,
};
pub use handout::HandoutPolicy;
pub use local::{Binaries, LocalOptions, LocalRun};
pub use placement::Residency;
pub use protocol::{Assignment, Handout, JobStatus, Joined, PROTOCOL_VERSION};
pub use rendezvous::{
    DirectRendezvous, DirectoryObjects, EnvRendezvous, FileRendezvous, ObjectRendezvous,
    ObjectStore, Record, Rendezvous,
};
pub use shared_volume::SharedVolumes;
pub use spec::{
    decompose, probe_job, probe_job_over, read_job, ChainSpec, FragmentPhaseSpec, JobSpec, OpSpec,
    ProbeWorkflows, SidecarSpec, StoreSpec, WorkflowFactory, WorkflowSpec,
};
pub use worker::{WorkerOptions, WorkerReport};
