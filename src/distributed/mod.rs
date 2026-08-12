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
pub use coordinator::{Coordinator, Job, DEFAULT_LEASE_MS, DEFAULT_LINGER_MS};
pub use handout::HandoutPolicy;
pub use local::{Binaries, LocalOptions, LocalRun};
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
