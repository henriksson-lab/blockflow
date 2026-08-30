// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// What a job *is*, on the wire, and how a worker turns one back into something
// runnable.
//
// The problem this solves
// -----------------------
// A worker executes an op chain and reads and writes arrays. Neither of those
// is serialisable — a `Chain` holds `Box<dyn BlockOp>` and an `Environment` is
// a live handle on storage. So a job spec cannot *contain* them; it can only
// **name** them in terms the receiving process already understands.
//
// Hence two halves, and the split is the same one the crate draws everywhere
// else:
//
// * **the binding half travels.** The `Decomposition` is integers, it is what
//   makes output reproducible, and it is what a worker must be *given* rather
//   than allowed to derive. It rides alongside this spec (see `wire`).
// * **the executable half is resolved locally**, by a [`WorkflowFactory`] the
//   worker process was built with. The spec says which workflow and with what
//   parameters; the factory says what that means here.
//
// That is also the extension seam. A consumer with translated kernels cannot
// put them in this crate, and does not have to: it links this crate, registers
// a factory that understands its own `kind`, and ships its own worker binary.
// The coordinator never learns what the ops do — it schedules `(block, phase)`
// and would schedule them identically for any chain.
//
// The built-in factory is the probes
// ----------------------------------
// `ProbeWorkflows` builds chains out of `probes`, which is exactly what that
// module is for: proving the framework without a real kernel. It means the
// distribution machinery is runnable and verifiable end to end with nothing
// but this crate — and an identity chain is a correctness oracle, because the
// expected output is the input whatever the worker count.

use std::path::PathBuf;
use std::time::Duration;

use serde_json::{json, Value};

use crate::decomposition::{Constraints, Decomposition};
use crate::dtype::Dtype;
use crate::env::{AccountingEnvironment, Environment};
use crate::error::{Error, Result};
use crate::fragment::{append_fragment_phase, FragmentOp};
use crate::op::Chain;
use crate::probes::{
    AffineOp, BlockSummaryOp, FragmentReduceOp, IdentityOp, NeighbourFoldOp, OpaqueOp, WindowSumOp,
};
use crate::sidecar::{check_stream_name, Lifecycle};
use crate::strategy::{Enumerating, Strategy, Workflow};

use super::handout::HandoutPolicy;
use super::shared_volume::SharedVolumes;
use super::wire::{
    array, decomposition_json, flag, get, millis_json, millis_or_none, number_or, real_or, text,
    text_or, triple, triple_or,
};

/// One op, named in terms a factory resolves.
#[derive(Debug, Clone, PartialEq)]
pub struct OpSpec {
    pub kind: String,
    pub name: String,
    pub reach: [usize; 3],
    pub cost: f64,
    pub order: Option<[usize; 3]>,
    /// `affine` only.
    pub scale: f64,
    pub offset: f64,
}

impl OpSpec {
    pub fn new(kind: &str, name: &str, reach: [usize; 3]) -> Self {
        Self {
            kind: kind.to_string(),
            name: name.to_string(),
            reach,
            cost: 1.0,
            order: None,
            scale: 1.0,
            offset: 0.0,
        }
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }

    pub fn with_order(mut self, order: [usize; 3]) -> Self {
        self.order = Some(order);
        self
    }

    pub fn affine(name: &str, scale: f64, offset: f64, reach: [usize; 3]) -> Self {
        let mut spec = Self::new("affine", name, reach);
        spec.scale = scale;
        spec.offset = offset;
        spec
    }

    pub fn to_json(&self) -> Value {
        let mut value = json!({
            "op": self.kind,
            "name": self.name,
            "reach": self.reach,
            "cost": self.cost,
            "scale": self.scale,
            "offset": self.offset,
        });
        if let Some(order) = self.order {
            value["order"] = json!(order);
        }
        value
    }

    pub fn from_json(value: &Value) -> Result<Self> {
        Ok(Self {
            kind: text(value, "op")?,
            name: text_or(value, "name", "op"),
            reach: triple_or(value, "reach", [0, 0, 0]),
            cost: real_or(value, "cost", 1.0),
            order: triple(value, "order").ok(),
            scale: real_or(value, "scale", 1.0),
            offset: real_or(value, "offset", 0.0),
        })
    }
}

/// Where a worker's data lives.
///
/// Two kinds, and the second is not a test double: an environment that only
/// accumulates cost is how a handout policy is scored over a real decomposition
/// with no data and no cluster.
#[derive(Debug, Clone, PartialEq)]
pub enum StoreSpec {
    /// One file per image on a filesystem every worker can see. This is the
    /// stand-in for the shared storage the design assumes; a deployment
    /// supplies its own environment through its own factory.
    Files { dir: PathBuf },
    /// No data at all. Counts reads, writes, voxels and chunks.
    Counting { emptiness: f64, fill_value: f64 },
}

impl StoreSpec {
    pub fn to_json(&self) -> Value {
        match self {
            Self::Files { dir } => json!({"kind": "files", "dir": dir.display().to_string()}),
            Self::Counting {
                emptiness,
                fill_value,
            } => json!({"kind": "counting", "emptiness": emptiness, "fill_value": fill_value}),
        }
    }

    pub fn from_json(value: &Value) -> Result<Self> {
        Ok(match text(value, "kind")?.as_str() {
            "files" => Self::Files {
                dir: PathBuf::from(text(value, "dir")?),
            },
            "counting" => Self::Counting {
                emptiness: real_or(value, "emptiness", 0.0),
                fill_value: real_or(value, "fill_value", 0.0),
            },
            other => {
                return Err(Error::invalid(format!(
                    "{other:?} is not a store this build knows. Built in: \"files\", \
                     \"counting\". A deployment with its own storage registers its own \
                     workflow factory rather than adding a kind here."
                )))
            }
        })
    }
}

/// A per-block sidecar stream the job's workers write.
///
/// Why a *job* says this rather than an op
/// ---------------------------------------
/// A `BlockOp` is handed a buffer and nothing else, so it has no environment to
/// write a fragment through — and widening it is the pending dtype work's
/// business, not this. Meanwhile the property worth demonstrating is a storage
/// property: **fragments written by N worker processes are readable by one
/// merging reader**, which needs producers on several nodes and does not care
/// what produced them.
///
/// So this is the same device `ProbeWorkflows` is: a producer that proves the
/// machinery without a real kernel. A worker that is given a stream writes one
/// [`task_fragment`] per task it completes, and a merge over the result is a
/// real global reduction over real per-block non-pixel output.
///
/// **Superseded for new work by [`FragmentPhaseSpec`]**, which names an actual
/// `fragment::FragmentOp` — an op that consumes and produces fragments through
/// the executor rather than a job-level side effect bolted onto every task.
/// This is kept because it is what the storage property was first demonstrated
/// with, and because it is orthogonal: a job may have both.
#[derive(Debug, Clone, PartialEq)]
pub struct SidecarSpec {
    pub stream: String,
    /// Stated in the job, because it is a decision and has no default.
    pub lifecycle: Lifecycle,
}

impl SidecarSpec {
    pub fn new(stream: impl Into<String>, lifecycle: Lifecycle) -> Self {
        Self {
            stream: stream.into(),
            lifecycle,
        }
    }

    pub fn to_json(&self) -> Value {
        json!({"stream": self.stream, "lifecycle": self.lifecycle.as_str()})
    }

    pub fn from_json(value: &Value) -> Result<Self> {
        let stream = text(value, "stream")?;
        check_stream_name(&stream)?;
        let name = text(value, "lifecycle")?;
        let lifecycle = Lifecycle::parse(&name).ok_or_else(|| {
            Error::invalid(format!(
                "{name:?} is not a sidecar lifecycle. It is \"delete-on-exit\" or \
                 \"persistent\", and there is no default."
            ))
        })?;
        Ok(Self { stream, lifecycle })
    }
}

/// One task's fragment: the block it ran and how many voxels it was trusted
/// for, little-endian.
///
/// Deliberately **not** a serialisation format the framework knows about — the
/// store takes bytes and this is a caller choosing an encoding, which is the
/// whole point of taking bytes. Four `u64`s, because a fixed width makes a
/// truncated fragment detectable by length alone.
pub fn task_fragment(block: [usize; 3], valid_voxels: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(32);
    for value in [block[0], block[1], block[2], valid_voxels] {
        out.extend_from_slice(&(value as u64).to_le_bytes());
    }
    out
}

/// The other half of [`task_fragment`], for whoever merges.
pub fn read_task_fragment(bytes: &[u8]) -> Result<([usize; 3], usize)> {
    if bytes.len() != 32 {
        return Err(Error::invalid(format!(
            "a task fragment is 32 bytes; this one is {}",
            bytes.len()
        )));
    }
    let mut values = [0usize; 4];
    for (slot, word) in bytes.chunks_exact(8).enumerate() {
        values[slot] = u64::from_le_bytes(word.try_into().expect("eight bytes")) as usize;
    }
    Ok(([values[0], values[1], values[2]], values[3]))
}

/// **The distributed barrier probe**: a `fragments -> volume` reduction that
/// declares [`FragmentOp::barrier`] and hoists its fold into
/// [`FragmentOp::reduce`].
///
/// It is [`FragmentReduceOp`] with the fold
/// moved, so the two are directly comparable and the only thing that differs is
/// where the work happens: that one re-derives the total in every block over a
/// whole-lattice fragment reach, this one derives it once for the phase and
/// reaches zero blocks. The answer is the same number and the test suite asserts
/// so.
///
/// **Why it lives here and not with the other probes.** It exists to exercise
/// the *distributed* half of a barrier — that every worker computes the same
/// blob from the shared fragment set with nothing transported — and the wiring
/// that makes that reachable is [`FragmentPhaseSpec`] and
/// [`ProbeWorkflows::fragment_ops`], both in this file. A probe whose whole
/// purpose is one path is easiest to read beside that path.
///
/// The payload is the summed voxel count, packed as one word, with a magic and
/// a version in front of it. That framing is not decoration: it is the mitigation
/// `FragmentOp::reduce` names, because a blob is the op's own encoding and the
/// op's own decode is the only place a mismatch can surface.
pub struct HoistedReduceOp {
    name: &'static str,
    input: String,
    input_phase: usize,
}

impl HoistedReduceOp {
    /// Magic and version, so a block that is handed the wrong bytes says so
    /// rather than reading a plausible number out of them.
    const MAGIC: u64 = 0x6862_6172_7231; // "hbarr1"

    pub fn new(name: &'static str, input: impl Into<String>, input_phase: usize) -> Self {
        Self {
            name,
            input: input.into(),
            input_phase,
        }
    }

    /// The blob's fields, for whoever reads one.
    pub fn read(bytes: &[u8]) -> Result<u64> {
        let words = crate::fragment::unpack_u64(bytes)?;
        match words.as_slice() {
            [magic, total] if *magic == Self::MAGIC => Ok(*total),
            [magic, ..] => Err(Error::invalid(format!(
                "a hoisted reduction begins {:#x}; this blob begins {magic:#x}",
                Self::MAGIC
            ))),
            _ => Err(Error::invalid(format!(
                "a hoisted reduction is two words; this blob is {}. An empty blob is what a \
                 block is handed when its phase declares no barrier, or when it was run \
                 through an entry point that carries no reduction.",
                words.len()
            ))),
        }
    }
}

impl FragmentOp for HoistedReduceOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// **Zero, and that is the point.** The pixel reach of the op this replaces
    /// is the whole volume, which is what forces a whole-volume halo and a
    /// whole-volume fetch in every block. With the answer computed once for the
    /// phase, a block needs nothing outside its own core.
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn writes_pixels(&self) -> bool {
        true
    }

    fn barrier(&self) -> bool {
        true
    }

    /// Nothing is gathered per block: the reach below is zero and the fold is in
    /// [`Self::reduce`].
    fn gathers(&self) -> bool {
        false
    }

    /// **Reach zero, and the stream still declared.** The declaration is what
    /// puts the stream in the plan and what lets `PhaseView` offer it; the reach
    /// is what a *block* gathers, and a block gathers none of it.
    fn inputs(&self) -> Vec<crate::fragment::FragmentInput> {
        vec![
            crate::fragment::FragmentInput::own(self.input.clone(), self.input_phase)
                .with_reach([0, 0, 0]),
        ]
    }

    /// Integer addition over the whole set: associative in the type it
    /// accumulates in, so the answer does not depend on the order the lattice is
    /// walked and the phase is decomposition-invariant. The executor checks that
    /// claim by reducing a second time over the reversed lattice.
    fn seam_fold(&self) -> Option<crate::fragment::SeamFold> {
        Some(crate::fragment::SeamFold::Unordered)
    }

    fn reduce(&self, at: &crate::fragment::PhaseView<'_>) -> Result<Vec<u8>> {
        let mut total = 0u64;
        let mut failed: Option<Error> = None;
        at.stream_fragments(&self.input, &mut |_, bytes| {
            match BlockSummaryOp::read(bytes) {
                Ok((_, voxels, _, _)) => total += voxels as u64,
                Err(err) => failed = Some(err),
            }
            Ok(())
        })?;
        if let Some(err) = failed {
            return Err(err);
        }
        Ok(crate::fragment::pack_u64(&[Self::MAGIC, total]))
    }

    fn apply(&self, at: &crate::fragment::BlockView<'_>) -> Result<crate::fragment::BlockOutput> {
        let total = Self::read(at.reduced)?;
        Ok(crate::fragment::BlockOutput::nothing().with_pixels(at.output_buffer(total as f64)?))
    }
}

/// A fragment phase appended to the job's chain, named in terms a factory
/// resolves.
///
/// The same split as [`OpSpec`], and for the same reason: a `FragmentOp` is not
/// serialisable, so the spec **names** one and the receiving process builds it.
/// The numbers that must not vary between nodes — which stream, from which
/// phase, how far in blocks — travel here, because a worker that reached a
/// different neighbourhood than its peers would produce a plan-dependent answer.
///
/// Fragment phases are appended **after** the chain's phases, which is where
/// `fragment::check_phase_work` requires them: a phase that writes only
/// fragments produces no image for a later pixel phase to read.
#[derive(Debug, Clone, PartialEq)]
pub struct FragmentPhaseSpec {
    /// Which fragment probe. `"summary"`, `"fold"` or `"reduce"`.
    pub kind: String,
    pub name: String,
    /// The stream written. Empty for a phase that writes pixels instead.
    pub stream: String,
    pub lifecycle: Lifecycle,
    /// The stream read, and the phase that wrote it. Empty for a producer.
    pub input: String,
    pub input_phase: usize,
    /// Neighbouring blocks read, in block units.
    pub reach: [usize; 3],
}

impl FragmentPhaseSpec {
    /// `volume -> fragments`: read pixels, write one fragment per block.
    pub fn summary(name: &str, stream: &str, lifecycle: Lifecycle) -> Self {
        Self {
            kind: "summary".to_string(),
            name: name.to_string(),
            stream: stream.to_string(),
            lifecycle,
            input: String::new(),
            input_phase: 0,
            reach: [0, 0, 0],
        }
    }

    /// `fragments -> fragments`: read a neighbourhood, write one fragment.
    pub fn fold(
        name: &str,
        input: &str,
        input_phase: usize,
        reach: [usize; 3],
        stream: &str,
        lifecycle: Lifecycle,
    ) -> Self {
        Self {
            kind: "fold".to_string(),
            name: name.to_string(),
            stream: stream.to_string(),
            lifecycle,
            input: input.to_string(),
            input_phase,
            reach,
        }
    }

    /// `fragments -> volume`: read every fragment, write pixels. `reach` is the
    /// input phase's blocks per axis.
    pub fn reduce(name: &str, input: &str, input_phase: usize, reach: [usize; 3]) -> Self {
        Self {
            kind: "reduce".to_string(),
            name: name.to_string(),
            stream: String::new(),
            lifecycle: Lifecycle::DeleteOnExit,
            input: input.to_string(),
            input_phase,
            reach,
        }
    }

    /// [`Self::reduce`] with a barrier and the fold hoisted: the same answer,
    /// derived once for the phase instead of once per block, over a fragment
    /// reach of zero instead of the whole lattice. See [`HoistedReduceOp`].
    ///
    /// No `reach` argument, and that absence is the declaration: this phase
    /// reaches nothing, which is what the barrier buys.
    pub fn hoisted(name: &str, input: &str, input_phase: usize) -> Self {
        Self {
            kind: "hoisted".to_string(),
            name: name.to_string(),
            stream: String::new(),
            lifecycle: Lifecycle::DeleteOnExit,
            input: input.to_string(),
            input_phase,
            reach: [0, 0, 0],
        }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "op": self.kind,
            "name": self.name,
            "stream": self.stream,
            "lifecycle": self.lifecycle.as_str(),
            "input": self.input,
            "input_phase": self.input_phase,
            "reach": self.reach,
        })
    }

    pub fn from_json(value: &Value) -> Result<Self> {
        let stream = text_or(value, "stream", "");
        if !stream.is_empty() {
            check_stream_name(&stream)?;
        }
        let input = text_or(value, "input", "");
        if !input.is_empty() {
            check_stream_name(&input)?;
        }
        let name = text_or(value, "lifecycle", Lifecycle::DeleteOnExit.as_str());
        let lifecycle = Lifecycle::parse(&name).ok_or_else(|| {
            Error::invalid(format!(
                "{name:?} is not a sidecar lifecycle. It is \"delete-on-exit\" or \
                 \"persistent\", and there is no default."
            ))
        })?;
        Ok(Self {
            kind: text(value, "op")?,
            name: text_or(value, "name", "fragment"),
            stream,
            lifecycle,
            input,
            input_phase: number_or(value, "input_phase", 0) as usize,
            reach: triple_or(value, "reach", [0, 0, 0]),
        })
    }

    /// The runnable op, for a process that has this crate's probes.
    pub fn build(&self, tag: u64) -> Result<Box<dyn FragmentOp>> {
        Ok(match self.kind.as_str() {
            "summary" => Box::new(
                BlockSummaryOp::new(static_name(&self.name), self.stream.clone(), self.lifecycle)
                    .with_tag(tag),
            ),
            "fold" => Box::new(NeighbourFoldOp::new(
                static_name(&self.name),
                self.input.clone(),
                self.input_phase,
                self.reach,
                self.stream.clone(),
                self.lifecycle,
            )),
            "reduce" => Box::new(FragmentReduceOp::new(
                static_name(&self.name),
                self.input.clone(),
                self.input_phase,
                self.reach,
            )),
            "hoisted" => Box::new(HoistedReduceOp::new(
                static_name(&self.name),
                self.input.clone(),
                self.input_phase,
            )),
            other => {
                return Err(Error::invalid(format!(
                    "{other:?} is not a fragment probe. Built in: summary, fold, reduce, \
                     hoisted. A deployment with its own fragment ops registers its own \
                     workflow factory."
                )))
            }
        })
    }
}

/// Everything a worker needs to rebuild the runnable half of a job.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkflowSpec {
    /// Which factory resolves this. `"probe"` is the built-in.
    pub kind: String,
    pub shape: [usize; 3],
    pub dtype: Dtype,
    /// The storage chunk lattice. Used for the read-cost model and for the
    /// coordinator's cache model; it is a property of the arrays, so both ends
    /// have to agree on it and it therefore travels.
    pub chunk: [usize; 3],
    /// The per-worker cache budget the coordinator *models*. Not an instruction
    /// to the worker — the worker's own budget is its own business — but the
    /// model is worthless if the two are wildly different, so it is stated once
    /// in the job rather than guessed per side.
    pub cache_bytes: u64,
    pub ops: Vec<OpSpec>,
    pub store: StoreSpec,
    /// Per-block output that is not a pixel region. `None` — the usual case —
    /// means the job produces nothing but images.
    pub sidecar: Option<SidecarSpec>,
    /// Fragment phases appended after the chain's. Empty — the usual case —
    /// means every phase is `region -> region`.
    pub fragment_phases: Vec<FragmentPhaseSpec>,
}

impl WorkflowSpec {
    pub fn to_json(&self) -> Value {
        let mut value = json!({
            "kind": self.kind,
            "shape": self.shape,
            "dtype": self.dtype.numpy_name(),
            "chunk": self.chunk,
            "cache_bytes": self.cache_bytes,
            "ops": self.ops.iter().map(OpSpec::to_json).collect::<Vec<_>>(),
            "store": self.store.to_json(),
        });
        if let Some(sidecar) = &self.sidecar {
            value["sidecar"] = sidecar.to_json();
        }
        if !self.fragment_phases.is_empty() {
            value["fragment_phases"] = json!(self
                .fragment_phases
                .iter()
                .map(FragmentPhaseSpec::to_json)
                .collect::<Vec<_>>());
        }
        value
    }

    pub fn from_json(value: &Value) -> Result<Self> {
        let name = text_or(value, "dtype", "float64");
        let dtype = Dtype::from_numpy_name(&name)
            .ok_or_else(|| Error::invalid(format!("{name:?} is not an element type")))?;
        Ok(Self {
            kind: text_or(value, "kind", "probe"),
            shape: triple(value, "shape")?,
            dtype,
            chunk: triple_or(value, "chunk", [1, 1, 1]),
            cache_bytes: number_or(value, "cache_bytes", 256 << 20),
            ops: array(value, "ops")?
                .iter()
                .map(OpSpec::from_json)
                .collect::<Result<Vec<_>>>()?,
            store: StoreSpec::from_json(get(value, "store")?)?,
            sidecar: match value.get("sidecar") {
                Some(sidecar) => Some(SidecarSpec::from_json(sidecar)?),
                None => None,
            },
            fragment_phases: match value.get("fragment_phases") {
                Some(entries) => array(value, "fragment_phases")
                    .map(|_| entries)
                    .and_then(|entries| {
                        entries
                            .as_array()
                            .ok_or_else(|| Error::invalid("fragment_phases is not an array"))
                    })?
                    .iter()
                    .map(FragmentPhaseSpec::from_json)
                    .collect::<Result<Vec<_>>>()?,
                None => Vec::new(),
            },
        })
    }
}

/// A job, as submitted and as handed to a worker.
#[derive(Debug, Clone, PartialEq)]
pub struct JobSpec {
    pub id: String,
    pub workflow: WorkflowSpec,
    /// How long a claim survives without a completion before the coordinator
    /// takes the task back and gives it to somebody else — **or `None`, which
    /// is the default and means never.**
    ///
    /// `None` rather than a very large number of milliseconds, and the
    /// difference is the point. A big number is still a deadline: it can be
    /// compared, added to, overflowed, and — worst — it can be *met*, so the
    /// reissue path stays live and the question becomes whether the number was
    /// chosen large enough. `None` is not a deadline at all; there is no
    /// arithmetic to get wrong because there is no operand. A claim handed out
    /// is a claim held until it completes.
    ///
    /// Set it only to opt in to reissue, which the module header explains is
    /// no longer the default deployment's answer to a lost node. When it is
    /// set it must exceed `(ahead + 1) x task duration`, because a worker holds
    /// `ahead` tasks it has not started; a lease shorter than that reissues
    /// work nobody lost. That contract being implicit and unenforceable is a
    /// large part of why the default is `None`.
    pub lease: Option<Duration>,
    pub policy: HandoutPolicy,
}

impl JobSpec {
    pub fn new(id: impl Into<String>, workflow: WorkflowSpec) -> Self {
        Self {
            id: id.into(),
            workflow,
            lease: None,
            policy: HandoutPolicy::default(),
        }
    }

    /// The spec plus the binding decomposition — what `Join` answers with.
    ///
    /// Fallible because the decomposition is: not every plan this crate can
    /// build can be *sent*, and the one that cannot — a phase with a per-block
    /// fetch region — is refused by the sender, where the plan is, rather than
    /// arriving as a fingerprint mismatch nobody can read.
    pub fn to_json(&self, decomposition: &Decomposition) -> Result<Value> {
        Ok(json!({
            "id": self.id,
            "lease_ms": millis_json(self.lease),
            "policy": self.policy.as_str(),
            "workflow": self.workflow.to_json(),
            "decomposition": decomposition_json(decomposition)?,
        }))
    }

    pub fn from_json(value: &Value) -> Result<Self> {
        let policy_name = text_or(value, "policy", HandoutPolicy::default().as_str());
        // `select` and not `parse`: this is a submitted job naming a policy, so
        // it is one of the two boundaries a caller crosses, and a policy that is
        // built but not calibrated is refused here with its reason rather than
        // accepted and scheduled against.
        let policy = HandoutPolicy::select(&policy_name)?;
        Ok(Self {
            id: text_or(value, "id", "job"),
            workflow: WorkflowSpec::from_json(get(value, "workflow")?)?,
            // Absent, `null` and a non-number all mean **no expiry**, which is
            // also the constructor's default, so a spec that never mentions a
            // lease and one that mentions it as `null` agree. The field's
            // shape on the wire is unchanged — a number when there is a lease
            // — so nothing here moves `PROTOCOL_VERSION`; what changed is what
            // its absence means, and it now means the safe thing rather than
            // thirty seconds nobody asked for.
            lease: millis_or_none(value, "lease_ms"),
            policy,
        })
    }

    /// The block edges and split axes a submitted job asks the strategy for.
    ///
    /// Part of the *submission*, not of the handout: choosing a decomposition is
    /// the binding half and happens once, in the coordinator, before any worker
    /// exists.
    pub fn constraints_from_json(value: &Value) -> Constraints {
        let mut constraints = Constraints::default();
        if let Ok(candidates) = super::wire::counts(value, "block_candidates") {
            if !candidates.is_empty() {
                constraints.block_candidates = candidates;
            }
        }
        if let Ok(axes) = super::wire::counts(value, "split_axes") {
            if !axes.is_empty() {
                constraints.split_axes = axes;
            }
        }
        if flag(value, "unbounded", false) {
            constraints.budget_bytes = None;
        }
        constraints
    }
}

/// Turns a [`WorkflowSpec`] into the two things that cannot travel.
///
/// A worker is constructed with one of these. It is the only place a worker
/// binary knows anything about what it is computing, which is what keeps the
/// coordinator, the protocol and the worker loop free of it.
pub trait WorkflowFactory: Send + Sync {
    fn chain(&self, spec: &WorkflowSpec) -> Result<Chain>;
    fn environment(&self, spec: &WorkflowSpec, n_phases: usize) -> Result<Box<dyn Environment>>;

    /// The fragment phases appended after the chain's, in order.
    ///
    /// Defaulted to none, so a factory written before fragment phases existed
    /// keeps compiling and keeps meaning "every phase is `region -> region`".
    /// `tag` identifies this process; a probe that stamps it on its fragments is
    /// what lets a reader tell which node produced one.
    fn fragment_ops(&self, _spec: &WorkflowSpec, _tag: u64) -> Result<Vec<Box<dyn FragmentOp>>> {
        Ok(Vec::new())
    }
}

/// The built-in factory: chains of `probes`, over files or over a counter.
#[derive(Debug, Default, Clone, Copy)]
pub struct ProbeWorkflows;

/// Probe constructors take `&'static str`, and a name that arrived over a wire
/// is not static.
///
/// Leaking is the honest answer here and a bounded one: the names in a job spec
/// are a handful of short strings, leaked once when the job starts, for the
/// life of a process that exists to run that job. The alternative — widening
/// every probe's name to `String` — would change a type the whole test suite
/// uses in order to serve one caller.
fn static_name(name: &str) -> &'static str {
    Box::leak(name.to_string().into_boxed_str())
}

impl WorkflowFactory for ProbeWorkflows {
    fn chain(&self, spec: &WorkflowSpec) -> Result<Chain> {
        if spec.kind != "probe" {
            return Err(Error::invalid(format!(
                "this worker builds {:?} workflows; the job asks for {:?}. A deployment \
                 with its own ops ships its own worker binary registering its own \
                 factory — see `WorkflowFactory`.",
                "probe", spec.kind
            )));
        }
        if spec.ops.is_empty() {
            return Err(Error::invalid("the workflow has no ops".to_string()));
        }
        let mut slots = Vec::with_capacity(spec.ops.len());
        for op in &spec.ops {
            let name = static_name(&op.name);
            slots.push(match op.kind.as_str() {
                "identity" => {
                    let mut built = IdentityOp::new(name, op.reach).with_cost(op.cost);
                    if let Some(order) = op.order {
                        built = built.with_order(order);
                    }
                    Chain::op(built)
                }
                "affine" => {
                    let mut built =
                        AffineOp::new(name, op.scale, op.offset, op.reach).with_cost(op.cost);
                    if let Some(order) = op.order {
                        built = built.with_order(order);
                    }
                    Chain::op(built)
                }
                "opaque" => Chain::op(OpaqueOp::new(name, op.reach).with_cost(op.cost)),
                "window_sum" => {
                    let mut built = WindowSumOp::new(name, op.reach).with_cost(op.cost);
                    if let Some(order) = op.order {
                        built = built.with_order(order);
                    }
                    Chain::op(built)
                }
                other => {
                    return Err(Error::invalid(format!(
                        "{other:?} is not a probe. Built in: identity, affine, opaque, \
                         window_sum."
                    )))
                }
            });
        }
        Ok(Chain::sequence(slots))
    }

    fn fragment_ops(&self, spec: &WorkflowSpec, tag: u64) -> Result<Vec<Box<dyn FragmentOp>>> {
        spec.fragment_phases
            .iter()
            .map(|phase| phase.build(tag))
            .collect()
    }

    fn environment(&self, spec: &WorkflowSpec, n_phases: usize) -> Result<Box<dyn Environment>> {
        Ok(match &spec.store {
            StoreSpec::Files { dir } => {
                Box::new(SharedVolumes::open(dir, spec.shape, spec.chunk, n_phases)?)
            }
            StoreSpec::Counting {
                emptiness,
                fill_value,
            } => Box::new(
                AccountingEnvironment::new(spec.shape, spec.chunk, spec.dtype.size_of() as u64)
                    .with_emptiness(*emptiness, *fill_value),
            ),
        })
    }
}

/// A chain of probes, for building jobs in tests and in the local runner.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainSpec(pub Vec<OpSpec>);

impl ChainSpec {
    /// The correctness oracle: whatever the worker count, the output must equal
    /// the input exactly.
    pub fn identity() -> Self {
        Self(vec![
            OpSpec::new("identity", "first", [2, 0, 0]),
            OpSpec::affine("scaled", 2.0, 1.0, [0, 0, 0]),
        ])
    }

    /// A chain whose output at each voxel is a function of its whole declared
    /// reach, so a short halo diverges at block edges. An identity op cannot
    /// show that, because it never reads its halo.
    pub fn window_sum(radius: [usize; 3]) -> Self {
        Self(vec![OpSpec::new("window_sum", "window", radius)])
    }
}

/// A job over probe ops, decomposed. The runnable example, and what the tests
/// and the local runner both build.
///
/// `blocks` is how many blocks along the split axis, which is what makes the
/// task count predictable in a test that is about scheduling rather than about
/// geometry.
pub fn probe_job(blocks: usize, phases: usize, chain: ChainSpec) -> (JobSpec, Decomposition) {
    probe_job_over(
        blocks,
        phases,
        chain,
        StoreSpec::Counting {
            emptiness: 0.0,
            fill_value: 0.0,
        },
    )
}

/// `probe_job`, against a named store.
pub fn probe_job_over(
    blocks: usize,
    phases: usize,
    chain: ChainSpec,
    store: StoreSpec,
) -> (JobSpec, Decomposition) {
    let edge = 8usize;
    let shape = [edge * blocks.max(1), 8, 8];
    // One op per phase when several are asked for, so the phase count is what
    // the caller said rather than whatever the cost model preferred.
    let mut ops = chain.0;
    while ops.len() < phases {
        let next = ops.len();
        ops.push(OpSpec::new("identity", "extra", [0, 0, 0]).with_order([next % 3, 1, 2]));
    }
    let workflow = WorkflowSpec {
        kind: "probe".to_string(),
        shape,
        dtype: Dtype::F64,
        chunk: [4, 8, 8],
        cache_bytes: 1 << 20,
        ops,
        store,
        sidecar: None,
        fragment_phases: Vec::new(),
    };
    let spec = JobSpec::new("job", workflow);
    let decomposition = decompose(&spec, phases).expect("a probe job decomposes");
    (spec, decomposition)
}

/// Choose the binding decomposition for a job.
///
/// In the coordinator, once, at submission. A worker never calls this — it is
/// given the answer, because "workers receive certainty".
pub fn decompose(spec: &JobSpec, min_phases: usize) -> Result<Decomposition> {
    let factory = ProbeWorkflows;
    let chain = factory.chain(&spec.workflow)?;
    let workflow = Workflow::new(chain, spec.workflow.shape, spec.workflow.dtype);
    let constraints = Constraints {
        block_candidates: vec![8],
        split_axes: vec![0],
        ..Default::default()
    };
    let mut decomposition = Enumerating::default().decompose(&workflow, &constraints)?;
    if decomposition.n_phases() < min_phases {
        // Force the split the caller asked for. Legitimate: the partition is a
        // cost-model choice, and every partition of the same chain produces the
        // same output — that is the property the conformance suite rests on.
        decomposition = split_every_op(&workflow, &constraints)?;
    }
    // Fragment phases go after the chain's, on the last phase's lattice, so the
    // block indices a fragment is keyed by mean the same thing on both sides.
    // Done here rather than in the worker because the decomposition is the
    // binding half: every worker must be *given* it, not derive it.
    for phase in &spec.workflow.fragment_phases {
        let op = phase.build(0)?;
        decomposition = append_fragment_phase(decomposition, op.as_ref())?;
    }
    Ok(decomposition)
}

fn split_every_op(workflow: &Workflow, constraints: &Constraints) -> Result<Decomposition> {
    use crate::decomposition::{summarise_slots, PhaseDecomposition};
    use crate::geometry::BlockGrid;
    let slots = workflow.chain.slots();
    let mut phases = Vec::with_capacity(slots.len());
    for slot in 0..slots.len() {
        let (reach, _, names, _) = summarise_slots(&slots, &[slot], workflow.shape)?;
        let grid = BlockGrid::along(
            workflow.shape,
            &constraints.split_axes,
            constraints.block_candidates[0],
        )?;
        phases.push(PhaseDecomposition::derive(
            vec![slot],
            names,
            reach.clone(),
            reach,
            grid,
        ));
    }
    let decomposition = Decomposition {
        volume: workflow.shape,
        dtype: workflow.dtype,
        phases,
        chain_reach: workflow.chain.reach3(&workflow.shape),
    };
    decomposition.check()?;
    Ok(decomposition)
}

/// Read a job spec, and its constraints, from a JSON document.
pub fn read_job(path: &std::path::Path) -> Result<(JobSpec, Decomposition)> {
    let text = std::fs::read_to_string(path)
        .map_err(|err| Error::backend(format!("reading {}: {err}", path.display())))?;
    let document: Value = serde_json::from_str(&text)
        .map_err(|err| Error::invalid(format!("{} is not JSON: {err}", path.display())))?;
    let spec = JobSpec::from_json(&document)?;
    let decomposition = match document.get("decomposition") {
        // A spec may carry a decomposition, in which case it is binding and is
        // used as given — that is how a run is reproduced exactly.
        Some(value) => super::wire::decomposition_from_json(value)?,
        None => {
            let factory = ProbeWorkflows;
            let chain = factory.chain(&spec.workflow)?;
            let workflow = Workflow::new(chain, spec.workflow.shape, spec.workflow.dtype);
            let constraints = JobSpec::constraints_from_json(&document);
            Enumerating::default().decompose(&workflow, &constraints)?
        }
    };
    Ok((spec, decomposition))
}

/// How many tasks the coordinator will have to hand out, without building the
/// graph. Used by the local runner to say what it is about to do.
pub fn task_count(decomposition: &Decomposition) -> usize {
    decomposition.n_tasks()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_job_spec_survives_the_wire_with_its_decomposition() {
        let (spec, decomposition) = probe_job(4, 2, ChainSpec::identity());
        let document = spec.to_json(&decomposition).unwrap();
        let rebuilt = JobSpec::from_json(&document).unwrap();
        assert_eq!(rebuilt, spec);
        let rebuilt_decomposition =
            super::super::wire::decomposition_from_json(document.get("decomposition").unwrap())
                .unwrap();
        assert_eq!(rebuilt_decomposition, decomposition);
    }

    #[test]
    fn the_factory_builds_the_chain_the_spec_names() {
        let (spec, _) = probe_job(2, 1, ChainSpec::identity());
        let chain = ProbeWorkflows.chain(&spec.workflow).unwrap();
        let names: Vec<String> = chain
            .slots()
            .iter()
            .map(|slot| slot.display_name())
            .collect();
        assert_eq!(names, vec!["first".to_string(), "scaled".to_string()]);
        assert_eq!(chain.reach3(&spec.workflow.shape), [2, 0, 0]);
    }

    #[test]
    fn a_workflow_kind_this_build_cannot_run_is_refused_by_name() {
        let (mut spec, _) = probe_job(2, 1, ChainSpec::identity());
        spec.workflow.kind = "deconvolution-pipeline".to_string();
        let error = match ProbeWorkflows.chain(&spec.workflow) {
            Ok(_) => panic!("a workflow kind this build cannot run should be refused"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("ships its own worker"),
            "{error}"
        );
    }

    #[test]
    fn a_store_this_build_cannot_open_is_refused_by_name() {
        let error = StoreSpec::from_json(&json!({"kind": "s3"})).unwrap_err();
        assert!(error.to_string().contains("registers its own"), "{error}");
    }

    #[test]
    fn asking_for_more_phases_gets_more_phases() {
        let (_, one) = probe_job(4, 1, ChainSpec::identity());
        let (_, three) = probe_job(4, 3, ChainSpec::identity());
        assert!(three.n_phases() >= 3, "{}", three.n_phases());
        assert!(three.n_tasks() > one.n_tasks());
        three.check().unwrap();
    }
}
