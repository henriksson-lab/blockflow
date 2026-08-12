// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// Block-keyed output that is not a pixel region.
//
// Why this exists
// ---------------
// The executor can write exactly one kind of thing: a valid region of a level.
// That covers every op whose output is an image, and it covers nothing else —
// yet several real steps produce, per block, a **fragment** that a later global
// step merges: incident lists, connected-component boundaries, per-block
// displacement estimates a least-squares solve consumes. Every one of them has
// the same shape — *a block produces a fragment; a global step merges
// fragments* — and every one of them has, until now, had nowhere to put its
// output. Which is why such work either stayed outside the executor entirely or
// was held in memory, and holding it in memory is precisely what pins a stage to
// one node.
//
// Three decisions, and each is what makes the rest tractable
// ----------------------------------------------------------
// **1. The key is the decomposition's own unit: `(stream, phase, block)`.** Not
// a fresh identifier scheme. The block index is what the planner, the task
// graph, the scheduler, the handout policy and the cache model already speak in,
// so a sidecar inherits all of it for free: the same task that computes a block
// writes that block's fragment, a reissued task overwrites its own key and
// nobody else's, and "did every block produce one?" is answered against the same
// grid every other coverage question is answered against.
//
// **2. The value is `&[u8]`, and no format is imposed.** Not serde, not JSON,
// not a self-describing container. The op knows what it is writing and the
// merge knows what it is reading; anything in between would be this crate
// having an opinion about a type it has never seen. It is the refusal to
// support *types* that makes supporting *arbitrary* types possible — a
// framework that took `T: Serialize` would support the types its author
// imagined, and this supports bytes.
//
// **3. Plain objects, not an array.** One object per `(stream, phase, block)`,
// on a filesystem or in an object store. A chunked array format can express
// variable-length data, but awkwardly, and none of the three things it would buy
// — a chunk grid, codecs, partial reads — applies to a blob whose length is
// whatever the op decided. A prefix and a name is the whole storage model, which
// is also why it ports to S3 without a design.
//
// Where merging happens: not here
// -------------------------------
// A merge is a **global reduction**, not block-parallel work. Expressing it in
// the task DAG would need a fan-in node the DAG does not have and does not
// otherwise want, and would put a sequential step inside a structure whose whole
// value is that its nodes are independent. So the caller runs the block phase,
// reads the fragments back, and merges them itself. Multi-node needs nothing
// extra for that: the fragments are on shared storage, so whichever process
// merges simply reads them, exactly as a worker reads an intermediate level
// another worker wrote.
//
// Lifecycle is declared, never defaulted
// --------------------------------------
// A stream is created as [`Lifecycle::DeleteOnExit`] or
// [`Lifecycle::Persistent`], and there is no default — `Lifecycle` deliberately
// does not implement `Default`. Keeping intermediate output deliberately, for
// debugging, and cleaning it up automatically are both legitimate, and which one
// a stream wants is a property of the stream. What is not legitimate is drifting
// into either by omission.
//
// The deletion has a bug to learn from
// ------------------------------------
// A recorded defect in the application this crate came from: a cleanup step
// called `remove_file` on an intermediate that had become a *directory*, dropped
// the resulting error, and so never deleted anything — in the default
// configuration, for as long as nobody looked. The failure was not the wrong
// call; it was that the wrong call was **silent**. So deletion here:
//
// * **reports what it removed** — [`Discarded`] carries every stream, its
//   fragment count and its bytes, and an event is emitted per stream, so a
//   caller that ignores the return value still leaves a record;
// * **fails rather than swallows** — any removal error is returned, naming the
//   path;
// * **confirms the absence afterwards**, because an error that is not raised
//   cannot be propagated, and the recorded bug was exactly a removal that
//   reported success in the sense that mattered to the caller.
//
// It is also why "delete on exit" is an explicit call rather than a `Drop`
// implementation. A `Drop` has nowhere to report to and no way to fail loudly,
// which is the shape of the bug, not the fix.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use crate::error::{Error, Result};
use crate::listener::EventListener;
use crate::log::Event;

/// What happens to a stream's fragments when the run is over.
///
/// No `Default`, on purpose: see the module header. A stream that nobody chose
/// a lifecycle for is a stream nobody decided about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Lifecycle {
    /// The caller will call [`Sidecars::discard`] and the fragments go away.
    DeleteOnExit,
    /// The fragments stay, deliberately — kept for inspection, for a later run,
    /// or because they are the output.
    Persistent,
}

impl Lifecycle {
    /// The token written into storage, so a different process can read a
    /// stream's lifecycle rather than being told it.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::DeleteOnExit => "delete-on-exit",
            Self::Persistent => "persistent",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        match text.trim() {
            "delete-on-exit" => Some(Self::DeleteOnExit),
            "persistent" => Some(Self::Persistent),
            _ => None,
        }
    }
}

/// Which fragment: a stream, a phase, and a block of that phase's grid.
///
/// Ordered stream-major so that iterating a stream's keys is a contiguous,
/// deterministic sweep — a merge that wants a reproducible reduction order gets
/// one without sorting on anything of its own.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FragmentKey {
    pub stream: String,
    pub phase: usize,
    pub block: [usize; 3],
}

impl FragmentKey {
    pub fn new(stream: impl Into<String>, phase: usize, block: [usize; 3]) -> Self {
        Self {
            stream: stream.into(),
            phase,
            block,
        }
    }
}

/// What removing one stream removed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamRemoval {
    pub stream: String,
    pub fragments: usize,
    pub bytes: u64,
    /// Where it was — a directory, a key prefix — so the report says what a
    /// person could have gone and looked at.
    pub location: String,
}

/// What a [`Sidecars::discard`] did, in full.
///
/// Returned *and* emitted as events, because the recorded bug this is arranged
/// against was a cleanup nobody could see had not happened.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Discarded {
    pub removed: Vec<StreamRemoval>,
    /// Streams left alone because they are [`Lifecycle::Persistent`]. Named, so
    /// that "why is this still here" has an answer in the same report.
    pub kept: Vec<String>,
    /// Present in the store with no declared lifecycle. Not removed — the
    /// lifecycle is the licence to remove — but reported rather than passed
    /// over, because a stream nobody claims is how storage fills up quietly.
    pub undeclared: Vec<String>,
}

impl Discarded {
    pub fn fragments(&self) -> usize {
        self.removed.iter().map(|entry| entry.fragments).sum()
    }

    pub fn bytes(&self) -> u64 {
        self.removed.iter().map(|entry| entry.bytes).sum()
    }

    /// One line a caller can log. Deliberately says the *numbers*: "cleaned up"
    /// with no count is indistinguishable from the bug.
    pub fn describe(&self) -> String {
        let mut parts = vec![format!(
            "removed {} fragment(s), {} byte(s), from {} stream(s)",
            self.fragments(),
            self.bytes(),
            self.removed.len()
        )];
        if !self.kept.is_empty() {
            parts.push(format!("kept {}", self.kept.join(", ")));
        }
        if !self.undeclared.is_empty() {
            parts.push(format!(
                "undeclared and therefore left: {}",
                self.undeclared.join(", ")
            ));
        }
        parts.join("; ")
    }
}

/// Bytes in, bytes out, keyed by `(stream, phase, block)`.
///
/// Implementations do storage and nothing else: declaration policy, event
/// emission and the discard report all live in [`Sidecars`], so a second
/// backend cannot get them subtly different.
pub trait SidecarBackend: Send + Sync {
    /// For messages: a path, a bucket, "memory".
    fn describe(&self) -> String;

    /// Record `stream`'s lifecycle **durably**, so another process can read it.
    ///
    /// Idempotent for the same lifecycle — every worker in a distributed run
    /// declares the streams it writes, and they must not have to coordinate.
    /// A *conflicting* lifecycle is an error: two callers disagreeing about
    /// whether output survives is a disagreement worth stopping on.
    fn declare(&self, stream: &str, lifecycle: Lifecycle) -> Result<()>;

    fn lifecycle(&self, stream: &str) -> Result<Option<Lifecycle>>;

    /// Every stream present, with its lifecycle if it has one. `None` means the
    /// store holds something nobody declared.
    fn streams(&self) -> Result<Vec<(String, Option<Lifecycle>)>>;

    /// Write one fragment, replacing any fragment already at that key.
    ///
    /// Must be atomic against a reader: a task that is reissued after a worker
    /// died rewrites its own key, and a merge that read a half-written fragment
    /// would be the well-formed-and-wrong failure this crate exists to prevent.
    fn put(&self, key: &FragmentKey, bytes: &[u8]) -> Result<()>;

    fn get(&self, key: &FragmentKey) -> Result<Option<Vec<u8>>>;

    /// Every key of `stream`, in [`FragmentKey`] order. A stream nobody wrote
    /// to is empty, not an error.
    fn keys(&self, stream: &str) -> Result<Vec<FragmentKey>>;

    /// Remove every fragment of `stream`, and the stream itself.
    ///
    /// Must report what it removed and must **fail rather than swallow**. See
    /// the module header for the bug this rule comes from.
    fn remove_stream(&self, stream: &str) -> Result<StreamRemoval>;
}

/// A stream name is a storage path component, so it is checked rather than
/// trusted.
///
/// Restrictive on purpose: this is the one place a caller-supplied string
/// becomes a filesystem path, and `..` reaching outside the store would be a
/// security problem rather than a naming one.
pub fn check_stream_name(stream: &str) -> Result<()> {
    let ok = !stream.is_empty()
        && stream.len() <= 128
        && !stream.starts_with('.')
        && stream
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.');
    if ok {
        Ok(())
    } else {
        Err(Error::invalid(format!(
            "{stream:?} is not a usable stream name. A stream name becomes a storage path \
             component, so it must be 1-128 characters of ASCII letters, digits, '_', '-' or \
             '.', and must not start with '.'."
        )))
    }
}

// ------------------------------------------------------------- the store --

struct Registered {
    listener: Arc<dyn EventListener>,
    faulted: AtomicBool,
}

/// The block-keyed sidecar store, as an [`crate::env::Environment`] exposes it.
///
/// Wraps a [`SidecarBackend`] with the three things that must not vary between
/// backends: a stream must be declared before it is written, every write and
/// read is reported through the crate's own event stream, and discarding
/// reports what it did.
pub struct Sidecars {
    backend: Box<dyn SidecarBackend>,
    /// Held behind a lock and attachable after construction, because an
    /// environment is built by a factory that has never heard of listeners —
    /// the distributed worker builds its environment from a job spec and only
    /// then knows where its events go. The lock is taken once per fragment,
    /// which is per block, not per voxel.
    listeners: RwLock<Vec<Registered>>,
    /// Streams this handle has seen declared, so the per-write "is it
    /// declared?" check is a set lookup rather than a storage round trip. It is
    /// a *cache*, never an authority: a stream missing from it falls through to
    /// the backend, which is what makes a process that declared nothing — a
    /// merging reader, say — behave correctly.
    ///
    /// Worth having because the check is per block: on a shared filesystem an
    /// uncached one would be an extra open and read per fragment, which is the
    /// kind of cost that only shows up on a cluster.
    known: RwLock<BTreeMap<String, Lifecycle>>,
}

impl Sidecars {
    pub fn new(backend: impl SidecarBackend + 'static) -> Self {
        Self {
            backend: Box::new(backend),
            listeners: RwLock::new(Vec::new()),
            known: RwLock::new(BTreeMap::new()),
        }
    }

    /// In-memory fragments. What every environment that is not backed by shared
    /// storage uses.
    pub fn in_memory() -> Self {
        Self::new(MemorySidecars::default())
    }

    /// Report this store's activity to `listener` from now on.
    ///
    /// Additive and idempotent-ish by construction: registering the same
    /// listener twice reports twice, which is the caller's business.
    pub fn attach(&self, listener: Arc<dyn EventListener>) {
        self.listeners
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(Registered {
                listener,
                faulted: AtomicBool::new(false),
            });
    }

    pub fn attach_all(&self, listeners: &[Arc<dyn EventListener>]) {
        for listener in listeners {
            self.attach(Arc::clone(listener));
        }
    }

    pub fn describe(&self) -> String {
        self.backend.describe()
    }

    /// Isolated exactly as the executor's dispatch is: a listener that panics
    /// is disabled and the run continues. Observation may not fail the thing it
    /// observes.
    fn emit(&self, event: Event) {
        let listeners = self
            .listeners
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for registered in listeners.iter() {
            if registered.faulted.load(Ordering::Relaxed) {
                continue;
            }
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                registered.listener.on_event(&event)
            }));
            if outcome.is_err() {
                registered.faulted.store(true, Ordering::Relaxed);
            }
        }
    }

    /// Create `stream`, or agree with what is already there.
    pub fn declare(&self, stream: &str, lifecycle: Lifecycle) -> Result<()> {
        check_stream_name(stream)?;
        self.backend.declare(stream, lifecycle)?;
        self.known
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(stream.to_string(), lifecycle);
        Ok(())
    }

    pub fn lifecycle(&self, stream: &str) -> Result<Option<Lifecycle>> {
        check_stream_name(stream)?;
        if let Some(known) = self
            .known
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(stream)
        {
            return Ok(Some(*known));
        }
        let found = self.backend.lifecycle(stream)?;
        if let Some(lifecycle) = found {
            self.known
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .insert(stream.to_string(), lifecycle);
        }
        Ok(found)
    }

    /// Forget what this handle believes about `stream`. Called after a removal,
    /// so a stream that comes back is looked up again rather than assumed.
    fn forget(&self, stream: &str) {
        self.known
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(stream);
    }

    pub fn streams(&self) -> Result<Vec<(String, Option<Lifecycle>)>> {
        self.backend.streams()
    }

    /// Write one block's fragment.
    ///
    /// Refuses an undeclared stream, and says why: the lifecycle is a decision
    /// somebody has to make, and inventing one here is how a default nobody
    /// chose becomes the behaviour everybody gets.
    pub fn write(&self, stream: &str, phase: usize, block: [usize; 3], bytes: &[u8]) -> Result<()> {
        if self.lifecycle(stream)?.is_none() {
            return Err(Error::invalid(format!(
                "sidecar stream {stream:?} has not been declared, so there is nothing to say \
                 whether its fragments survive the run. Declare it first, with \
                 Lifecycle::DeleteOnExit or Lifecycle::Persistent — the choice is deliberate \
                 and has no default."
            )));
        }
        let key = FragmentKey::new(stream, phase, block);
        let started = Instant::now();
        self.backend.put(&key, bytes)?;
        self.emit(Event::SidecarWritten {
            stream: stream.to_string(),
            phase,
            index: block,
            bytes: bytes.len() as u64,
            duration_ns: started.elapsed().as_nanos() as u64,
        });
        Ok(())
    }

    /// One block's fragment, or `None` if that block wrote none.
    pub fn read(&self, stream: &str, phase: usize, block: [usize; 3]) -> Result<Option<Vec<u8>>> {
        check_stream_name(stream)?;
        let key = FragmentKey::new(stream, phase, block);
        let started = Instant::now();
        let found = self.backend.get(&key)?;
        self.emit(Event::SidecarRead {
            stream: stream.to_string(),
            phase,
            index: block,
            bytes: found.as_ref().map(|bytes| bytes.len() as u64).unwrap_or(0),
            found: found.is_some(),
            duration_ns: started.elapsed().as_nanos() as u64,
        });
        Ok(found)
    }

    /// Every key in `stream`. A stream nobody wrote to is empty, not an error —
    /// a phase in which no block had anything to say is a normal outcome.
    pub fn keys(&self, stream: &str) -> Result<Vec<FragmentKey>> {
        check_stream_name(stream)?;
        self.backend.keys(stream)
    }

    /// Every fragment in `stream`, keyed, in key order. The merge side's whole
    /// interface: read them, reduce them, done.
    pub fn fragments(&self, stream: &str) -> Result<Vec<(FragmentKey, Vec<u8>)>> {
        let mut out = Vec::new();
        for key in self.keys(stream)? {
            let started = Instant::now();
            let Some(bytes) = self.backend.get(&key)? else {
                // Listed and then gone: somebody is removing fragments while a
                // merge is reading them, which is a coordination bug in the
                // caller and not something to paper over with a skip.
                return Err(Error::backend(format!(
                    "sidecar fragment {key:?} was listed by {} and then could not be read. \
                     A stream must not be discarded while it is being merged.",
                    self.backend.describe()
                )));
            };
            self.emit(Event::SidecarRead {
                stream: key.stream.clone(),
                phase: key.phase,
                index: key.block,
                bytes: bytes.len() as u64,
                found: true,
                duration_ns: started.elapsed().as_nanos() as u64,
            });
            out.push((key, bytes));
        }
        Ok(out)
    }

    /// Remove every [`Lifecycle::DeleteOnExit`] stream, and say what went.
    ///
    /// This is the "exit" in delete-on-exit, and it is a call rather than a
    /// `Drop` for the reason the module header gives: a destructor has nowhere
    /// to report to and no way to fail loudly.
    pub fn discard(&self) -> Result<Discarded> {
        let mut report = Discarded::default();
        for (stream, lifecycle) in self.backend.streams()? {
            match lifecycle {
                Some(Lifecycle::DeleteOnExit) => {
                    let removal = self.backend.remove_stream(&stream)?;
                    self.forget(&stream);
                    self.emit(Event::SidecarDiscarded {
                        stream: removal.stream.clone(),
                        fragments: removal.fragments,
                        bytes: removal.bytes,
                    });
                    report.removed.push(removal);
                }
                Some(Lifecycle::Persistent) => report.kept.push(stream),
                None => report.undeclared.push(stream),
            }
        }
        Ok(report)
    }

    /// Remove one stream whatever its lifecycle, and say what went.
    ///
    /// For a caller that has finished with a stream early. Same reporting, same
    /// loudness; the only difference is that the licence is this call rather
    /// than the declaration.
    pub fn remove(&self, stream: &str) -> Result<StreamRemoval> {
        check_stream_name(stream)?;
        let removal = self.backend.remove_stream(stream)?;
        self.forget(stream);
        self.emit(Event::SidecarDiscarded {
            stream: removal.stream.clone(),
            fragments: removal.fragments,
            bytes: removal.bytes,
        });
        Ok(removal)
    }
}

impl std::fmt::Debug for Sidecars {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Sidecars")
            .field("backend", &self.backend.describe())
            .finish()
    }
}

// ------------------------------------------------------------ in memory --

/// Fragments in a map. What an in-process environment uses.
///
/// Not a stub: the simulated environment counts sidecar traffic like any other,
/// and a fragment's bytes are the *caller's* — unlike a voxel, they cannot be
/// fabricated from a region, so a store that discarded them would break any
/// caller that read one back.
#[derive(Debug, Default)]
pub struct MemorySidecars {
    declared: RwLock<BTreeMap<String, Lifecycle>>,
    fragments: RwLock<BTreeMap<FragmentKey, Vec<u8>>>,
}

impl SidecarBackend for MemorySidecars {
    fn describe(&self) -> String {
        "memory".to_string()
    }

    fn declare(&self, stream: &str, lifecycle: Lifecycle) -> Result<()> {
        let mut declared = self
            .declared
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match declared.get(stream) {
            Some(existing) if *existing != lifecycle => Err(conflict(stream, *existing, lifecycle)),
            Some(_) => Ok(()),
            None => {
                declared.insert(stream.to_string(), lifecycle);
                Ok(())
            }
        }
    }

    fn lifecycle(&self, stream: &str) -> Result<Option<Lifecycle>> {
        Ok(self
            .declared
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(stream)
            .copied())
    }

    fn streams(&self) -> Result<Vec<(String, Option<Lifecycle>)>> {
        Ok(self
            .declared
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .map(|(stream, lifecycle)| (stream.clone(), Some(*lifecycle)))
            .collect())
    }

    fn put(&self, key: &FragmentKey, bytes: &[u8]) -> Result<()> {
        self.fragments
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(key.clone(), bytes.to_vec());
        Ok(())
    }

    fn get(&self, key: &FragmentKey) -> Result<Option<Vec<u8>>> {
        Ok(self
            .fragments
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(key)
            .cloned())
    }

    fn keys(&self, stream: &str) -> Result<Vec<FragmentKey>> {
        Ok(self
            .fragments
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .keys()
            .filter(|key| key.stream == stream)
            .cloned()
            .collect())
    }

    fn remove_stream(&self, stream: &str) -> Result<StreamRemoval> {
        let mut fragments = self
            .fragments
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let going: Vec<FragmentKey> = fragments
            .keys()
            .filter(|key| key.stream == stream)
            .cloned()
            .collect();
        let mut bytes = 0u64;
        for key in &going {
            if let Some(value) = fragments.remove(key) {
                bytes += value.len() as u64;
            }
        }
        drop(fragments);
        self.declared
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(stream);
        Ok(StreamRemoval {
            stream: stream.to_string(),
            fragments: going.len(),
            bytes,
            location: "memory".to_string(),
        })
    }
}

fn conflict(stream: &str, existing: Lifecycle, asked: Lifecycle) -> Error {
    Error::invalid(format!(
        "sidecar stream {stream:?} was declared {} and is now being declared {}. Whether \
         output survives a run is a deliberate decision, so two callers disagreeing about it \
         is a disagreement to settle rather than to resolve by last-writer-wins.",
        existing.as_str(),
        asked.as_str()
    ))
}

// ------------------------------------------------------------ on a disk --

/// One object per fragment, under a directory every process can see.
///
/// Layout, chosen so it ports to an object store unchanged:
///
/// ```text
/// <root>/<stream>/lifecycle                    "delete-on-exit" | "persistent"
/// <root>/<stream>/phase-<p>/block-<i>-<j>-<k>.bin
/// ```
///
/// The lifecycle lives **in storage** rather than in the declaring process,
/// which is what lets a different process — the one that merges, or a person
/// cleaning up after a failed run — know whether a stream is meant to survive.
#[derive(Debug)]
pub struct FileSidecars {
    root: PathBuf,
}

const LIFECYCLE_FILE: &str = "lifecycle";
const FRAGMENT_PREFIX: &str = "block-";
const FRAGMENT_SUFFIX: &str = ".bin";

impl FileSidecars {
    /// Open, creating the root if it is not there.
    pub fn at(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        fs::create_dir_all(&root)
            .map_err(|err| Error::backend(format!("creating {}: {err}", root.display())))?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn stream_dir(&self, stream: &str) -> PathBuf {
        self.root.join(stream)
    }

    fn phase_dir(&self, stream: &str, phase: usize) -> PathBuf {
        self.stream_dir(stream).join(format!("phase-{phase}"))
    }

    fn fragment_path(&self, key: &FragmentKey) -> PathBuf {
        self.phase_dir(&key.stream, key.phase).join(format!(
            "{FRAGMENT_PREFIX}{}-{}-{}{FRAGMENT_SUFFIX}",
            key.block[0], key.block[1], key.block[2]
        ))
    }

    fn parse_phase(name: &str) -> Option<usize> {
        name.strip_prefix("phase-")?.parse().ok()
    }

    fn parse_block(name: &str) -> Option<[usize; 3]> {
        let body = name
            .strip_prefix(FRAGMENT_PREFIX)?
            .strip_suffix(FRAGMENT_SUFFIX)?;
        let mut parts = body.split('-');
        let block = [
            parts.next()?.parse().ok()?,
            parts.next()?.parse().ok()?,
            parts.next()?.parse().ok()?,
        ];
        parts.next().is_none().then_some(block)
    }

    /// Remove a path, using the call that fits what is actually there, and
    /// **confirm it is gone**.
    ///
    /// The two halves are both the lesson from the recorded bug: `remove_file`
    /// on a directory fails, and a failure that is dropped is a cleanup that
    /// never happened and never said so. Here the error is returned with the
    /// path in it, and the absence is checked afterwards so that a backend that
    /// somehow succeeds without removing anything is caught too.
    fn remove_confirmed(path: &Path) -> Result<()> {
        let metadata = match fs::symlink_metadata(path) {
            Ok(metadata) => metadata,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(err) => {
                return Err(Error::backend(format!(
                    "inspecting {} before removing it: {err}",
                    path.display()
                )))
            }
        };
        let outcome = if metadata.is_dir() {
            fs::remove_dir_all(path)
        } else {
            fs::remove_file(path)
        };
        outcome.map_err(|err| {
            Error::backend(format!(
                "removing {} ({}): {err}. A sidecar stream declared delete-on-exit must \
                 actually go; this is reported rather than dropped because a cleanup that \
                 silently does nothing is indistinguishable from one that worked.",
                path.display(),
                if metadata.is_dir() {
                    "a directory"
                } else {
                    "a file"
                }
            ))
        })?;
        if fs::symlink_metadata(path).is_ok() {
            return Err(Error::backend(format!(
                "removing {} reported success and the path is still there. Refusing to \
                 report a cleanup that did not happen.",
                path.display()
            )));
        }
        Ok(())
    }

    fn read_dir(path: &Path) -> Result<Vec<fs::DirEntry>> {
        let mut entries = Vec::new();
        let listing = match fs::read_dir(path) {
            Ok(listing) => listing,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(entries),
            Err(err) => return Err(Error::backend(format!("listing {}: {err}", path.display()))),
        };
        for entry in listing {
            entries.push(
                entry
                    .map_err(|err| Error::backend(format!("listing {}: {err}", path.display())))?,
            );
        }
        entries.sort_by_key(fs::DirEntry::file_name);
        Ok(entries)
    }
}

impl SidecarBackend for FileSidecars {
    fn describe(&self) -> String {
        self.root.display().to_string()
    }

    fn declare(&self, stream: &str, lifecycle: Lifecycle) -> Result<()> {
        if let Some(existing) = self.lifecycle(stream)? {
            if existing != lifecycle {
                return Err(conflict(stream, existing, lifecycle));
            }
            return Ok(());
        }
        let dir = self.stream_dir(stream);
        fs::create_dir_all(&dir)
            .map_err(|err| Error::backend(format!("creating {}: {err}", dir.display())))?;
        let marker = dir.join(LIFECYCLE_FILE);
        // Written through a rename, so a second process either sees no marker
        // or sees a complete one. Several workers declaring the same stream at
        // the same instant is the normal case, not a race to avoid.
        let temporary = dir.join(format!(
            ".{LIFECYCLE_FILE}-{}-{}",
            std::process::id(),
            nanos()
        ));
        fs::write(&temporary, lifecycle.as_str())
            .map_err(|err| Error::backend(format!("writing {}: {err}", temporary.display())))?;
        fs::rename(&temporary, &marker).map_err(|err| {
            let _ = fs::remove_file(&temporary);
            Error::backend(format!("publishing {}: {err}", marker.display()))
        })?;
        // Two processes may have written different lifecycles concurrently, in
        // which case one rename won. Read it back rather than assume.
        match self.lifecycle(stream)? {
            Some(settled) if settled == lifecycle => Ok(()),
            Some(settled) => Err(conflict(stream, settled, lifecycle)),
            None => Err(Error::backend(format!(
                "declaring sidecar stream {stream:?} left no lifecycle at {}",
                marker.display()
            ))),
        }
    }

    fn lifecycle(&self, stream: &str) -> Result<Option<Lifecycle>> {
        let marker = self.stream_dir(stream).join(LIFECYCLE_FILE);
        match fs::read_to_string(&marker) {
            Ok(text) => Lifecycle::parse(&text).map(Some).ok_or_else(|| {
                Error::invalid(format!(
                    "{} says {text:?}, which is not a sidecar lifecycle",
                    marker.display()
                ))
            }),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(Error::backend(format!(
                "reading {}: {err}",
                marker.display()
            ))),
        }
    }

    fn streams(&self) -> Result<Vec<(String, Option<Lifecycle>)>> {
        let mut out = Vec::new();
        for entry in Self::read_dir(&self.root)? {
            if !entry.path().is_dir() {
                continue;
            }
            let Some(name) = entry.file_name().to_str().map(str::to_string) else {
                continue;
            };
            if check_stream_name(&name).is_err() {
                continue;
            }
            let lifecycle = self.lifecycle(&name)?;
            out.push((name, lifecycle));
        }
        Ok(out)
    }

    fn put(&self, key: &FragmentKey, bytes: &[u8]) -> Result<()> {
        let dir = self.phase_dir(&key.stream, key.phase);
        fs::create_dir_all(&dir)
            .map_err(|err| Error::backend(format!("creating {}: {err}", dir.display())))?;
        // Write-then-rename: a reader either sees the previous fragment or the
        // whole new one, never a prefix. The case that needs it is a task
        // reissued after a worker died, which rewrites a key a merge may
        // already be reading.
        let temporary = dir.join(format!(
            ".tmp-{}-{}-{}-{}-{}",
            std::process::id(),
            nanos(),
            key.block[0],
            key.block[1],
            key.block[2]
        ));
        fs::write(&temporary, bytes)
            .map_err(|err| Error::backend(format!("writing {}: {err}", temporary.display())))?;
        let path = self.fragment_path(key);
        fs::rename(&temporary, &path).map_err(|err| {
            let _ = fs::remove_file(&temporary);
            Error::backend(format!("publishing {}: {err}", path.display()))
        })
    }

    fn get(&self, key: &FragmentKey) -> Result<Option<Vec<u8>>> {
        let path = self.fragment_path(key);
        match fs::read(&path) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(Error::backend(format!("reading {}: {err}", path.display()))),
        }
    }

    fn keys(&self, stream: &str) -> Result<Vec<FragmentKey>> {
        let mut out = Vec::new();
        for phase_entry in Self::read_dir(&self.stream_dir(stream))? {
            let Some(phase) = phase_entry.file_name().to_str().and_then(Self::parse_phase) else {
                continue;
            };
            for fragment in Self::read_dir(&phase_entry.path())? {
                let Some(block) = fragment.file_name().to_str().and_then(Self::parse_block) else {
                    continue;
                };
                out.push(FragmentKey::new(stream, phase, block));
            }
        }
        out.sort();
        Ok(out)
    }

    fn remove_stream(&self, stream: &str) -> Result<StreamRemoval> {
        // Counted before removal, because the report is the point and a count
        // taken afterwards would be zero by construction.
        let keys = self.keys(stream)?;
        let mut bytes = 0u64;
        for key in &keys {
            let path = self.fragment_path(key);
            if let Ok(metadata) = fs::symlink_metadata(&path) {
                bytes += metadata.len();
            }
        }
        let dir = self.stream_dir(stream);
        Self::remove_confirmed(&dir)?;
        Ok(StreamRemoval {
            stream: stream.to_string(),
            fragments: keys.len(),
            bytes,
            location: dir.display().to_string(),
        })
    }
}

fn nanos() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|since| since.as_nanos())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::listener::EventListener;
    use std::sync::Mutex;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "blockflow-sidecar-{}-{}-{name}",
            std::process::id(),
            nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[derive(Default)]
    struct Recorder {
        seen: Mutex<Vec<Event>>,
    }

    impl EventListener for Recorder {
        fn on_event(&self, event: &Event) {
            self.seen
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(event.clone());
        }
    }

    fn stores() -> Vec<(&'static str, Sidecars)> {
        vec![
            ("memory", Sidecars::in_memory()),
            (
                "files",
                Sidecars::new(FileSidecars::at(scratch("round-trip")).unwrap()),
            ),
        ]
    }

    #[test]
    fn bytes_written_are_bytes_read_across_streams_phases_and_blocks() {
        for (what, store) in stores() {
            store.declare("left", Lifecycle::Persistent).unwrap();
            store.declare("right", Lifecycle::DeleteOnExit).unwrap();
            let mut expected: Vec<(FragmentKey, Vec<u8>)> = Vec::new();
            for stream in ["left", "right"] {
                for phase in 0..3usize {
                    for block in 0..4usize {
                        let index = [block, phase, 0];
                        // Deliberately not a fixed width and not valid UTF-8 at
                        // every position: the store must not be inspecting it.
                        let bytes: Vec<u8> = (0..=(block as u8 + phase as u8 * 7))
                            .map(|value| value.wrapping_mul(31))
                            .collect();
                        store.write(stream, phase, index, &bytes).unwrap();
                        expected.push((FragmentKey::new(stream, phase, index), bytes));
                    }
                }
            }
            expected.sort();
            for (key, bytes) in &expected {
                assert_eq!(
                    store
                        .read(&key.stream, key.phase, key.block)
                        .unwrap()
                        .as_ref(),
                    Some(bytes),
                    "{what}: {key:?}"
                );
            }
            let left: Vec<(FragmentKey, Vec<u8>)> = expected
                .iter()
                .filter(|(key, _)| key.stream == "left")
                .cloned()
                .collect();
            assert_eq!(store.fragments("left").unwrap(), left, "{what}");
            // Streams do not leak into each other.
            assert_eq!(store.keys("left").unwrap().len(), 12, "{what}");
            assert_eq!(store.keys("right").unwrap().len(), 12, "{what}");
        }
    }

    #[test]
    fn a_stream_nobody_wrote_to_reads_as_empty_rather_than_failing() {
        for (what, store) in stores() {
            store.declare("quiet", Lifecycle::Persistent).unwrap();
            assert!(store.keys("quiet").unwrap().is_empty(), "{what}");
            assert!(store.fragments("quiet").unwrap().is_empty(), "{what}");
            assert_eq!(store.read("quiet", 0, [0, 0, 0]).unwrap(), None, "{what}");
            // And so does a stream that was never even declared, when only
            // *read* — reading is a question, and the answer is "nothing".
            assert!(store.keys("absent").unwrap().is_empty(), "{what}");
            assert_eq!(store.read("absent", 0, [0, 0, 0]).unwrap(), None, "{what}");
        }
    }

    #[test]
    fn an_empty_fragment_is_a_fragment_and_not_an_absence() {
        for (what, store) in stores() {
            store.declare("s", Lifecycle::Persistent).unwrap();
            store.write("s", 0, [0, 0, 0], &[]).unwrap();
            assert_eq!(
                store.read("s", 0, [0, 0, 0]).unwrap(),
                Some(Vec::new()),
                "{what}: a zero-length fragment must not read as missing"
            );
        }
    }

    #[test]
    fn writing_to_an_undeclared_stream_says_the_lifecycle_has_no_default() {
        for (what, store) in stores() {
            let error = store.write("nobody", 0, [0, 0, 0], b"x").unwrap_err();
            assert!(
                error.to_string().contains("has no default"),
                "{what}: {error}"
            );
        }
    }

    #[test]
    fn declaring_the_same_stream_twice_agrees_and_disagreeing_is_an_error() {
        for (what, store) in stores() {
            store.declare("s", Lifecycle::DeleteOnExit).unwrap();
            store.declare("s", Lifecycle::DeleteOnExit).unwrap();
            let error = store.declare("s", Lifecycle::Persistent).unwrap_err();
            assert!(error.to_string().contains("disagreeing"), "{what}: {error}");
        }
    }

    #[test]
    fn a_stream_name_that_would_escape_the_store_is_refused() {
        for name in ["../elsewhere", "", ".hidden", "with/slash", "with space"] {
            assert!(
                check_stream_name(name).is_err(),
                "{name:?} should not be a stream name"
            );
        }
        for name in ["fragments", "chain-fragments", "a.b_c-1"] {
            check_stream_name(name).unwrap();
        }
    }

    #[test]
    fn discarding_removes_the_delete_on_exit_stream_reports_it_and_keeps_the_other() {
        let dir = scratch("lifecycle");
        let store = Sidecars::new(FileSidecars::at(&dir).unwrap());
        let recorder = Arc::new(Recorder::default());
        store.attach(recorder.clone());
        store.declare("scratch", Lifecycle::DeleteOnExit).unwrap();
        store.declare("keepsake", Lifecycle::Persistent).unwrap();
        for block in 0..3usize {
            store
                .write("scratch", 0, [block, 0, 0], &[1, 2, 3, 4])
                .unwrap();
            store.write("keepsake", 0, [block, 0, 0], &[9]).unwrap();
        }

        let report = store.discard().unwrap();
        assert_eq!(report.removed.len(), 1);
        assert_eq!(report.removed[0].stream, "scratch");
        assert_eq!(report.removed[0].fragments, 3);
        assert_eq!(report.removed[0].bytes, 12);
        assert_eq!(report.kept, vec!["keepsake".to_string()]);
        assert!(report.undeclared.is_empty());
        assert!(
            report.describe().contains("3 fragment(s), 12 byte(s)"),
            "{}",
            report.describe()
        );

        // Gone from storage, not merely from the report.
        assert!(!dir.join("scratch").exists());
        assert!(dir.join("keepsake").join("phase-0").exists());
        assert!(store.keys("scratch").unwrap().is_empty());
        assert_eq!(store.keys("keepsake").unwrap().len(), 3);

        // And the removal is on the event stream, so a caller that drops the
        // report still leaves a record.
        let discarded: Vec<Event> = recorder
            .seen
            .lock()
            .unwrap()
            .iter()
            .filter(|event| matches!(event, Event::SidecarDiscarded { .. }))
            .cloned()
            .collect();
        assert_eq!(
            discarded,
            vec![Event::SidecarDiscarded {
                stream: "scratch".to_string(),
                fragments: 3,
                bytes: 12,
            }]
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_stream_present_but_undeclared_is_reported_rather_than_passed_over() {
        let dir = scratch("undeclared");
        fs::create_dir_all(dir.join("orphan")).unwrap();
        let store = Sidecars::new(FileSidecars::at(&dir).unwrap());
        store.declare("mine", Lifecycle::DeleteOnExit).unwrap();
        store.write("mine", 0, [0, 0, 0], b"z").unwrap();
        let report = store.discard().unwrap();
        assert_eq!(report.undeclared, vec!["orphan".to_string()]);
        assert!(report.describe().contains("undeclared"));
        assert!(dir.join("orphan").exists(), "an unclaimed stream is left");
        fs::remove_dir_all(&dir).ok();
    }

    /// A backend whose removal fails. The portable half of "a deletion that
    /// cannot happen fails loudly" — no permissions, no platform.
    struct Unremovable(MemorySidecars);

    impl SidecarBackend for Unremovable {
        fn describe(&self) -> String {
            "unremovable".to_string()
        }
        fn declare(&self, stream: &str, lifecycle: Lifecycle) -> Result<()> {
            self.0.declare(stream, lifecycle)
        }
        fn lifecycle(&self, stream: &str) -> Result<Option<Lifecycle>> {
            self.0.lifecycle(stream)
        }
        fn streams(&self) -> Result<Vec<(String, Option<Lifecycle>)>> {
            self.0.streams()
        }
        fn put(&self, key: &FragmentKey, bytes: &[u8]) -> Result<()> {
            self.0.put(key, bytes)
        }
        fn get(&self, key: &FragmentKey) -> Result<Option<Vec<u8>>> {
            self.0.get(key)
        }
        fn keys(&self, stream: &str) -> Result<Vec<FragmentKey>> {
            self.0.keys(stream)
        }
        fn remove_stream(&self, stream: &str) -> Result<StreamRemoval> {
            Err(Error::backend(format!("{stream} is held open")))
        }
    }

    #[test]
    fn a_discard_that_cannot_remove_fails_instead_of_reporting_success() {
        let store = Sidecars::new(Unremovable(MemorySidecars::default()));
        store.declare("scratch", Lifecycle::DeleteOnExit).unwrap();
        store.write("scratch", 0, [0, 0, 0], b"x").unwrap();
        let error = store.discard().unwrap_err();
        assert!(error.to_string().contains("held open"), "{error}");
        // The fragment is still there, which is the honest outcome: the caller
        // was told the cleanup did not happen and the data agrees.
        assert_eq!(store.keys("scratch").unwrap().len(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn a_file_deletion_the_filesystem_refuses_names_the_path() {
        use std::os::unix::fs::MetadataExt;
        use std::os::unix::fs::PermissionsExt;

        let dir = scratch("refused");
        let store = Sidecars::new(FileSidecars::at(&dir).unwrap());
        store.declare("scratch", Lifecycle::DeleteOnExit).unwrap();
        store.write("scratch", 0, [0, 0, 0], b"abcd").unwrap();

        // Root ignores directory permissions, so the provocation would not
        // provoke. Say so rather than pass vacuously.
        if fs::metadata(&dir).map(|meta| meta.uid()).unwrap_or(0) == 0 {
            println!("running as root: a permission-refused removal cannot be provoked");
            fs::remove_dir_all(&dir).ok();
            return;
        }

        let phase = dir.join("scratch").join("phase-0");
        fs::set_permissions(&phase, fs::Permissions::from_mode(0o500)).unwrap();
        let error = store.discard().unwrap_err();
        fs::set_permissions(&phase, fs::Permissions::from_mode(0o700)).unwrap();
        assert!(
            error.to_string().contains("scratch"),
            "the error must name what could not go: {error}"
        );
        assert!(
            error.to_string().contains("silently does nothing"),
            "{error}"
        );
        // Still there, and the caller knows.
        assert_eq!(store.keys("scratch").unwrap().len(), 1);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_rewritten_fragment_replaces_the_previous_one() {
        // The reissue case: a worker died mid-task and another worker ran the
        // same block. One key, one fragment, the later one.
        for (what, store) in stores() {
            store.declare("s", Lifecycle::Persistent).unwrap();
            store.write("s", 1, [2, 0, 0], b"first attempt").unwrap();
            store.write("s", 1, [2, 0, 0], b"second").unwrap();
            assert_eq!(
                store.read("s", 1, [2, 0, 0]).unwrap(),
                Some(b"second".to_vec()),
                "{what}"
            );
            assert_eq!(store.keys("s").unwrap().len(), 1, "{what}");
        }
    }

    #[test]
    fn a_second_handle_on_one_directory_reads_what_the_first_wrote() {
        // The property that makes this worth building: no message between the
        // writer and the reader, only shared storage.
        let dir = scratch("two-handles");
        let writer = Sidecars::new(FileSidecars::at(&dir).unwrap());
        writer
            .declare("fragments", Lifecycle::DeleteOnExit)
            .unwrap();
        writer.write("fragments", 0, [3, 1, 0], b"payload").unwrap();

        let reader = Sidecars::new(FileSidecars::at(&dir).unwrap());
        assert_eq!(
            reader.lifecycle("fragments").unwrap(),
            Some(Lifecycle::DeleteOnExit),
            "the lifecycle must travel through storage, not through the process"
        );
        assert_eq!(
            reader.fragments("fragments").unwrap(),
            vec![(
                FragmentKey::new("fragments", 0, [3, 1, 0]),
                b"payload".to_vec()
            )]
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn writes_and_reads_appear_on_the_event_stream() {
        let store = Sidecars::in_memory();
        let recorder = Arc::new(Recorder::default());
        store.attach(recorder.clone());
        store.declare("s", Lifecycle::Persistent).unwrap();
        store.write("s", 2, [1, 0, 0], b"abcd").unwrap();
        store.read("s", 2, [1, 0, 0]).unwrap();
        store.read("s", 2, [9, 9, 9]).unwrap();
        let seen = recorder.seen.lock().unwrap().clone();
        assert!(matches!(
            seen[0],
            Event::SidecarWritten {
                phase: 2,
                index: [1, 0, 0],
                bytes: 4,
                ..
            }
        ));
        assert!(matches!(
            seen[1],
            Event::SidecarRead {
                bytes: 4,
                found: true,
                ..
            }
        ));
        assert!(matches!(
            seen[2],
            Event::SidecarRead {
                bytes: 0,
                found: false,
                ..
            }
        ));
    }

    #[test]
    fn a_listener_that_panics_is_disabled_and_does_not_fail_the_write() {
        struct Broken;
        impl EventListener for Broken {
            fn on_event(&self, _event: &Event) {
                panic!("a broken observer");
            }
        }
        let store = Sidecars::in_memory();
        store.attach(Arc::new(Broken));
        let recorder = Arc::new(Recorder::default());
        store.attach(recorder.clone());
        store.declare("s", Lifecycle::Persistent).unwrap();
        // Panics are noisy by default; the hook is silenced for the duration so
        // the test output says what the test is about.
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        store.write("s", 0, [0, 0, 0], b"x").unwrap();
        store.write("s", 0, [1, 0, 0], b"y").unwrap();
        std::panic::set_hook(previous);
        assert_eq!(store.keys("s").unwrap().len(), 2);
        assert_eq!(recorder.seen.lock().unwrap().len(), 2);
    }

    #[test]
    fn the_file_layout_is_one_object_per_fragment() {
        let dir = scratch("layout");
        let store = Sidecars::new(FileSidecars::at(&dir).unwrap());
        store.declare("s", Lifecycle::Persistent).unwrap();
        store.write("s", 1, [4, 5, 6], b"hello").unwrap();
        let path = dir.join("s").join("phase-1").join("block-4-5-6.bin");
        assert_eq!(fs::read(&path).unwrap(), b"hello");
        assert_eq!(
            fs::read_to_string(dir.join("s").join(LIFECYCLE_FILE)).unwrap(),
            "persistent"
        );
        fs::remove_dir_all(&dir).ok();
    }
}
