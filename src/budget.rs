// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// Why it exists
// -------------
// The motivating dataset is 93.9 Gvoxel. At the bytes-per-voxel measured on one
// of its tiles the image stages need ~0.42 TB if the volume is held whole, so
// on any real node the question is not "how fast" but "how many
// blocks may be resident at once". That is a *byte* question, and a count of
// slabs cannot answer it: a slab's footprint varies by stage, by dtype and by
// block size. So everything large takes a lease denominated in bytes, and
// concurrency falls out of the budget rather than being set beside it.
//
// The two classes exist because the failure mode they guard against is
// asymmetric. Compute must be able to wait — a blocked worker is slow, an
// unadmitted one is a crash. Cache and prefetch must never wait, because a
// prefetcher that can starve the thing it exists to accelerate is worse than no
// prefetcher. Hence `Reserved` blocks and `Opportunistic` fails fast.
//
// What this deliberately does not do
// ----------------------------------
// It does not evict. `Opportunistic` holders are told to release
// ([`Lease::is_revoked`]) and are expected to cooperate; nothing here reaches
// into another thread's buffer. Revocation is therefore advisory, and the
// property that actually protects compute is the stronger one: while a
// `Reserved` request is waiting, no new opportunistic lease is granted at all.
// A budget that promised true eviction would need a callback per lease and a
// guarantee that the callback cannot deadlock against the holder's own locks;
// that is a larger design than Phase 1 needs, and an advisory flag is honest
// about what it provides.
//
// It also does not track actual RSS. A lease is a *promise* about a buffer the
// caller is about to allocate, so the budget is only as good as the
// bytes-per-voxel estimate fed to it. That estimate is measured, not guessed
// (`progress.json` records `bytes_per_voxel` per stage), but it is an estimate,
// and the accounting says so rather than pretending to observe the allocator.
//
// What the planner asks this budget for, and what it used to leave out
// --------------------------------------------------------------------------
// The figure `strategy` checks against a budget is
// `PhaseCost::working_set_bytes_per_block`: `resident_voxels x bytes_per_voxel`
// times **every block buffer alive while one block is in flight** — the phase's
// own input and output, one per source image, and one per buffer the chain holds
// between them (`Chain::resident_block_buffers`).
//
// It was a literal `x 2.0` — one input buffer and one output buffer — and the
// table below records the allocator measurements against it, in units of one
// `f64` block buffer. Those rows are now an equality; they are kept as the
// record of a gap that ran from `1.00x` to `3.56x` over the shapes listed and to
// **46.5x** for the one that was not:
// a fan-in whose combine cannot fold holds one buffer per arm, and the feature
// stack of `docs/design/pixel-classification.md` has ninety-one of them.
//
// ```text
// one in, one out                2.00x   (the shape the formula is for)
// sequence of two maps           3.00x
// sequence from a source         3.00x
// sequence of four maps          4.00x
// fan-in, 2 computed arms        4.00x
// fan-in, 3 computed arms        4.12x
// fan-in, 7 computed arms        4.12x
// fan-in, 1 arm + 1 source       4.00x
// fan-in, 2 source arms          4.00x
// fan-in, 1 arm + 2 sources      5.13x
// ```
//
// The excess is `Chain::apply_placed`'s own allocations — a `Sequence` holds the
// intermediate it is reading and the one it is writing, a `Parallel` holds
// branch buffers, and a `Chain::Source` arm is handed a fetched buffer besides.
// None of it is in the `x 2.0`.
//
// **A source arm is no longer one of them.** A `Chain::Source` leaf's answer is
// the stored image at the block's read extent — a buffer the executor fetched
// before the chain was entered and owns until after it returns. Where that
// answer is an *operand* rather than an output (a fan-in branch, the head of a
// sequence) the walk hands the reference on and allocates nothing, so a fan-in
// of two source arms holds the input, the output and the two fetched buffers
// and not one byte besides. It is the same category as the two below: bytes
// that were written once and then stood in for bytes the run already had. The
// fetched buffer itself is of course still resident, and it is the executor's,
// which is why the source rows fall by exactly the copies and not to nothing:
//
// ```text
//                                 copying   borrowing
// fan-in, 1 arm + 1 source          5.00x       4.00x
// fan-in, 1 arm + 2 sources         6.13x       5.13x
// fan-in, 2 source arms             6.00x       4.00x
// sequence from a source            4.00x       3.00x
// ```
//
// Both columns are the same table run twice, the left one with the copy put
// back. **`fan-in, 2 source arms` is the shape to read**: every branch's answer
// already existed, so the walk now allocates nothing at all and what is left is
// the executor's four buffers. It is also the shape that made the harness's
// accounting find a term nobody had needed before — with no block buffer of its
// own, the tallest thing that chain allocates is the shape fold
// `Chain::apply_tallied` does before its tally exists.
//
// **One shape did not move and is worth as much as the ones that did.**
// `fan-in, rank arm + 2 sources` measures `7.00x` either way, because for that
// chain the copies were never the peak: the worst moment is the rank filter's
// own scratch while its branch buffer is live, and the source arms are joined
// after it has finished. A fix that removes bytes does not always remove the
// bytes that bound the figure.
//
// **The two sequence rows differ, and they used to be one number: `4.00x` and
// `4.00x`, measured.** A `Sequence` began with `input.clone()` — a whole block
// buffer, written once and then read only where the caller's own block would
// have served. It is gone, and what that is worth is a two-child sequence:
// `4.00x` to `3.00x`. From three children on, the intermediate being read and
// the one being written are the tall term and the clone had already been freed,
// so `sequence of four maps` is `4.00x` either way. A whole block buffer off
// every two-child sequence and nothing off a longer one, which is not what "a
// sequence clones its input" would have suggested — the before figures are the
// same table run with the clone put back.
//
// **That `fan-in, 3 computed arms` and `fan-in, 7 computed arms` measure the
// same is the point of both rows.** A `Parallel` used to hold **one buffer per
// branch at once**, so its figure grew with the arity and at one block each of
// those buffers was a whole volume. It now folds as its branches are computed
// wherever the combine declares itself a left fold over pairs
// (`Combine::fold_carrier`), and holds three buffers whatever the arity: the
// partial, the branch just finished, and the buffer their join goes into. Seven arms therefore measure exactly what
// three do. The buffers that went away were **computed and then not read** —
// from the moment a branch finished until the combine ran they were bytes and
// nothing else — so this is not a trade against locality, and
// the byte-for-byte identity property is the only reason this is allowed.
// Three combines decline the declaration and say why in their own words:
// `Arithmetic::Subtract` and `Arithmetic::Divide` are not folds, neither is a
// difference, and a tree walk needs every channel at a voxel at once.
//
// **So a lease granted against that figure is not a bound on what the block
// holds, and this module must not be read as if it were.** Two consequences,
// both stated here because this is where the promise is made:
//
// * the shortfall is **shape-dependent**, `1.00x` to `2.56x` across ordinary
//   chains, so it is not a constant a caller can pre-multiply away and have the
//   planner still rank candidates on comparable numbers;
// * even a figure corrected for every buffer above would still not be a bound,
//   because an op may allocate whatever it likes inside `BlockOp::apply` and
//   nothing declares it. Measured on the *same* chain shape and the same block —
//   two framework buffers in every case — a voxelwise map holds `2.00x`, a
//   `5^3` morphological open `2.38x`, and a `5^3` rank filter `4.00x`.
//
// The honest position is therefore that this is an **estimate that is known to
// run low**, not a ceiling; a caller sizing a machine should read
// the shape-derived residency model for the factor its own chain shape earns.
// Making the framework's half exact is a change to `price_phase` and to what the
// planner is handed, and it moves which plans are affordable — that is a budget
// review, and the numbers it needs are in the same file.
//
// **The framework's half is now observable.** `Chain::apply_observing` runs a
// block and hands back `BlockResidency`, the high-water mark of the buffers the
// walk itself allocated plus the input, the output and the distinct source
// buffers. It is measured rather than forecast — the same walk that allocates
// keeps the tally, so a buffer that is not counted is a buffer that was not
// allocated. The original residency audit checked it against the allocator over
// the same execution, accounting for every byte as an equality rather than a
// window.
//
// **It does not make a lease a promise, and nothing here should be read as if
// it had.** Two things stay outside any such figure: what an op allocates inside
// `BlockOp::apply` — the three figures above — and what a `Combine` allocates
// inside its own. A residency observation is also scoped to the chain
// it was taken on and refuses to answer for another, which is why it cannot
// simply be substituted for the shape-derived count at plan time without the
// planner deciding what to do when it has no observation. That decision belongs with
// the budget review; the sentence that must survive it is the one above — an
// estimate known to run low, not a ceiling.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, Weak};
use std::time::Duration;

/// Environment override consulted by [`default_budget_bytes`].
///
/// Accepts a plain byte count, a `K`/`M`/`G`/`T` suffix, or `auto` to derive it
/// from `/proc/meminfo`.
///
/// An embedding application that has its own settled name for this should call
/// [`budget_bytes_from`] with both, rather than making its users learn a second
/// one — see that function.
pub const BUDGET_ENV: &str = "BLOCKFLOW_MEMORY_BUDGET_BYTES";

/// What the budget assumes when it cannot read `/proc/meminfo`.
///
/// Deliberately small. The stated deployment floor is a 32 GB node, and a
/// default that is wrong there is worse than one that is merely conservative on
/// a large one — an over-large budget admits blocks the machine cannot hold,
/// which is the exact failure this module exists to prevent.
const FALLBACK_BUDGET_BYTES: u64 = 8 << 30;

/// Never hand out less than this, whatever the machine reports.
///
/// Below a gibibyte a single tile-scale block cannot be admitted at all and
/// every acquire degenerates into the oversubscription path.
const MIN_BUDGET_BYTES: u64 = 1 << 30;

/// What class of consumer a lease belongs to.
///
/// The distinction is not bookkeeping: it decides whether a request may wait.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Class {
    /// Compute working set. Blocks until granted; never revoked.
    Reserved,
    /// Cache and prefetch. Granted only from slack, never blocks, and is asked
    /// to release as soon as a `Reserved` request is waiting.
    Opportunistic,
}

#[derive(Debug, Default)]
struct State {
    reserved: u64,
    opportunistic: u64,
    high_water: u64,
    /// `Reserved` requests currently blocked. While this is non-zero no
    /// opportunistic lease is granted, which is what keeps rule 1 of §2 true
    /// without an eviction callback.
    waiting_reserved: usize,
    /// Monotonic; every grant and every release bumps it. Used only for
    /// diagnostics.
    grants: u64,
    /// Grants that exceeded the whole budget and were let through anyway to
    /// keep the run alive. A non-zero value means the budget is too small for
    /// the block size, not that anything went wrong.
    oversubscribed: u64,
}

struct Inner {
    /// Atomic so that a process budget created before the cap was stated can
    /// be resized once it is ([`set_process_cap`]).
    total: AtomicU64,
    state: Mutex<State>,
    released: Condvar,
    /// Set whenever a `Reserved` request is waiting; read by opportunistic
    /// holders through [`Lease::is_revoked`] without taking the lock.
    revoke: AtomicBool,
    /// High-water mark, duplicated outside the lock so instrumentation can read
    /// it from a signal handler or a progress thread without blocking.
    high_water: AtomicU64,
    /// Holders of opportunistic bytes that can give them back on request
    /// ([`Reclaim`]): the chunk caches leasing from this budget.
    reclaimers: Mutex<Vec<Weak<dyn Reclaim>>>,
}

impl std::fmt::Debug for Inner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Inner")
            .field("total", &self.total.load(Ordering::Relaxed))
            .field("high_water", &self.high_water.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl Inner {
    fn total(&self) -> u64 {
        self.total.load(Ordering::Relaxed)
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        // A panic while holding the budget lock must not wedge every other
        // worker: the invariant the lock protects is two integers, and a
        // panicking thread's `Lease` still runs its `Drop` on the way out, so
        // the counters are consistent by the time anyone else sees them.
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn in_use(state: &State) -> u64 {
        state.reserved + state.opportunistic
    }

    fn note_high_water(&self, state: &mut State) {
        let in_use = Self::in_use(state);
        if in_use > state.high_water {
            state.high_water = in_use;
            self.high_water.store(in_use, Ordering::Relaxed);
        }
    }

    fn release(&self, bytes: u64, class: Class) {
        let mut state = self.lock();
        match class {
            Class::Reserved => state.reserved = state.reserved.saturating_sub(bytes),
            Class::Opportunistic => state.opportunistic = state.opportunistic.saturating_sub(bytes),
        }
        drop(state);
        // Wake everyone: a release of `n` bytes may satisfy one large waiter or
        // several small ones, and which is not knowable from here.
        self.released.notify_all();
    }
}

/// One global, byte-denominated budget.
///
/// Cheap to clone — clones share one pool, which is the point; a budget that
/// could be duplicated would be no budget at all.
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    inner: Arc<Inner>,
}

/// One pool is equal only to itself: clones are the same budget.
impl PartialEq for MemoryBudget {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl Eq for MemoryBudget {}

/// A holder of [`Class::Opportunistic`] bytes that can give some back when
/// compute needs them: a chunk cache, which evicts.
///
/// **This is the eviction the module header says it does not do, and it is
/// still not done *by* the budget.** The budget asks; the holder decides what
/// to drop, under its own lock, and the bytes come back through the leases it
/// drops. See [`MemoryBudget::reclaim`] for the lock order.
pub trait Reclaim: Send + Sync {
    /// Free at least `bytes` if possible; answer how many were freed.
    fn reclaim(&self, bytes: u64) -> u64;
}

impl MemoryBudget {
    /// A budget of exactly `total_bytes`.
    pub fn new(total_bytes: u64) -> Self {
        Self {
            inner: Arc::new(Inner {
                total: AtomicU64::new(total_bytes.max(1)),
                state: Mutex::new(State::default()),
                released: Condvar::new(),
                revoke: AtomicBool::new(false),
                high_water: AtomicU64::new(0),
                reclaimers: Mutex::new(Vec::new()),
            }),
        }
    }

    /// A budget sized by [`default_budget_bytes`].
    pub fn detected() -> Self {
        Self::new(default_budget_bytes())
    }

    /// Total bytes this budget may hand out.
    pub fn total(&self) -> u64 {
        self.inner.total()
    }

    /// Change what this budget may hand out. Leases already granted are kept;
    /// a smaller total only refuses new ones until enough have returned.
    pub fn set_total(&self, total_bytes: u64) {
        self.inner
            .total
            .store(total_bytes.max(1), Ordering::Relaxed);
        self.inner.released.notify_all();
    }

    /// Bytes not leased to anybody right now.
    pub fn free(&self) -> u64 {
        let state = self.inner.lock();
        self.inner.total().saturating_sub(Inner::in_use(&state))
    }

    /// Register a holder of opportunistic bytes that [`Self::reclaim`] may ask
    /// to give some back. Held weakly: a dropped cache unregisters itself.
    pub fn register_reclaimer(&self, reclaimer: Weak<dyn Reclaim>) {
        let mut list = self
            .inner
            .reclaimers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        list.retain(|weak| weak.strong_count() > 0);
        list.push(reclaimer);
    }

    /// Ask the registered opportunistic holders to give back `bytes`, in
    /// registration order, and answer how many they freed.
    ///
    /// **Never called with this budget's lock held**, and that is the whole of
    /// the deadlock argument: a reclaimer drops leases, and a lease's drop takes
    /// this lock. A reclaimer takes its own lock and then, through the drops,
    /// this one — the same order the cache's admission path takes them in.
    pub fn reclaim(&self, bytes: u64) -> u64 {
        if bytes == 0 {
            return 0;
        }
        let live: Vec<Arc<dyn Reclaim>> = {
            let list = self
                .inner
                .reclaimers
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            list.iter().filter_map(Weak::upgrade).collect()
        };
        let mut freed = 0u64;
        for reclaimer in live {
            if freed >= bytes {
                break;
            }
            freed = freed.saturating_add(reclaimer.reclaim(bytes - freed));
        }
        freed
    }

    /// Lease `bytes` for compute **without waiting on another compute lease**:
    /// granted from the free bytes, after asking the opportunistic holders for
    /// whatever is missing, and refused (`None`) if that is still not enough.
    ///
    /// The executor's admission call. It runs blocks in waves and holds no
    /// compute lease of its own while it forms one, so waiting could only be
    /// waiting on a cache — which is what the reclaim is for — or on another
    /// executor sharing the budget, which may hold its leases for a whole wave.
    pub fn try_acquire_reclaiming(&self, bytes: u64) -> Option<Lease> {
        if let Some(lease) = self.try_acquire(bytes) {
            return Some(lease);
        }
        let short = bytes.saturating_sub(self.free());
        self.reclaim(short);
        self.try_acquire(bytes)
    }

    /// Bytes currently leased, both classes.
    pub fn in_use(&self) -> u64 {
        let state = self.inner.lock();
        Inner::in_use(&state)
    }

    /// Bytes currently leased as [`Class::Reserved`].
    pub fn reserved_in_use(&self) -> u64 {
        self.inner.lock().reserved
    }

    /// Bytes currently leased as [`Class::Opportunistic`].
    pub fn opportunistic_in_use(&self) -> u64 {
        self.inner.lock().opportunistic
    }

    /// The largest simultaneous total this budget has ever held.
    ///
    /// This is what §8 wants in `progress.json` as `budget_high_water_bytes`.
    pub fn high_water(&self) -> u64 {
        self.inner.high_water.load(Ordering::Relaxed)
    }

    /// How many `Reserved` requests are blocked right now.
    pub fn waiting(&self) -> usize {
        self.inner.lock().waiting_reserved
    }

    /// How many grants exceeded the whole budget and were admitted anyway.
    ///
    /// Non-zero means the budget cannot hold one unit of work, so the run
    /// continued at concurrency one rather than deadlocking. It is a signal to
    /// shrink the block, not an error.
    pub fn oversubscribed(&self) -> u64 {
        self.inner.lock().oversubscribed
    }

    /// Total grants made, for instrumentation.
    pub fn grants(&self) -> u64 {
        self.inner.lock().grants
    }

    /// Acquire `bytes` for compute. Blocks until they are available.
    ///
    /// A request larger than the whole budget cannot ever be satisfied by
    /// waiting, so it is admitted as soon as the budget is otherwise empty and
    /// counted in [`MemoryBudget::oversubscribed`]. The alternative — refusing
    /// or blocking forever — turns a memory setting into a correctness cliff,
    /// and §2's rule is that pressure reduces concurrency rather than failing.
    pub fn acquire(&self, bytes: u64) -> Lease {
        if bytes == 0 {
            return self.zero_lease(Class::Reserved);
        }
        // Ask the caches first, without the lock: a request that does not fit
        // beside what they hold is the moment they exist to give it up.
        let short = bytes.saturating_sub(self.free());
        if short > 0 {
            self.reclaim(short);
        }
        let mut state = self.inner.lock();
        let oversized = bytes > self.inner.total();
        state.waiting_reserved += 1;
        self.inner.revoke.store(true, Ordering::Relaxed);
        loop {
            let in_use = Inner::in_use(&state);
            let grantable = if oversized {
                in_use == 0
            } else {
                in_use + bytes <= self.inner.total()
            };
            if grantable {
                state.reserved += bytes;
                state.grants += 1;
                if oversized {
                    state.oversubscribed += 1;
                }
                state.waiting_reserved -= 1;
                if state.waiting_reserved == 0 {
                    self.inner.revoke.store(false, Ordering::Relaxed);
                }
                self.inner.note_high_water(&mut state);
                return Lease {
                    inner: Some(Arc::clone(&self.inner)),
                    bytes,
                    class: Class::Reserved,
                };
            }
            // A timeout rather than a plain wait: `Drop` notifies, so the
            // timeout is not load-bearing for correctness, but it means a lost
            // wakeup costs a stall rather than a hang.
            let (next, _) = self
                .inner
                .released
                .wait_timeout(state, Duration::from_millis(250))
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state = next;
        }
    }

    /// Acquire `bytes` for compute without waiting.
    ///
    /// Useful where a caller has a cheaper alternative to blocking — running the
    /// unit smaller, or in one piece.
    pub fn try_acquire(&self, bytes: u64) -> Option<Lease> {
        if bytes == 0 {
            return Some(self.zero_lease(Class::Reserved));
        }
        let mut state = self.inner.lock();
        if Inner::in_use(&state) + bytes > self.inner.total() {
            return None;
        }
        state.reserved += bytes;
        state.grants += 1;
        self.inner.note_high_water(&mut state);
        Some(Lease {
            inner: Some(Arc::clone(&self.inner)),
            bytes,
            class: Class::Reserved,
        })
    }

    /// Acquire `bytes` from slack, for a cache or a prefetch. Never blocks.
    ///
    /// Returns `None` while any `Reserved` request is waiting, so a prefetcher
    /// cannot take the bytes a stalled compute worker is queueing for. That is
    /// §2 rule 1, and it is enforced here rather than left to the caller's
    /// discipline.
    pub fn try_acquire_opportunistic(&self, bytes: u64) -> Option<Lease> {
        if bytes == 0 {
            return Some(self.zero_lease(Class::Opportunistic));
        }
        let mut state = self.inner.lock();
        if state.waiting_reserved > 0 {
            return None;
        }
        if Inner::in_use(&state) + bytes > self.inner.total() {
            return None;
        }
        state.opportunistic += bytes;
        state.grants += 1;
        self.inner.note_high_water(&mut state);
        Some(Lease {
            inner: Some(Arc::clone(&self.inner)),
            bytes,
            class: Class::Opportunistic,
        })
    }

    /// Whether opportunistic holders are being asked to let go.
    pub fn revoking(&self) -> bool {
        self.inner.revoke.load(Ordering::Relaxed)
    }

    fn zero_lease(&self, class: Class) -> Lease {
        Lease {
            inner: Some(Arc::clone(&self.inner)),
            bytes: 0,
            class,
        }
    }
}

/// A granted claim on `bytes` of the budget.
///
/// RAII, and that is the whole contract: the bytes return on drop, on every
/// path including a panic unwinding through the worker that holds it. Nothing
/// here can fail, so nothing here can panic during an unwind.
#[derive(Debug)]
pub struct Lease {
    inner: Option<Arc<Inner>>,
    bytes: u64,
    class: Class,
}

impl Lease {
    /// Bytes this lease holds.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Which class it was granted under.
    pub fn class(&self) -> Class {
        self.class
    }

    /// Whether an opportunistic holder should release now.
    ///
    /// Always `false` for [`Class::Reserved`]: compute is never revoked.
    pub fn is_revoked(&self) -> bool {
        if self.class == Class::Reserved {
            return false;
        }
        self.inner
            .as_ref()
            .is_some_and(|inner| inner.revoke.load(Ordering::Relaxed))
    }

    /// Give the bytes back now rather than at end of scope.
    ///
    /// Equivalent to `drop`, and named so a release that matters at a particular
    /// point reads as deliberate rather than as an accident of scope.
    pub fn release(self) {
        drop(self);
    }

    /// A lease that owns nothing, for call sites that are not budgeted.
    ///
    /// Exists so a caller can pass `Lease` unconditionally instead of
    /// `Option<Lease>` — the write-back queue takes one per buffer, and a test
    /// or a small path that never allocates enough to matter should not have to
    /// build a budget to use it.
    pub fn unbudgeted() -> Self {
        Self {
            inner: None,
            bytes: 0,
            class: Class::Reserved,
        }
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        if self.bytes == 0 {
            return;
        }
        if let Some(inner) = self.inner.take() {
            inner.release(self.bytes, self.class);
        }
    }
}

/// The default budget: the environment if it says, otherwise a fraction of what
/// the kernel reports — and never more than the process's memory cap leaves
/// for blocks in flight, when a cap was stated ([`process_cap`]).
///
/// The fraction is `min(MemTotal / 2, MemAvailable * 3 / 4)`. Two terms because
/// they fail differently: `MemTotal / 2` bounds us on an idle machine, where
/// `MemAvailable` is nearly everything and taking it would leave nothing for
/// page cache — and this pipeline reads far more than it holds, so page cache is
/// not slack. `MemAvailable * 3 / 4` bounds us on a busy one, where `MemTotal`
/// is a fiction. On the stated 32 GB floor this is 16 GiB, which leaves the
/// other half for the page cache, the allocator's fragmentation and every
/// allocation this module does not see.
pub fn default_budget_bytes() -> u64 {
    budget_bytes_from(&[BUDGET_ENV])
}

/// The same rule, but consulting `names` in order before falling back.
///
/// Exists because an embedding application usually already has a name its users
/// set, and forcing them onto a second one is a needless break. It passes its
/// own name first and [`BUDGET_ENV`] second; the first name that parses wins,
/// and an unparseable value is skipped rather than treated as zero.
///
/// A **binding** cap ([`MemoryCap::is_binding`]: stated, or a cgroup limit)
/// clamps whatever the names say to [`MemoryCap::usable_bytes`]: a budget for
/// blocks in flight larger than the memory the process may hold is a budget
/// the kernel enforces instead.
pub fn budget_bytes_from(names: &[&str]) -> u64 {
    let cap = process_cap();
    for name in names {
        if let Ok(raw) = std::env::var(name) {
            if let Some(bytes) = parse_budget(raw.trim()) {
                return match cap.is_binding() {
                    true => bytes.min(cap.usable_bytes()),
                    false => bytes,
                };
            }
        }
    }
    cap.usable_bytes()
}

/// The `auto` rule, or `None` where `/proc/meminfo` cannot be read.
pub fn auto_budget_bytes() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    machine_rule(&meminfo)
}

fn machine_rule(meminfo: &str) -> Option<u64> {
    let total = meminfo_field(meminfo, "MemTotal:")?;
    let available = meminfo_field(meminfo, "MemAvailable:").unwrap_or(total);
    Some((total / 2).min(available / 4 * 3).max(MIN_BUDGET_BYTES))
}

// --------------------------------------------------------- the memory cap --

/// Environment variable naming the **memory cap**: the most this process may
/// hold resident, in bytes, with the same `K`/`M`/`G`/`T` suffixes as
/// [`BUDGET_ENV`].
///
/// Not the same quantity as [`BUDGET_ENV`], and both exist for that reason.
/// The budget is what blocks in flight may hold; the cap is everything — the
/// blocks in flight, the chunk caches that fill the rest, and the process's own
/// baseline. A run under a cap plans its blocks in flight into
/// [`MemoryCap::usable_bytes`], and the caches fit themselves into what the
/// blocks leave.
pub const CAP_ENV: &str = "BLOCKFLOW_MEMORY_CAP";

/// Where a [`MemoryCap`] came from, in the order they are consulted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CapSource {
    /// The caller said so: a command-line flag ([`MemoryCap::resolve`]).
    Explicit,
    /// [`CAP_ENV`], or else [`BUDGET_ENV`] read as a cap.
    Environment,
    /// The process's cgroup: v2 `memory.max`, or v1 `memory.limit_in_bytes`,
    /// the smallest along the path from the process's group to the root.
    Cgroup,
    /// Nothing stated: `min(MemTotal / 2, MemAvailable * 3 / 4)`, the rule
    /// [`default_budget_bytes`] has always used.
    Machine,
    /// `/proc/meminfo` could not be read: [`FALLBACK_BUDGET_BYTES`].
    Fallback,
}

/// The most this process may hold resident, and where that figure came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryCap {
    pub bytes: u64,
    pub source: CapSource,
}

/// The least a binding cap holds back from the blocks in flight. See
/// [`MemoryCap::reserve_bytes`].
pub const MIN_CAP_RESERVE_BYTES: u64 = 96 << 20;

impl MemoryCap {
    /// The cap from the environment, the cgroup or the machine, in that order.
    pub fn detect() -> Self {
        Self::resolve(None)
    }

    /// `explicit` (a command-line flag) if given, else [`Self::detect`]'s
    /// order. A cgroup limit below a stated cap still binds: it is the kernel's
    /// and the run cannot exceed it by saying otherwise.
    pub fn resolve(explicit: Option<u64>) -> Self {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok();
        Self::from_sources(
            explicit,
            std::env::var(CAP_ENV).ok().as_deref(),
            std::env::var(BUDGET_ENV).ok().as_deref(),
            cgroup_memory_limit(),
            meminfo.as_deref(),
        )
    }

    /// The precedence, on its inputs: `explicit`, then `cap_env`, then
    /// `budget_env` (the older variable, read as a cap), then `cgroup`, then the
    /// machine rule on `meminfo`, then [`FALLBACK_BUDGET_BYTES`]. An
    /// unparseable or zero variable is skipped. A cgroup limit at or above the
    /// machine's `MemTotal` is no limit, and one below a stated cap wins.
    pub fn from_sources(
        explicit: Option<u64>,
        cap_env: Option<&str>,
        budget_env: Option<&str>,
        cgroup: Option<u64>,
        meminfo: Option<&str>,
    ) -> Self {
        let total = meminfo.and_then(|text| meminfo_field(text, "MemTotal:"));
        let cgroup = cgroup.filter(|&limit| limit > 0 && total.is_none_or(|total| limit < total));
        let stated = explicit
            .filter(|&bytes| bytes > 0)
            .map(|bytes| (bytes, CapSource::Explicit))
            .or_else(|| {
                [cap_env, budget_env]
                    .into_iter()
                    .flatten()
                    .find_map(|raw| parse_cap(raw.trim()))
                    .map(|bytes| (bytes, CapSource::Environment))
            });
        let cap = match (stated, cgroup) {
            (Some((bytes, _)), Some(limit)) if limit < bytes => (limit, CapSource::Cgroup),
            (Some(stated), _) => stated,
            (None, Some(limit)) => (limit, CapSource::Cgroup),
            (None, None) => match meminfo.and_then(machine_rule) {
                Some(bytes) => (bytes, CapSource::Machine),
                None => (FALLBACK_BUDGET_BYTES, CapSource::Fallback),
            },
        };
        Self {
            bytes: cap.0,
            source: cap.1,
        }
    }

    /// A cap somebody stated or the kernel enforces, rather than this module's
    /// estimate of a polite share of the machine.
    pub fn is_binding(&self) -> bool {
        matches!(
            self.source,
            CapSource::Explicit | CapSource::Environment | CapSource::Cgroup
        )
    }

    /// What a binding cap holds back from the blocks in flight: a tenth, and at
    /// least [`MIN_CAP_RESERVE_BYTES`] (never more than half the cap).
    ///
    /// For what no plan sees: the binary, its thread stacks, the allocator's
    /// retained and fragmented heap, and the moment between two samples of the
    /// resident set during which a block can allocate more than it was charged.
    /// The executor measures the first three as they are
    /// ([`resident_bytes`]); this is the margin for the last. Zero for the
    /// machine rule, which already leaves half the machine.
    pub fn reserve_bytes(&self) -> u64 {
        if !self.is_binding() {
            return 0;
        }
        (self.bytes / 10)
            .max(MIN_CAP_RESERVE_BYTES)
            .min(self.bytes / 2)
    }

    /// The bytes the blocks in flight and the caches may share between them.
    pub fn usable_bytes(&self) -> u64 {
        self.bytes.saturating_sub(self.reserve_bytes()).max(1)
    }

    /// The least binding cap whose usable bytes are at least `usable`: what a
    /// plan needing `usable` bytes for its blocks in flight has to be given.
    pub fn needed_for(usable: u64) -> u64 {
        // The reserve is one of three pieces (half, the floor, a tenth), so the
        // answer is the least of the three inversions that actually holds.
        [
            usable.saturating_mul(2),
            usable.saturating_add(MIN_CAP_RESERVE_BYTES),
            usable.saturating_mul(10).div_ceil(9),
        ]
        .into_iter()
        .filter(|&bytes| {
            Self {
                bytes,
                source: CapSource::Explicit,
            }
            .usable_bytes()
                >= usable
        })
        .min()
        .unwrap_or(u64::MAX)
    }
}

impl std::fmt::Display for MemoryCap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let source = match self.source {
            CapSource::Explicit => "stated",
            CapSource::Environment => "environment",
            CapSource::Cgroup => "cgroup",
            CapSource::Machine => "machine",
            CapSource::Fallback => "fallback",
        };
        write!(f, "{} MiB ({source})", self.bytes >> 20)
    }
}

/// A cap value: a byte count with an optional `K`/`M`/`G`/`T` suffix. Unlike
/// [`BUDGET_ENV`], `auto` means "not stated" here, so the cgroup and the
/// machine still get their say.
fn parse_cap(raw: &str) -> Option<u64> {
    if raw.eq_ignore_ascii_case("auto") {
        return None;
    }
    parse_budget(raw)
}

/// The memory limit of this process's cgroup, or `None` where there is none
/// (or no cgroup filesystem to ask).
pub fn cgroup_memory_limit() -> Option<u64> {
    let membership = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    cgroup_limit_with(&membership, &|path| std::fs::read_to_string(path).ok())
}

/// [`cgroup_memory_limit`] on a `/proc/self/cgroup` text and a file reader, so
/// both hierarchies can be tested without one.
///
/// The smallest limit from the process's group up to the root, because a
/// limit on a parent slice binds every group inside it. v2 first (the
/// `0::<path>` line, `memory.max`, `max` for none), then v1 (the line whose
/// controllers include `memory`, `memory.limit_in_bytes`, where unlimited is a
/// page-rounded `i64::MAX`). Inside a container the group path is often not
/// visible from its own mount, so the mount's root is also asked.
pub fn cgroup_limit_with(membership: &str, read: &dyn Fn(&str) -> Option<String>) -> Option<u64> {
    let parse = |text: Option<String>| -> Option<u64> {
        let text = text?;
        let text = text.trim();
        if text == "max" {
            return None;
        }
        text.parse::<u64>()
            .ok()
            .filter(|&v| v > 0 && v < (1u64 << 62))
    };
    let smallest_along = |root: &str, path: &str, file: &str| -> Option<u64> {
        let mut best: Option<u64> = None;
        let mut current = path.trim_end_matches('/').to_string();
        loop {
            let candidate = format!("{root}{current}/{file}");
            if let Some(limit) = parse(read(&candidate)) {
                best = Some(best.map_or(limit, |b| b.min(limit)));
            }
            match current.rfind('/') {
                Some(at) => current.truncate(at),
                None => break,
            }
            if current.is_empty() {
                if let Some(limit) = parse(read(&format!("{root}/{file}"))) {
                    best = Some(best.map_or(limit, |b| b.min(limit)));
                }
                break;
            }
        }
        best
    };
    for line in membership.lines() {
        let mut fields = line.splitn(3, ':');
        let (Some(_id), Some(controllers), Some(path)) =
            (fields.next(), fields.next(), fields.next())
        else {
            continue;
        };
        if controllers.is_empty() {
            if let Some(limit) = smallest_along("/sys/fs/cgroup", path, "memory.max") {
                return Some(limit);
            }
        }
    }
    for line in membership.lines() {
        let mut fields = line.splitn(3, ':');
        let (Some(_id), Some(controllers), Some(path)) =
            (fields.next(), fields.next(), fields.next())
        else {
            continue;
        };
        if controllers.split(',').any(|c| c == "memory") {
            return smallest_along("/sys/fs/cgroup/memory", path, "memory.limit_in_bytes");
        }
    }
    None
}

static PROCESS_CAP: Mutex<Option<MemoryCap>> = Mutex::new(None);
static PROCESS_BUDGET: OnceLock<MemoryBudget> = OnceLock::new();

/// The memory cap this process runs under: what [`set_process_cap`] stated,
/// else [`MemoryCap::detect`], read once and kept.
pub fn process_cap() -> MemoryCap {
    let mut cap = PROCESS_CAP
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    *cap.get_or_insert_with(MemoryCap::detect)
}

/// State the process's memory cap — a command-line flag, typically, at the
/// top of `main` — and resize [`process_budget`] to it.
pub fn set_process_cap(cap: MemoryCap) {
    *PROCESS_CAP
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(cap);
    if let Some(budget) = PROCESS_BUDGET.get() {
        budget.set_total(cap.usable_bytes());
    }
}

/// The one budget every chunk cache and the executor's admission lease from
/// unless told otherwise, sized at [`process_cap`]'s usable bytes.
///
/// One pool for both on purpose: compute leases [`Class::Reserved`] for its
/// blocks in flight, the caches lease [`Class::Opportunistic`] for what they
/// retain, and a compute lease that does not fit beside the caches makes them
/// give bytes back ([`Reclaim`]). So a cache is as large as the cap leaves
/// after the active set, and shrinks when the active set grows.
pub fn process_budget() -> MemoryBudget {
    PROCESS_BUDGET
        .get_or_init(|| MemoryBudget::new(process_cap().usable_bytes()))
        .clone()
}

/// This process's resident set in bytes (`VmRSS` of `/proc/self/status`), or
/// `None` where that cannot be read. One small read; cheap enough to take per
/// wave of blocks.
pub fn resident_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    meminfo_field(&status, "VmRSS:")
}

/// Hand the heap's free pages back to the system (glibc's `malloc_trim`); a
/// no-op elsewhere.
///
/// glibc keeps freed block buffers in its arenas, so without this the resident
/// set after a wave of blocks is the wave's high-water mark rather than what is
/// live, and an admission that measures the resident set would charge the next
/// wave for buffers nobody holds. Milliseconds; called between waves only when
/// a run is near its cap.
pub fn release_freed_heap() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    // SAFETY: `malloc_trim` only returns free heap pages to the system.
    unsafe {
        libc::malloc_trim(0);
    }
}

/// A thread that samples [`resident_bytes`] every few milliseconds and keeps
/// the largest it has seen since the last [`Self::reset`]: the peak of a wave
/// of blocks, which the executor compares with what the wave was charged.
///
/// One small file read per sample; the thread exists only while a run near its
/// cap is executing, and stops when dropped.
pub struct ResidentSampler {
    peak: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

/// How often [`ResidentSampler`] reads the resident set.
pub const SAMPLE_INTERVAL: Duration = Duration::from_millis(2);

impl ResidentSampler {
    /// `None` where the resident set cannot be read.
    pub fn start() -> Option<Self> {
        let first = resident_bytes()?;
        let peak = Arc::new(AtomicU64::new(first));
        let stop = Arc::new(AtomicBool::new(false));
        let handle = {
            let peak = Arc::clone(&peak);
            let stop = Arc::clone(&stop);
            std::thread::Builder::new()
                .name("blockflow-rss".into())
                .spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        if let Some(now) = resident_bytes() {
                            peak.fetch_max(now, Ordering::Relaxed);
                        }
                        std::thread::sleep(SAMPLE_INTERVAL);
                    }
                })
                .ok()?
        };
        Some(Self {
            peak,
            stop,
            handle: Some(handle),
        })
    }

    /// Start a new window at `now`.
    pub fn reset(&self, now: u64) {
        self.peak.store(now, Ordering::Relaxed);
    }

    /// The largest resident set since the last reset, this moment included.
    pub fn peak(&self) -> u64 {
        if let Some(now) = resident_bytes() {
            self.peak.fetch_max(now, Ordering::Relaxed);
        }
        self.peak.load(Ordering::Relaxed)
    }
}

impl Drop for ResidentSampler {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

// ------------------------------------------------ what admission charges --

/// The margin over an **assumed** framework figure, on a first run.
///
/// `PhaseCost::working_set_bytes_per_block` was `resident_voxels x
/// bytes_per_voxel x 2.0` — one input buffer and one output buffer — when these
/// were measured against what a block really holds. **As multiples of that
/// assumed charge**, which is the unit this constant is in:
///
/// The framework half of that gap is closed — the charge now counts the chain's
/// own buffers — so what a re-measurement would find is only the **op** half,
/// the scratch an `apply` allocates inside itself. This constant has not been
/// re-fitted, and until it is it is larger than its evidence rather than
/// smaller, which is the safe direction for a margin.
///
/// ```text
/// one in, one out                 1.00x     fan-in, 1 arm + 1 source    2.00x
/// sequence of two maps            1.50x     fan-in, 2 source arms       2.00x
/// sequence from a source          1.50x     fan-in, 1 arm + 2 sources   2.56x
/// sequence of four maps           2.00x     rank filter alone           2.00x
/// fan-in, 2 computed arms         2.00x     morphological open alone    1.19x
/// fan-in, 3 computed arms         2.06x     sequence of four ranks      3.00x
/// fan-in, 7 computed arms         2.06x     fan-in, rank arm + 2 src  3.5003x <- widest
/// ```
///
/// **That table is the old one, taken while the charge was a literal `x 2.0`.**
/// It is kept because it is what the constant was fitted to when it was `3.6`,
/// and because the difference between it and the table below — where the same
/// shapes measure `1.00x` — is the whole of what closing that gap bought. The
/// figure now is **`2.1`**, and the smallest-tenth rule is unchanged; see the
/// re-fit section below.
///
/// The rule is deliberately two-sided: a margin above its evidence is headroom
/// nobody can point at, and one below it is the failure it exists to prevent.
///
/// **A tenth and not a whole number**, which this constant was first written as
/// and had to be corrected. Rounding up to whole numbers reads well until a
/// measurement lands just over one: [`UNOBSERVED_OP_MARGIN`]'s widest op is
/// `2.0002x`, and `3` would have been fifty per cent of headroom bought with two
/// ten-thousandths of evidence. A tenth is fine enough that the rounding is not
/// an argument and coarse enough that a constant does not move on noise.
///
/// **The right-hand rows include combinations, and they are why the margin is
/// derived from a measurement rather than from the parts.** A chain may be
/// expensive in its framework buffers *and* in its op, and a margin justified by
/// the worse of two separate measurements need not cover one that is both.
///
/// **A combination is now the worst, and it was not before.** While a fan-in
/// held every branch result at once, its own buffers were the tall term: the
/// peak of `fan-in, rank arm + 2 src` fell at the combine, the rank filter's
/// transient scratch never showed, and that chain measured the same `3.56x` as
/// the cheap-arm fan-in beside it. Folding a fan-in's branches as they are
/// computed took its own buffers down, the peak moved to the arm, and the
/// combination is the widest shape here by nearly a unit. Nothing about the rank
/// filter changed. That is the argument for measuring combinations rather than
/// reasoning about the parts, stated by the one time the reasoning was wrong.
///
/// **It then survived a second fix that should have moved it, and that is the
/// same lesson from the other side.** Borrowing a source arm instead of copying
/// it took a whole block buffer off every source row here — and left
/// `fan-in, rank arm + 2 src` at `3.5003x`, unmoved, because for *that* chain
/// the copies were never the peak: the worst moment is the rank filter's own
/// scratch while its branch buffer is live, and the source arms are joined after
/// it has finished. So this constant has now been re-derived across two fixes,
/// and the shape that binds it is the same one for a reason that has changed
/// twice. **A margin defined as "the smallest tenth covering every shape
/// measured" has to be re-derived whenever a fix relocates a peak**, whether or
/// not the number comes out the same.
///
/// It is a **margin fitted to those measurements, not a bound**. An op that
/// allocates more, at a moment when the rest of the chain is also at its
/// widest, would exceed it, and nothing declares that none does — see
/// [`crate::op::BlockResidency`].
///
/// # It was 3.6, and the re-fit is the interesting part
///
/// Most of what this used to cover was not the op at all: it was the framework's
/// own buffers, which `PhaseCost::working_set_bytes_per_block` did not count
/// while it was a literal `x 2.0`. Now that it counts them, the same fourteen
/// shapes measure like this against their own charge:
///
/// ```text
/// one in, one out                 1.0000x     fan-in, 2 source arms       1.0000x
/// sequence of two maps            1.0001x     sequence from a source      1.0001x
/// sequence of four maps           1.0001x     rank filter alone           2.0002x
/// fan-in, 2 computed arms         1.0000x     morphological open alone    1.1875x
/// fan-in, 3 computed arms         1.0312x     fan-in, rank arm + 2 src    1.4001x
/// fan-in, 7 computed arms         1.0312x     sequence of four ranks      1.5002x
/// fan-in, 1 arm + 1 source        1.0000x
/// fan-in, 1 arm + 2 sources       1.0250x
/// ```
///
/// **Every shape whose cost is framework buffers is now `1.00x`** — the charge
/// is exact for it — and every shape above one has an op in it. So the residual
/// this margin covers is the op's own scratch and nothing else, and the widest
/// is the rank filter at `2.0002x`, giving `2.1` by the same smallest-tenth rule.
///
/// # And that is the same number [`UNOBSERVED_OP_MARGIN`] carries
///
/// Not a coincidence and worth stating rather than leaving for a reader to
/// notice: the two constants existed to cover different residuals — this one the
/// framework's *and* the op's, that one the op's alone — and closing the
/// framework half left them measuring the same thing. They are fitted by
/// separate tests over separate fixtures, fourteen shapes here and three ops
/// there, and they agree to the tenth.
///
/// **A consequence for [`admission_bytes`]: its two branches now charge
/// identically.** [`FrameworkFigure`]'s variants still mean different things —
/// one is derived from a chain's shape, the other observed on a run — and the
/// arithmetic would separate again the moment a shape is found that the
/// derivation misses. Keeping both, and keeping them fitted independently, is
/// what would make that visible; collapsing them into one constant would hide
/// the very measurement that is currently agreeing.
pub const UNOBSERVED_SHAPE_MARGIN: f64 = 2.1;

/// The margin over an **exact** framework figure.
///
/// Once the framework's half is exact — an observation, or the shape-derived
/// figure that would replace it at plan time — the only thing left unpriced is
/// what an op allocates inside `BlockOp::apply`. Measured on one chain shape and
/// one block, against the two framework buffers that shape holds: a voxelwise
/// map `1.0000x`, a `5^3` morphological open `1.1875x`, a `5^3` rank filter
/// `2.0002x`. `2.1` is the smallest tenth that covers them, by the same rule and
/// asserted the same way.
///
/// **The `2.0002` is the reason the rule is a tenth**, not a decoration on it: a
/// rank filter holds two block buffers of its own *and a little more*, so a rule
/// that rounded to whole numbers would have charged `3` for it. See
/// [`UNOBSERVED_SHAPE_MARGIN`].
///
/// Fitted to three ops, and a margin for the same reason as above.
pub const UNOBSERVED_OP_MARGIN: f64 = 2.1;

/// The best figure available for the framework's half of one block's residency.
///
/// The variants are ordered by how much is known, and the policy in
/// [`admission_bytes`] is a function of which one a caller can supply.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FrameworkFigure {
    /// `PhaseCost::working_set_bytes_per_block`, which counts **every block
    /// buffer alive while one block is in flight**: the phase's own input and
    /// output, one per source image, and one per buffer the chain holds between
    /// them.
    ///
    /// **It used to assume one image in and one out — a literal `x 2.0` — and
    /// this doc used to record how wrong that was**, at `1.00x` to `3.56x` over
    /// the shapes then measured. What closed it was a shape nobody had measured:
    /// a fan-in whose combine cannot declare a
    /// [`Combine::fold_carrier`](crate::op::Combine::fold_carrier) holds one
    /// block buffer **per arm**, so the 91-arm feature stack of
    /// `docs/design/pixel-classification.md` held 93 where the budget charged 2.
    /// A 46.5x under-charge, in the direction that admits a plan the run cannot
    /// afford, is not a gap a margin absorbs — `UNOBSERVED_SHAPE_MARGIN` was 3.6
    /// at the time, and is 2.1 now that it no longer has to cover the framework.
    ///
    /// The count is [`Chain::resident_block_buffers`](crate::op::Chain::resident_block_buffers),
    /// derived from the chain's shape and originally checked against allocator
    /// measurements over representative shapes.
    ///
    /// **It is computed where the price is taken, not recorded on the plan**,
    /// and that is a decision worth stating because the other way was tried
    /// first. A field on `PhaseDecomposition` with a `declare`/`check` pair —
    /// the arrangement `dtype` and `source_images` have — has no safe default:
    /// zero under-charges, so `check` has to refuse every plan that did not
    /// declare, and that made eighteen test files' hand-built plans invalid at a
    /// stroke. The quantity is a function of the chain and every caller that
    /// prices a budget already holds the chain, so it is derived like
    /// `compute_per_voxel` rather than recorded like `source_images`. Nothing
    /// downstream reads `working_set_bytes_per_block` except
    /// `Constraints::affords_working_set`, and every path to it has the slots.
    ///
    /// **What it still does not know is the op's own scratch**, which is why
    /// this is `Assumed` and still takes a margin. See [`Self::Exact`].
    Assumed(f64),
    /// The framework's half, exact: a [`crate::op::BlockResidency`] observed for
    /// this chain at this block, or the shape-derived figure that stands in for
    /// one before a first run.
    ///
    /// **The shape-derived figure now exists and is wired in**, and did not when
    /// this was written:
    /// [`Chain::resident_block_buffers`](crate::op::Chain::resident_block_buffers)
    /// counts what a chain holds inside itself from its structure alone. The
    /// whole quantity a plan can now state is
    ///
    /// ```text
    /// buffers = images_read + writes_an_image + resident_buffers
    /// ```
    ///
    /// and [`Self::Assumed`] is that, priced. So the gap between the two
    /// variants is no longer the framework's arithmetic — it is only what an
    /// **op** allocates inside its own `apply`, which no declaration reaches and
    /// which `an_ops_own_working_buffers_are_not_visible_to_any_declaration`
    /// measures.
    ///
    /// **Exact about the framework and silent about the op**, which is why it
    /// still takes a margin.
    Exact(u64),
}

/// **What admission charges for one block, and the policy behind it.**
///
/// The question this settles is narrow: on a first run, with nothing observed,
/// what does the budget charge? Three answers were available and each fails
/// differently.
///
/// The decision
/// ------------
/// **Charge a stated margin over the best framework figure there is.** With
/// nothing but the plan, that is [`UNOBSERVED_SHAPE_MARGIN`] over the assumed
/// figure; with the framework's half exact, it is the smaller
/// [`UNOBSERVED_OP_MARGIN`], because the exactness has already absorbed the part
/// the larger margin was covering. Both are measured, both are fitted, and
/// neither is a ceiling.
///
/// **Why a flat factor is legitimate here and would not be for the cost.**
/// `PhaseCost`'s own header sets the test for anything added to it: "is the size
/// of the over-estimate the same question for every candidate", and an error
/// that varies with the candidate "does not make the planner cautious — it
/// reorders the candidates". That rule governs the number the search *ranks* on.
/// This is not that number. The same header says so: `working_set_bytes_per_block`
/// "is allowed to over-state, because it feeds a budget and never a comparison".
/// A budget figure that is uniformly high refuses some plans and mis-ranks none,
/// which is exactly the trade this margin makes.
///
/// **What it costs when wrong, in each direction.** The asymmetry is the whole
/// argument:
///
/// * **too low** — the run holds more than the budget promised. Measured today,
///   with no margin at all: up to `2.50x` over budget in the original residency
///   sweep, and worst exactly where the
///   planner has just fitted a large candidate, which is the situation a
///   memory-constrained run is in by definition. That is a killed run;
/// * **too high** — the planner takes a smaller block off the candidate ladder.
///   It costs read amplification and more tasks. It cannot make a run
///   unrunnable and it cannot move a voxel. Counted over a sweep of nine
///   budgets **on a ladder of powers of two**: the cold-start charge costs one
///   rung at **three** of them and nothing at the other six. It was six of nine
///   while the margin was `3.6`, and re-fitting it to `2.1` — see
///   [`UNOBSERVED_SHAPE_MARGIN`] — halved the cost of being careful, because
///   most of what the old margin covered is now counted exactly rather than
///   guarded against. That count is a property of that spacing rather than of
///   the margin, which is the distinction the next paragraph exists to make.
///
/// **And the ceiling on that cost is arithmetic rather than luck — but it is a
/// ceiling on *volume*, not on rungs.** Every margin here is under eight: `2.1`
/// on its own, and `2.0002 x 2.1 = 4.20` for the worst measured shape under the
/// exact figure — both were larger, `3.6` and `7.48`, while the charge was a
/// literal `x 2.0` and the margin had to cover the framework's own buffers. A block that fitted without a margin either has a rung at an
/// eighth of its volume, which then fits *with* one, or the ladder has bottomed
/// out above that point and there is nothing smaller to fall to. Either way:
/// **neither branch can move the admitted block by more than `8x` in volume, at
/// any budget.**
///
/// **On a ladder of powers of two that is exactly one rung**, which is how this
/// was first stated and why. It is **not** the same sentence on
/// [`crate::decomposition::refined_ladder`], where a rung is `2.37x` or `3.375x`
/// in volume: `2.1` fits inside one of the finer rungs, but the exact branch's
/// `4.20` spans two of them — and `3.6` spanned two of either. The step count was
/// a proxy that happened to equal the volume ratio while the spacing was octaves;
/// the volume ratio is what survives a change of spacing.
///
/// It is also the **better** bound and not merely the more durable one: the
/// planner stops at the largest rung that fits, so a finer ladder lands the
/// correction closer and never further — which is the refinement's own win, seen
/// from the margin's side.
///
/// This argument was checked against the real [`admission_bytes`] and the real
/// `price_phase` at both coarse and refined ladder spacings. Two spacings is
/// what makes it an invariant rather than a coincidence.
///
/// **At the coarse spacing the bound is slack, measurably so**, and the refined
/// test is therefore the sharper of the two. A block admitted without a margin
/// sits below its rung's ceiling by whatever the ladder's coarseness left it —
/// up to a full rung — and that headroom absorbs a margin somewhat past eight
/// before any rung is lost: a hypothetical `9.0` still moves nothing more than
/// `8x`. The coarse test's negative control uses two full rungs for that reason,
/// which is the amount no headroom under one rung can absorb.
///
/// That headroom is thin, and deliberately so: it is what says the two margins
/// cannot both be raised much further without the affordability argument
/// changing shape. A future measurement that needs a wider margin needs these
/// paragraphs rewritten, not quietly exceeded.
///
/// The two that were not chosen
/// ----------------------------
/// * **Charge the framework figure with no margin.** It is exact for what it
///   covers and silent about the op, so it would still under-charge a rank
///   filter by `2.00x` — the same failure in the same direction, merely smaller.
///   Being exact about one half is not a reason to price the other half at zero.
/// * **Refuse to admit fine cuts until something is observed.** This one is
///   backwards, and worth writing down so it is not re-proposed: admission takes
///   the *largest* block that fits, and residency grows with block size, so the
///   conservative move is a **smaller** block — which is precisely what a larger
///   charge already produces. A separate refusal would be a second control for
///   the same effect, and it would buy nothing the margin does not while costing
///   read amplification on every phase rather than on the ones near their
///   budget.
///
/// What must not be read into this
/// -------------------------------
/// **A lease is still an estimate known to run low, not a ceiling.** The margin
/// makes it run low less often; it does not make it a promise, and an observed
/// figure does not either — [`crate::op::BlockResidency`] measures the
/// framework's half exactly and is silent about the op's, which is why the
/// exact branch still takes a margin. Nothing here should be quoted as a bound
/// on what a block holds. The module header above is the statement of record.
pub fn admission_bytes(figure: FrameworkFigure) -> u64 {
    let (bytes, margin) = match figure {
        FrameworkFigure::Assumed(bytes) => (bytes.max(0.0), UNOBSERVED_SHAPE_MARGIN),
        FrameworkFigure::Exact(bytes) => (bytes as f64, UNOBSERVED_OP_MARGIN),
    };
    let charged = bytes * margin;
    // A budget is compared against this, so a figure that is not a number would
    // admit everything. Saturating at `u64::MAX` refuses everything instead,
    // which is the safe direction for a quantity whose whole job is to say no.
    if charged.is_finite() {
        charged.round() as u64
    } else {
        u64::MAX
    }
}

fn meminfo_field(meminfo: &str, field: &str) -> Option<u64> {
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix(field) {
            let kib = rest
                .trim()
                .trim_end_matches("kB")
                .trim()
                .parse::<u64>()
                .ok()?;
            return Some(kib * 1024);
        }
    }
    None
}

/// Parse `auto`, a plain byte count, or one with a `K`/`M`/`G`/`T` suffix.
///
/// Returns `None` for anything unrecognised *and for zero*: a budget of zero
/// admits nothing except through the oversubscription path, which would turn
/// every block into a serialisation point and look like a hang.
fn parse_budget(raw: &str) -> Option<u64> {
    if raw.eq_ignore_ascii_case("auto") {
        return auto_budget_bytes();
    }
    let (digits, scale) = match raw.as_bytes().last()? {
        b'K' | b'k' => (&raw[..raw.len() - 1], 1u64 << 10),
        b'M' | b'm' => (&raw[..raw.len() - 1], 1u64 << 20),
        b'G' | b'g' => (&raw[..raw.len() - 1], 1u64 << 30),
        b'T' | b't' => (&raw[..raw.len() - 1], 1u64 << 40),
        _ => (raw, 1u64),
    };
    digits
        .trim()
        .parse::<u64>()
        .ok()
        .and_then(|value| value.checked_mul(scale))
        .filter(|value| *value > 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::thread;

    #[test]
    fn a_lease_returns_its_bytes_on_drop() {
        let budget = MemoryBudget::new(1000);
        {
            let lease = budget.acquire(400);
            assert_eq!(lease.bytes(), 400);
            assert_eq!(budget.in_use(), 400);
        }
        assert_eq!(budget.in_use(), 0);
        assert_eq!(budget.high_water(), 400);
    }

    #[test]
    fn bytes_return_when_the_holder_panics() {
        let budget = MemoryBudget::new(1000);
        let held = budget.clone();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _lease = held.acquire(600);
            panic!("worker died holding a lease");
        }));
        assert!(outcome.is_err());
        assert_eq!(
            budget.in_use(),
            0,
            "a panicking worker must not leak budget"
        );
        // And the budget still works afterwards, i.e. the mutex is not wedged.
        let lease = budget.acquire(1000);
        assert_eq!(lease.bytes(), 1000);
    }

    #[test]
    fn reserved_blocks_until_bytes_are_available() {
        let budget = MemoryBudget::new(1000);
        let held = budget.acquire(800);
        let (tx, rx) = mpsc::channel();
        let waiter = budget.clone();
        let worker = thread::spawn(move || {
            let lease = waiter.acquire(500);
            tx.send(lease.bytes()).unwrap();
        });
        // It cannot have been granted: 800 + 500 > 1000.
        assert!(rx.recv_timeout(Duration::from_millis(200)).is_err());
        drop(held);
        assert_eq!(rx.recv_timeout(Duration::from_secs(5)).unwrap(), 500);
        worker.join().unwrap();
    }

    #[test]
    fn opportunistic_never_blocks_and_only_takes_slack() {
        let budget = MemoryBudget::new(1000);
        let _compute = budget.acquire(900);
        // Bound, not asserted-and-dropped: the point is that the *next* request
        // sees the reduced slack.
        let _cache = budget
            .try_acquire_opportunistic(50)
            .expect("50 bytes of slack were available");
        assert!(
            budget.try_acquire_opportunistic(100).is_none(),
            "slack is 50, so 100 must be refused rather than waited for"
        );
        let _rest = budget
            .try_acquire_opportunistic(50)
            .expect("the last 50 bytes of slack");
        assert!(
            budget.try_acquire_opportunistic(1).is_none(),
            "slack is exhausted"
        );
    }

    #[test]
    fn a_waiting_reserved_request_shuts_out_new_opportunistic_grants() {
        let budget = MemoryBudget::new(1000);
        let held = budget.acquire(900);
        let waiter = budget.clone();
        let worker = thread::spawn(move || {
            let _lease = waiter.acquire(1000);
        });
        // Wait for the acquire to register as a waiter.
        let mut spins = 0;
        while budget.waiting() == 0 && spins < 1000 {
            thread::sleep(Duration::from_millis(2));
            spins += 1;
        }
        assert_eq!(budget.waiting(), 1);
        assert!(budget.revoking(), "holders should be told to let go");
        assert!(
            budget.try_acquire_opportunistic(10).is_none(),
            "prefetch must not take bytes a stalled compute worker is queueing for"
        );
        drop(held);
        worker.join().unwrap();
        assert!(!budget.revoking());
    }

    #[test]
    fn an_opportunistic_lease_reports_revocation_and_a_reserved_one_does_not() {
        let budget = MemoryBudget::new(1000);
        let cache = budget.try_acquire_opportunistic(400).unwrap();
        let compute = budget.acquire(100);
        assert!(!cache.is_revoked());
        assert!(!compute.is_revoked());
        let waiter = budget.clone();
        let worker = thread::spawn(move || {
            let _lease = waiter.acquire(900);
        });
        let mut spins = 0;
        while !budget.revoking() && spins < 1000 {
            thread::sleep(Duration::from_millis(2));
            spins += 1;
        }
        assert!(cache.is_revoked());
        assert!(!compute.is_revoked(), "compute is never revoked");
        drop(cache);
        drop(compute);
        worker.join().unwrap();
    }

    #[test]
    fn a_request_larger_than_the_budget_is_admitted_alone_rather_than_deadlocking() {
        let budget = MemoryBudget::new(100);
        let small = budget.acquire(50);
        let waiter = budget.clone();
        let worker = thread::spawn(move || {
            let lease = waiter.acquire(4096);
            lease.bytes()
        });
        thread::sleep(Duration::from_millis(50));
        drop(small);
        assert_eq!(worker.join().unwrap(), 4096);
        assert_eq!(budget.oversubscribed(), 1);
        assert_eq!(budget.in_use(), 0);
    }

    #[test]
    fn admission_caps_concurrency_rather_than_thrashing() {
        // Eight workers, each needing a quarter of the budget: at most four can
        // be resident at once, and the peak must show it.
        let budget = MemoryBudget::new(4000);
        let peak = Arc::new(AtomicU64::new(0));
        let mut workers = Vec::new();
        for _ in 0..8 {
            let budget = budget.clone();
            let peak = Arc::clone(&peak);
            workers.push(thread::spawn(move || {
                let _lease = budget.acquire(1000);
                let now = budget.in_use();
                peak.fetch_max(now, Ordering::Relaxed);
                thread::sleep(Duration::from_millis(20));
            }));
        }
        for worker in workers {
            worker.join().unwrap();
        }
        assert!(peak.load(Ordering::Relaxed) <= 4000);
        assert!(budget.high_water() <= 4000);
        assert_eq!(budget.in_use(), 0);
        assert_eq!(budget.grants(), 8);
    }

    #[test]
    fn try_acquire_refuses_rather_than_waiting() {
        let budget = MemoryBudget::new(100);
        let _held = budget.acquire(80);
        assert!(budget.try_acquire(30).is_none());
        assert!(budget.try_acquire(20).is_some());
    }

    #[test]
    fn zero_byte_leases_are_free_and_always_granted() {
        let budget = MemoryBudget::new(1);
        let _big = budget.acquire(1);
        let free = budget.acquire(0);
        assert_eq!(free.bytes(), 0);
        assert!(budget.try_acquire_opportunistic(0).is_some());
        assert_eq!(budget.in_use(), 1);
    }

    #[test]
    fn an_unbudgeted_lease_belongs_to_no_pool() {
        let lease = Lease::unbudgeted();
        assert_eq!(lease.bytes(), 0);
        assert!(!lease.is_revoked());
    }

    #[test]
    fn budget_sizes_parse() {
        assert_eq!(parse_budget("1024"), Some(1024));
        assert_eq!(parse_budget("4K"), Some(4 << 10));
        assert_eq!(parse_budget("16M"), Some(16 << 20));
        assert_eq!(parse_budget("2G"), Some(2 << 30));
        assert_eq!(parse_budget("1t"), Some(1 << 40));
        assert_eq!(parse_budget("0"), None);
        assert_eq!(parse_budget("nonsense"), None);
        assert_eq!(parse_budget(""), None);
    }

    #[test]
    fn the_default_budget_is_safe_at_the_thirty_two_gigabyte_floor() {
        // The rule, evaluated by hand at the stated floor rather than on
        // whatever machine happens to run the test.
        let total: u64 = 32 << 30;
        for available in [total, total * 3 / 4, total / 2, total / 8] {
            let budget = (total / 2).min(available / 4 * 3).max(MIN_BUDGET_BYTES);
            assert!(
                budget <= total / 2,
                "budget {budget} must never exceed half of a 32 GB node"
            );
            assert!(budget >= MIN_BUDGET_BYTES);
        }
        // And the live one, whatever this machine is, obeys the same bounds.
        let live = default_budget_bytes();
        assert!(live >= MIN_BUDGET_BYTES);
    }

    const MEMINFO: &str =
        "MemTotal:       65536000 kB\nMemFree:  1 kB\nMemAvailable:   32768000 kB\n";

    #[test]
    fn a_cap_is_explicit_then_environment_then_cgroup_then_machine() {
        let machine = machine_rule(MEMINFO).unwrap();
        let cap = |e: Option<u64>, env: Option<&str>, old: Option<&str>, cg: Option<u64>| {
            MemoryCap::from_sources(e, env, old, cg, Some(MEMINFO))
        };
        assert_eq!(
            cap(Some(3 << 30), Some("2G"), Some("1G"), None),
            MemoryCap {
                bytes: 3 << 30,
                source: CapSource::Explicit
            }
        );
        assert_eq!(
            cap(None, Some("2G"), Some("1G"), None),
            MemoryCap {
                bytes: 2 << 30,
                source: CapSource::Environment
            }
        );
        // The older budget variable is read as a cap when the cap's is absent
        // or unreadable.
        assert_eq!(cap(None, Some("lots"), Some("1G"), None).bytes, 1 << 30);
        assert_eq!(
            cap(None, None, None, Some(5 << 30)),
            MemoryCap {
                bytes: 5 << 30,
                source: CapSource::Cgroup
            }
        );
        assert_eq!(
            cap(None, None, None, None),
            MemoryCap {
                bytes: machine,
                source: CapSource::Machine
            }
        );
        assert_eq!(
            MemoryCap::from_sources(None, None, None, None, None),
            MemoryCap {
                bytes: FALLBACK_BUDGET_BYTES,
                source: CapSource::Fallback
            }
        );
        // `auto` and zero mean "not stated".
        assert_eq!(
            cap(None, Some("auto"), Some("0"), None).source,
            CapSource::Machine
        );
        assert_eq!(cap(Some(0), None, None, None).source, CapSource::Machine);
    }

    #[test]
    fn a_cgroup_limit_below_a_stated_cap_still_binds_and_one_above_the_machine_does_not() {
        let stated_above =
            MemoryCap::from_sources(Some(8 << 30), None, None, Some(2 << 30), Some(MEMINFO));
        assert_eq!(
            stated_above,
            MemoryCap {
                bytes: 2 << 30,
                source: CapSource::Cgroup
            }
        );
        let stated_below =
            MemoryCap::from_sources(Some(1 << 30), None, None, Some(2 << 30), Some(MEMINFO));
        assert_eq!(stated_below.source, CapSource::Explicit);
        // 64 GB of machine: a 1 TB cgroup limit is no limit.
        let roomy = MemoryCap::from_sources(None, None, None, Some(1 << 40), Some(MEMINFO));
        assert_eq!(roomy.source, CapSource::Machine);
    }

    #[test]
    fn a_binding_cap_holds_back_a_reserve_and_the_machine_rule_does_not() {
        let machine = MemoryCap {
            bytes: 16 << 30,
            source: CapSource::Machine,
        };
        assert_eq!(machine.usable_bytes(), 16 << 30);
        let big = MemoryCap {
            bytes: 10 << 30,
            source: CapSource::Explicit,
        };
        assert_eq!(big.reserve_bytes(), (10 << 30) / 10);
        let small = MemoryCap {
            bytes: 512 << 20,
            source: CapSource::Cgroup,
        };
        assert_eq!(small.reserve_bytes(), MIN_CAP_RESERVE_BYTES);
        let tiny = MemoryCap {
            bytes: 100 << 20,
            source: CapSource::Explicit,
        };
        assert_eq!(tiny.reserve_bytes(), 50 << 20, "never more than half");
        // `needed_for` inverts `usable_bytes`.
        for usable in [10u64 << 20, 500 << 20, 3 << 30, 40 << 30] {
            let cap = MemoryCap {
                bytes: MemoryCap::needed_for(usable),
                source: CapSource::Explicit,
            };
            assert!(cap.usable_bytes() >= usable, "{usable}");
            let less = MemoryCap {
                bytes: cap.bytes - (2 << 20),
                source: CapSource::Explicit,
            };
            assert!(less.usable_bytes() < usable + (2 << 20), "{usable}");
        }
    }

    #[test]
    fn cgroup_v2_takes_the_smallest_limit_from_the_group_to_the_root() {
        let files: std::collections::HashMap<&str, &str> = [
            ("/sys/fs/cgroup/user.slice/memory.max", "max\n"),
            (
                "/sys/fs/cgroup/user.slice/app.scope/memory.max",
                "2147483648\n",
            ),
            (
                "/sys/fs/cgroup/user.slice/app.scope/run/memory.max",
                "max\n",
            ),
            ("/sys/fs/cgroup/memory.max", "8589934592"),
        ]
        .into_iter()
        .collect();
        let read = |path: &str| files.get(path).map(|text| text.to_string());
        assert_eq!(
            cgroup_limit_with("0::/user.slice/app.scope/run\n", &read),
            Some(2 << 30)
        );
        let none = |_: &str| -> Option<String> { Some("max".into()) };
        assert_eq!(cgroup_limit_with("0::/user.slice\n", &none), None);
    }

    #[test]
    fn cgroup_v1_reads_the_memory_controller_and_ignores_its_unlimited_value() {
        let membership = "12:cpu,cpuacct:/a\n4:memory:/docker/abc\n1:name=systemd:/x\n";
        let limited = |path: &str| {
            (path == "/sys/fs/cgroup/memory/docker/abc/memory.limit_in_bytes")
                .then(|| "1073741824\n".to_string())
        };
        assert_eq!(cgroup_limit_with(membership, &limited), Some(1 << 30));
        let unlimited = |_: &str| Some("9223372036854771712\n".to_string());
        assert_eq!(cgroup_limit_with(membership, &unlimited), None);
        assert_eq!(cgroup_limit_with("", &unlimited), None);
    }

    struct Holder {
        budget: MemoryBudget,
        held: Mutex<Vec<Lease>>,
    }

    impl Reclaim for Holder {
        fn reclaim(&self, bytes: u64) -> u64 {
            let mut held = self.held.lock().unwrap();
            let mut freed = 0;
            while freed < bytes {
                match held.pop() {
                    Some(lease) => freed += lease.bytes(),
                    None => break,
                }
            }
            freed
        }
    }

    #[test]
    fn a_compute_lease_that_does_not_fit_beside_a_cache_makes_it_give_bytes_back() {
        let budget = MemoryBudget::new(1000);
        let holder = Arc::new(Holder {
            budget: budget.clone(),
            held: Mutex::new(Vec::new()),
        });
        for _ in 0..9 {
            let lease = holder.budget.try_acquire_opportunistic(100).unwrap();
            holder.held.lock().unwrap().push(lease);
        }
        let weak: Weak<dyn Reclaim> = Arc::downgrade(&holder) as Weak<dyn Reclaim>;
        budget.register_reclaimer(weak);
        assert!(budget.try_acquire(300).is_none(), "900 of 1000 are cached");
        let lease = budget
            .try_acquire_reclaiming(300)
            .expect("the cache gives back 200");
        assert_eq!(lease.bytes(), 300);
        assert_eq!(budget.opportunistic_in_use(), 700);
        // And the blocking acquire asks too, rather than waiting forever.
        let more = budget.acquire(600);
        assert_eq!(more.bytes(), 600);
        assert_eq!(budget.opportunistic_in_use(), 100);
        // A request beyond everything is refused rather than blocked.
        assert!(budget.try_acquire_reclaiming(2000).is_none());
    }

    #[test]
    fn a_budget_can_be_resized_and_is_equal_only_to_its_clones() {
        let budget = MemoryBudget::new(1000);
        let _held = budget.acquire(800);
        budget.set_total(500);
        assert_eq!(budget.total(), 500);
        assert_eq!(budget.free(), 0);
        assert!(budget.try_acquire(1).is_none());
        assert_eq!(budget, budget.clone());
        assert_ne!(budget, MemoryBudget::new(1000));
    }

    #[test]
    fn the_resident_set_is_readable_and_the_sampler_sees_a_peak() {
        let Some(now) = resident_bytes() else {
            return;
        };
        assert!(now > 0);
        let sampler = ResidentSampler::start().unwrap();
        sampler.reset(0);
        let big = std::hint::black_box(vec![1u8; 64 << 20]);
        std::thread::sleep(SAMPLE_INTERVAL * 5);
        assert!(sampler.peak() >= 64 << 20, "{}", sampler.peak());
        drop(std::hint::black_box(big));
    }
}
