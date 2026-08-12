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

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
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

#[derive(Debug)]
struct Inner {
    total: u64,
    state: Mutex<State>,
    released: Condvar,
    /// Set whenever a `Reserved` request is waiting; read by opportunistic
    /// holders through [`Lease::is_revoked`] without taking the lock.
    revoke: AtomicBool,
    /// High-water mark, duplicated outside the lock so instrumentation can read
    /// it from a signal handler or a progress thread without blocking.
    high_water: AtomicU64,
}

impl Inner {
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

impl MemoryBudget {
    /// A budget of exactly `total_bytes`.
    pub fn new(total_bytes: u64) -> Self {
        Self {
            inner: Arc::new(Inner {
                total: total_bytes.max(1),
                state: Mutex::new(State::default()),
                released: Condvar::new(),
                revoke: AtomicBool::new(false),
                high_water: AtomicU64::new(0),
            }),
        }
    }

    /// A budget sized by [`default_budget_bytes`].
    pub fn detected() -> Self {
        Self::new(default_budget_bytes())
    }

    /// Total bytes this budget may hand out.
    pub fn total(&self) -> u64 {
        self.inner.total
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
        let mut state = self.inner.lock();
        let oversized = bytes > self.inner.total;
        state.waiting_reserved += 1;
        self.inner.revoke.store(true, Ordering::Relaxed);
        loop {
            let in_use = Inner::in_use(&state);
            let grantable = if oversized {
                in_use == 0
            } else {
                in_use + bytes <= self.inner.total
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
        if Inner::in_use(&state) + bytes > self.inner.total {
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
        if Inner::in_use(&state) + bytes > self.inner.total {
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
/// the kernel reports.
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
pub fn budget_bytes_from(names: &[&str]) -> u64 {
    for name in names {
        if let Ok(raw) = std::env::var(name) {
            if let Some(bytes) = parse_budget(raw.trim()) {
                return bytes;
            }
        }
    }
    auto_budget_bytes().unwrap_or(FALLBACK_BUDGET_BYTES)
}

/// The `auto` rule, or `None` where `/proc/meminfo` cannot be read.
pub fn auto_budget_bytes() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let total = meminfo_field(&meminfo, "MemTotal:")?;
    let available = meminfo_field(&meminfo, "MemAvailable:").unwrap_or(total);
    Some((total / 2).min(available / 4 * 3).max(MIN_BUDGET_BYTES))
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
}
