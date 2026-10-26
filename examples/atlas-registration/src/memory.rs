// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Leases of the process's memory budget for the volumes this crate holds
//! resident: the registration lattices and the fit's copies of them, the
//! images read whole to be resampled or warped, the placement search's
//! pyramid, and a prepared atlas opened for lookups.
//!
//! None of these is planned: each is one array as large as its image, so the
//! only choice is whether it fits. A lease is taken before the array is
//! allocated and dropped once it is freed, so that the chunk caches and any
//! planned phase running beside it see the bytes as taken. Under a stated or
//! cgroup cap a lease that does not fit is refused, naming the cap that would
//! hold it; under the machine's own share it is let through with a warning.

use blockflow::budget::{process_budget, process_cap, Lease, MemoryCap};
use blockflow::error::{Error, Result};

/// Lease `bytes` of the process's memory budget for `what`, which stays
/// resident until the lease is dropped (see the module notes for a lease
/// that does not fit).
pub fn lease(bytes: u64, what: &str) -> Result<Option<Lease>> {
    let budget = process_budget();
    if let Some(lease) = budget.try_acquire_reclaiming(bytes) {
        return Ok(Some(lease));
    }
    let cap = process_cap();
    let held = budget.reserved_in_use();
    let needed = MemoryCap::needed_for(held.saturating_add(bytes));
    let message = format!(
        "{what} needs about {} MiB resident, and the memory cap ({cap}) leaves {} MiB beside \
         the {} MiB already held; a cap of about {} MiB holds it.",
        bytes.div_ceil(1 << 20),
        budget.free() >> 20,
        held >> 20,
        needed.div_ceil(1 << 20)
    );
    if cap.is_binding() {
        return Err(Error::invalid(message));
    }
    eprintln!("atlas-registration: warning: {message}");
    Ok(None)
}

/// Voxels of a `[z, y, x]` shape, as a byte multiplier.
pub fn voxels(shape: [usize; 3]) -> u64 {
    shape.iter().map(|&n| n as u64).product()
}
