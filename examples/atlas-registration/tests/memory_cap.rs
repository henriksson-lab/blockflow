// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The resident volumes' leases under a stated cap: one that fits is granted
//! and returned on drop, one that does not is refused naming the cap that
//! would hold it. Its own test binary, since the process cap is global.

use blockflow::budget::{process_budget, set_process_cap, CapSource, MemoryCap};
use blockflow_atlas_registration::memory::lease;

#[test]
fn a_lease_over_a_stated_cap_is_refused_naming_the_cap_that_holds_it() {
    set_process_cap(MemoryCap {
        bytes: 256 << 20,
        source: CapSource::Explicit,
    });
    let usable = process_budget().total();
    assert_eq!(usable, 128 << 20);

    let held = lease(100 << 20, "the first volume").unwrap().unwrap();
    assert_eq!(process_budget().reserved_in_use(), 100 << 20);

    let error = lease(64 << 20, "the second volume")
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("the second volume needs about 64 MiB"),
        "{error}"
    );
    // 164 MiB usable needs a cap of 164 + 128 MiB.
    assert!(error.contains("a cap of about 292 MiB holds it"), "{error}");

    drop(held);
    assert_eq!(process_budget().reserved_in_use(), 0);
    assert!(lease(64 << 20, "the second volume").unwrap().is_some());
}
