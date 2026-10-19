// SPDX-License-Identifier: MIT
// Original work for this crate.

//! What a resident run of the segmentation holds, per voxel, at three volumes
//! of a tile's cross-section: the process's peak resident set over the
//! volume, read from `/proc/self/status` (Linux only).
//!
//! The in-memory runner holds every image of the plan, so this is the figure
//! that says how large a volume fits in memory; an out-of-core run through
//! `ZarrEnvironment` holds blocks instead.

use blockflow_vessel_graph::segment::synthetic::vessel_volume;
use blockflow_vessel_graph::segment::{segment, SegmentParams};

fn status_field(name: &str) -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with(name))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib * 1024)
}

fn reset_high_water() -> bool {
    std::fs::write("/proc/self/clear_refs", "5").is_ok()
}

#[test]
#[ignore = "allocates gigabytes and takes minutes; Linux only"]
fn the_resident_cost_is_a_constant_per_voxel() {
    let params = SegmentParams::with_channels(1);
    for depth in [6usize, 12, 24] {
        let shape = [depth, 1304, 404];
        let voxels: usize = shape.iter().product();
        let volume = vessel_volume(shape, 1);
        let before = status_field("VmRSS").unwrap_or(0);
        let reset = reset_high_water();
        segment(vec![volume], &params, [depth, 256, 256]).expect("a run");
        let peak = status_field("VmHWM").unwrap_or(0);
        println!(
            "{shape:?}: peak {:.2} GiB, {:.1} bytes a voxel (resident before: {:.2} GiB){}",
            peak as f64 / f64::from(1 << 30),
            peak as f64 / voxels as f64,
            before as f64 / f64::from(1 << 30),
            if reset {
                ""
            } else {
                " (high-water mark not reset)"
            }
        );
    }
}
