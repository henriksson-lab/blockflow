// SPDX-License-Identifier: MIT
// Original work for this crate.

//! What the hole filling costs, measured from outside it.
//!
//! The fill is two fragment phases: one labels each block's background, one
//! merges the labellings and fills. The merge is hoisted into the second
//! phase's barrier `reduce`, so the fragment set is read once for the phase —
//! linear in the blocks — rather than once per block, which would be
//! quadratic. No pixel counter can tell the two placements apart; the
//! fragment counters can.

use blockflow::log::Stats;
use blockflow::ops::Connectivity;
use blockflow::voxels::Voxels;
use blockflow_vessel_graph::segment::{mask_steps, MaskSteps, PostProcess};
use ndarray::Array3;

/// A deterministic speckle with a foreground shell, so the enclosed background
/// is genuinely enclosed.
fn speckled(volume: [usize; 3], seed: u64, foreground_in_16: u64) -> Voxels {
    let mut state = seed | 1;
    let mask = Array3::from_shape_fn((volume[0], volume[1], volume[2]), |(z, y, x)| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let on_shell = z == 0
            || y == 0
            || x == 0
            || z + 1 == volume[0]
            || y + 1 == volume[1]
            || x + 1 == volume[2];
        on_shell || (state >> 33) % 16 < foreground_in_16
    });
    Voxels::Bool(mask)
}

fn fill(mask: &Voxels, edge: usize) -> (Stats, Voxels) {
    let steps = MaskSteps {
        post: Some(PostProcess {
            smooth: None,
            fill: Some(Connectivity::Faces),
        }),
        skeleton: None,
    };
    let (stats, filled, _) = mask_steps(mask.clone(), &steps, [edge; 3]).expect("a fill");
    (stats, filled)
}

fn blocks(volume: [usize; 3], edge: usize) -> u64 {
    volume.iter().map(|len| len.div_ceil(edge) as u64).product()
}

/// Fragment reads grow linearly with the block count, not quadratically.
#[test]
fn the_merge_is_hoisted_out_of_the_block_loop() {
    let mask = speckled([64, 64, 64], 55, 6);
    let mut reads = Vec::new();
    for edge in [32usize, 16] {
        let (stats, _) = fill(&mask, edge);
        let n = blocks(mask.shape(), edge);
        println!(
            "block {edge}: {n} blocks, {} fragments read, {} written",
            stats.sidecar_reads, stats.sidecar_writes
        );
        assert!(
            stats.sidecar_reads <= 4 * n,
            "{} fragment reads over {n} blocks; a per-block merge would read {}",
            stats.sidecar_reads,
            n * n
        );
        reads.push((n, stats.sidecar_reads));
    }
    let growth = reads[1].1 as f64 / reads[0].1 as f64;
    let block_growth = reads[1].0 as f64 / reads[0].0 as f64;
    assert!(growth <= block_growth * 1.5, "reads grew {growth:.1}x");
}

/// A fragment-only plan reports its work in `fragment_applications`, not in
/// `ops_applied`.
#[test]
fn a_fragment_only_run_reports_its_work_somewhere() {
    let mask = speckled([32, 32, 32], 55, 6);
    let (stats, _) = fill(&mask, 16);
    let n = blocks(mask.shape(), 16);
    assert_eq!(n, 8);
    assert_eq!(stats.ops_applied, 0, "no chain slot in a fill");
    assert_eq!(
        stats.fragment_applications,
        2 * n,
        "two phases, once per block"
    );
    assert_eq!(
        stats.sidecar_writes, n,
        "the labelling writes one fragment per block"
    );
    // Once for the phase, once more for the reversed-order check an unordered
    // seam fold runs.
    assert_eq!(stats.sidecar_reads, 2 * n);
}

/// The counters move with the lattice; the answer and the pixel reads do not.
#[test]
fn the_fragment_counters_move_with_the_lattice() {
    let mask = speckled([32, 32, 32], 55, 6);
    let (whole, resident) = fill(&mask, 32);
    let (cut, decomposed) = fill(&mask, 16);
    assert_eq!(resident.widened(), decomposed.widened());
    assert_eq!(whole.sidecar_writes, 1);
    assert_eq!(cut.sidecar_writes, 8);
    assert_eq!(whole.fragment_applications, 2);
    assert_eq!(cut.fragment_applications, 16);
    assert_eq!(whole.read_voxels, cut.read_voxels);
}
