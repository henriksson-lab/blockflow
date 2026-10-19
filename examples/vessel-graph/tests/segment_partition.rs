// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The binarisation's plan, priced without running it: what moves its shape
//! and what does not.
//!
//! Most phases are fixed by the arms — a boundary another arm names is a
//! requirement, so those are `PlanBuilder::pixels` phases. The deconvolve and
//! vesselize arms are sequences the planner may cut, so their phase counts are
//! the planner's answer and are bounded here rather than pinned.

use blockflow::Dtype;
use blockflow_vessel_graph::segment::{binarize_plan, BinarizeParams};

const VOLUME: [usize; 3] = [12, 256, 256];

/// Population, not-low, clip, lightsheet, median, equalize gain, seed, the
/// deconvolve sink, the adaptive sink and the equalize sink.
const FIXED_PHASES: usize = 10;

/// An arm's name, how to switch it off, and how many phases go with it.
type Removal = (&'static str, fn(&mut BinarizeParams), usize);

fn priced(params: &BinarizeParams, dtype: Dtype, block: [usize; 3]) -> (usize, Vec<usize>) {
    let (assembly, _) = binarize_plan(VOLUME, dtype, params, block).expect("the plan assembles");
    (
        assembly.n_phases(),
        assembly.decomposition.exact_read_voxels(),
    )
}

/// The source's element type is an accounting choice and not a numerical one:
/// every op reading image 0 widens what it reads, so a `u16` source plans the
/// same phases and reads the same voxels as an `f64` one — in a quarter of the
/// bytes.
#[test]
fn the_sources_width_moves_no_read() {
    let params = BinarizeParams::default();
    for block in [[12, 256, 256], [12, 64, 64], [6, 32, 32]] {
        assert_eq!(
            priced(&params, Dtype::U16, block),
            priced(&params, Dtype::F64, block),
            "at {block:?}"
        );
    }
}

/// Every arm switched off takes its phases with it, and the two planned arms
/// are at least one phase each.
#[test]
fn the_plans_shape_moves_with_the_arms() {
    let block = [12, 64, 64];
    let full = BinarizeParams::default();
    let (all, _) = priced(&full, Dtype::U16, block);
    assert!(
        all >= FIXED_PHASES + 2,
        "{all} phases with every arm on; at least {} expected",
        FIXED_PHASES + 2
    );
    let without = |change: fn(&mut BinarizeParams)| {
        let mut params = full.clone();
        change(&mut params);
        priced(&params, Dtype::U16, block).0
    };
    // At least this many phases go with each arm.
    let removals: [Removal; 6] = [
        ("lightsheet", |p| p.lightsheet = None, 1),
        ("median", |p| p.median = None, 2),
        ("deconvolve", |p| p.deconvolve = None, 2),
        ("adaptive", |p| p.adaptive = None, 1),
        ("equalize", |p| p.equalize = None, 2),
        ("vesselize", |p| p.vesselize = None, 1),
    ];
    for (arm, change, at_least) in removals {
        let fewer = without(change);
        assert!(
            fewer + at_least <= all,
            "turning {arm} off left {fewer} of {all} phases; it owns at least {at_least}"
        );
    }
    // With every arm and the median off, nothing reads either population and
    // the plan is the clip and the seed.
    let (bare, _) = priced(&BinarizeParams::seed_only(full.clip), Dtype::U16, block);
    assert_eq!(bare, 2);
}

/// What the plan reads at four blockings, relative to the volume. Printed for
/// the record; asserted only to be finite and at least one read per phase.
#[test]
fn the_read_amplification_across_four_blockings() {
    let voxels: usize = VOLUME.iter().product();
    for block in [[12, 256, 256], [12, 128, 128], [12, 64, 64], [6, 32, 32]] {
        let (phases, reads) = priced(&BinarizeParams::default(), Dtype::U16, block);
        let amplification = reads.iter().sum::<usize>() as f64 / voxels as f64;
        println!("{block:?}: {phases} phases, {amplification:.3}x the volume");
        assert!(amplification.is_finite() && amplification >= phases as f64);
    }
}
