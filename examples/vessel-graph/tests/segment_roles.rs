// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Per-channel binarisation against the reference's, role by role, on volumes
//! the reference binarised with `reference-clearmap/binarize_roles.py`: the
//! recorded 32 x 96 x 96 crop (both roles over its one channel) and a
//! 64 x 256 x 256 region of a vasculature tile (the vessel role over the
//! all-vessel channel, the artery role over the artery channel). Both sides
//! are the binarisation alone, before smoothing and filling.

use std::path::{Path, PathBuf};
use std::time::Instant;

use blockflow::npy::{read_array_file, OrderPolicy};
use blockflow::voxels::Voxels;
use blockflow_vessel_graph::segment::{binarize_volume, ChannelRole};
use ndarray::{Array3, Ix3};

fn reference() -> PathBuf {
    PathBuf::from(
        std::env::var("VESSEL_REFERENCE")
            .unwrap_or_else(|_| "/husky/henriksson/lightsheet-reference/vessels".to_string()),
    )
}

fn load<T: Clone + blockflow::npy::NpyElement>(path: &Path) -> Array3<T> {
    let array: ndarray::ArrayD<T> = read_array_file(path, OrderPolicy::Either)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    array.into_dimensionality::<Ix3>().expect("a volume")
}

/// Binarise `source` (`[z, y, x]`) at `role`'s defaults and compare with the
/// reference's mask; returns the Dice coefficient.
fn compare(case: &str, source: Array3<u16>, role: ChannelRole, recorded: &Path) -> f64 {
    let recorded: Array3<bool> = load(recorded);
    let shape = [source.shape()[0], source.shape()[1], source.shape()[2]];
    assert_eq!(recorded.shape(), &shape[..], "{case}: shapes");
    let started = Instant::now();
    let (_, binary) = binarize_volume(Voxels::U16(source), &role.binarize(), shape).unwrap();
    let seconds = started.elapsed().as_secs_f64();
    let ours = binary.widened().mapv(|value| value != 0.0);
    let count = |mask: &Array3<bool>| mask.iter().filter(|set| **set).count();
    let both = ours
        .iter()
        .zip(&recorded)
        .filter(|(a, b)| **a && **b)
        .count();
    let dice = 2.0 * both as f64 / (count(&ours) + count(&recorded)).max(1) as f64;
    println!(
        "{case} {:6} {shape:?}: ours {} ({seconds:.2}s), reference {}, both {both}, Dice {dice:.3}",
        role.name(),
        count(&ours),
        count(&recorded)
    );
    dice
}

#[test]
#[ignore = "needs the reference's role binarisations under $VESSEL_REFERENCE \
            (default /husky/henriksson/lightsheet-reference/vessels), made by \
            reference-clearmap/run.sh"]
fn each_role_binarises_close_to_the_reference() {
    let root = reference();
    let fixture = root.join("fixture");
    // Recorded `[x, y, z]`.
    let crop: Array3<u16> = load(&fixture.join("source_xyz.npy"));
    let crop = crop
        .permuted_axes([2, 1, 0])
        .as_standard_layout()
        .into_owned();
    let mut results = Vec::new();
    for role in [ChannelRole::Vessel, ChannelRole::Artery] {
        let recorded = fixture.join(format!("clearmap_{}.npy", role.name()));
        results.push(("crop", role, compare("crop", crop.clone(), role, &recorded)));
    }
    let region = root.join("tile05x05-z1600");
    for (channel, role) in [(0, ChannelRole::Vessel), (1, ChannelRole::Artery)] {
        let source: Array3<u16> = load(&region.join(format!("source_c{channel}.npy")));
        let recorded = region.join(format!("clearmap_{}.npy", role.name()));
        results.push(("tile", role, compare("tile", source, role, &recorded)));
    }
    for (case, role, dice) in results {
        assert!(dice > 0.95, "{case} {}: Dice {dice:.3}", role.name());
    }
}
