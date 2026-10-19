// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The binarisation's arms, one at a time and together.
//!
//! The sink is a union, and a union hides an arm that does nothing: an arm can
//! deposit voxels and still be worth nothing because another arm covers them.
//! So each arm is checked twice — on its own over the saturated seed, where it
//! has to find vessel, and as the one arm missing from the full chain, where
//! its worth is how many voxels the chain loses without it.

use std::path::PathBuf;

use blockflow::voxels::Voxels;
use blockflow_vessel_graph::segment::synthetic::{tube_mask, vessel_volume};
use blockflow_vessel_graph::segment::{binarize_volume, BinarizeParams, ClipParams};
use ndarray::{Array3, Axis, Ix3};

const SHAPE: [usize; 3] = [24, 64, 64];
const BLOCK: [usize; 3] = [12, 32, 32];

fn mask(voxels: &Voxels) -> Array3<bool> {
    voxels.widened().mapv(|value| value != 0.0)
}

fn count(mask: &Array3<bool>) -> usize {
    mask.iter().filter(|set| **set).count()
}

fn hits(mask: &Array3<bool>, truth: &Array3<bool>) -> usize {
    mask.iter().zip(truth).filter(|(a, b)| **a && **b).count()
}

type Arm = (&'static str, fn(&mut BinarizeParams, &BinarizeParams));

const ARMS: [Arm; 4] = [
    ("deconvolve", |p, full| p.deconvolve = full.deconvolve),
    ("adaptive", |p, full| p.adaptive = full.adaptive),
    ("equalize", |p, full| p.equalize = full.equalize),
    ("vesselize", |p, full| p.vesselize = full.vesselize),
];

/// Each arm on its own — over the seed, after the median — finds vessel the
/// seed alone does not, and most of what it adds is vessel.
#[test]
fn every_arm_finds_vessel_on_its_own() {
    let volume = vessel_volume(SHAPE, 7);
    let truth = tube_mask(SHAPE, 2.5);
    let full = BinarizeParams::default();
    let mut seed = BinarizeParams::seed_only(full.clip);
    seed.median = full.median;
    let (_, binary) = binarize_volume(volume.clone(), &seed, BLOCK).unwrap();
    let base = mask(&binary);
    assert!(count(&base) > 0, "the fixture has a saturated core to seed");
    for (name, add) in ARMS {
        let mut params = seed.clone();
        add(&mut params, &full);
        let (_, binary) = binarize_volume(volume.clone(), &params, BLOCK).unwrap();
        let arm = mask(&binary);
        let added = count(&arm) - count(&base);
        let added_hits = hits(&arm, &truth) - hits(&base, &truth);
        println!("{name:10}: adds {added} voxels, {added_hits} on vessel");
        assert!(added_hits > 100, "{name} found {added_hits} vessel voxels");
        assert!(
            2 * added_hits > added,
            "{name}: {added_hits} of the {added} voxels it adds are vessel"
        );
        // The seed is a subset of every arm's sink.
        assert!(base.iter().zip(&arm).all(|(b, a)| !*b || *a), "{name}");
    }
}

/// The full chain loses voxels without each of the adaptive, equalize and
/// vesselize arms; the deconvolve arm's own voxels are covered by the others
/// on this fixture, which is reported rather than asserted.
#[test]
fn what_each_arm_is_worth_in_the_full_chain() {
    let volume = vessel_volume(SHAPE, 7);
    let full = BinarizeParams::default();
    let (_, binary) = binarize_volume(volume.clone(), &full, BLOCK).unwrap();
    let all = mask(&binary);
    for (name, _) in ARMS {
        let mut params = full.clone();
        match name {
            "deconvolve" => params.deconvolve = None,
            "adaptive" => params.adaptive = None,
            "equalize" => params.equalize = None,
            _ => params.vesselize = None,
        }
        let (_, binary) = binarize_volume(volume.clone(), &params, BLOCK).unwrap();
        let without = mask(&binary);
        let worth = all.iter().zip(&without).filter(|(a, b)| a != b).count();
        println!("{name:10}: worth {worth} voxels of {}", count(&all));
        if name != "deconvolve" {
            assert!(worth > 0, "the chain does not need {name}");
        }
    }
}

/// A clip range whose top is above the data saturates nothing, so the seed is
/// empty and so is a chain with no other arm.
#[test]
fn a_clip_above_the_data_seeds_nothing() {
    let volume = vessel_volume(SHAPE, 7);
    let params = BinarizeParams::seed_only(ClipParams {
        range: (400.0, 65_535.5),
        bins: 4096,
    });
    let (_, binary) = binarize_volume(volume, &params, BLOCK).unwrap();
    assert_eq!(count(&mask(&binary)), 0);
}

/// The lightsheet correction runs under the median, so it changes the volume
/// every later arm reads; and it never raises a voxel.
#[test]
fn the_lightsheet_correction_moves_the_filtered_volume_down() {
    let volume = vessel_volume(SHAPE, 7);
    let full = BinarizeParams::default();
    let mut without = full.clone();
    without.lightsheet = None;
    let (with, _) = binarize_volume(volume.clone(), &full, BLOCK).unwrap();
    let (plain, _) = binarize_volume(volume, &without, BLOCK).unwrap();
    let with = with.widened();
    let plain = plain.widened();
    let lowered = with.iter().zip(plain.iter()).filter(|(a, b)| a < b).count();
    assert!(lowered > SHAPE.iter().product::<usize>() / 2);
    assert!(with.iter().zip(plain.iter()).all(|(a, b)| a <= b));
}

fn fixtures() -> PathBuf {
    PathBuf::from(
        std::env::var("VESSEL_GRAPH_FIXTURES")
            .unwrap_or_else(|_| "/husky/henriksson/clearmap-img/outputs".to_string()),
    )
}

/// `(x, y, z)` arrays as recorded, to this crate's `(z, y, x)`.
fn zyx<T: Clone>(array: ndarray::ArrayD<T>) -> Array3<T> {
    let array = array.into_dimensionality::<Ix3>().expect("a volume");
    array
        .permuted_axes([2, 1, 0])
        .as_standard_layout()
        .into_owned()
}

/// A recorded vasculature crop: the binarisation at the defaults against the
/// recorded mask of the same crop, as a Dice coefficient. Not a parity check —
/// the arms here differ in rounding, percentile convention, lattice mapping
/// and tubeness arithmetic — but the two masks should mostly agree.
#[test]
#[ignore = "needs a recorded crop (source.npy, binary.npy) under $VESSEL_GRAPH_FIXTURES/vasc-binarize"]
fn a_recorded_crop_binarises_close_to_its_recording() {
    use blockflow::npy::{read_array_file, OrderPolicy};
    let dir = fixtures().join("vasc-binarize");
    let source: ndarray::ArrayD<u16> =
        read_array_file(&dir.join("source.npy"), OrderPolicy::Either).expect("source.npy");
    let recorded: ndarray::ArrayD<bool> =
        read_array_file(&dir.join("binary.npy"), OrderPolicy::Either).expect("binary.npy");
    let source = zyx(source);
    let recorded = zyx(recorded);
    let shape = [
        source.len_of(Axis(0)),
        source.len_of(Axis(1)),
        source.len_of(Axis(2)),
    ];
    let (_, binary) =
        binarize_volume(Voxels::U16(source), &BinarizeParams::default(), shape).unwrap();
    let ours = mask(&binary);
    let both = hits(&ours, &recorded);
    let dice = 2.0 * both as f64 / (count(&ours) + count(&recorded)) as f64;
    println!(
        "{shape:?}: ours {}, recorded {}, both {both}, Dice {dice:.3}",
        count(&ours),
        count(&recorded)
    );
    assert!(dice > 0.8, "Dice {dice:.3}");
}
