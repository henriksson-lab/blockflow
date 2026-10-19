// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Real data, ignored: a cFos crop through the osf-cztq2 chain registration
//! (signal mosaic -> autofluorescence -> atlas) into the prepared atlas, and
//! the per-cell regions compared with the reference's for the same cells
//! (`reference-atlas/compare_atlas.py` writes `cells_reference.csv` from a run
//! with the same crop and parameters; the run is deterministic, so label ids
//! pair up).
//!
//! Needs `/husky/otherdataset/teresa/clearmap_em/osf-cztq2/cFos/tile_01x02.zarr`,
//! the registration under `/husky/henriksson/lightsheet-reference/registration/
//! ours/chain`, the atlas `.../atlas/ABA_25um_2017_1_2_3_z0-256.ome.zarr` and
//! `.../cells-atlas/clearmap/tile_01x02_crop/cells_reference.csv`. Writes under
//! `.../cells-atlas/ignored-test/`. About two minutes on 40 threads.
//!
//! The tile's offset into the mosaic: tile `(r, c) = (1, 2)` starts at mosaic
//! voxel `round([0, r * 1471, c * 1368] / 25) = [0, 59, 109]`, and the mosaic
//! holds the tile's level 4 (translation `[9, 12.1875, 12.1875]` µm) resampled
//! to 25 µm, so a tile position `p` sits at mosaic position
//! `p + 25 * start - t4 = p + [-9, 1462.8125, 2712.8125]` µm.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use blockflow::object_table::TableReader;
use cell_density::cli::{run, Args};
use clap::Parser;

const REFERENCE: &str = "/husky/henriksson/lightsheet-reference";
const TILE: &str = "/husky/otherdataset/teresa/clearmap_em/osf-cztq2/cFos/tile_01x02.zarr";

#[test]
#[ignore = "reads a real cFos tile, a registration and the reference's mapped cells; run with --release --ignored"]
fn a_real_crop_lands_in_the_references_regions() {
    let root = PathBuf::from(REFERENCE);
    let registration = root.join("registration/ours/chain");
    let atlas = root.join("atlas/ABA_25um_2017_1_2_3_z0-256.ome.zarr");
    let reference = root.join("cells-atlas/clearmap/tile_01x02_crop/cells_reference.csv");
    for path in [&PathBuf::from(TILE), &registration, &atlas, &reference] {
        if !path.exists() {
            eprintln!("skipping: {} is not on this machine", path.display());
            return;
        }
    }
    let output = root.join("cells-atlas/ignored-test");
    let density = root.join("cells-atlas/ignored-test-density.ome.zarr");
    let args = Args::parse_from([
        "cell-density",
        "--input",
        TILE,
        "--output",
        output.to_str().unwrap(),
        "--crop-start",
        "256,192,192",
        "--crop-shape",
        "512,640,640",
        "--registration",
        registration.to_str().unwrap(),
        "--atlas",
        atlas.to_str().unwrap(),
        "--sample-offset-um=-9,1462.8125,2712.8125",
        "--density",
        density.to_str().unwrap(),
        "--overwrite",
    ]);
    let started = Instant::now();
    let report = run(&args).unwrap();
    eprintln!(
        "run {:.1} s (detect {:.1} s), mapping {}",
        started.elapsed().as_secs_f64(),
        report["seconds"]["detect"],
        report["atlas_mapping"]
    );

    let table = TableReader::open(output.join("tables/cell-density-filtered")).unwrap();
    let n = table.spec().row_count;
    let ids = table.read_u64("label_id", 0..n).unwrap();
    let regions = table.read_i64("region", 0..n).unwrap();
    let hemispheres = table.read_i64("hemisphere", 0..n).unwrap();
    let ours: BTreeMap<u64, (i64, i64)> = ids
        .into_iter()
        .zip(regions.into_iter().zip(hemispheres))
        .collect();

    let text = std::fs::read_to_string(&reference).unwrap();
    let (mut paired, mut same_region, mut same_hemisphere) = (0usize, 0usize, 0usize);
    let mut counts: BTreeMap<(i64, i64), (i64, i64)> = BTreeMap::new();
    for line in text.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        let id: u64 = fields[0].parse().unwrap();
        let region: i64 = fields[4].parse().unwrap();
        let hemisphere: i64 = fields[5].parse().unwrap();
        let Some(&(our_region, our_hemisphere)) = ours.get(&id) else {
            continue;
        };
        paired += 1;
        same_region += usize::from(our_region == region);
        same_hemisphere += usize::from(our_hemisphere == hemisphere);
        counts.entry((our_region, our_hemisphere)).or_default().0 += 1;
        counts.entry((region, hemisphere)).or_default().1 += 1;
    }
    let difference: i64 = counts.values().map(|(a, b)| (a - b).abs()).sum();
    let region_fraction = same_region as f64 / paired as f64;
    let count_difference = difference as f64 / paired as f64;
    eprintln!(
        "{paired} of {n} cells paired; same region {region_fraction:.4}, same hemisphere \
         {:.4}; sum |count difference| / cells {count_difference:.4} over {} regions",
        same_hemisphere as f64 / paired as f64,
        counts.len()
    );
    // Recorded 2026-10-19: 21 997 paired, 0.903 same finest region, 1.0 same
    // hemisphere, 0.061 summed count difference. One atlas voxel (25 µm, the
    // two registrations' median disagreement here) moves a tenth of the
    // finest labels, so these bounds are about the registrations, not the
    // labelling.
    assert_eq!(
        paired as u64, n,
        "the run must reproduce the recorded cells"
    );
    assert!(region_fraction > 0.85, "{region_fraction}");
    assert!(same_hemisphere as f64 / paired as f64 > 0.99);
    assert!(count_difference < 0.1, "{count_difference}");
}
