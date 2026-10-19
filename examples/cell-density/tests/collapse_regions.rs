// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The region collapse on constructed tables: counts, mean sizes, the full
//! region list, volumes and densities, and that the fold does not depend on
//! the cut or on the order the rows arrive in.

use std::collections::BTreeMap;

use atlas_regions::{Ontology, RegionNames};
use cell_density::collapse::{collapse, RegionCell, RegionContext, RegionRow};

fn cell(at: [usize; 3], region: i64, hemisphere: i64, size: f64) -> RegionCell {
    RegionCell {
        at,
        region,
        hemisphere,
        size,
    }
}

/// Twelve cells over two regions and two hemispheres, scattered so that a
/// fine cut splits every group across blocks.
fn table() -> Vec<RegionCell> {
    vec![
        cell([0, 0, 0], 4, 0, 50.0),
        cell([7, 7, 7], 4, 0, 70.0),
        cell([3, 5, 1], 4, 0, 60.0),
        cell([1, 6, 2], 4, 255, 100.0),
        cell([6, 1, 6], 4, 255, 200.0),
        cell([2, 2, 7], 9, 0, 80.0),
        cell([5, 3, 0], 9, 255, 90.0),
        cell([4, 4, 4], 9, 255, 91.0),
        cell([0, 7, 3], 9, 255, 92.0),
        cell([7, 0, 5], 9, 255, 93.0),
        cell([3, 3, 3], 0, -1, 55.0),
        cell([6, 6, 0], 0, -1, 65.0),
    ]
}

const VOLUME: [usize; 3] = [8, 8, 8];

fn ontology() -> Ontology {
    let mut tree = Ontology::new([
        (1, Some(0), None),
        (4, Some(1), Some(1)),
        (9, Some(1), Some(1)),
    ])
    .expect("a tree");
    for (id, name, acronym) in [
        (1, "root", "root"),
        (4, "four, with a comma", "IV"),
        (9, "nine", "IX"),
    ] {
        tree.set_names(
            id,
            RegionNames {
                name: Some(name.into()),
                acronym: Some(acronym.into()),
            },
        )
        .expect("a known id");
    }
    tree
}

fn find(rows: &[RegionRow], region: i64, hemisphere: i64) -> &RegionRow {
    rows.iter()
        .find(|row| row.region == region && row.hemisphere == hemisphere)
        .unwrap_or_else(|| panic!("no row for ({region}, {hemisphere})"))
}

#[test]
fn the_counts_and_means_are_per_region_and_hemisphere() {
    let rows = collapse(&table(), VOLUME, VOLUME, RegionContext::default()).unwrap();
    assert_eq!(rows.len(), 5, "the observed pairs only");
    let keys: Vec<(i64, i64)> = rows
        .iter()
        .map(|row| (row.region, row.hemisphere))
        .collect();
    assert_eq!(keys, [(0, -1), (4, 0), (4, 255), (9, 0), (9, 255)]);
    let left = find(&rows, 4, 0);
    assert_eq!((left.cells, left.mean_size), (3, Some(60.0)));
    let right = find(&rows, 4, 255);
    assert_eq!((right.cells, right.mean_size), (2, Some(150.0)));
    let nine = find(&rows, 9, 255);
    assert_eq!((nine.cells, nine.mean_size), (4, Some(91.5)));
    assert!(rows
        .iter()
        .all(|row| row.name.is_none() && row.volume_mm3.is_none()));
    assert_eq!(rows.iter().map(|row| row.cells).sum::<u64>(), 12);
}

#[test]
fn the_collapse_does_not_depend_on_the_cut_or_the_row_order() {
    let whole = collapse(&table(), VOLUME, VOLUME, RegionContext::default()).unwrap();
    for block in [[4, 4, 4], [2, 8, 1], [1, 1, 1], [3, 5, 7]] {
        let cut = collapse(&table(), VOLUME, block, RegionContext::default()).unwrap();
        assert_eq!(cut, whole, "block {block:?}");
    }
    let mut reversed = table();
    reversed.reverse();
    assert_eq!(
        collapse(&reversed, VOLUME, [2, 2, 2], RegionContext::default()).unwrap(),
        whole
    );
}

#[test]
fn every_region_is_listed_with_names_volumes_and_densities() {
    let tree = ontology();
    let volumes = BTreeMap::from([
        ((4, 0), 1.5),
        ((4, 255), 0.5),
        ((9, 0), 2.0),
        ((9, 255), 4.0),
    ]);
    let rows = collapse(
        &table(),
        VOLUME,
        [4, 4, 4],
        RegionContext {
            ontology: Some(&tree),
            volumes: Some(&volumes),
            voxels: None,
            all_regions: true,
        },
    )
    .unwrap();
    // Every tree id on every hemisphere seen, plus the observed outside row.
    let keys: Vec<(i64, i64)> = rows
        .iter()
        .map(|row| (row.region, row.hemisphere))
        .collect();
    assert_eq!(
        keys,
        [
            (0, -1),
            (1, -1),
            (1, 0),
            (1, 255),
            (4, -1),
            (4, 0),
            (4, 255),
            (9, -1),
            (9, 0),
            (9, 255)
        ]
    );
    let root = find(&rows, 1, 0);
    assert_eq!(
        (root.cells, root.mean_size),
        (0, None),
        "an empty region counts zero"
    );
    assert_eq!(root.acronym.as_deref(), Some("root"));
    let four = find(&rows, 4, 255);
    assert_eq!(four.name.as_deref(), Some("four, with a comma"));
    assert_eq!(four.volume_mm3, Some(0.5));
    assert_eq!(four.density_per_mm3, Some(4.0));
    let nine = find(&rows, 9, 0);
    assert_eq!(nine.density_per_mm3, Some(0.5));
    assert_eq!(
        find(&rows, 0, -1).density_per_mm3,
        None,
        "no volume for outside"
    );
}

#[test]
fn listing_every_region_needs_the_tree() {
    let failed = collapse(
        &table(),
        VOLUME,
        VOLUME,
        RegionContext {
            all_regions: true,
            ..RegionContext::default()
        },
    )
    .unwrap_err();
    assert!(failed.to_string().contains("region tree"), "{failed}");
}

#[test]
fn a_row_no_block_owns_or_a_fractional_size_is_refused() {
    let outside = vec![cell([9, 0, 0], 1, 0, 50.0)];
    let failed = collapse(&outside, [4, 4, 4], [2, 2, 2], RegionContext::default()).unwrap_err();
    assert!(failed.to_string().contains("no block owns it"), "{failed}");
    let fractional = vec![cell([0, 0, 0], 1, 0, 50.5)];
    assert!(collapse(&fractional, [4, 4, 4], [2, 2, 2], RegionContext::default()).is_err());
}

#[test]
fn no_cells_is_an_empty_table() {
    assert!(collapse(&[], VOLUME, [4, 4, 4], RegionContext::default())
        .unwrap()
        .is_empty());
}
