// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The atlas columns and the crust filter on a synthetic brain: a ball of
//! region 2 inside a shell of region 1, split into two hemispheres along `x`,
//! on a grid four times coarser than the image.

use std::sync::Arc;

use atlas_regions::Ontology;
use blockflow::voxels::Voxels;
use cell_density::atlas::{annotate_cells, crust_keeps, AtlasFrame};
use cell_density::detect::CellRow;
use ndarray::Array3;

const ATLAS: [usize; 3] = [20, 20, 20];
const FACTOR: usize = 4;
const IMAGE: [usize; 3] = [80, 80, 80];
/// Image voxels of 2.5 µm, so atlas voxels of 10 µm.
const VOXEL_UM: [f64; 3] = [2.5, 2.5, 2.5];

fn annotation() -> Voxels {
    let mut labels = Array3::<u16>::zeros(ATLAS);
    for ((z, y, x), label) in labels.indexed_iter_mut() {
        let r2 = [z, y, x]
            .iter()
            .map(|c| (*c as f64 - 9.5).powi(2))
            .sum::<f64>();
        if r2 <= 4.0f64.powi(2) {
            *label = 2;
        } else if r2 <= 8.0f64.powi(2) {
            *label = 1;
        }
    }
    Voxels::U16(labels)
}

fn hemispheres() -> Voxels {
    Voxels::U8(Array3::from_shape_fn(
        ATLAS,
        |(_, _, x)| if x < 10 { 0 } else { 255 },
    ))
}

fn frame() -> AtlasFrame {
    AtlasFrame::covering(annotation(), IMAGE, VOXEL_UM)
        .unwrap()
        .with_hemispheres(hemispheres())
        .unwrap()
        .with_surface_distance()
        .unwrap()
}

/// A grid of cells every 6 image voxels over the whole image, all passing the
/// bands.
fn cells() -> Vec<CellRow> {
    let mut rows = Vec::new();
    for z in (1..IMAGE[0]).step_by(6) {
        for y in (1..IMAGE[1]).step_by(6) {
            for x in (1..IMAGE[2]).step_by(6) {
                rows.push(CellRow {
                    label_id: rows.len() as u64 + 1,
                    at: [z, y, x],
                    size: 100,
                    source: 1000.0,
                    background: 900.0,
                    passes_filter: true,
                    atlas: None,
                });
            }
        }
    }
    rows
}

fn kept(crust_um: f64) -> usize {
    let mut rows = cells();
    annotate_cells(&mut rows, &frame(), [0; 3], crust_um).unwrap();
    rows.iter().filter(|row| row.passes_filter).count()
}

#[test]
fn the_image_maps_onto_the_atlas_by_voxel_centres() {
    let frame = frame();
    assert_eq!(frame.scale, [0.25; 3]);
    assert_eq!(frame.voxel_size_um, [10.0; 3]);
    // Image voxels 0..4 lie under atlas voxel 0, whose centre is at 1.5.
    assert_eq!(frame.to_atlas([1.5, 0.0, 3.0]), [0.0, -0.375, 0.375]);
    assert_eq!(frame.to_atlas([41.5; 3]), [10.0; 3]);
    assert!((frame.voxel_volume_mm3() - 1e-6).abs() < 1e-18);
}

#[test]
fn every_cell_gets_its_region_hemisphere_and_depth() {
    let mut rows = cells();
    annotate_cells(&mut rows, &frame(), [0; 3], 0.0).unwrap();
    let mut seen = std::collections::BTreeSet::new();
    for row in &rows {
        let atlas = row.atlas.expect("annotated");
        seen.insert(atlas.region);
        let centre = row.at.map(|c| (c as f64 + 0.5) / FACTOR as f64 - 0.5);
        let r = centre.iter().map(|c| (c - 9.5).powi(2)).sum::<f64>().sqrt();
        if atlas.region == 2 {
            assert!(r < 5.0, "region 2 at radius {r}");
            assert!(atlas.distance_to_surface > 30.0, "{atlas:?}");
        }
        if atlas.region == 0 {
            assert_eq!(
                atlas.distance_to_surface, 0.0,
                "outside the brain is at depth 0"
            );
        }
        if r < 7.0 {
            let expected = if centre[2].round_ties_even() < 10.0 {
                0
            } else {
                255
            };
            assert_eq!(atlas.hemisphere, expected, "{row:?}");
        }
        assert!(row.passes_filter, "no crust, no change");
    }
    assert_eq!(seen.into_iter().collect::<Vec<_>>(), vec![0, 1, 2]);
}

/// A filter that dropped nothing would pass every other assertion.
#[test]
fn the_crust_actually_filters_and_deeper_keeps_fewer() {
    let all = cells().len();
    let inside = {
        let mut rows = cells();
        annotate_cells(&mut rows, &frame(), [0; 3], 0.0).unwrap();
        rows.iter()
            .filter(|row| row.atlas.unwrap().region > 0)
            .count()
    };
    assert_eq!(kept(0.0), all, "the arm off keeps every row");
    let shallow = kept(5.0);
    let deep = kept(40.0);
    assert!(
        shallow <= inside && shallow > deep && deep > 0,
        "{all} {inside} {shallow} {deep}"
    );
    assert_eq!(kept(1000.0), 0, "past the deepest point nothing is left");
}

#[test]
fn the_comparison_is_strict_and_no_distance_fails() {
    assert!(!crust_keeps(30.0, 30.0));
    assert!(crust_keeps(30.000_001, 30.0));
    assert!(!crust_keeps(f64::NAN, 0.0));
}

#[test]
fn a_cell_outside_the_annotation_has_no_region_and_fails_the_crust() {
    let mut rows = cells();
    rows.truncate(1);
    // The crop origin puts the cell past the image's (and the atlas') far edge.
    annotate_cells(&mut rows, &frame(), [200, 0, 0], 5.0).unwrap();
    let atlas = rows[0].atlas.unwrap();
    assert_eq!((atlas.region, atlas.hemisphere), (0, -1));
    assert!(atlas.distance_to_surface.is_nan());
    assert!(!rows[0].passes_filter);
}

#[test]
fn a_crust_without_a_distance_field_is_refused() {
    let bare = AtlasFrame::covering(annotation(), IMAGE, VOXEL_UM).unwrap();
    let mut rows = cells();
    assert!(annotate_cells(&mut rows, &bare, [0; 3], 5.0).is_err());
    // With the arm off it runs, and the distance column is unknown.
    annotate_cells(&mut rows, &bare, [0; 3], 0.0).unwrap();
    assert!(rows
        .iter()
        .all(|row| row.atlas.unwrap().distance_to_surface.is_nan()));
}

/// The verdict is a function of the point alone: chunking or reordering the
/// rows changes nothing.
#[test]
fn the_verdict_does_not_depend_on_chunking_or_order() {
    let frame = frame();
    let mut whole = cells();
    annotate_cells(&mut whole, &frame, [0; 3], 20.0).unwrap();
    for chunk in [1, 7, 64] {
        let mut pieces = cells();
        for slice in pieces.chunks_mut(chunk) {
            annotate_cells(slice, &frame, [0; 3], 20.0).unwrap();
        }
        assert_eq!(pieces, whole, "chunks of {chunk}");
    }
    let mut reversed = cells();
    reversed.reverse();
    annotate_cells(&mut reversed, &frame, [0; 3], 20.0).unwrap();
    reversed.reverse();
    assert_eq!(reversed, whole);
}

/// The bands and the crust are both row-wise predicates, so the order they
/// are applied in is not a choice.
#[test]
fn the_bands_and_the_crust_commute() {
    let frame = frame();
    let band = |row: &CellRow| !row.label_id.is_multiple_of(3);
    let mut crust_first = cells();
    annotate_cells(&mut crust_first, &frame, [0; 3], 20.0).unwrap();
    let crust_first: Vec<u64> = crust_first
        .iter()
        .filter(|row| row.passes_filter && band(row))
        .map(|row| row.label_id)
        .collect();
    let mut bands_first: Vec<CellRow> = cells();
    for row in &mut bands_first {
        row.passes_filter = band(row);
    }
    annotate_cells(&mut bands_first, &frame, [0; 3], 20.0).unwrap();
    let bands_first: Vec<u64> = bands_first
        .iter()
        .filter(|row| row.passes_filter)
        .map(|row| row.label_id)
        .collect();
    assert_eq!(crust_first, bands_first);
    assert!(!crust_first.is_empty() && crust_first.len() < cells().len() * 2 / 3);
}

#[test]
fn region_volumes_add_up_and_collapse_through_the_tree() {
    let frame = frame();
    let volumes = frame.region_volumes().unwrap();
    let voxels = |region: u16| {
        annotation()
            .view::<u16>()
            .unwrap()
            .iter()
            .filter(|label| **label == region)
            .count() as f64
    };
    let total = |region: i64| {
        volumes
            .iter()
            .filter(|((id, _), _)| *id == region)
            .map(|(_, volume)| volume)
            .sum::<f64>()
    };
    assert!((total(2) - voxels(2) * 1e-6).abs() < 1e-12);
    assert!((total(1) - voxels(1) * 1e-6).abs() < 1e-12);
    assert!(volumes
        .keys()
        .all(|(_, hemisphere)| [0, 255].contains(hemisphere)));

    // Collapsed onto level 1, region 2 (a child of 1) folds into 1.
    let tree = Ontology::new([(1, Some(1), None), (2, Some(2), Some(1))]).unwrap();
    let collapsed = frame.with_ontology(Arc::new(tree), Some(1));
    let volumes = collapsed.region_volumes().unwrap();
    assert!(volumes.keys().all(|(id, _)| [0, 1].contains(id)));
    let one: f64 = volumes
        .iter()
        .filter(|((id, _), _)| *id == 1)
        .map(|(_, volume)| volume)
        .sum();
    assert!((one - (voxels(1) + voxels(2)) * 1e-6).abs() < 1e-12);
    let mut rows = cells();
    annotate_cells(&mut rows, &collapsed, [0; 3], 0.0).unwrap();
    assert!(rows.iter().all(|row| row.atlas.unwrap().region <= 1));
}
