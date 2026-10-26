// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The alignment from stored tiles, box by box, against the alignment of the
//! same tiles held whole: the rigid step reads ahead the per-slice step's cuts
//! (and keeps them) where that decodes fewer stored chunks, and neither that
//! nor the memory cap changes an answer.

mod common;

use blockflow::budget::{process_cap, set_process_cap, MemoryCap};
use blockflow::voxels::Voxels;
use blockflow::zarr_env::{PyramidSpec, ZarrEnvironment};
use common::{crop, grid_layout, scratch, OVERLAP_X, OVERLAP_Y, TILE};
use tile_stitching::io::open_tile;
use tile_stitching::rigid::{align_layout_keeping, RigidParams};
use tile_stitching::tiles::StoredTiles;
use tile_stitching::wobbly::{cut_boxes_within, measure_layout_boxes, measure_layout_kept};
use tile_stitching::{align, align_stored, StitchParams};

/// A 2 x 2 grid cut from the scene with small offsets and a drift, each tile
/// stored with chunks small enough that the facing slabs and the cuts share
/// some.
fn tiles() -> (Vec<Voxels>, StoredTiles, tile_stitching::layout::TileLayout) {
    let grid = [[0, 0], [1, 0], [0, 1], [1, 1]];
    let step = [TILE[1] as isize - OVERLAP_Y, TILE[2] as isize - OVERLAP_X];
    let offsets = [[0, 0, 0], [1, 2, -1], [0, -1, 2], [1, 1, 1]];
    let volumes = grid
        .iter()
        .zip(offsets)
        .map(|(g, offset)| {
            crop(
                [
                    offset[0],
                    g[1] * step[0] + offset[1],
                    g[0] * step[1] + offset[2],
                ],
                TILE,
                |k| [0, (k as isize / 8) % 2],
            )
        })
        .collect::<Vec<_>>();
    let dir = scratch("stitch-stored");
    let spec = PyramidSpec::new(Vec::new())
        .expect("a one-level spec")
        .with_chunk([8, 16, 16]);
    let infos = volumes
        .iter()
        .enumerate()
        .map(|(index, volume)| {
            let path = dir.join(format!("tile_{index}.zarr"));
            ZarrEnvironment::build_multiscale(&path, volume, &spec).expect("a stored tile");
            open_tile(&path, 0, 0, None).expect("an opened tile")
        })
        .collect::<Vec<_>>();
    let stored = StoredTiles::new(&infos).expect("stored tiles");
    (volumes, stored, grid_layout(&grid))
}

fn params() -> StitchParams {
    let mut params = StitchParams {
        rigid: RigidParams::from_overlaps(&[OVERLAP_X as usize, OVERLAP_Y as usize], &[2, 1]),
        ..StitchParams::default()
    };
    params.rigid.max_shifts = [(-3, 3), (-4, 4), (-4, 4)];
    params.measure.max_shifts = [[-3, 3], [-3, 3]];
    params
}

/// One test, since it sets the process's memory cap.
#[test]
fn stored_tiles_align_as_resident_ones_with_and_without_reading_ahead() {
    let (volumes, stored, layout) = tiles();
    let params = params();
    let resident = format!(
        "{:?}",
        align(&volumes, layout.clone(), &params).expect("aligned")
    );

    // The stored alignment, and its two steps by hand, reading ahead.
    let previous = process_cap();
    let check = |what: &str| {
        let aligned = align_stored(&stored, layout.clone(), &params).expect("aligned");
        assert_eq!(format!("{aligned:?}"), resident, "{what}");

        let mut rigid = layout.clone();
        let pairs = rigid
            .alignments()
            .iter()
            .map(|alignment| (alignment.pre, alignment.post))
            .collect::<Vec<_>>();
        let margins = vec![[(-6, 6); 3]; pairs.len()];
        let ahead = cut_boxes_within(
            rigid.positions(),
            &pairs,
            &stored.shapes,
            0,
            &params.measure,
            &margins,
        );
        assert!(
            ahead.iter().all(Option::is_some),
            "{what}: every pair has boxes"
        );
        let kept = align_layout_keeping::<u16, _>(&mut rigid, &stored, &params.rigid, &ahead)
            .expect("rigid");
        rigid
            .place_in_place(&params.rigid_placement)
            .expect("placed");
        let count = kept.iter().filter(|pair| pair.is_some()).count();
        let measured =
            measure_layout_kept(rigid.positions(), &pairs, &stored, 0, &params.measure, kept)
                .expect("measured");
        let read =
            measure_layout_boxes::<u16, _>(rigid.positions(), &pairs, &stored, 0, &params.measure)
                .expect("measured");
        assert_eq!(format!("{measured:?}"), format!("{read:?}"), "{what}");
        count
    };
    let uncapped = check("uncapped");
    assert!(uncapped > 0, "some pair reads its cuts ahead");
    // A cap with room for little: fewer pairs, or none, keep their boxes.
    set_process_cap(MemoryCap::resolve(Some(64 << 10)));
    let capped = check("capped");
    assert!(
        capped < uncapped,
        "{capped} of {uncapped} kept under the cap"
    );
    set_process_cap(previous);
}
