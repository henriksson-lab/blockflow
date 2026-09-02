//! The viewer's own bytes, rasterised.
//!
//! `tests/data/viewer_fragments.bftable` was written by the OME-Zarr viewer's
//! `annotations::fragments` and committed here unchanged. It is the only thing
//! that checks the two halves of this pipeline against each other: the viewer
//! does not depend on this crate (it reimplements the table format rather than
//! pull burn, candle and a CUDA toolchain behind it), so nothing else can
//! compare the layout the producer writes with the layout this op reads.
//!
//! The counterpart is the viewer's `server/tests/fragments_golden.rs`, which
//! asserts its writer still produces exactly these bytes. A change on either
//! side that the other does not expect fails one of the two tests. Regenerate
//! there, then copy the file here.

use blockflow::error::Result;
use blockflow::ops::rasterise::{rasterise_into, shapes_of, vertex_schema};
use blockflow::region::Region;
use blockflow::BlockGrid;
use ndarray::Array3;

const FIXTURE: &[u8] = include_bytes!("data/viewer_fragments.bftable");

/// The scene the viewer encoded: a polygon with a hole, an open stroke with a
/// width, and a dense region. Described in the viewer's own fixture test.
const VOLUME: [usize; 3] = [8, 128, 128];

fn rasterise() -> Result<Array3<u64>> {
    let grid = BlockGrid::whole(VOLUME)?;
    let mut out = Array3::<u64>::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    rasterise_into(
        &[([0, 0, 0], FIXTURE.to_vec())],
        &grid,
        VOLUME,
        u64::MAX,
        &Region::whole(&VOLUME),
        out.view_mut(),
    )?;
    Ok(out)
}

#[test]
fn the_viewers_columns_are_the_ones_this_op_reads() {
    // Named, ordered and typed. A consumer reads columns positionally, so a
    // reordering upstream is not a rename — it swaps two meanings silently.
    let schema = vertex_schema();
    let names: Vec<&str> = schema.columns().iter().map(|c| c.name()).collect();
    assert_eq!(
        names,
        vec![
            "shape",
            "ring",
            "vertex",
            "class",
            "closed",
            "dense",
            "z_extent",
            "x",
            "y",
            "half_width",
        ]
    );
    // And the blob really parses under it, which the name list alone does not
    // prove: a type disagreement passes the check above and fails here.
    let grid = BlockGrid::whole(VOLUME).unwrap();
    let shapes = shapes_of(&[([0, 0, 0], FIXTURE.to_vec())], &grid, VOLUME, u64::MAX)
        .expect("the viewer's blob parses under this op's schema");
    assert_eq!(shapes.len(), 3, "three shapes were drawn");
}

#[test]
fn a_polygon_with_a_hole_still_has_its_hole_after_a_round_trip() {
    // The case the op exists for, carried the whole way: outline-plus-flood-fill
    // would close this hole, and so would a rasteriser that lost the ring index
    // somewhere in the blob.
    let volume = rasterise().expect("rasterising the viewer's fragments");
    let plane = volume.index_axis(ndarray::Axis(0), 2);

    // The scene's first shape is a 40-wide square at (10, 10) with a 12-wide
    // hole at (24, 24); coordinates are (x, y) and the volume is (z, y, x).
    assert_ne!(plane[[12, 12]], 0, "inside the ring is filled");
    assert_ne!(plane[[45, 45]], 0, "and so is the far corner of it");
    assert_eq!(plane[[30, 30]], 0, "the hole is not");
    assert_eq!(plane[[5, 5]], 0, "and neither is outside the shape");
}

#[test]
fn a_stroke_covers_its_width_and_a_bare_outline_does_not() {
    let volume = rasterise().expect("rasterising the viewer's fragments");
    // The stroke sits on the default plane, z = 0.
    let plane = volume.index_axis(ndarray::Axis(0), 0);

    // The second shape is an open path from (60.5, 12.25) with half_width 3.5.
    assert_ne!(plane[[12, 60]], 0, "the stroke covers the path itself");
    assert_ne!(plane[[14, 61]], 0, "and the pixels within its half width");
    assert_eq!(plane[[25, 60]], 0, "and nothing far from it");
}

#[test]
fn a_shapes_z_extent_survives_the_blob() {
    // The polygon was drawn on plane 2 spanning three further planes. A
    // `z_extent` lost in the encoding would show as a shape one plane deep.
    let volume = rasterise().expect("rasterising the viewer's fragments");
    let covered: Vec<usize> = (0..VOLUME[0])
        .filter(|&z| volume.index_axis(ndarray::Axis(0), z)[[12, 12]] != 0)
        .collect();
    assert_eq!(covered, vec![2, 3, 4, 5], "plane 2 and the three it names");
}
