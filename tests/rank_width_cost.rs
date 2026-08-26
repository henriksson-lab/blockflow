//! **What the element's width costs the rank filter.**
//!
//! A consumer's cell chain plans in `Dtype::F64` where the reference
//! implementation it reproduces works in `uint16` — four times the bytes per
//! tap. Whether that is
//! worth anything is a measurement rather than an argument, and it is cheaper to
//! take here, on the kernel alone, than to find out by converting a stage's
//! whole plan.
//!
//! Interleaved and best-of, because the absolutes on this machine are not stable
//! and the ratio is what the question needs.
use std::time::Instant;

use blockflow::ops::element::{ElementShape, Rank, StructuringElement};
use blockflow::ops::rank::{rank_filter_f64_into, rank_filter_into};
use ndarray::Array3;

const SHAPE: [usize; 3] = [64, 128, 128];

#[test]
#[ignore = "a measurement; run with --release --ignored --nocapture"]
fn what_the_element_width_costs() {
    let voxels = SHAPE.iter().product::<usize>();
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 48) as u16
    };
    let source: Array3<u16> = Array3::from_shape_fn((SHAPE[0], SHAPE[1], SHAPE[2]), |_| next());
    let wide: Array3<f64> = source.mapv(f64::from);
    // The background element that motivates this: a `10 x 10` disk, in plane.
    let disk = StructuringElement::from_radius(ElementShape::Ellipsoid, [0, 5, 5]);

    let mut narrow_out: Array3<u16> = Array3::zeros((SHAPE[0], SHAPE[1], SHAPE[2]));
    let mut wide_out: Array3<f64> = Array3::zeros((SHAPE[0], SHAPE[1], SHAPE[2]));

    let (mut narrow, mut widest) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..5 {
        let start = Instant::now();
        rank_filter_into(source.view(), &disk, Rank::lowest(), narrow_out.view_mut())
            .expect("u16 erosion");
        narrow = narrow.min(start.elapsed().as_secs_f64());

        let start = Instant::now();
        rank_filter_f64_into(wide.view(), &disk, Rank::lowest(), wide_out.view_mut())
            .expect("f64 erosion");
        widest = widest.min(start.elapsed().as_secs_f64());
    }

    eprintln!(
        "\nerosion by a 10x10 disk over {voxels} voxels, best of 5, interleaved\n  \
         u16 {narrow:.4} s ({:.1} ns/voxel)\n  f64 {widest:.4} s ({:.1} ns/voxel)\n  \
         f64 / u16 = {:.2}x",
        narrow * 1e9 / voxels as f64,
        widest * 1e9 / voxels as f64,
        widest / narrow
    );

    // The answers must agree where the widening is exact, or the comparison is
    // between two different computations.
    let differing = narrow_out
        .iter()
        .zip(wide_out.iter())
        .filter(|(a, b)| f64::from(**a) != **b)
        .count();
    assert_eq!(
        differing, 0,
        "the two widths disagree at {differing} voxels; a `u16` widened to `f64` is exact, so \
         this is a bug in one of the two paths rather than a rounding difference"
    );
}
