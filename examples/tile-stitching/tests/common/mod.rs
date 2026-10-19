// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Synthetic tile sets: overlapping tiles cropped out of one volume at known
//! offsets, so a stitch can be checked against the truth.

#![allow(dead_code)]

use std::path::PathBuf;

use blockflow::voxels::Voxels;
use ndarray::Array3;
use tile_stitching::layout::{TileLayout, TileSource};

/// A deterministic hash, so the scene has no symmetry a correlation could
/// latch onto.
pub fn hash(z: isize, y: isize, x: isize) -> u64 {
    let mut h = (z as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
        .wrapping_add((x as u64).wrapping_mul(0x1656_67B1_9E37_79F9));
    h ^= h >> 31;
    h = h.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    h ^ (h >> 29)
}

/// The scene's value at a global voxel: a dim textured background and sparse
/// bright points, in the range a 16-bit light-sheet tile has.
pub fn scene(z: isize, y: isize, x: isize) -> u16 {
    let h = hash(z, y, x);
    let background = 300 + (h % 200) as u16;
    if h.is_multiple_of(61) {
        background + 2000 + (h % 500) as u16
    } else {
        background
    }
}

/// A tile cropped from the scene: plane `k` (along z) is taken at `origin +
/// (k, 0, drift(k))`.
pub fn crop(origin: [isize; 3], shape: [usize; 3], drift: impl Fn(usize) -> [isize; 2]) -> Voxels {
    Voxels::U16(Array3::from_shape_fn(
        (shape[0], shape[1], shape[2]),
        |(k, j, i)| {
            let [dy, dx] = drift(k);
            scene(
                origin[0] + k as isize,
                origin[1] + j as isize + dy,
                origin[2] + i as isize + dx,
            )
        },
    ))
}

pub const TILE: [usize; 3] = [24, 80, 70];
pub const OVERLAP_X: isize = 30;
pub const OVERLAP_Y: isize = 24;

/// Two tiles side by side in x: the second's true origin is the nominal one
/// moved by `offset`, and its planes drift in-plane by `drift`.
pub fn pair(offset: [isize; 3], drift: impl Fn(usize) -> [isize; 2]) -> (Vec<Voxels>, TileLayout) {
    let nominal_x = TILE[2] as isize - OVERLAP_X;
    let tiles = vec![
        crop([0, 0, 0], TILE, |_| [0, 0]),
        crop([offset[0], offset[1], nominal_x + offset[2]], TILE, drift),
    ];
    (tiles, grid_layout(&[[0, 0], [1, 0]]))
}

/// A layout of `TILE`-shaped tiles at the given `(x, y)` grid indices.
pub fn grid_layout(grid: &[[isize; 2]]) -> TileLayout {
    TileLayout::on_grid(
        grid.iter()
            .map(|g| TileSource {
                shape: TILE,
                tile_position: g.to_vec(),
            })
            .collect(),
        &[OVERLAP_X, OVERLAP_Y],
        &[2, 1],
    )
    .expect("a grid layout")
}

/// A scratch directory under the target directory, emptied.
pub fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

/// Where the real tiles live: `TILE_STITCHING_TILES`, or the moved OSF
/// `sa3x8` conversion.
pub fn real_tiles_root() -> PathBuf {
    std::env::var("TILE_STITCHING_TILES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/husky/otherdataset/teresa/clearmap_em/osf-sa3x8/tiles"))
}
