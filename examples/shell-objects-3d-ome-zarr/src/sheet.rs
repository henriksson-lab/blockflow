//! Review contact sheets: three orthogonal slices through each object's fitted
//! centre with its boundary drawn, green when accepted and red when not.

use std::path::Path;

use blockflow::ome_zarr::OmeVolumePyramid;
use blockflow::sphere::RaySet;
use blockflow::{Error, Result};
use image::{Rgb, RgbImage};
use rayon::prelude::*;

pub struct Tile<'a> {
    pub center: [f64; 3],
    pub radii: &'a [f32],
    pub accepted: bool,
}

const FONT: [[u8; 5]; 10] = [
    [7, 5, 5, 5, 7],
    [2, 6, 2, 2, 7],
    [7, 1, 7, 4, 7],
    [7, 1, 7, 1, 7],
    [5, 5, 7, 1, 1],
    [7, 4, 7, 1, 7],
    [7, 4, 7, 5, 7],
    [7, 1, 1, 1, 1],
    [7, 5, 7, 5, 7],
    [7, 5, 7, 1, 7],
];

fn number(img: &mut RgbImage, x0: u32, y0: u32, n: usize) {
    for (k, ch) in n.to_string().chars().enumerate() {
        let d = ch.to_digit(10).expect("a digit") as usize;
        for (row, bits) in FONT[d].iter().enumerate() {
            for col in 0..3u32 {
                if bits & (4 >> col) != 0 {
                    for s in 0..4 {
                        let (x, y) = (x0 + (k as u32 * 4 + col) * 2 + s % 2, y0 + row as u32 * 2 + s / 2);
                        if x < img.width() && y < img.height() {
                            img.put_pixel(x, y, Rgb([255, 255, 0]));
                        }
                    }
                }
            }
        }
    }
}

/// Tiles of `2 * half + 1` voxels, four objects per row, numbered by position.
/// Objects are read and drawn in parallel.
pub fn contact_sheet(pyramid: &OmeVolumePyramid, path: &Path, tiles: &[Tile<'_>], rays: &RaySet, half: usize) -> Result<()> {
    let t = 2 * half + 1;
    let per_row = 4;
    let tile_w = 3 * t + 6;
    let rows = tiles.len().div_ceil(per_row).max(1);
    let rendered = tiles
        .par_iter()
        .map(|tile| render(pyramid, tile, rays, half))
        .collect::<Result<Vec<_>>>()?;
    let mut img = RgbImage::new((per_row * tile_w) as u32, (rows * (t + 4)) as u32);
    for (n, piece) in rendered.iter().enumerate() {
        let bx = ((n % per_row) * tile_w) as u32;
        let by = ((n / per_row) * (t + 4)) as u32;
        image::imageops::replace(&mut img, piece, bx as i64, by as i64);
        number(&mut img, bx + 2, by + 2, n);
    }
    img.save(path).map_err(Error::backend)
}

/// One object's three slices, side by side.
fn render(pyramid: &OmeVolumePyramid, tile: &Tile<'_>, rays: &RaySet, half: usize) -> Result<RgbImage> {
    let volume = pyramid.levels()[0].shape;
    let t = 2 * half + 1;
    let start: [usize; 3] = std::array::from_fn(|a| {
        (tile.center[a].round() as isize - half as isize).clamp(0, volume[a] as isize - 1) as usize
    });
    let shape: [usize; 3] = std::array::from_fn(|a| t.min(volume[a] - start[a]));
    let crop = pyramid.attached_level(0)?.read_region(start, shape)?.widened();
    let mut sorted: Vec<f64> = crop.iter().copied().collect();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let lo = sorted[sorted.len() / 100];
    let hi = sorted[sorted.len() * 995 / 1000].max(lo + 1.0);
    let c: [isize; 3] = std::array::from_fn(|a| tile.center[a].round() as isize);
    let inside = |p: [isize; 3]| {
        let d = [p[0] as f64 - tile.center[0], p[1] as f64 - tile.center[1], p[2] as f64 - tile.center[2]];
        ((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() as f32) <= rays.interpolate(tile.radii, d)
    };
    let colour = if tile.accepted { Rgb([60, 255, 60]) } else { Rgb([255, 60, 60]) };
    let mut img = RgbImage::new((3 * t + 4) as u32, t as u32);
    for plane in 0..3 {
        let at = |i: isize, j: isize| match plane {
            0 => [c[0], c[1] + i, c[2] + j],
            1 => [c[0] + i, c[1], c[2] + j],
            _ => [c[0] + i, c[1] + j, c[2]],
        };
        for i in -(half as isize)..=half as isize {
            for j in -(half as isize)..=half as isize {
                let p = at(i, j);
                let local: Option<[usize; 3]> = (0..3)
                    .all(|a| p[a] >= start[a] as isize && p[a] < (start[a] + shape[a]) as isize)
                    .then(|| std::array::from_fn(|a| p[a] as usize - start[a]));
                let v = local.map_or(lo, |q| crop[q]);
                let g = (((v - lo) / (hi - lo)).clamp(0.0, 1.0) * 255.0) as u8;
                let edge = inside(p)
                    && !(inside(at(i + 1, j)) && inside(at(i - 1, j)) && inside(at(i, j + 1)) && inside(at(i, j - 1)));
                let x = (plane * (t + 2)) as u32 + (j + half as isize) as u32;
                let y = (i + half as isize) as u32;
                img.put_pixel(x, y, if edge { colour } else { Rgb([g, g, g]) });
            }
        }
    }
    Ok(img)
}
