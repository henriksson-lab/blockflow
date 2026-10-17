//! QC renders: XZ sections with surfaces and beads, and height maps.

use std::path::Path;

use blockflow::{Error, Result};
use image::{Rgb, RgbImage};
use ndarray::{Array2, Array3, Axis};

use crate::measure::{Class, HeightMaps, Measured};

fn percentile(a: &Array3<f32>, q: f64) -> f32 {
    let mut v: Vec<f32> = a.iter().step_by(11).copied().collect();
    let k = ((v.len() as f64 * q) as usize).min(v.len() - 1);
    let (_, kth, _) = v.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).unwrap());
    *kth
}

/// `channels` = [red, green, blue] as `(nz, ys.len(), nx)` stacks of the XZ
/// planes at `ys` (read as windows, never whole volumes); `objects` likewise.
/// One panel per y, z stretched by `zstretch` rows per slice.
pub fn xz_sections(
    channels: [&Array3<f32>; 3],
    h: &HeightMaps,
    beads: &[Measured],
    objects: Option<&Array3<u64>>,
    ys: &[usize],
    zstretch: usize,
    path: &Path,
) -> Result<()> {
    let (nz, _, nx) = channels[0].dim();
    let ranges: Vec<(f32, f32)> = channels
        .iter()
        .enumerate()
        .map(|(i, c)| {
            (
                percentile(c, 0.5),
                percentile(c, if i == 2 { 0.9995 } else { 0.995 }),
            )
        })
        .collect();
    let ph = nz * zstretch;
    let gap = 6;
    let mut img = RgbImage::new(nx as u32, (ys.len() * (ph + gap)) as u32);
    for (p, &y) in ys.iter().enumerate() {
        let oy = p * (ph + gap);
        for z in 0..nz {
            for x in 0..nx {
                let mut px = [0u8; 3];
                for (k, c) in channels.iter().enumerate() {
                    let (lo, hi) = ranges[k];
                    let v = ((c[[z, p, x]] - lo) / (hi - lo).max(1e-6)).clamp(0.0, 1.0);
                    px[k] = (v * 255.0) as u8;
                }
                if let Some(o) = objects {
                    if o[[z, p, x]] > 0 {
                        // outline-ish tint: keep red, boost towards orange
                        px = [px[0].max(200), px[1].max(110), px[2] / 2];
                    }
                }
                for r in 0..zstretch {
                    img.put_pixel(x as u32, (oy + z * zstretch + r) as u32, Rgb(px));
                }
            }
        }
        for x in 0..nx {
            for (surf, col) in [(&h.mucus, [255, 255, 0]), (&h.epi, [255, 255, 255])] {
                let zz = surf[[y, x]] as usize;
                if zz < nz {
                    img.put_pixel(x as u32, (oy + zz * zstretch) as u32, Rgb(col));
                }
            }
        }
        for m in beads {
            if (m.c.y - y as f64).abs() > 3.0 {
                continue;
            }
            let col = match m.class {
                _ if m.kept() => [0, 255, 255],
                Class::Bead => [128, 128, 128], // manually rejected
                Class::InTissue => [255, 0, 255],
                Class::Dim | Class::Spike => [128, 128, 128],
                Class::InLargeObject => [255, 128, 0],
            };
            let cx = m.c.x.round() as i64;
            let cz = (oy as f64 + (m.c.z + 0.5) * zstretch as f64) as i64;
            for d in -4i64..=4 {
                for (xx, zz) in [(cx + d, cz), (cx, cz + d)] {
                    if d.abs() >= 2
                        && xx >= 0
                        && (xx as usize) < nx
                        && zz >= oy as i64
                        && zz < (oy + ph) as i64
                    {
                        img.put_pixel(xx as u32, zz as u32, Rgb(col));
                    }
                }
            }
        }
    }
    img.save(path).map_err(Error::backend)
}

pub fn height_map(h: &Array2<f32>, nz: usize, path: &Path) -> Result<()> {
    let (ny, nx) = h.dim();
    let mut img = RgbImage::new(nx as u32, ny as u32);
    for ((y, x), &v) in h.indexed_iter() {
        let t = (v / nz as f32).clamp(0.0, 1.0);
        // simple dark-blue -> yellow ramp
        let px = [
            (255.0 * t) as u8,
            (64.0 + 160.0 * t) as u8,
            (160.0 * (1.0 - t)) as u8,
        ];
        img.put_pixel(x as u32, y as u32, Rgb(px));
    }
    img.save(path).map_err(Error::backend)
}

/// Max projection of a volume along z, for quick looks.
#[allow(dead_code)]
pub fn max_z(a: &Array3<f32>) -> Array2<f32> {
    a.fold_axis(Axis(0), f32::MIN, |m, &v| m.max(v))
}

/// Half extent (z, y, x) of the crop [`gallery`] shows per candidate.
pub const CROP_HALF: [usize; 3] = [8, 12, 12];

/// Tile of candidate crops: per candidate an XY patch (max over z±1) and an XZ
/// patch, mapped linearly from `range`. `crops[i]` is centred on `rows[i]`
/// with half extent [`CROP_HALF`], padded outside the volume.
pub fn gallery(
    rows: &[&Measured],
    crops: &[Array3<f32>],
    range: (f32, f32),
    path: &Path,
) -> Result<()> {
    const H: i64 = CROP_HALF[1] as i64; // half width in xy
    const ZH: i64 = CROP_HALF[0] as i64; // half height in z
    const ZS: i64 = 2; // z stretch
    let tile_w = (2 * H + 1) as u32 * 2 + 2;
    let tile_h = ((2 * ZH + 1) * ZS).max(2 * H + 1) as u32 + 2;
    let cols = 8u32;
    let nrows = (rows.len() as u32).div_ceil(cols).max(1);
    let mut img = RgbImage::from_pixel(cols * tile_w, nrows * tile_h, Rgb([40, 0, 0]));

    let g = |v: f32| {
        let t = ((v - range.0) / (range.1 - range.0)).clamp(0.0, 1.0);
        let b = (t.sqrt() * 255.0) as u8;
        Rgb([b, b, b])
    };
    for (n, crop) in crops.iter().enumerate().take(rows.len()) {
        let (ox, oy) = ((n as u32 % cols) * tile_w, (n as u32 / cols) * tile_h);
        let val =
            |z: i64, y: i64, x: i64| crop[[(z + ZH) as usize, (y + H) as usize, (x + H) as usize]];
        let (cz, cy, cx) = (0i64, 0i64, 0i64);
        for dy in -H..=H {
            for dx in -H..=H {
                let v = (-1..=1)
                    .map(|dz| val(cz + dz, cy + dy, cx + dx))
                    .fold(f32::MIN, f32::max);
                img.put_pixel(ox + (dx + H) as u32, oy + (dy + H) as u32, g(v));
            }
        }
        let ox2 = ox + (2 * H + 1) as u32 + 1;
        for dz in -ZH..=ZH {
            for dx in -H..=H {
                let v = val(cz + dz, cy, cx + dx);
                for r in 0..ZS {
                    img.put_pixel(
                        ox2 + (dx + H) as u32,
                        oy + ((dz + ZH) * ZS + r) as u32,
                        g(v),
                    );
                }
            }
        }
    }
    let (w, h) = img.dimensions();
    image::imageops::resize(&img, w * 3, h * 3, image::imageops::FilterType::Nearest)
        .save(path)
        .map_err(Error::backend)
}
