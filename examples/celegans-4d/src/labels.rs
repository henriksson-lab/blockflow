// SPDX-License-Identifier: MIT

//! Nuclei as label spheres: `labels/<layer>`, one `uint64` plane per frame
//! (time chunk one), every level of the source image rasterised directly —
//! a sphere is cheap to evaluate at any resolution, so no pyramid
//! downsampling is needed and coarse levels stay sphere-shaped.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use blockflow::time_series::{LabelStack, TimeSeriesImage};
use blockflow::track::DetectionSet;
use blockflow::{Error, Result};
use ndarray::Array3;

use crate::import::grid;

/// Write the label stack at `root` (which must not exist). Label value =
/// detection id; a voxel claimed by two spheres goes to the one whose centre
/// is nearer relative to its radius. `radius` is in physical units, by
/// detection id. Frames without detections stay unwritten (background).
pub fn rasterize(
    root: &Path,
    image_path: &Path,
    image: &TimeSeriesImage,
    detections: &DetectionSet,
    radius: &BTreeMap<u64, f64>,
    spatial_chunk: usize,
    jobs: usize,
) -> Result<()> {
    let stack = LabelStack::create(root, image, [spatial_chunk; 2], "../../")?;
    let level0 = grid(image_path, image, 0)?;
    let levels = (0..stack.n_levels())
        .map(|k| grid(image_path, image, k))
        .collect::<Result<Vec<_>>>()?;
    let frames: Vec<u64> = detections
        .frames()
        .iter()
        .map(|f| f.t_index)
        .filter(|&t| !detections.in_frame(t).is_empty())
        .collect();
    let next = AtomicUsize::new(0);
    let failure: std::sync::Mutex<Option<Error>> = std::sync::Mutex::new(None);
    std::thread::scope(|scope| {
        for _ in 0..jobs.max(1) {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(&t) = frames.get(i) else { break };
                let outcome = (|| -> Result<()> {
                    for (k, level) in levels.iter().enumerate() {
                        let [nz, ny, nx] = stack.frame_shape(k)?;
                        let mut plane = Array3::<u64>::zeros((nz, ny, nx));
                        let mut best = Array3::<f32>::from_elem((nz, ny, nx), f32::INFINITY);
                        for d in detections.in_frame(t) {
                            let r = radius[&d.id];
                            if r <= 0.0 {
                                continue;
                            }
                            // Level-0 voxel -> physical -> level-k voxel.
                            let centre: [f64; 3] = std::array::from_fn(|a| {
                                (d.native[a] * level0.scale[a] + level0.translation[a]
                                    - level.translation[a])
                                    / level.scale[a]
                            });
                            let extent = [nz, ny, nx];
                            let range: [(usize, usize); 3] = std::array::from_fn(|a| {
                                let reach = r / level.scale[a];
                                let lo = (centre[a] - reach).ceil().max(0.0);
                                let hi = (centre[a] + reach).floor().min(extent[a] as f64 - 1.0);
                                if hi < lo {
                                    (1, 0)
                                } else {
                                    (lo as usize, hi as usize + 1)
                                }
                            });
                            for z in range[0].0..range[0].1 {
                                let dz = (z as f64 - centre[0]) * level.scale[0];
                                for y in range[1].0..range[1].1 {
                                    let dy = (y as f64 - centre[1]) * level.scale[1];
                                    for x in range[2].0..range[2].1 {
                                        let dx = (x as f64 - centre[2]) * level.scale[2];
                                        let q = ((dz * dz + dy * dy + dx * dx) / (r * r)) as f32;
                                        if q <= 1.0 && q < best[[z, y, x]] {
                                            best[[z, y, x]] = q;
                                            plane[[z, y, x]] = d.id;
                                        }
                                    }
                                }
                            }
                        }
                        stack.write_plane(k, t as usize, &plane)?;
                    }
                    Ok(())
                })();
                if let Err(e) = outcome {
                    failure
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .get_or_insert(e);
                    break;
                }
            });
        }
    });
    match failure.into_inner().unwrap_or_else(|p| p.into_inner()) {
        Some(e) => Err(e),
        None => Ok(()),
    }
}
