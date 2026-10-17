//! Small windowed reads for QC images. Everything the analysis itself reads
//! goes through the planner; these never load a whole volume.

use std::path::Path;
use std::sync::Arc;

use blockflow::{Error, Result};
use ndarray::{s, Array3};
use zarrs::array::{Array, ArraySubset};
use zarrs::filesystem::FilesystemStore;

fn open(root: &Path, path: &str) -> Result<Array<FilesystemStore>> {
    let store = Arc::new(FilesystemStore::new(root).map_err(Error::backend)?);
    Array::open(store, path).map_err(Error::backend)
}

/// `[z, y, x]` window of one channel of level 0 of a `t,c,z,y,x` image, as
/// `f32`, clipped to the volume; voxels outside are `fill`.
pub fn read_channel_window(
    image_root: &Path,
    channel: usize,
    start: [i64; 3],
    shape: [usize; 3],
    fill: f32,
) -> Result<Array3<f32>> {
    let a = open(image_root, "/0")?;
    let full = a.shape().to_vec();
    if full.len() != 5 {
        return Err(Error::invalid("expected a t,c,z,y,x image"));
    }
    read_clipped(shape, start, [full[2], full[3], full[4]], fill, |lo, n| {
        let subset = ArraySubset::new_with_start_shape(
            vec![0, channel as u64, lo[0], lo[1], lo[2]],
            vec![1, 1, n[0], n[1], n[2]],
        )
        .map_err(Error::backend)?;
        let v: Vec<u16> = a.retrieve_array_subset(&subset).map_err(Error::backend)?;
        Ok(v.into_iter().map(f32::from).collect())
    })
}

/// `[z, y, x]` window of level 0 of a published `uint64` label image
/// (`<image>/labels/<name>`), clipped; voxels outside are 0.
pub fn read_label_window(
    label_root: &Path,
    start: [i64; 3],
    shape: [usize; 3],
) -> Result<Array3<u64>> {
    let a = open(label_root, "/0")?;
    let full = a.shape().to_vec();
    read_clipped(shape, start, [full[0], full[1], full[2]], 0, |lo, n| {
        let subset =
            ArraySubset::new_with_start_shape(lo.to_vec(), n.to_vec()).map_err(Error::backend)?;
        a.retrieve_array_subset(&subset).map_err(Error::backend)
    })
}

fn read_clipped<T: Copy>(
    shape: [usize; 3],
    start: [i64; 3],
    volume: [u64; 3],
    fill: T,
    read: impl Fn([u64; 3], [u64; 3]) -> Result<Vec<T>>,
) -> Result<Array3<T>> {
    let mut out = Array3::from_elem((shape[0], shape[1], shape[2]), fill);
    let lo: [i64; 3] = std::array::from_fn(|a| start[a].max(0));
    let hi: [i64; 3] = std::array::from_fn(|a| (start[a] + shape[a] as i64).min(volume[a] as i64));
    if (0..3).any(|a| hi[a] <= lo[a]) {
        return Ok(out);
    }
    let n: [u64; 3] = std::array::from_fn(|a| (hi[a] - lo[a]) as u64);
    let values = read(lo.map(|v| v as u64), n)?;
    let block = Array3::from_shape_vec((n[0] as usize, n[1] as usize, n[2] as usize), values)
        .map_err(Error::backend)?;
    let o: [usize; 3] = std::array::from_fn(|a| (lo[a] - start[a]) as usize);
    out.slice_mut(s![
        o[0]..o[0] + n[0] as usize,
        o[1]..o[1] + n[1] as usize,
        o[2]..o[2] + n[2] as usize
    ])
    .assign(&block);
    Ok(out)
}
