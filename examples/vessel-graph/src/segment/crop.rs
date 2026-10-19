// SPDX-License-Identifier: MIT
// Original work for this crate.

//! A crop of an OME-Zarr image, copied into a new one-level `c, z, y, x`
//! OME-Zarr image that keeps the source's physical frame: its translation is
//! moved to the crop's first voxel, so a point in the crop has the physical
//! position it had in the source, and a registration of the source applies
//! to the crop unchanged.
//!
//! The published labels must match the image's level 0, so `segment` and
//! `tract` run on a crop by copying it first (`--crop`, `--crop-out`) and then
//! running on the copy. The copy goes in z slabs; one slab of one channel is
//! resident at a time.

use std::fs;
use std::path::Path;
use std::sync::Arc;

use atlas_regions::Crop;
use blockflow::ome_zarr::OmeVolumePyramid;
use blockflow::voxels::Voxels;
use blockflow::{Error, Result};
use serde_json::json;
use zarrs::array::{data_type, ArrayBuilder, ArraySubset, FillValue};
use zarrs::filesystem::FilesystemStore;

use crate::graph::cli::{image_unit, level_lattice};

/// z planes copied at once.
const SLAB: usize = 32;

/// What [`write_crop`] wrote.
#[derive(Debug, Clone, PartialEq)]
pub struct CropWritten {
    /// The crop's first voxel in the source's level 0, `[z, y, x]`.
    pub start: [usize; 3],
    pub shape: [usize; 3],
    /// The crop's level-0 translation (the source's, moved to `start`).
    pub translation: [f64; 3],
}

/// Copy `crop` (`"z0:z1,y0:y1,x0:x1"`, Python slices, `:` for a whole axis)
/// of level 0 of `channels` at time `time` of the image at `source` into a new
/// image at `out`, its channels renumbered `0..` in the order given.
/// Replaces `out`.
pub fn write_crop(
    source: &Path,
    channels: &[usize],
    time: usize,
    crop: &str,
    out: &Path,
) -> Result<CropWritten> {
    if channels.is_empty() {
        return Err(Error::invalid("a crop needs at least one channel"));
    }
    let pyramids = channels
        .iter()
        .map(|&channel| OmeVolumePyramid::open(source, channel, time))
        .collect::<Result<Vec<_>>>()?;
    let level = &pyramids[0].levels()[0];
    let (start, shape) = Crop::parse_zyx(crop)?.resolve(level.shape)?;
    if shape.contains(&0) {
        return Err(Error::invalid(format!("the crop {crop:?} is empty")));
    }
    let lattice = level_lattice(level);
    let translation: [f64; 3] = std::array::from_fn(|axis| {
        lattice.translation[axis] + start[axis] as f64 * lattice.scale[axis]
    });
    let attached = pyramids
        .iter()
        .map(|pyramid| pyramid.attached_level(0))
        .collect::<Result<Vec<_>>>()?;
    let (dtype, _) = attached[0].metadata()?;
    for (image, channel) in attached.iter().zip(channels) {
        let (other, volume) = image.metadata()?;
        if other != dtype || volume != level.shape {
            return Err(Error::invalid(format!(
                "channel {channel} is {volume:?} {} and channel {} {:?} {}; a crop's \
                 channels share a lattice and a type",
                other.numpy_name(),
                channels[0],
                level.shape,
                dtype.numpy_name()
            )));
        }
    }

    if out.exists() {
        fs::remove_dir_all(out).map_err(Error::backend)?;
    }
    fs::create_dir_all(out.join("0")).map_err(Error::backend)?;
    let unit = image_unit(source);
    let space = |name: &str| match unit.as_str() {
        "voxel" => json!({"name": name, "type": "space"}),
        unit => json!({"name": name, "type": "space", "unit": unit}),
    };
    let attributes = json!({"ome": {"version": "0.5", "multiscales": [{
        "axes": [{"name": "c", "type": "channel"}, space("z"), space("y"), space("x")],
        "datasets": [{"path": "0", "coordinateTransformations": [
            {"type": "scale", "scale": [1.0, lattice.scale[0], lattice.scale[1], lattice.scale[2]]},
            {"type": "translation", "translation": [0.0, translation[0], translation[1], translation[2]]}
        ]}],
        "metadata": {"method": "vessel-graph crop", "source": source.display().to_string(),
                     "channels": channels, "start_zyx": start, "shape_zyx": shape},
    }]}});
    fs::write(
        out.join("zarr.json"),
        serde_json::to_vec_pretty(
            &json!({"zarr_format": 3, "node_type": "group", "attributes": attributes}),
        )
        .map_err(Error::backend)?,
    )
    .map_err(Error::backend)?;

    let store = Arc::new(FilesystemStore::new(out).map_err(Error::backend)?);
    let array_shape = vec![
        channels.len() as u64,
        shape[0] as u64,
        shape[1] as u64,
        shape[2] as u64,
    ];
    let chunk = vec![
        1,
        shape[0].min(SLAB) as u64,
        shape[1].min(256) as u64,
        shape[2].min(256) as u64,
    ];
    let (zarr_type, fill) = match dtype.numpy_name() {
        "uint8" => (data_type::uint8(), FillValue::from(0u8)),
        "uint16" => (data_type::uint16(), FillValue::from(0u16)),
        "uint32" => (data_type::uint32(), FillValue::from(0u32)),
        "float32" => (data_type::float32(), FillValue::from(0f32)),
        "float64" => (data_type::float64(), FillValue::from(0f64)),
        other => {
            return Err(Error::invalid(format!(
                "a crop copies uint8, uint16, uint32, float32 or float64 images; this one is \
                 {other}"
            )))
        }
    };
    let array = ArrayBuilder::new(array_shape, chunk, zarr_type, fill)
        .build(store, "/0")
        .map_err(Error::backend)?;
    array.store_metadata().map_err(Error::backend)?;
    for (index, image) in attached.iter().enumerate() {
        for z in (0..shape[0]).step_by(SLAB) {
            let depth = SLAB.min(shape[0] - z);
            let slab = image.read_region(
                [start[0] + z, start[1], start[2]],
                [depth, shape[1], shape[2]],
            )?;
            let subset = ArraySubset::new_with_start_shape(
                vec![index as u64, z as u64, 0, 0],
                vec![1, depth as u64, shape[1] as u64, shape[2] as u64],
            )
            .map_err(Error::backend)?;
            let stored = match &slab {
                Voxels::U8(v) => {
                    array.store_array_subset(&subset, v.iter().copied().collect::<Vec<_>>())
                }
                Voxels::U16(v) => {
                    array.store_array_subset(&subset, v.iter().copied().collect::<Vec<_>>())
                }
                Voxels::U32(v) => {
                    array.store_array_subset(&subset, v.iter().copied().collect::<Vec<_>>())
                }
                Voxels::F32(v) => {
                    array.store_array_subset(&subset, v.iter().copied().collect::<Vec<_>>())
                }
                Voxels::F64(v) => {
                    array.store_array_subset(&subset, v.iter().copied().collect::<Vec<_>>())
                }
                other => {
                    return Err(Error::invalid(format!(
                        "a crop read {} voxels, which it does not copy",
                        other.dtype().numpy_name()
                    )))
                }
            };
            stored.map_err(Error::backend)?;
        }
    }
    Ok(CropWritten {
        start,
        shape,
        translation,
    })
}
