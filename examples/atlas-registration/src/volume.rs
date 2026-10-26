// SPDX-License-Identifier: MIT
// Original work for this crate.

//! OME-Zarr in and out: voxel frames from multiscale metadata, a planned
//! resample onto a registration lattice, a resident warp through a point
//! transform, and an OME-Zarr image writer.

use std::path::Path;

use blockflow::assemble::PlanBuilder;
use blockflow::error::{Error, Result};
use blockflow::geometry::BlockGrid;
use blockflow::ome_zarr::{OmeVolumeLevel, OmeVolumePyramid};
use blockflow::ops::{
    append_warp_phase, remap_into, TransformBoundary, TransformInterpolation, WarpOp,
};
use blockflow::strategy::{execute_phases, Hints};
use blockflow::voxels::Voxels;
use blockflow::zarr_env::{PhysicalFrame, PyramidSpec, ZarrEnvironment};
use ndarray::{s, Array3, Axis};
use rayon::prelude::*;
use serde_json::Value;

use crate::rescale::{Frame, Rescale};

/// One channel and time point of an OME-Zarr image, with a frame per level.
pub struct Volume {
    pyramid: OmeVolumePyramid,
    frames: Vec<Frame>,
}

impl Volume {
    pub fn open(root: &Path, channel: usize) -> Result<Self> {
        let pyramid = OmeVolumePyramid::open(root, channel, 0)?;
        let frames = pyramid
            .levels()
            .iter()
            .map(level_frame)
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { pyramid, frames })
    }

    pub fn levels(&self) -> usize {
        self.frames.len()
    }

    pub fn shape(&self, level: usize) -> [usize; 3] {
        self.pyramid.levels()[level].shape
    }

    pub fn frame(&self, level: usize) -> Frame {
        self.frames[level]
    }

    /// The coarsest level whose voxels are no larger than `spacing` on every
    /// axis, so that sampling it onto a `spacing` lattice never skips more than
    /// one voxel. Level 0 when even that is coarser.
    pub fn level_for(&self, spacing: [f64; 3]) -> usize {
        let fits = |frame: &Frame| (0..3).all(|a| frame.scale[a] <= spacing[a] * (1.0 + 1e-9));
        (0..self.frames.len())
            .rev()
            .find(|level| fits(&self.frames[*level]))
            .unwrap_or(0)
    }

    /// The bytes [`Self::read`] allocates for `level`, as stored.
    pub fn stored_bytes(&self, level: usize) -> Result<u64> {
        let (dtype, shape) = self.pyramid.attached_level(level)?.metadata()?;
        Ok(crate::memory::voxels(shape) * dtype.size_of() as u64)
    }

    pub fn read(&self, level: usize) -> Result<Voxels> {
        self.pyramid.attached_level(level)?.read()
    }

    /// The spatial unit the image declares, if any.
    pub fn unit(&self) -> Option<String> {
        ome_unit(self.pyramid.image_root())
    }

    /// Level `level` sampled onto `target`, planned over the stored image.
    ///
    /// The map from a target voxel to a source voxel is the diagonal affine
    /// between the two frames, so this is `ops::transform`'s warp: it handles a
    /// non-rational factor, and a target that extends past the source reads
    /// zero there rather than stretching the source to fit.
    pub fn sample_onto(
        &self,
        level: usize,
        target: &Frame,
        target_shape: [usize; 3],
        interpolation: TransformInterpolation,
        work: &Path,
        block: usize,
    ) -> Result<Voxels> {
        let image = self.pyramid.attached_level(level)?;
        let (dtype, source_shape) = image.metadata()?;
        let to_source = Rescale::frames(target, &self.frames[level]);
        let (factor, offset) = (to_source.factor(), to_source.offset());
        let matrix = std::array::from_fn(|row| {
            std::array::from_fn(|col| if row == col { factor[row] } else { 0.0 })
        });
        let edge = block.max(1);
        let mut builder = PlanBuilder::new(
            source_shape,
            dtype,
            BlockGrid::new(source_shape, [edge; 3])?,
        );
        append_warp_phase(
            &mut builder,
            WarpOp::affine(
                "atlas-registration-sample",
                matrix,
                offset,
                target_shape,
                interpolation,
                TransformBoundary::Constant(0.0),
            )?,
            source_shape,
            BlockGrid::new(target_shape, [edge; 3])?,
        )?;
        let assembly = builder.finish()?;
        if work.exists() {
            std::fs::remove_dir_all(work).map_err(Error::backend)?;
        }
        let env = ZarrEnvironment::attach(work, &[image])?;
        execute_phases(
            "atlas-registration-sample",
            &assembly.workflow,
            &assembly.decomposition,
            &Hints {
                concurrency: rayon::current_num_threads(),
                ..Hints::default()
            },
            &env,
            &[],
            &assembly.work(),
        )?;
        let out = env.output()?;
        drop(env);
        std::fs::remove_dir_all(work).map_err(Error::backend)?;
        Ok(out)
    }
}

/// A level's frame from its `scale` and `translation` transformations.
pub fn level_frame(level: &OmeVolumeLevel) -> Result<Frame> {
    let mut scale = [1.0; 3];
    let mut translation = [0.0; 3];
    let rows = level
        .coordinate_transformations
        .as_array()
        .cloned()
        .unwrap_or_default();
    for row in rows {
        let (key, into) = match row.get("type").and_then(Value::as_str) {
            Some("scale") => ("scale", &mut scale),
            Some("translation") => ("translation", &mut translation),
            _ => continue,
        };
        let values = row.get(key).and_then(Value::as_array).ok_or_else(|| {
            Error::invalid(format!(
                "OME level {} has a {key} without values",
                level.path
            ))
        })?;
        for axis in 0..3 {
            into[axis] = values.get(axis).and_then(Value::as_f64).ok_or_else(|| {
                Error::invalid(format!("OME level {} has a short {key}", level.path))
            })?;
        }
    }
    Frame::new(scale, translation)
}

/// A lattice of `spacing` voxels over the same physical box as `shape` voxels
/// of `frame`.
pub fn lattice_over(
    frame: &Frame,
    shape: [usize; 3],
    spacing: [f64; 3],
) -> Result<(Frame, [usize; 3])> {
    let lattice = frame.relattice(spacing)?;
    let shape = std::array::from_fn(|a| {
        ((shape[a] as f64 * frame.scale[a] / spacing[a]).round() as usize).max(1)
    });
    Ok((lattice, shape))
}

/// `source` (on `source_frame`) resampled onto `target`, through a point map
/// from target physical coordinates to source physical coordinates.
///
/// Resident: the registration lattice is small by construction. Planes are
/// computed in parallel straight into the output, a row at a time with
/// `ops::transform::remap_into`, so the only memory beside the source and the
/// output is [`warp_scratch_bytes`].
pub fn warp(
    source: &Array3<f64>,
    source_frame: &Frame,
    target: &Frame,
    target_shape: [usize; 3],
    map: &(dyn Fn([f64; 3]) -> [f64; 3] + Sync),
    interpolation: TransformInterpolation,
) -> Result<Array3<f64>> {
    let [nz, ny, nx] = target_shape;
    let mut out = Array3::<f64>::zeros((nz, ny, nx));
    let planes: Vec<_> = out.axis_iter_mut(Axis(0)).enumerate().collect();
    planes
        .into_par_iter()
        .try_for_each(|(z, mut plane)| -> Result<()> {
            let mut coordinates = [
                Array3::<f64>::zeros((1, 1, nx)),
                Array3::<f64>::zeros((1, 1, nx)),
                Array3::<f64>::zeros((1, 1, nx)),
            ];
            for y in 0..ny {
                for x in 0..nx {
                    let physical = target.to_physical([z as f64, y as f64, x as f64]);
                    let at = source_frame.to_index(map(physical));
                    for axis in 0..3 {
                        coordinates[axis][[0, 0, x]] = at[axis];
                    }
                }
                let [cz, cy, cx] = &coordinates;
                let row = plane.slice_mut(s![y..y + 1, ..]).insert_axis(Axis(0));
                remap_into(
                    source.view(),
                    [cz.view(), cy.view(), cx.view()],
                    interpolation,
                    TransformBoundary::Constant(0.0),
                    row,
                )?;
            }
            Ok(())
        })?;
    Ok(out)
}

/// What [`warp`] holds beside its source and its output: three rows of
/// coordinates per worker.
pub fn warp_scratch_bytes(target_shape: [usize; 3]) -> u64 {
    (rayon::current_num_threads() * 3 * target_shape[2] * 8) as u64
}

pub fn write_image(root: &Path, image: &Voxels, frame: &Frame, unit: Option<&str>) -> Result<()> {
    let shape = image.shape();
    let mut levels = 1;
    while levels < 6 && shape.iter().all(|&n| n >> levels >= 16) {
        levels += 1;
    }
    if root.exists() {
        std::fs::remove_dir_all(root).map_err(Error::backend)?;
    }
    let physical = PhysicalFrame::new(frame.scale, unit).with_origin(frame.translation);
    let spec = PyramidSpec::powers_of_two(levels)?.with_physical(physical);
    ZarrEnvironment::build_multiscale(root, image, &spec)?;
    Ok(())
}

/// The spatial unit (of the `z` axis) the OME-Zarr image at `root` declares,
/// if it can be read and declares one.
pub fn ome_unit(root: &Path) -> Option<String> {
    let text = std::fs::read(root.join("zarr.json")).ok()?;
    let json: Value = serde_json::from_slice(&text).ok()?;
    let multiscale = json
        .pointer("/attributes/ome/multiscales/0")
        .or_else(|| json.pointer("/attributes/multiscales/0"))?;
    multiscale
        .get("axes")?
        .as_array()?
        .iter()
        .find(|axis| axis.get("name").and_then(Value::as_str) == Some("z"))?
        .get("unit")?
        .as_str()
        .map(str::to_string)
}
