// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Points from a sample into an atlas, through one registration or a chain.
//!
//! **Conventions.**
//! - Points are `[z, y, x]` in **level-0 voxel indices** (voxel centres at
//!   integers) of the image they belong to.
//! - [`PointTransform::apply`] takes points of the **first fixed image** (the
//!   sample: its level 0, whatever level the registration ran on) to
//!   continuous voxel indices of the **last moving image** (the atlas, level
//!   0). This is the direction an elastix transform maps, and the direction
//!   the reference carries cell coordinates in (`transformix -def`).
//! - Between steps of a chain the point is handed on in level-0 voxels of the
//!   shared image: step `k`'s moving image is step `k + 1`'s fixed image, and
//!   both must record the same level-0 shape and frame.
//! - Outside a B-spline's grid the deformation is the identity, as in
//!   elastix; no point is dropped and no point becomes `NaN`.
//! - [`PointTransform::apply_inverse`] goes the other way (atlas voxels to
//!   sample voxels) by Newton's method on the forward map; a point whose
//!   iteration does not converge to `1e-3` voxels is `None`.

use std::fs;
use std::path::{Path, PathBuf};

use blockflow::error::{Error, Result};
use rayon::prelude::*;
use serde_json::Value;

use crate::pipeline::{registration_transform, Units, CHAIN_JSON, REGISTRATION_JSON};
use crate::rescale::Frame;
use crate::volume::ome_unit;
use crate::AtlasTransform;

/// One image of a registration, level 0, as its sidecar records it.
#[derive(Debug, Clone, PartialEq)]
pub struct RegisteredImage {
    /// The image the registration read, as written in the sidecar.
    pub path: Option<PathBuf>,
    /// Level-0 shape, `[z, y, x]`.
    pub shape: [usize; 3],
    /// Level-0 physical frame: `physical = translation + index * scale`.
    pub frame: Frame,
    /// The spatial unit of `frame`: the sidecar's `unit`, else the one the
    /// image's OME-Zarr declares when it can be read, else `None`.
    pub unit: Option<String>,
}

/// One registration's point map in level-0 voxels, with the images' frames.
#[derive(Debug, Clone)]
struct Step {
    map: AtlasTransform,
    fixed: RegisteredImage,
    moving: RegisteredImage,
}

/// Sample level-0 voxels to atlas level-0 voxels, through one registration
/// directory or a chain of them.
#[derive(Debug, Clone)]
pub struct PointTransform {
    steps: Vec<Step>,
}

impl PointTransform {
    /// From a directory `atlas-registration register` wrote
    /// (`registration.json`) or one `atlas-registration chain` wrote
    /// (`chain.json`, whose steps are applied in order).
    pub fn from_registration_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let chain = dir.join(CHAIN_JSON);
        let directories = if chain.exists() {
            let text = fs::read_to_string(&chain).map_err(Error::backend)?;
            let json: Value = serde_json::from_str(&text).map_err(Error::backend)?;
            let steps = json
                .get("steps")
                .and_then(Value::as_array)
                .ok_or_else(|| Error::invalid(format!("{}: no `steps`", chain.display())))?;
            steps
                .iter()
                .map(|step| {
                    step.as_str().map(|name| dir.join(name)).ok_or_else(|| {
                        Error::invalid(format!("{}: a step is not a name", chain.display()))
                    })
                })
                .collect::<Result<Vec<_>>>()?
        } else if dir.join(REGISTRATION_JSON).exists() {
            vec![dir.to_path_buf()]
        } else {
            return Err(Error::invalid(format!(
                "{}: neither {REGISTRATION_JSON} nor {CHAIN_JSON}; not a registration directory",
                dir.display()
            )));
        };
        let steps = directories
            .iter()
            .map(|directory| step(directory))
            .collect::<Result<Vec<_>>>()?;
        for (index, pair) in steps.windows(2).enumerate() {
            let (moving, fixed) = (&pair[0].moving, &pair[1].fixed);
            let same_frame = (0..3).all(|a| {
                (moving.frame.scale[a] - fixed.frame.scale[a]).abs() < 1e-9
                    && (moving.frame.translation[a] - fixed.frame.translation[a]).abs() < 1e-9
            });
            if moving.shape != fixed.shape || !same_frame {
                return Err(Error::invalid(format!(
                    "{}: step {index}'s moving image (shape {:?}, frame {:?}) is not step {}'s \
                     fixed image (shape {:?}, frame {:?}); a chain hands points on in level-0 \
                     voxels of one shared image",
                    dir.display(),
                    moving.shape,
                    (moving.frame.scale, moving.frame.translation),
                    index + 1,
                    fixed.shape,
                    (fixed.frame.scale, fixed.frame.translation)
                )));
            }
        }
        Ok(Self { steps })
    }

    /// The number of registrations.
    pub fn steps(&self) -> usize {
        self.steps.len()
    }

    /// Level-0 shape of the sample (the first fixed image), `[z, y, x]`.
    pub fn sample_shape(&self) -> [usize; 3] {
        self.sample_frame().shape
    }

    /// Level-0 shape of the atlas (the last moving image), `[z, y, x]`.
    pub fn atlas_shape(&self) -> [usize; 3] {
        self.atlas_frame().shape
    }

    /// The sample, the first fixed image: its path, level-0 shape, physical
    /// frame and unit. [`Self::apply`] takes this frame's voxel indices.
    pub fn sample_frame(&self) -> &RegisteredImage {
        &self.steps[0].fixed
    }

    /// The atlas, the last moving image: its path, level-0 shape, physical
    /// frame and unit. [`Self::apply`] gives this frame's voxel indices.
    pub fn atlas_frame(&self) -> &RegisteredImage {
        &self.steps[self.steps.len() - 1].moving
    }

    /// One sample voxel to atlas voxels.
    pub fn apply_one(&self, point: [f64; 3]) -> [f64; 3] {
        self.steps.iter().fold(point, |at, step| step.map.apply(at))
    }

    /// Sample level-0 voxels `[z, y, x]` to atlas level-0 voxels, in order.
    pub fn apply(&self, points: &[[f64; 3]]) -> Vec<[f64; 3]> {
        points.par_iter().map(|p| self.apply_one(*p)).collect()
    }

    /// Atlas level-0 voxels to sample level-0 voxels: the point the forward
    /// map takes to each, by Newton's method from the inverse of the forward
    /// map's linearisation at the sample centre. `None` where it does not
    /// converge.
    pub fn apply_inverse(&self, points: &[[f64; 3]]) -> Vec<Option<[f64; 3]>> {
        let centre = self.sample_shape().map(|n| (n as f64 - 1.0) / 2.0);
        let start = self.linearisation(centre);
        points
            .par_iter()
            .map(|target| {
                let guess = start.as_ref().map(|(f0, inverse)| {
                    let d: [f64; 3] = std::array::from_fn(|a| target[a] - f0[a]);
                    std::array::from_fn(|r| {
                        centre[r] + (0..3).map(|k| inverse[r][k] * d[k]).sum::<f64>()
                    })
                })?;
                self.newton(*target, guess)
            })
            .collect()
    }

    /// `f(c)` and the inverse of the Jacobian of the forward map at `c`.
    fn linearisation(&self, at: [f64; 3]) -> Option<([f64; 3], [[f64; 3]; 3])> {
        let value = self.apply_one(at);
        let jacobian = self.jacobian(at);
        invert(jacobian).map(|inverse| (value, inverse))
    }

    fn jacobian(&self, at: [f64; 3]) -> [[f64; 3]; 3] {
        let h = 0.5;
        let mut jacobian = [[0.0; 3]; 3];
        for c in 0..3 {
            let mut up = at;
            up[c] += h;
            let mut down = at;
            down[c] -= h;
            let (fu, fd) = (self.apply_one(up), self.apply_one(down));
            for r in 0..3 {
                jacobian[r][c] = (fu[r] - fd[r]) / (2.0 * h);
            }
        }
        jacobian
    }

    fn newton(&self, target: [f64; 3], mut x: [f64; 3]) -> Option<[f64; 3]> {
        let residual = |x: [f64; 3]| -> ([f64; 3], f64) {
            let f = self.apply_one(x);
            let r: [f64; 3] = std::array::from_fn(|a| f[a] - target[a]);
            (r, (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt())
        };
        let (mut r, mut norm) = residual(x);
        for _ in 0..50 {
            if norm < 1e-3 {
                return Some(x);
            }
            let inverse = invert(self.jacobian(x))?;
            let step: [f64; 3] =
                std::array::from_fn(|a| (0..3).map(|k| inverse[a][k] * r[k]).sum::<f64>());
            // Halve the step until the residual falls.
            let mut scale = 1.0;
            loop {
                let next: [f64; 3] = std::array::from_fn(|a| x[a] - scale * step[a]);
                let (next_r, next_norm) = residual(next);
                if next_norm < norm || scale < 1e-3 {
                    x = next;
                    r = next_r;
                    norm = next_norm;
                    break;
                }
                scale *= 0.5;
            }
        }
        (norm < 1e-3).then_some(x)
    }
}

fn step(directory: &Path) -> Result<Step> {
    let text = fs::read_to_string(directory.join(REGISTRATION_JSON)).map_err(|error| {
        Error::invalid(format!(
            "{}: {REGISTRATION_JSON}: {error}",
            directory.display()
        ))
    })?;
    let json: Value = serde_json::from_str(&text).map_err(Error::backend)?;
    let shape = |side: &str| -> Result<[usize; 3]> {
        let values = json
            .pointer(&format!("/{side}/level0_shape"))
            .and_then(Value::as_array)
            .filter(|v| v.len() == 3)
            .ok_or_else(|| {
                Error::invalid(format!("{REGISTRATION_JSON}: no {side}.level0_shape"))
            })?;
        let mut out = [0usize; 3];
        for a in 0..3 {
            out[a] = values[a].as_u64().ok_or_else(|| {
                Error::invalid(format!("{REGISTRATION_JSON}: {side}.level0_shape"))
            })? as usize;
        }
        Ok(out)
    };
    let frame = |side: &str| -> Result<Frame> {
        let triple = |key: &str| -> Result<[f64; 3]> {
            let values = json
                .pointer(&format!("/{side}/level0_frame/{key}"))
                .and_then(Value::as_array)
                .filter(|v| v.len() == 3)
                .ok_or_else(|| {
                    Error::invalid(format!("{REGISTRATION_JSON}: no {side}.level0_frame.{key}"))
                })?;
            let mut out = [0.0; 3];
            for a in 0..3 {
                out[a] = values[a].as_f64().unwrap_or(f64::NAN);
            }
            Ok(out)
        };
        Ok(Frame {
            scale: triple("scale")?,
            translation: triple("translation")?,
        })
    };
    let image = |side: &str| -> Result<RegisteredImage> {
        let path = json
            .pointer(&format!("/{side}/path"))
            .and_then(Value::as_str)
            .map(PathBuf::from);
        // Sidecars before `unit` was recorded: the image's own metadata.
        let unit = json
            .pointer(&format!("/{side}/unit"))
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| path.as_deref().and_then(ome_unit));
        Ok(RegisteredImage {
            path,
            shape: shape(side)?,
            frame: frame(side)?,
            unit,
        })
    };
    Ok(Step {
        map: registration_transform(directory, Units::Voxel)?,
        fixed: image("fixed")?,
        moving: image("moving")?,
    })
}

fn invert(m: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-12 || !det.is_finite() {
        return None;
    }
    Some(std::array::from_fn(|r| {
        std::array::from_fn(|c| {
            let (r0, r1) = ((c + 1) % 3, (c + 2) % 3);
            let (c0, c1) = ((r + 1) % 3, (r + 2) % 3);
            (m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0]) / det
        })
    }))
}
