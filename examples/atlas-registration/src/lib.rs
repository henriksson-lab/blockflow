// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Atlas registration: block-parallel affine and B-spline registration of a
//! volume to an atlas, and point transforms between the two spaces.
//!
//! * [`compute`] fits a transform with `blockflow::ops::align` and writes it as
//!   elastix `TransformParameters.N.txt` text; it also reads elastix
//!   registration parameter files into fit settings, and [`stochastic`] reads
//!   the stochastic ones (Mattes mutual information, random samplers, elastix's
//!   gradient descents), the reference's shipped files among them.
//! * [`PointTransform`] carries sample level-0 voxels to atlas level-0 voxels
//!   through a registration directory or a chain of them, and back.
//! * [`ElastixTransform`] reads those files (or elastix's own) and applies them to
//!   points, with every convention measured against transformix.
//! * [`pipeline`] is what the CLI runs: OME-Zarr in, transform files, a
//!   registration sidecar and optionally a warped OME-Zarr volume out, and CSV
//!   point tables carried between the two images.
//! * [`place`] finds where a fixed image lies in a moving one by a
//!   coarse-to-fine correlation search, to start a registration from.
//! * [`atlas_space`] maps an image's points through a registration into the
//!   prepared atlas it registered to, checks the two belong together, and
//!   reads regions, hemispheres, distance to the surface and densities there.
//! * [`memory`] leases the volumes held resident from the process's memory
//!   budget, and refuses one a stated cap cannot hold.
//! * [`report`] measures the fit against a known answer or a recorded elastix
//!   run.

pub mod atlas_space;
pub mod compute;
pub mod memory;
pub mod parameters;
pub mod pipeline;
pub mod place;
pub mod points;
pub mod register;
pub mod report;
pub mod rescale;
pub mod similarity;
pub mod stochastic;
pub mod transform;
pub mod volume;

pub use parameters::ParameterMap;
pub use points::{PointTransform, RegisteredImage};
pub use register::{transform_file, AtlasTransform, ElastixTransform, PointSpace, RegisterParams};
pub use rescale::{Frame, Rescale};
pub use transform::{
    BSpline, Chain, Combine, ImageGeometry, IndexRounding, Link, Transform, NO_INITIAL_TRANSFORM,
};
