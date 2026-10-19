// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Atlas region lookups: an annotation volume and its region tree, applied to
//! points and masks.
//!
//! - [`Ontology`]: the region hierarchy, built from `(id, level, parent)`
//!   values or read from an Allen-style JSON-lines file, with
//!   [`Ontology::collapse`] onto the ancestor at a level.
//! - [`Annotator`]: the region, hemisphere and distance to the surface at each
//!   of a set of points in atlas voxel coordinates, as [`AtlasColumns`].
//! - [`mask_from_annotation`] and [`distance_to_surface`]: the brain mask of an
//!   annotation and its distance-to-surface field.
//! - [`polyline_minima`]: a segmented minimum, for folding a per-point column
//!   onto the polylines (edges) the points belong to.
//! - [`Orientation`]: signed axis permutations of volumes, points, shapes and
//!   resolutions, with the reference's `(x, y, z)` tuples read by
//!   [`Orientation::from_xyz`].
//! - [`prepare_atlas`]: orient, crop and optionally keep one hemisphere of the
//!   atlas volumes, recording the [`PreparedWindow`].
//! - [`atlas_zarr`] (feature `zarr`): read the source volumes from OME-Zarr and
//!   write or open a prepared atlas as one OME-Zarr image.

mod annotate;
#[cfg(feature = "zarr")]
pub mod atlas_zarr;
mod ontology;
mod orientation;
mod prepare;
mod surface;

pub use annotate::{
    field_at, label_at, polyline_minima, voxel_index, AnnotateParams, Annotator, AtlasColumns,
    AtlasVolumes, OutOfBounds, Rounding,
};
pub use ontology::{Ontology, OntologyLevel, RegionNames, RegionNode};
pub use orientation::Orientation;
pub use prepare::{
    prepare_atlas, AtlasStack, AxisRange, Crop, HemisphereSelection, PrepareParams, PreparedWindow,
};
pub use surface::{distance_to_surface, mask_from_annotation};
