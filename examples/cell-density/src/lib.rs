// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Cell detection in whole-organ light-sheet volumes, per-region counts and
//! density maps, written as an NGFF object table.
//!
//! - [`detect`]: the detector as one planned run (background removal,
//!   optional band-pass, h-maxima, maxima, seeded watershed, per-basin size
//!   and intensity), [`detect::CellPlan`] for any environment and
//!   [`detect::detect_cells`] in memory.
//! - [`params`]: what the detector decides, with defaults for 16-bit data.
//! - [`atlas`]: region, hemisphere and distance-to-surface columns from an
//!   annotation in the sample's frame, or from a prepared atlas through a
//!   registration (`blockflow_atlas_registration::atlas_space`), and the
//!   crust filter.
//! - [`collapse`]: per-region counts, volumes and densities.
//! - [`density`]: a density volume from the cell positions.
//! - [`table`]: the object table, the region CSV and the density image.
//! - [`cli`]: the `cell-density` command over one OME-Zarr image.

pub mod atlas;
pub mod cli;
pub mod collapse;
pub mod density;
pub mod detect;
pub mod extreme;
pub mod params;
pub mod table;
