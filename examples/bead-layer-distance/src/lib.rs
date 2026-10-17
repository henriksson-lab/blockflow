// SPDX-License-Identifier: MIT
//
// Original work for this crate.

//! Beads on a mucus layer over an epithelium: where are they, and how far
//! from the epithelium?
//!
//! Three channels of an OME-Zarr image: C-Blue (1 µm beads, PSF-blurred; it
//! also stains epithelial nuclei faintly), AF488 (outer epithelium; beads are
//! bright in it too) and AF594 (top of the mucus). The analysis is one
//! blockflow plan over level 0 ([`pipeline`]):
//!
//! 1. **Layers.** Per channel a smooth → log → z-filter cost image; the mucus
//!    top and the epithelium apical surface are found jointly as one exact
//!    minimum cut with step and separation limits (`blockflow::ops::surfaces`),
//!    which is what keeps them off bright clutter.
//! 2. **Large AF488 objects** (probable dying cells) inside the mucus band:
//!    robust thresholds, planned connected components, a volume filter
//!    ([`cells`]). Beads touching one are flagged, not dropped.
//! 3. **Beads.** DoG response on C-Blue, a noise-referenced level, exact
//!    regional maxima, minimum-distance suppression, a PSF-shape test,
//!    vertical and Euclidean distance to the epithelium top, a brightness
//!    split ([`beads`], [`measure`]).
//!
//! Output: label images and object tables published into the image for a
//! viewer such as newvolim (`bjorn-layers`, `bjorn-surfaces`, `bjorn-beads`,
//! `bjorn-large-af488`; colours in [`style`]), CSVs and QC images. A run
//! replaces all previous output.

pub mod beads;
pub mod cells;
pub mod measure;
pub mod ops;
pub mod pipeline;
pub mod qc;
pub mod style;
pub mod zio;

pub use pipeline::{run, Config, Summary};
