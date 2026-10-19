// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Vessel channel(s) to a binary mask and its skeleton, and tract maps.
//!
//! * [`params`] — per-channel parameter sets: a role's defaults (vessel,
//!   artery, vein) and a JSON file of overrides;
//! * [`binarize`] — one channel to a mask: clip, lightsheet correction, median,
//!   and the deconvolve / adaptive / equalize / vesselize arms OR-ed into a
//!   sink seeded with the saturated voxels;
//! * [`smooth`] — the topology-preserving 3x3x3 configuration rule;
//! * [`pipeline`] — channels binarised, smoothed and filled, united,
//!   post-processed again, and skeletonised ([`skeleton`]), as one plan;
//! * [`tract`] — a tract channel binarised at a clip range taken from its own
//!   intensity percentiles, its points voxelised into a density;
//! * [`cli`] — the `segment` and `tract` subcommands over OME-Zarr;
//! * [`crop`] — a crop of an image copied into a new one in the same physical
//!   frame, which `--crop` runs the subcommands on.
//!
//! Each `append_*` function adds phases to a caller's [`PlanBuilder`] through a
//! [`plan::Cursor`]; the in-memory functions ([`pipeline::segment`],
//! [`pipeline::binarize`], [`pipeline::mask_steps`], [`tract::tract_map`]) are
//! the same plans run over arrays.
//!
//! [`PlanBuilder`]: blockflow::assemble::PlanBuilder

pub mod binarize;
pub mod cli;
pub mod clip;
pub mod crop;
pub mod params;
pub mod pipeline;
pub mod plan;
pub mod skeleton;
pub mod smooth;
pub mod synthetic;
pub mod tract;

pub use binarize::{
    Adaptive, BinarizeParams, Binarized, Deconvolve, Equalize, Lightsheet, Vesselize,
};
pub use clip::ClipParams;
pub use params::{ChannelOverrides, ChannelRole, PostOverrides, SegmentOverrides};
pub use pipeline::{
    binarize as binarize_volume, binarize_plan, mask_steps, segment, segment_plan, ChannelParams,
    MaskSteps, PostProcess, SegmentParams, Segmentation,
};
pub use skeleton::SkeletonParams;
pub use smooth::{Rule, SmoothParams};
pub use tract::{tract_map, TractMap, TractParams};
