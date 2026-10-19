// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Vessel segmentation, skeleton-to-graph extraction, graph cleanup, radii,
//! artery/vein classification and branch density.
//!
//! [`segment`] turns a vessel channel into a binary mask and its skeleton;
//! [`graph`] turns a skeleton into an annotated vessel graph. Either maps into
//! a prepared atlas through a registration with
//! `blockflow_atlas_registration::atlas_space`.

pub mod graph;
pub mod profile;
pub mod segment;
