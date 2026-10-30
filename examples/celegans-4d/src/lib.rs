// SPDX-License-Identifier: MIT

//! *C. elegans* 4-D DIC recordings (Endrov / S-BIAD191) on Blockflow. See
//! `README.md` and the project's `GOAL.md`.
//!
//! | module | what |
//! |---|---|
//! | [`endrov`] | Endrov `<nuclineage>` parsing and Endrov's interpolation rules — keyframes vs interpolated kept explicit |
//! | [`import`] | a lineage as `plate-timelapse-tracking`-format detection / track / link tables inside the image |
//! | [`labels`] | optional nucleus spheres as a `uint64` label stack |
//! | [`benchmark`] | held-out-keyframe benchmark: the [`benchmark::Refiner`] trait, the linear baseline, statistics |
//! | [`metrics`] | tracking metrics against a lineage: detection, division, link, naming; partial-truth modes |
//!
//! The Endrov reader lives here rather than in Blockflow's `src/`: it is a
//! file format, not an op, has no use for the framework's block machinery,
//! and every consumer so far (this import, the held-out-keyframe benchmark)
//! is part of this project. It depends only on `roxmltree`/`zip` and can move
//! to a crate of its own when a second project needs it.

pub mod benchmark;
pub mod endrov;
pub mod import;
pub mod labels;
pub mod metrics;
