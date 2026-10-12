// SPDX-License-Identifier: MIT

//! Segment, measure and track cells in multiwell-plate time-lapse OME-Zarr
//! data. See `README.md` and the repository's `TRACKING_PLAN.md`.
//!
//! | module | stage |
//! |---|---|
//! | [`run`] | enumerate series, validate, run all frames on one bounded pool, finalise each series, write the collection index |
//! | [`series`] | per-series staging, linking, pyramids, tables, exports, atomic publish |
//! | [`frame`] | per-frame segmentation into the staged label stack, measurement, durable artifact |
//! | [`segment`] | segmenter loading and normalisation |
//! | [`tables`] | detection / track / link object tables |
//! | [`synth`] | a synthetic plate with ground truth |
//! | [`score`] | scoring against that truth |

pub mod frame;
pub mod params;
pub mod pool;
pub mod run;
pub mod score;
pub mod segment;
pub mod series;
pub mod synth;
pub mod tables;
