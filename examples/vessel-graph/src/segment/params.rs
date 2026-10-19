// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Per-channel parameter sets: a channel's **role** picks its defaults, and a
//! small JSON file (or the command line) overrides any of them.
//!
//! A vasculature study images more than one vessel channel and they do not
//! share a dynamic range. The all-vessel stain (role `vessel`) is dim and fine,
//! and it is binarised with every arm on over a narrow raw range; the
//! large-vessel stains (roles `artery`, `vein`) are bright and coarse, and they
//! are binarised over a higher range with a lower deconvolution threshold and
//! without the equalize and vessel-filter arms, which would pick up the
//! capillaries the vessel channel already has. The defaults per role are those
//! of the reference pipeline's vasculature configuration:
//!
//! | role | clip range | deconvolve threshold | equalize, vesselize |
//! |---|---|---|---|
//! | `vessel` | `(350, 1000)` | 750 | on |
//! | `artery`, `vein` | `(1000, 8000)` | 450 | off |
//!
//! Everything else (lightsheet, median, adaptive window, smoothing passes, fill)
//! is the same for every role, except the deep filling: on for every role in a
//! build with the `deep-fill` feature (off without it), resampled by 1 for
//! `vessel` and by 2 for `artery` and `vein`, as the reference's
//! configuration has it. Note that [`BinarizeParams::default`] is the
//! reference's *library* default, `(400, 60000)` with every arm on, which is
//! not any role's default.
//!
//! The parameter file is JSON, one entry per `--channel` in order; every key
//! but `role` is optional and overrides that role's default:
//!
//! ```json
//! {
//!   "channels": [
//!     {"role": "vessel"},
//!     {"role": "artery", "clip_range": [1000, 8000], "deconvolve_threshold": 450,
//!      "equalize": false, "vesselize": false, "smooth_passes": 6, "fill": true,
//!      "deep_fill": true, "deep_fill_resample": 2, "deep_fill_threshold": 0.5}
//!   ],
//!   "combined": {"smooth_passes": 6, "fill": true}
//! }
//! ```

use std::path::Path;

use blockflow::ops::Connectivity;
use blockflow::{Error, Result};
use serde::Deserialize;

use super::binarize::{BinarizeParams, Deconvolve, Equalize, Lightsheet, Vesselize};
use super::clip::ClipParams;
use super::deep_fill::{DeepFillParams, AVAILABLE as DEEP_FILL};
use super::pipeline::{ChannelParams, PostProcess, SegmentParams};
use super::smooth::SmoothParams;

/// What a channel stains, which picks its binarisation defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChannelRole {
    /// Every vessel, capillaries included.
    Vessel,
    /// Arteries (a smooth-muscle stain): the large-vessel defaults.
    Artery,
    /// Veins: the large-vessel defaults.
    Vein,
}

impl ChannelRole {
    pub fn name(self) -> &'static str {
        match self {
            Self::Vessel => "vessel",
            Self::Artery => "artery",
            Self::Vein => "vein",
        }
    }

    pub fn parse(text: &str) -> Result<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "vessel" | "vessels" => Ok(Self::Vessel),
            "artery" | "arteries" => Ok(Self::Artery),
            "vein" | "veins" => Ok(Self::Vein),
            other => Err(Error::invalid(format!(
                "`{other}` is not a channel role; the roles are vessel, artery and vein"
            ))),
        }
    }

    /// The role of channel `index` of a run when none is stated: the first
    /// channel is the all-vessel stain, every other one a large-vessel stain.
    pub fn default_for(index: usize) -> Self {
        if index == 0 {
            Self::Vessel
        } else {
            Self::Artery
        }
    }

    /// This role's binarisation.
    pub fn binarize(self) -> BinarizeParams {
        match self {
            Self::Vessel => BinarizeParams {
                clip: ClipParams {
                    range: (350.0, 1000.0),
                    ..ClipParams::default()
                },
                ..BinarizeParams::default()
            },
            Self::Artery | Self::Vein => BinarizeParams {
                clip: ClipParams {
                    range: (1000.0, 8000.0),
                    ..ClipParams::default()
                },
                deconvolve: Some(Deconvolve {
                    threshold: 450.0,
                    ..Deconvolve::default()
                }),
                equalize: None,
                vesselize: None,
                ..BinarizeParams::default()
            },
        }
    }

    /// This role's deep filling, whether or not the build can run it.
    pub fn deep_fill(self) -> DeepFillParams {
        match self {
            Self::Vessel => DeepFillParams::resampled(1),
            Self::Artery | Self::Vein => DeepFillParams::resampled(2),
        }
    }

    /// This role's binarisation and post-processing; the deep filling only in
    /// a build that has it.
    pub fn channel(self) -> ChannelParams {
        ChannelParams {
            binarize: self.binarize(),
            post: PostProcess::default(),
            deep_fill: DEEP_FILL.then(|| self.deep_fill()),
        }
    }
}

impl SegmentParams {
    /// One channel per role, each at its role's defaults.
    pub fn for_roles(roles: &[ChannelRole]) -> Self {
        Self {
            channels: roles.iter().map(|role| role.channel()).collect(),
            ..Self::with_channels(0)
        }
    }
}

/// Overrides of one channel's parameters; `None` keeps the value below it.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChannelOverrides {
    pub role: Option<ChannelRole>,
    pub clip_range: Option<(f64, f64)>,
    pub bins: Option<u32>,
    pub lightsheet: Option<bool>,
    pub median: Option<bool>,
    pub deconvolve: Option<bool>,
    pub adaptive: Option<bool>,
    pub equalize: Option<bool>,
    pub vesselize: Option<bool>,
    /// In clipped bins.
    pub deconvolve_threshold: Option<f64>,
    pub equalize_threshold: Option<f64>,
    pub vesselize_threshold: Option<f64>,
    /// 0 turns the channel's smoothing off.
    pub smooth_passes: Option<usize>,
    pub fill: Option<bool>,
    /// Switch the deep filling on (at the role's resampling) or off.
    pub deep_fill: Option<bool>,
    pub deep_fill_resample: Option<usize>,
    pub deep_fill_threshold: Option<f64>,
}

/// Overrides of a smooth-and-fill.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PostOverrides {
    /// 0 turns the smoothing off.
    pub smooth_passes: Option<usize>,
    pub fill: Option<bool>,
}

/// An arm switched on or off, or left; an arm switched on that was off starts
/// at its own defaults.
fn switch<T>(arm: Option<T>, on: Option<bool>, default: T) -> Option<T> {
    match on {
        None => arm,
        Some(false) => None,
        Some(true) => Some(arm.unwrap_or(default)),
    }
}

/// Set an arm's threshold, refusing one for an arm that is off.
fn threshold<T>(
    arm: &mut Option<T>,
    value: Option<f64>,
    what: &str,
    slot: impl Fn(&mut T) -> &mut f64,
) -> Result<()> {
    let Some(value) = value else {
        return Ok(());
    };
    match arm {
        Some(arm) => {
            *slot(arm) = value;
            Ok(())
        }
        None => Err(Error::invalid(format!(
            "a {what} threshold was given for a channel whose {what} arm is off"
        ))),
    }
}

impl PostOverrides {
    pub fn apply(&self, post: &mut PostProcess) {
        if let Some(passes) = self.smooth_passes {
            post.smooth = (passes > 0).then_some(SmoothParams {
                passes,
                ..post.smooth.unwrap_or_default()
            });
        }
        if let Some(fill) = self.fill {
            post.fill = fill.then_some(Connectivity::Faces);
        }
    }
}

impl ChannelOverrides {
    /// Apply these overrides to `channel`, of `role`. The role is not applied
    /// here: it picks the `channel` passed in, and the resampling a deep
    /// filling switched on starts at.
    pub fn apply(&self, channel: &mut ChannelParams, role: ChannelRole) -> Result<()> {
        let b = &mut channel.binarize;
        if let Some(range) = self.clip_range {
            b.clip.range = range;
        }
        if let Some(bins) = self.bins {
            b.clip.bins = bins;
        }
        b.lightsheet = switch(b.lightsheet, self.lightsheet, Lightsheet::default());
        b.median = switch(b.median, self.median, [3, 3, 3]);
        b.deconvolve = switch(b.deconvolve, self.deconvolve, Deconvolve::default());
        b.adaptive = switch(b.adaptive, self.adaptive, Default::default());
        b.equalize = switch(b.equalize, self.equalize, Equalize::default());
        b.vesselize = switch(b.vesselize, self.vesselize, Vesselize::default());
        threshold(
            &mut b.deconvolve,
            self.deconvolve_threshold,
            "deconvolve",
            |arm| &mut arm.threshold,
        )?;
        threshold(
            &mut b.equalize,
            self.equalize_threshold,
            "equalize",
            |arm| &mut arm.threshold,
        )?;
        threshold(
            &mut b.vesselize,
            self.vesselize_threshold,
            "vesselize",
            |arm| &mut arm.threshold,
        )?;
        PostOverrides {
            smooth_passes: self.smooth_passes,
            fill: self.fill,
        }
        .apply(&mut channel.post);
        channel.deep_fill = switch(channel.deep_fill, self.deep_fill, role.deep_fill());
        if self.deep_fill_resample.is_some() || self.deep_fill_threshold.is_some() {
            let Some(deep_fill) = channel.deep_fill.as_mut() else {
                return Err(Error::invalid(
                    "a deep fill resample or threshold was given for a channel whose deep \
                     filling is off",
                ));
            };
            if let Some(resample) = self.deep_fill_resample {
                deep_fill.resample = resample;
            }
            if let Some(value) = self.deep_fill_threshold {
                deep_fill.threshold = value;
            }
            deep_fill.validate()?;
        }
        Ok(())
    }
}

/// The parameter file.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SegmentOverrides {
    /// One entry per channel, in `--channel` order; may be empty.
    #[serde(default)]
    pub channels: Vec<ChannelOverrides>,
    /// After the union.
    #[serde(default)]
    pub combined: PostOverrides,
}

impl SegmentOverrides {
    pub fn from_json(text: &str) -> Result<Self> {
        serde_json::from_str(text)
            .map_err(|error| Error::invalid(format!("the segment parameter file: {error}")))
    }

    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|error| Error::invalid(format!("reading {}: {error}", path.display())))?;
        Self::from_json(&text)
            .map_err(|error| Error::invalid(format!("{}: {error}", path.display())))
    }

    /// The roles of `n` channels: a role in the file, else the default for the
    /// channel's position.
    pub fn roles(&self, n: usize) -> Vec<ChannelRole> {
        (0..n)
            .map(|index| {
                self.channels
                    .get(index)
                    .and_then(|channel| channel.role)
                    .unwrap_or_else(|| ChannelRole::default_for(index))
            })
            .collect()
    }

    /// The parameters of `roles.len()` channels of the given roles, with these
    /// overrides applied on top of the roles' defaults.
    pub fn params(&self, roles: &[ChannelRole]) -> Result<SegmentParams> {
        if self.channels.len() > roles.len() {
            return Err(Error::invalid(format!(
                "the parameter file lists {} channels and the run has {}",
                self.channels.len(),
                roles.len()
            )));
        }
        let mut params = SegmentParams::for_roles(roles);
        for ((channel, overrides), role) in
            params.channels.iter_mut().zip(&self.channels).zip(roles)
        {
            overrides.apply(channel, *role)?;
            channel.binarize.validate()?;
        }
        self.combined.apply(&mut params.combined);
        Ok(params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_roles_carry_the_reference_defaults() {
        let vessel = ChannelRole::Vessel.binarize();
        assert_eq!(vessel.clip.range, (350.0, 1000.0));
        assert_eq!(vessel.deconvolve.unwrap().threshold, 750.0);
        assert!(vessel.equalize.is_some() && vessel.vesselize.is_some());
        let artery = ChannelRole::Artery.binarize();
        assert_eq!(artery.clip.range, (1000.0, 8000.0));
        assert_eq!(artery.deconvolve.unwrap().threshold, 450.0);
        assert!(artery.equalize.is_none() && artery.vesselize.is_none());
        assert_eq!(ChannelRole::Vein.binarize(), artery);
        assert!(artery.adaptive.is_some() && artery.lightsheet.is_some());
    }

    #[test]
    fn a_file_overrides_its_channels_only() {
        let file = SegmentOverrides::from_json(
            r#"{"channels": [{}, {"role": "vein", "clip_range": [900, 7000],
                 "vesselize": true, "vesselize_threshold": 90, "smooth_passes": 0}],
                "combined": {"fill": false}}"#,
        )
        .unwrap();
        let roles = file.roles(3);
        assert_eq!(
            roles,
            [ChannelRole::Vessel, ChannelRole::Vein, ChannelRole::Artery]
        );
        let params = file.params(&roles).unwrap();
        assert_eq!(params.channels[0], ChannelRole::Vessel.channel());
        assert_eq!(params.channels[2], ChannelRole::Artery.channel());
        let vein = &params.channels[1];
        assert_eq!(vein.binarize.clip.range, (900.0, 7000.0));
        assert_eq!(vein.binarize.vesselize.unwrap().threshold, 90.0);
        assert!(vein.binarize.equalize.is_none());
        assert!(vein.post.smooth.is_none());
        assert!(params.combined.fill.is_none() && params.combined.smooth.is_some());
    }

    #[test]
    fn the_deep_filling_follows_the_role_and_the_build() {
        assert_eq!(ChannelRole::Vessel.deep_fill().resample, 1);
        assert_eq!(ChannelRole::Artery.deep_fill().resample, 2);
        assert_eq!(
            ChannelRole::Vein.deep_fill(),
            ChannelRole::Artery.deep_fill()
        );
        for role in [ChannelRole::Vessel, ChannelRole::Artery] {
            assert_eq!(role.channel().deep_fill.is_some(), DEEP_FILL);
        }
        let file = SegmentOverrides::from_json(
            r#"{"channels": [{"deep_fill": false},
                 {"role": "artery", "deep_fill": true, "deep_fill_threshold": 0.4}]}"#,
        )
        .unwrap();
        let params = file.params(&file.roles(2)).unwrap();
        assert_eq!(params.channels[0].deep_fill, None);
        assert_eq!(
            params.channels[1].deep_fill,
            Some(DeepFillParams {
                resample: 2,
                threshold: 0.4
            })
        );
        let off = SegmentOverrides::from_json(
            r#"{"channels": [{"deep_fill": false, "deep_fill_threshold": 0.4}]}"#,
        )
        .unwrap();
        assert!(off.params(&off.roles(1)).is_err());
        let bad = SegmentOverrides::from_json(
            r#"{"channels": [{"deep_fill": true, "deep_fill_resample": 0}]}"#,
        )
        .unwrap();
        assert!(bad.params(&bad.roles(1)).is_err());
    }

    #[test]
    fn a_threshold_for_an_arm_that_is_off_is_refused() {
        let file = SegmentOverrides::from_json(
            r#"{"channels": [{"role": "artery", "equalize_threshold": 1.2}]}"#,
        )
        .unwrap();
        assert!(file.params(&file.roles(1)).is_err());
        assert!(SegmentOverrides::from_json(r#"{"channels": [{"clip": [1, 2]}]}"#).is_err());
        assert!(SegmentOverrides::from_json(r#"{"channels": [{}, {}]}"#)
            .unwrap()
            .params(&[ChannelRole::Vessel])
            .is_err());
    }
}
