// SPDX-License-Identifier: MIT
// Original work for this crate.

//! One vessel channel to a binary mask: a clip, two corrections, and five arms
//! whose verdicts are OR-ed into one sink.
//!
//! ```text
//! volumes  raw -> clip -> lightsheet -> median -+-> deconvolve -> read by adaptive
//!                                               \-> equalize ---> read by vesselize
//!
//! sink     high mask -> OR(deconvolve > t) -> OR(adaptive) -> OR(equalize > t)
//!                    -> OR(vesselize > t)
//! ```
//!
//! It is a fan-out and fan-in rather than a pipe: one source feeds several
//! arms and the mask is the union of their verdicts. Three arms also change
//! what a later arm reads — the lightsheet correction is under the median and
//! so under everything, `deconvolve` replaces the volume `adaptive` thresholds,
//! and `equalize` replaces the volume `vesselize` filters — so turning one off
//! reroutes the arms after it back to the median.
//!
//! Three masks of the clip are in play and they are not interchangeable: the
//! saturated voxels seed the sink, `not low` is the median's population, and
//! `in range` (neither low nor saturated) is the population of the lightsheet,
//! equalize and vessel-background estimates. Excluding saturated voxels there
//! keeps a bright vessel from raising its own background.
//!
//! Every step is a blockflow op: `voxelwise` maps, masks and joins
//! (`FnCombine` for the three-way arithmetic), `rank` for the median and the
//! masked percentiles, `local` for the sampled window statistics and the
//! adaptive isodata threshold, `smooth` for the deconvolution's flat blur and
//! `ridge` (`RatioResponse`) for the tubeness. The two arms with an expensive
//! op inside them — the blur in `deconvolve`, the disk percentile and the
//! Hessian in `vesselize` — are sequences handed to `PlanBuilder::partition`,
//! so the planner may give the wide-halo step a phase of its own instead of
//! fusing every halo into one.
//!
//! Axis order is `(z, y, x)`. The lightsheet correction runs along `x`, the
//! deconvolution blur and the vessel-background disk are in the `(y, x)` plane.

use blockflow::op::Chain;
use blockflow::ops::element::{ElementShape, Rank, StructuringElement};
use blockflow::ops::local::{
    AdaptiveThresholdOp, Isodata, LocalStatistic, LocalStatisticOp, Sampling, Statistic,
};
use blockflow::ops::normalise::bounded_gain_value;
use blockflow::ops::rank::MaskedRankFilterOp;
use blockflow::ops::ridge::{Boundary, Polarity, RatioResponse, RidgeFilterOp, ScaleSpace};
use blockflow::ops::smooth::{Gaussian, SmoothOp};
use blockflow::ops::voxelwise::{
    CarryOp, FnCombine, Logic, LogicCombine, VoxelwiseMapOp, VoxelwiseMaskOp,
};
use blockflow::{Dtype, Error, Result};

use super::clip::ClipParams;
use super::plan::{chain_of, source_f64, Cursor, Image};

/// Lightsheet correction: remove the stripe pattern of the illumination by
/// subtracting a low percentile along the light path, capped by a scaled
/// percentile of a wide background window.
///
/// ```text
/// corrected = round(value - min(value, min(line, versus * background)))
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lightsheet {
    /// The percentile both estimates take, `0 ..= 1`.
    pub percentile: f64,
    /// The line the illumination percentile is taken over: long on `x`.
    pub line: [usize; 3],
    /// The background window.
    pub background: [usize; 3],
    /// The background's sample lattice: the window is evaluated this far apart
    /// and interpolated in between.
    pub background_spacing: [usize; 3],
    /// The stride inside the background window, which decimates what it gathers.
    pub background_step: [usize; 3],
    /// What the background estimate is scaled by before the `min`.
    pub versus: f64,
}

impl Default for Lightsheet {
    fn default() -> Self {
        Self {
            percentile: 0.25,
            line: [1, 1, 150],
            background: [1, 200, 200],
            background_spacing: [1, 25, 25],
            background_step: [1, 2, 2],
            versus: 2.0,
        }
    }
}

/// Deconvolution: the voxels already in the sink are blurred in `(y, x)` and
/// subtracted from everything outside the sink; a residual above `threshold`
/// is foreground. The residual volume is also what `adaptive` thresholds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Deconvolve {
    /// The blur's sigma in `y` and `x`; it is flat in `z`.
    pub sigma: f64,
    /// In clipped bins, strict.
    pub threshold: f64,
}

impl Default for Deconvolve {
    fn default() -> Self {
        Self {
            sigma: 10.0,
            threshold: 750.0,
        }
    }
}

/// A local isodata threshold over a window, evaluated on a lattice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Adaptive {
    pub selem: [usize; 3],
    pub spacing: [usize; 3],
}

impl Default for Adaptive {
    fn default() -> Self {
        Self {
            selem: [3, 250, 250],
            spacing: [3, 50, 50],
        }
    }
}

/// Equalisation: each voxel scaled by `1 / max(p_lo, 1)`, capped so the local
/// `p_hi` maps no higher than `max_value`. The scaled value above `threshold`
/// is foreground; the same number also clamps and rescales the volume
/// `vesselize` reads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Equalize {
    pub percentile: (f64, f64),
    pub max_value: f64,
    pub selem: [usize; 3],
    pub spacing: [usize; 3],
    pub threshold: f64,
}

impl Default for Equalize {
    fn default() -> Self {
        Self {
            percentile: (0.4, 0.975),
            max_value: 1.5,
            selem: [5, 200, 200],
            spacing: [5, 50, 50],
            threshold: 1.1,
        }
    }
}

/// Vesselisation: the positive residual over a median background disk, then a
/// Hessian tubeness, thresholded.
///
/// The tubeness is `ops::ridge::RatioResponse`:
/// `|e3| * (e2 / e3)^gamma12 * (1 + e1 / |e2|)^gamma23` for a bright tube
/// (both cross-section curvatures negative), with `alpha` the penalty for an
/// along-axis curvature of the wrong sign.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vesselize {
    /// The background disk's extent in `(y, x)`; one plane thick.
    pub disk: [usize; 2],
    pub sigma: f64,
    pub gamma12: f64,
    pub gamma23: f64,
    pub alpha: f64,
    /// Strict.
    pub threshold: f64,
}

impl Default for Vesselize {
    fn default() -> Self {
        Self {
            disk: [30, 30],
            sigma: 1.0,
            gamma12: 0.0,
            gamma23: 1.0,
            alpha: 0.25,
            threshold: 120.0,
        }
    }
}

/// The percentile of the vessel background disk.
const VESSEL_BACKGROUND_PERCENTILE: f64 = 0.5;

/// The isodata histogram's bin count.
const ISODATA_BINS: usize = 256;

/// The equalisation gain's floor: `1 / max(p_lo, FLOOR)`.
const EQUALIZE_FLOOR: f64 = 1.0;

/// Gaussians are cut at four sigma.
const TRUNCATE: f64 = 4.0;

/// The binarisation of one channel. `None` turns an arm off.
#[derive(Debug, Clone, PartialEq)]
pub struct BinarizeParams {
    pub clip: ClipParams,
    pub lightsheet: Option<Lightsheet>,
    /// The median element's size, odd on every axis.
    pub median: Option<[usize; 3]>,
    pub deconvolve: Option<Deconvolve>,
    pub adaptive: Option<Adaptive>,
    pub equalize: Option<Equalize>,
    pub vesselize: Option<Vesselize>,
}

impl Default for BinarizeParams {
    /// Every arm on, at values for a whole-brain light-sheet vasculature scan.
    fn default() -> Self {
        Self {
            clip: ClipParams::default(),
            lightsheet: Some(Lightsheet::default()),
            median: Some([3, 3, 3]),
            deconvolve: Some(Deconvolve::default()),
            adaptive: Some(Adaptive::default()),
            equalize: Some(Equalize::default()),
            vesselize: Some(Vesselize::default()),
        }
    }
}

impl BinarizeParams {
    /// The clip and the saturated seed alone, every arm off.
    pub fn seed_only(clip: ClipParams) -> Self {
        Self {
            clip,
            lightsheet: None,
            median: None,
            deconvolve: None,
            adaptive: None,
            equalize: None,
            vesselize: None,
        }
    }

    /// Refuse a point the ops would refuse later, here, naming the parameter.
    pub fn validate(&self) -> Result<()> {
        self.clip.validate()?;
        if let Some(size) = self.median {
            if size.iter().any(|&edge| edge == 0 || edge % 2 == 0) || size == [1, 1, 1] {
                return Err(Error::invalid(format!(
                    "binarize: the median element is {size:?}; it needs an odd size on every \
                     axis and more than one voxel (leave it out with `median: None`)"
                )));
            }
        }
        let finite = |what: &str, value: f64| -> Result<()> {
            if value.is_finite() {
                Ok(())
            } else {
                Err(Error::invalid(format!("binarize: {what} is {value}")))
            }
        };
        if let Some(arm) = self.lightsheet {
            finite("the lightsheet balance", arm.versus)?;
            Rank::ceiling_percentile(arm.percentile)?;
            if arm.background_spacing.contains(&0) {
                return Err(Error::invalid(
                    "binarize: a lightsheet background spacing of zero lays out no lattice",
                ));
            }
        }
        if let Some(arm) = self.deconvolve {
            finite("the deconvolve threshold", arm.threshold)?;
            if arm.sigma <= 0.0 || !arm.sigma.is_finite() {
                return Err(Error::invalid(format!(
                    "binarize: the deconvolve sigma is {}; leave the arm out instead of a zero blur",
                    arm.sigma
                )));
            }
        }
        if let Some(arm) = self.adaptive {
            if arm.spacing.contains(&0) {
                return Err(Error::invalid(
                    "binarize: an adaptive spacing of zero lays out no lattice",
                ));
            }
        }
        if let Some(arm) = self.equalize {
            Rank::ceiling_percentile(arm.percentile.0)?;
            Rank::ceiling_percentile(arm.percentile.1)?;
            finite("the equalize ceiling", arm.max_value)?;
            if !arm.threshold.is_finite() || arm.threshold <= 0.0 {
                return Err(Error::invalid(format!(
                    "binarize: the equalize threshold is {}; it also divides the rescale the \
                     vessel arm reads, so it must be positive",
                    arm.threshold
                )));
            }
            if arm.spacing.contains(&0) {
                return Err(Error::invalid(
                    "binarize: an equalize spacing of zero lays out no lattice",
                ));
            }
        }
        if let Some(arm) = self.vesselize {
            finite("the vesselize threshold", arm.threshold)?;
            if arm.disk.contains(&0) || arm.disk == [1, 1] {
                return Err(Error::invalid(format!(
                    "binarize: a vessel background disk of {:?} removes nothing or everything",
                    arm.disk
                )));
            }
            RatioResponse::new(arm.gamma12, arm.gamma23, arm.alpha, Polarity::Ridge)?;
            ScaleSpace::isotropic(&[arm.sigma], TRUNCATE, 0.0)?;
        }
        Ok(())
    }
}

/// The images one channel's binarisation leaves behind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Binarized {
    /// The median-filtered volume (the clip itself when there is no median).
    pub filtered: Image,
    /// The sink, `Bool`.
    pub binary: Image,
}

fn or() -> Result<Box<LogicCombine>> {
    Ok(Box::new(
        LogicCombine::new("or", Logic::Or).producing(Dtype::Bool)?,
    ))
}

fn statistic_percentile(
    name: &'static str,
    element: StructuringElement,
    spacing: [usize; 3],
    percentile: f64,
    population: Image,
) -> Result<LocalStatisticOp> {
    Ok(LocalStatisticOp::new(
        name,
        LocalStatistic::sampled(
            element,
            Sampling::every(spacing),
            Statistic::Rank(Rank::ceiling_percentile(percentile)?),
        )?,
    )
    .masked_by(population.id))
}

/// Append one channel's binarisation, reading the raw channel `raw`.
pub fn append_binarize_phases(
    c: &mut Cursor<'_>,
    raw: Image,
    params: &BinarizeParams,
) -> Result<Binarized> {
    params.validate()?;
    let clip = params.clip;

    // The two populations and the clip all read the raw volume, so they sit
    // together at the top; the clip is last so that it is the input of the
    // lightsheet phase below it.
    let population =
        if params.lightsheet.is_some() || params.equalize.is_some() || params.vesselize.is_some() {
            Some(c.pixels(c.on_f64(raw, vec![clip.in_range_node()]))?)
        } else {
            None
        };
    let not_low = match params.median {
        Some(_) => Some(c.pixels(c.on_f64(raw, vec![clip.not_low_node()]))?),
        None => None,
    };
    let clipped = c.pixels(c.on_f64(raw, clip.clip_nodes()?))?;

    let corrected = match (params.lightsheet, population) {
        (Some(arm), Some(population)) => {
            let line = MaskedRankFilterOp::percentile(
                "lightsheet line",
                StructuringElement::from_size(ElementShape::Box, arm.line)?,
                arm.percentile,
                population.id,
            )?
            .filling_excluded_centres(0.0);
            let background = statistic_percentile(
                "lightsheet background",
                StructuringElement::from_size_stepped(
                    ElementShape::Box,
                    arm.background,
                    arm.background_step,
                )?,
                arm.background_spacing,
                arm.percentile,
                population,
            )?;
            let versus = arm.versus;
            // Written as `u16` directly: the answer is rounded, and the clipped
            // range is the only range it can land in.
            let correct = FnCombine::new("lightsheet correction", 3, move |x| {
                let combined = x[1].min(versus * x[2]);
                (x[0] - x[0].min(combined)).round()
            })?
            .producing(Dtype::U16)?;
            Some(c.pixels(Chain::parallel(
                vec![
                    c.on(clipped, vec![]),
                    c.on(clipped, vec![Chain::op(line)]),
                    c.on_f64(clipped, vec![Chain::op(background)]),
                ],
                Box::new(correct),
            )?)?)
        }
        _ => None,
    };
    let volume = corrected.unwrap_or(clipped);

    let filtered = match (params.median, not_low) {
        (Some(size), Some(not_low)) => {
            let element = StructuringElement::from_size(ElementShape::Box, size)?;
            let rank = Rank::median(&element);
            // A dark voxel is left at zero rather than given its neighbours'
            // median.
            let median = MaskedRankFilterOp::new("median", element, rank, not_low.id)
                .filling_excluded_centres(0.0);
            c.pixels(c.on(volume, vec![Chain::op(median)]))?
        }
        _ => volume,
    };

    // The equalize gain sits directly under the median, the image it reads;
    // its threshold joins the sink further down.
    let equalized = match (params.equalize, population) {
        (Some(arm), Some(population)) => {
            let element = StructuringElement::from_size(ElementShape::Box, arm.selem)?;
            let low = statistic_percentile(
                "equalize low",
                element.clone(),
                arm.spacing,
                arm.percentile.0,
                population,
            )?;
            let high = statistic_percentile(
                "equalize high",
                element,
                arm.spacing,
                arm.percentile.1,
                population,
            )?;
            let ceiling = arm.max_value;
            let gain = FnCombine::new("equalize gain", 3, move |x| {
                bounded_gain_value(x[0], x[1], x[2], EQUALIZE_FLOOR, ceiling)
            })?;
            Some(c.pixels(Chain::parallel(
                vec![
                    c.on_f64(filtered, vec![]),
                    c.on_f64(filtered, vec![Chain::op(low)]),
                    c.on_f64(filtered, vec![Chain::op(high)]),
                ],
                Box::new(gain),
            )?)?)
        }
        _ => None,
    };

    let seed = c.pixels(c.on_f64(raw, vec![clip.high_node()]))?;

    // Deconvolve: blur the sink's voxels, subtract outside the sink.
    let (deconvolved, mut sink) = match params.deconvolve {
        None => (None, seed),
        Some(arm) => {
            let blur = Gaussian::new([0.0, arm.sigma, arm.sigma], TRUNCATE)?
                .with_boundary(Boundary::Reflect);
            let masked = FnCombine::new("deconvolve masked source", 2, |x| {
                if x[1] != 0.0 {
                    x[0]
                } else {
                    0.0
                }
            })?;
            let residual = FnCombine::new("deconvolve residual", 3, |x| {
                if x[2] != 0.0 {
                    x[0]
                } else {
                    x[0] - x[0].min(x[1])
                }
            })?;
            let chain = Chain::sequence(vec![
                Chain::parallel(
                    vec![c.on_f64(filtered, vec![]), c.on(seed, vec![])],
                    Box::new(masked),
                )?,
                Chain::op(SmoothOp::new("deconvolve blur", blur)),
                // Inside a sequence a branch sees the stream (the blur), so the
                // volume and the sink are named.
                Chain::parallel(
                    vec![
                        chain_of(source_f64(filtered)),
                        Chain::op(CarryOp::new("blurred")),
                        Chain::source(seed.id, seed.dtype),
                    ],
                    Box::new(residual),
                )?,
            ]);
            let deconvolved = c.partition(chain)?;
            let sink = c.pixels(Chain::parallel(
                vec![
                    c.on(seed, vec![]),
                    c.on(
                        deconvolved,
                        vec![Chain::op(VoxelwiseMaskOp::threshold(
                            "deconvolved",
                            arm.threshold,
                        ))],
                    ),
                ],
                or()?,
            )?)?;
            (Some(deconvolved), sink)
        }
    };

    if let Some(arm) = params.adaptive {
        let statistic = LocalStatistic::sampled(
            StructuringElement::from_size(ElementShape::Box, arm.selem)?,
            Sampling::every(arm.spacing),
            Statistic::Isodata(Isodata::new(ISODATA_BINS, 0.0)?),
        )?;
        let volume = deconvolved.unwrap_or(filtered);
        sink = c.pixels(Chain::parallel(
            vec![
                c.on(sink, vec![]),
                c.on_f64(
                    volume,
                    vec![Chain::op(AdaptiveThresholdOp::new(
                        "adaptive", statistic, 1.0, 0.0,
                    ))],
                ),
            ],
            or()?,
        )?)?;
    }

    if let (Some(arm), Some(equalized)) = (params.equalize, equalized) {
        sink = c.pixels(Chain::parallel(
            vec![
                c.on(sink, vec![]),
                c.on(
                    equalized,
                    vec![Chain::op(VoxelwiseMaskOp::threshold(
                        "equalized",
                        arm.threshold,
                    ))],
                ),
            ],
            or()?,
        )?)?;
    }

    if let (Some(arm), Some(population)) = (params.vesselize, population) {
        // What the arm reads: the equalized volume clamped at the equalize
        // threshold and rescaled onto the bins, or the median.
        let mut nodes = match (params.equalize, equalized) {
            (Some(eq), Some(equalized)) => {
                let threshold = eq.threshold;
                let scale = clip.top() / threshold;
                vec![
                    Chain::source(equalized.id, equalized.dtype),
                    Chain::op(VoxelwiseMapOp::new("equalize rescale", move |v: f64| {
                        v.min(threshold) * scale
                    })),
                ]
            }
            _ => source_f64(filtered),
        };
        let disk =
            StructuringElement::from_size(ElementShape::Ellipsoid, [1, arm.disk[0], arm.disk[1]])?;
        let background = MaskedRankFilterOp::percentile(
            "vessel background",
            disk,
            VESSEL_BACKGROUND_PERCENTILE,
            population.id,
        )?
        .filling_excluded_centres(0.0);
        let tubeness = RidgeFilterOp::new(
            "tubeness",
            ScaleSpace::isotropic(&[arm.sigma], TRUNCATE, 0.0)?.with_boundary(Boundary::Reflect),
            RatioResponse::new(arm.gamma12, arm.gamma23, arm.alpha, Polarity::Ridge)?,
        );
        nodes.extend([
            Chain::op(VoxelwiseMapOp::new("clamp", |v: f64| {
                v.clamp(0.0, f64::from(u16::MAX)).trunc()
            })),
            Chain::parallel(
                vec![Chain::op(CarryOp::new("clamped")), Chain::op(background)],
                Box::new(FnCombine::new("vessel residual", 2, |x| {
                    x[0] - x[0].min(x[1])
                })?),
            )?,
            Chain::op(tubeness),
            Chain::op(VoxelwiseMaskOp::threshold("vessels", arm.threshold)),
            Chain::parallel(
                vec![
                    Chain::source(sink.id, sink.dtype),
                    Chain::op(CarryOp::new("vessels")),
                ],
                or()?,
            )?,
        ]);
        sink = c.partition(Chain::sequence(nodes))?;
    }

    Ok(Binarized {
        filtered,
        binary: sink,
    })
}
