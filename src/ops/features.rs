// SPDX-License-Identifier: MIT
//
// Original work for this crate.

//! **The pixel-classification feature stack**: sigmas in, a list of `Chain`s
//! out, one per channel a random forest will see.
//!
//! This is step 2 of `docs/design/pixel-classification.md` and it computes
//! nothing of its own. Every channel is an op that already existed or one the
//! same plan added; what is here is the *list* — which filters, at which scales,
//! in which order — and the arithmetic that says how long it is.
//!
//! # The list is Labkit's, and it is copied rather than invented
//!
//! Given sigmas `s_1 .. s_n`, per scale unless noted:
//!
//! | family | channels, 3-D |
//! |---|---|
//! | the original image | 1, once |
//! | Gaussian blur | 1 |
//! | difference of Gaussians, pairwise | `C(n,2)` total |
//! | Gaussian gradient magnitude | 1 |
//! | Laplacian of Gaussian | 1 |
//! | Hessian eigenvalues | 3 |
//! | structure tensor eigenvalues, at `gamma = 1, 3` | 6 |
//! | morphological min, max, mean, deviation | 4 |
//!
//! At five sigmas that is **91 channels**, which is the number the design
//! document's memory arithmetic is built on and the number
//! [`FeatureStack::len`] must agree with.
//!
//! **One deliberate departure.** Labkit lists *variance* where this emits the
//! standard *deviation*, because that is the statistic `ops::local` has. It
//! makes no difference to the consumer and the reason is worth stating rather
//! than leaving as an accident: a decision tree splits on `feature <= t`, so any
//! strictly increasing transform of a feature induces exactly the same set of
//! partitions and therefore the same fitted forest, up to the thresholds' units.
//! A square root on a non-negative quantity is such a transform. Squaring it
//! here to match a name would cost a pass over every voxel and change nothing
//! downstream.
//!
//! # What this deliberately does not do
//!
//! It does not join the branches. [`FeatureStack::branches`] hands back a `Vec`
//! and the caller supplies the [`Combine`](crate::op::Combine) — which for the real workload is
//! forest predictor, the thing that turns 91 images into one. A builder that
//! chose the combine would have had to know what the stack was for, and the
//! stack is also wanted for *training*, where the join is a sampler rather than
//! a predictor.
//!
//! # The 2-D mode is a scale of zero, not a second code path
//!
//! [`Geometry::PlaneWise`] sets every scale to zero on the plane normal and
//! every structuring element to one voxel thick there. Nothing else changes:
//! `gaussian_radius(0.0, t)` is zero, so the reach on that axis falls to what
//! the stencil alone needs, and the planner sees a phase that can be cut freely
//! along it. This is what `docs/design/pixel-classification.md` asks for under
//! its risks — that the mode "fall out of the element parameterisation rather
//! than becoming a second code path" — and it is why the channel *count* differs
//! between the modes only where the mathematics differs: a 2-D Hessian has two
//! eigenvalues and a 2-D structure tensor has two, so those families shrink and
//! the rest do not.

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::op::{Anchor, Chain, Combine, Slicing};
use crate::voxels::Voxels;

use super::convolve::{ConvolveOp, Kernel, Sense};
use super::element::{ElementShape, Rank, StructuringElement};
use super::local::{LocalStatistic, LocalStatisticOp, Statistic};
use super::rank::RankFilterOp;
use super::ridge::{Boundary, HessianEigenvalueOp};
use super::smooth::{Gaussian, SmoothOp};
use super::structure_tensor::{
    Eigenvalue, GradientMagnitudeOp, StructureTensor, StructureTensorOp,
};
use super::voxelwise::{Arithmetic, ArithmeticCombine, VoxelwiseMapOp};

/// Whether the stack is three-dimensional or plane-wise, and if plane-wise,
/// which axis is the plane normal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Geometry {
    /// Filters reach in all three axes.
    Volumetric,
    /// Filters reach within a plane only. `normal` is the axis they do **not**
    /// reach along — the slice axis.
    PlaneWise { normal: usize },
}

impl Geometry {
    /// The scale, per axis, for a caller-supplied isotropic sigma.
    fn scale(self, sigma: f64) -> [f64; 3] {
        let mut scale = [sigma; 3];
        if let Geometry::PlaneWise { normal } = self {
            scale[normal] = 0.0;
        }
        scale
    }

    /// The radius, per axis, for a caller-supplied isotropic radius.
    fn radius(self, radius: usize) -> [usize; 3] {
        let mut sides = [radius; 3];
        if let Geometry::PlaneWise { normal } = self {
            sides[normal] = 0;
        }
        sides
    }

    /// How many axes the filters work in. **Two, in the plane-wise case, and
    /// that is what shrinks the eigenvalue families**: a symmetric 2x2 matrix
    /// has two eigenvalues, so a plane-wise Hessian contributes two channels and
    /// not three.
    fn dimensions(self) -> usize {
        match self {
            Geometry::Volumetric => 3,
            Geometry::PlaneWise { .. } => 2,
        }
    }

    fn normal(self) -> Option<usize> {
        match self {
            Geometry::Volumetric => None,
            Geometry::PlaneWise { normal } => Some(normal),
        }
    }
}

/// The families of the stack, as a set the caller can narrow.
///
/// Named individually rather than offered as one flag because they are not
/// equally expensive and a caller who knows their data often knows which they
/// want — and, more to the point here, because a measurement of the fan-in wants
/// to be able to vary the arm count without changing anything else.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Family {
    /// The image itself, once, at no scale.
    Original,
    Gaussian,
    /// Every unordered pair of the Gaussians, so `C(n,2)` channels and not `n`.
    DifferenceOfGaussians,
    GradientMagnitude,
    LaplacianOfGaussian,
    /// One channel per eigenvalue: three in 3-D, two plane-wise.
    Hessian,
    /// Two integration scales times the eigenvalues: six in 3-D, four
    /// plane-wise.
    StructureTensor,
    /// Minimum, maximum, mean and deviation over a box of radius
    /// `floor(1 + 2 sigma)`.
    Morphological,
}

impl Family {
    /// Labkit's whole stack, in the order the channels come out.
    pub const ALL: [Family; 8] = [
        Family::Original,
        Family::Gaussian,
        Family::DifferenceOfGaussians,
        Family::GradientMagnitude,
        Family::LaplacianOfGaussian,
        Family::Hessian,
        Family::StructureTensor,
        Family::Morphological,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Family::Original => "original",
            Family::Gaussian => "gaussian",
            Family::DifferenceOfGaussians => "dog",
            Family::GradientMagnitude => "gradient",
            Family::LaplacianOfGaussian => "log",
            Family::Hessian => "hessian",
            Family::StructureTensor => "structure",
            Family::Morphological => "morphology",
        }
    }
}

/// One channel: what it is, and the chain that computes it.
pub struct FeatureChannel {
    /// Unique within a stack, and readable: `"hessian/1.6/largest"`. This is the
    /// name a trained forest's split refers to, so it is the thing that has to
    /// stay stable — see [`FeatureStack::channel_names`].
    pub name: String,
    pub family: Family,
    /// The scale this channel was computed at, where it has one. `None` for the
    /// original image; the difference of Gaussians carries the **smaller** of
    /// its pair, with both in the name.
    pub sigma: Option<f64>,
    /// The chain from the input image to this channel's image.
    pub chain: Chain,
}

/// The stack: a scale list, a geometry, and which families are wanted.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureStack {
    sigmas: Vec<f64>,
    geometry: Geometry,
    truncate: f64,
    families: Vec<Family>,
    exact_deviation: bool,
}

impl FeatureStack {
    /// Labkit's stack at the given sigmas, volumetric, every family on.
    ///
    /// `truncate` is 3.0 rather than a parameter of this constructor: it is the
    /// value `ops::ridge` and `ops::smooth` default to and the one every reach
    /// in this crate's tests is quoted against. [`Self::with_truncate`] changes
    /// it, and doing so changes every reach in the stack, which is the reason it
    /// is a separate call rather than a positional argument that could be
    /// mistaken for a scale.
    pub fn labkit(sigmas: &[f64]) -> Result<Self> {
        if sigmas.is_empty() {
            return Err(Error::InvalidArgument(
                "a feature stack needs at least one sigma; with none it is the original \
                 image and nothing else, which is not a stack"
                    .to_string(),
            ));
        }
        for &sigma in sigmas {
            if !(sigma > 0.0) || !sigma.is_finite() {
                return Err(Error::InvalidArgument(format!(
                    "every sigma of a feature stack must be positive and finite; got {sigma}"
                )));
            }
        }
        let mut sorted = sigmas.to_vec();
        sorted.sort_by(f64::total_cmp);
        // Ascending and distinct, so that a difference of Gaussians is
        // `wide - narrow` for every pair without a sign convention to remember,
        // and so that `C(n,2)` counts pairs rather than including a self-pair
        // that is identically zero.
        if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(Error::InvalidArgument(format!(
                "the sigmas {sigmas:?} repeat a value. A difference of Gaussians between a \
                 scale and itself is the zero image, and two identical Gaussian channels are \
                 one channel counted twice — either way the forest is handed a column that \
                 carries nothing."
            )));
        }
        Ok(Self {
            sigmas: sorted,
            geometry: Geometry::Volumetric,
            truncate: 3.0,
            families: Family::ALL.to_vec(),
            exact_deviation: false,
        })
    }

    pub fn with_geometry(mut self, geometry: Geometry) -> Result<Self> {
        if let Geometry::PlaneWise { normal } = geometry {
            if normal > 2 {
                return Err(Error::InvalidArgument(format!(
                    "the plane normal must be an axis, 0, 1 or 2; got {normal}"
                )));
            }
        }
        self.geometry = geometry;
        Ok(self)
    }

    /// Compute the deviation channel with a **two-pass window** rather than the
    /// separable two-moment form.
    ///
    /// The default is the separable form, and the trade is set out on
    /// [`separable_deviation`]: the two-moment identity is `1,000` times cheaper
    /// at Labkit's widest sigma and loses precision when the local mean is large
    /// beside the local spread. This turns it off for a caller whose data is in
    /// that corner — a volume with a large constant offset and a small
    /// modulation is the case — and the cost is the one measured there.
    pub fn with_exact_deviation(mut self, exact: bool) -> Self {
        self.exact_deviation = exact;
        self
    }

    pub fn with_truncate(mut self, truncate: f64) -> Result<Self> {
        if !truncate.is_finite() || truncate <= 0.0 {
            return Err(Error::InvalidArgument(format!(
                "truncate is {truncate}; it is how many standard deviations a Gaussian is cut \
                 off at and must be positive"
            )));
        }
        self.truncate = truncate;
        Ok(self)
    }

    /// Narrow the stack to these families, in [`Family::ALL`]'s order however
    /// they are listed here — so the channel order is a property of the stack
    /// and not of the call that built it.
    pub fn with_families(mut self, families: &[Family]) -> Result<Self> {
        if families.is_empty() {
            return Err(Error::InvalidArgument(
                "a feature stack with no families has no channels".to_string(),
            ));
        }
        self.families = Family::ALL
            .into_iter()
            .filter(|family| families.contains(family))
            .collect();
        Ok(self)
    }

    pub fn sigmas(&self) -> &[f64] {
        &self.sigmas
    }

    pub fn geometry(&self) -> Geometry {
        self.geometry
    }

    pub fn truncate(&self) -> f64 {
        self.truncate
    }

    pub fn families(&self) -> &[Family] {
        &self.families
    }

    /// **How many channels, without building any of them.**
    ///
    /// Closed form, and it exists so that the design document's arithmetic can
    /// be checked against the code rather than against a comment: at five sigmas
    /// in 3-D with every family it is 91, and the test below asserts exactly
    /// that decomposition term by term.
    pub fn len(&self) -> usize {
        let n = self.sigmas.len();
        let eigen = self.geometry.dimensions();
        self.families
            .iter()
            .map(|family| match family {
                Family::Original => 1,
                Family::Gaussian => n,
                Family::DifferenceOfGaussians => n * (n - 1) / 2,
                Family::GradientMagnitude => n,
                Family::LaplacianOfGaussian => n,
                Family::Hessian => n * eigen,
                Family::StructureTensor => n * eigen * INTEGRATION_SCALES.len(),
                Family::Morphological => n * 4,
            })
            .sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The channel names, in channel order, without building the chains.
    ///
    /// **The stable identity of a column.** A trained forest stores split
    /// thresholds against column *indices*; these names are what makes an index
    /// mean something, and what lets a forest trained on one stack be refused
    /// against a different one instead of silently reading the wrong feature.
    pub fn channel_names(&self) -> Result<Vec<String>> {
        Ok(self.channels()?.into_iter().map(|c| c.name).collect())
    }

    /// The branches, in channel order. What a caller hands to
    /// [`Chain::parallel`] alongside their own combine.
    pub fn branches(&self) -> Result<Vec<Chain>> {
        Ok(self.channels()?.into_iter().map(|c| c.chain).collect())
    }

    /// The whole stack joined by `combine` — the shape a plan is built from.
    ///
    /// Fallible for [`Chain::parallel`]'s reasons, and one of its refusals is
    /// worth knowing here: a stack of exactly one channel is not a fan-in, and a
    /// caller who narrowed to `[Family::Original]` gets that error rather than a
    /// degenerate `Parallel` node.
    pub fn into_chain(&self, combine: Box<dyn Combine>) -> Result<Chain> {
        Chain::parallel(self.branches()?, combine)
    }

    /// Every channel, in order: family by family in [`Family::ALL`]'s order,
    /// and within a family scale by scale ascending.
    pub fn channels(&self) -> Result<Vec<FeatureChannel>> {
        let mut channels = Vec::with_capacity(self.len());
        for &family in &self.families {
            match family {
                Family::Original => channels.push(FeatureChannel {
                    name: "original".to_string(),
                    family,
                    sigma: None,
                    chain: Chain::op(VoxelwiseMapOp::identity("original")),
                }),
                Family::Gaussian => {
                    for &sigma in &self.sigmas {
                        channels.push(FeatureChannel {
                            name: format!("gaussian/{sigma}"),
                            family,
                            sigma: Some(sigma),
                            chain: self.smoothed(sigma, "gaussian")?,
                        });
                    }
                }
                Family::DifferenceOfGaussians => {
                    for (index, &narrow) in self.sigmas.iter().enumerate() {
                        for &wide in &self.sigmas[index + 1..] {
                            channels.push(FeatureChannel {
                                name: format!("dog/{narrow}-{wide}"),
                                family,
                                sigma: Some(narrow),
                                // `narrow - wide`, in that branch order. The
                                // sign is a convention and this is Labkit's:
                                // subtracting the blurrier image leaves a
                                // band-pass that is positive on structures
                                // brighter than their surround.
                                chain: Chain::parallel(
                                    vec![
                                        self.smoothed(narrow, "dog.narrow")?,
                                        self.smoothed(wide, "dog.wide")?,
                                    ],
                                    Box::new(ArithmeticCombine::new("dog", Arithmetic::Subtract)),
                                )?,
                            });
                        }
                    }
                }
                Family::GradientMagnitude => {
                    for &sigma in &self.sigmas {
                        channels.push(FeatureChannel {
                            name: format!("gradient/{sigma}"),
                            family,
                            sigma: Some(sigma),
                            chain: Chain::op(GradientMagnitudeOp::new(
                                "gradient",
                                self.geometry.scale(sigma),
                                self.truncate,
                            )?),
                        });
                    }
                }
                Family::LaplacianOfGaussian => {
                    for &sigma in &self.sigmas {
                        channels.push(FeatureChannel {
                            name: format!("log/{sigma}"),
                            family,
                            sigma: Some(sigma),
                            // A sequence, and therefore a reach that **adds**:
                            // the Gaussian's radius and the stencil's one voxel.
                            // `Chain::Sequence` folds that itself, which is the
                            // reason this is two ops rather than one op that
                            // would have had to state the sum by hand.
                            chain: Chain::sequence(vec![
                                self.smoothed(sigma, "log.smooth")?,
                                Chain::op(ConvolveOp::new(
                                    "log.laplacian",
                                    self.laplacian_kernel()?,
                                    // Symmetric, so the two senses agree; stated
                                    // as correlation because that is what the
                                    // stencil is written as.
                                    Sense::Correlate,
                                    Boundary::Clamp,
                                )),
                            ]),
                        });
                    }
                }
                Family::Hessian => {
                    for &sigma in &self.sigmas {
                        for which in self.eigenvalues() {
                            channels.push(FeatureChannel {
                                name: format!("hessian/{sigma}/{}", which.as_str()),
                                family,
                                sigma: Some(sigma),
                                chain: Chain::op(HessianEigenvalueOp::new(
                                    "hessian",
                                    self.geometry.scale(sigma),
                                    self.truncate,
                                    which,
                                )?),
                            });
                        }
                    }
                }
                Family::StructureTensor => {
                    for &sigma in &self.sigmas {
                        for &gamma in INTEGRATION_SCALES {
                            for which in self.eigenvalues() {
                                channels.push(FeatureChannel {
                                    name: format!("structure/{sigma}/g{gamma}/{}", which.as_str()),
                                    family,
                                    sigma: Some(sigma),
                                    chain: Chain::op(StructureTensorOp::new(
                                        "structure",
                                        StructureTensor::at_gamma(
                                            self.geometry.scale(sigma),
                                            gamma,
                                            self.truncate,
                                        )?,
                                        which,
                                    )),
                                });
                            }
                        }
                    }
                }
                Family::Morphological => {
                    for &sigma in &self.sigmas {
                        let radius = self.geometry.radius(morphological_radius(sigma));
                        for (label, rank) in [("min", Extreme::Min), ("max", Extreme::Max)] {
                            channels.push(FeatureChannel {
                                name: format!("morphology/{sigma}/{label}"),
                                family,
                                sigma: Some(sigma),
                                chain: separable_extreme(radius, rank)?,
                            });
                        }
                        channels.push(FeatureChannel {
                            name: format!("morphology/{sigma}/mean"),
                            family,
                            sigma: Some(sigma),
                            chain: separable_mean(radius)?,
                        });
                        channels.push(FeatureChannel {
                            name: format!("morphology/{sigma}/deviation"),
                            family,
                            sigma: Some(sigma),
                            chain: if self.exact_deviation {
                                Chain::op(LocalStatisticOp::new(
                                    "morphology.deviation",
                                    // Spacing one: every voxel gets its own
                                    // window. `ops::local`'s decimation exists
                                    // for background estimation, where the
                                    // answer varies slowly by assumption; a
                                    // classifier feature interpolated between
                                    // samples would hand the forest a smoothed
                                    // version of the statistic it was trained
                                    // on.
                                    LocalStatistic::new(
                                        box_element(radius)?,
                                        [1, 1, 1],
                                        Statistic::Deviation,
                                    )?,
                                ))
                            } else {
                                separable_deviation(radius)?
                            },
                        });
                    }
                }
            }
        }
        debug_assert_eq!(
            channels.len(),
            self.len(),
            "`len` and `channels` disagree, so the closed form is wrong"
        );
        Ok(channels)
    }

    fn smoothed(&self, sigma: f64, name: &'static str) -> Result<Chain> {
        Ok(Chain::op(SmoothOp::new(
            name,
            Gaussian::new(self.geometry.scale(sigma), self.truncate)?,
        )))
    }

    /// Which eigenvalues a channel family emits, which is a question about the
    /// geometry and not about the op: the ops always decompose a 3x3 matrix,
    /// and plane-wise that matrix has a zero row and column, so its smallest
    /// eigenvalue is identically zero and carries nothing.
    ///
    /// **Which one is dropped depends on the sign, and that is why this is not
    /// simply "the last two".** A Hessian is indefinite — a plane-wise one has
    /// eigenvalues that may straddle zero — so the identically-zero eigenvalue
    /// is the *middle* one there, while a structure tensor is positive
    /// semi-definite and its zero is the *smallest*. Taking the same two indices
    /// for both would hand the forest a constant column in one of them.
    fn eigenvalues(&self) -> Vec<Eigenvalue> {
        match self.geometry {
            Geometry::Volumetric => Eigenvalue::ALL.to_vec(),
            Geometry::PlaneWise { .. } => vec![Eigenvalue::Largest, Eigenvalue::Smallest],
        }
    }

    /// The discrete Laplacian: `-2d` at the centre and `+1` at each of its `2d`
    /// face neighbours, where `d` is the number of axes the stack works in.
    ///
    /// Plane-wise this is the five-point stencil and the normal axis has no
    /// members at all, so the kernel's own reach on that axis is zero — which is
    /// the whole point of the mode, and is derived from the element rather than
    /// declared.
    fn laplacian_kernel(&self) -> Result<Kernel> {
        let sides = self.geometry.radius(1);
        let extent = [2 * sides[0] + 1, 2 * sides[1] + 1, 2 * sides[2] + 1];
        let centre = sides;
        let mut weights = vec![0.0; extent[0] * extent[1] * extent[2]];
        let at = |place: [usize; 3]| (place[0] * extent[1] + place[1]) * extent[2] + place[2];
        weights[at(centre)] = -2.0 * self.geometry.dimensions() as f64;
        for axis in 0..3 {
            if Some(axis) == self.geometry.normal() {
                continue;
            }
            for step in [-1isize, 1] {
                let mut place = centre;
                place[axis] = (centre[axis] as isize + step) as usize;
                weights[at(place)] = 1.0;
            }
        }
        Kernel::from_radius(sides, weights)
    }
}

/// The two ends of a box, as the separable form takes them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Extreme {
    Min,
    Max,
}

/// Labkit's rule: a box of radius `floor(1 + 2 sigma)`.
fn morphological_radius(sigma: f64) -> usize {
    (1.0 + 2.0 * sigma).floor() as usize
}

fn box_element(radius: [usize; 3]) -> Result<StructuringElement> {
    StructuringElement::from_size(
        ElementShape::Box,
        [2 * radius[0] + 1, 2 * radius[1] + 1, 2 * radius[2] + 1],
    )
}

/// One axis of a box, as an element. A radius of zero gives the one-voxel
/// element, which every op treats as the identity — so the plane-wise mode drops
/// out with no branch here.
fn line_element(radius: usize, axis: usize) -> Result<StructuringElement> {
    let mut size = [1usize; 3];
    size[axis] = 2 * radius + 1;
    StructuringElement::from_size(ElementShape::Box, size)
}

/// **A box erosion or dilation, as three one-dimensional passes.**
///
/// This is the single most consequential thing in this file and it is worth the
/// space. A minimum over a `(2r+1)^3` box costs `(2r+1)^3` reads per voxel done
/// directly, and at Labkit's widest sigma of 16 that is `67^3 = 300,763`.
/// Measured, on the machine this crate was developed on: **1,005,839 ns per
/// voxel** for one such arm, against 6,528 ns for the whole forest predictor at
/// 100 trees. One feature of ninety-one was a hundred and fifty times the
/// classifier.
///
/// A box is a product of intervals, so the minimum over it is the minimum along
/// x of the minimum along y of the minimum along z — `3(2r+1)` reads instead of
/// `(2r+1)^3`, which is **1,496 times fewer** at `r = 33`. That is not an
/// approximation and it is not a different filter:
///
/// * the identity `min over A x B = min over A of (min over B)` is exact for any
///   totally ordered element type, which is what a `Rank::lowest` reads;
/// * it survives **truncation at the volume boundary**, which is the part worth
///   checking rather than assuming. A box clipped by the volume is still a
///   product of clipped intervals, so the composition clips to exactly the same
///   set of voxels the whole box would have.
///
/// `Chain::sequence` folds the three reaches by addition — `r + r + r` per axis —
/// which is the same halo the single box declared, so nothing about the plan
/// changes except the price.
///
/// **Only the extremes.** A median is not separable and neither is any other
/// interior rank; this is a property of the minimum and the maximum, not of rank
/// filters. `tests/feature_stack.rs` checks the composition against the direct
/// box for both extremes and for a range of radii.
fn separable_extreme(radius: [usize; 3], extreme: Extreme) -> Result<Chain> {
    let mut passes = Vec::new();
    for axis in 0..3 {
        if radius[axis] == 0 {
            continue;
        }
        let element = line_element(radius[axis], axis)?;
        let rank = match extreme {
            Extreme::Min => Rank::lowest(),
            Extreme::Max => Rank::highest(&element),
        };
        passes.push(Chain::op(RankFilterOp::new(
            match extreme {
                Extreme::Min => "morphology.min",
                Extreme::Max => "morphology.max",
            },
            element,
            rank,
        )));
    }
    if passes.is_empty() {
        // Every radius is zero, which is the identity. A `Sequence` of nothing
        // has no defined answer, so the identity is written down.
        return Ok(Chain::op(VoxelwiseMapOp::identity("morphology.identity")));
    }
    Ok(Chain::sequence(passes))
}

/// **A box mean, as three one-dimensional means**, for the same reason and with
/// one caveat the extremes do not have.
///
/// A normalised box kernel is separable, so the composition computes the same
/// quantity — but *not the same bits*: three one-dimensional sums accumulate in
/// a different order than one three-dimensional sum, and floating-point addition
/// is not associative. The difference is at the level of rounding and it is on a
/// feature a forest thresholds, so it cannot change a decision that was not
/// already at the threshold. It is stated because "the same filter" is a claim
/// this crate holds to bits elsewhere, and here it does not.
fn separable_mean(radius: [usize; 3]) -> Result<Chain> {
    let mut passes = Vec::new();
    for axis in 0..3 {
        if radius[axis] == 0 {
            continue;
        }
        passes.push(Chain::op(LocalStatisticOp::new(
            "morphology.mean",
            LocalStatistic::new(
                line_element(radius[axis], axis)?,
                [1, 1, 1],
                Statistic::Mean,
            )?,
        )));
    }
    if passes.is_empty() {
        return Ok(Chain::op(VoxelwiseMapOp::identity("morphology.identity")));
    }
    Ok(Chain::sequence(passes))
}

/// **A box standard deviation, as two separable means.**
///
/// `var = E[x^2] - E[x]^2`, so both moments are box means and both are
/// separable. The chain is a fan-in of two arms — one squaring the input before
/// averaging, one averaging it directly — joined by [`DeviationCombine`].
///
/// # Why this is the default, in one number
///
/// Directly, a deviation over a `(2r+1)^3` box reads `(2r+1)^3` voxels per
/// voxel. At Labkit's widest sigma that arm declared **589,498** against a total
/// of 705,725 for the whole 91-channel stack — 83.5% of it, and the four
/// deviation channels together were 97.8%. This form reads `6(2r+1)` and is
/// about a thousandfold cheaper at that radius.
///
/// # What it costs, stated rather than discovered
///
/// The two-moment identity is the textbook example of catastrophic
/// cancellation. `E[x^2]` and `E[x]^2` are both near `mean^2` and their
/// difference is the variance, so the relative error in the variance is roughly
/// `eps * mean^2 / var` — that is, `eps * (mean / sd)^2`. At a mean of 30,000
/// and a standard deviation of 1 that is `2e-16 * 9e8 = 2e-7` on a variance of
/// 1, which is nothing; at a standard deviation of `0.001` on the same mean it
/// is a 20% error. **A volume with a large constant offset and a small
/// modulation riding on it is the case that breaks it**, which is an ordinary
/// shape for a measured signal rather than a contrived one.
///
/// Three things make this the right default anyway. The reference tools do the
/// same, through integral images, so a caller comparing against ilastik gets the
/// same quantity. The failure is confined to one channel of ninety-one, and a
/// forest that finds it uninformative simply stops splitting on it. And
/// [`FeatureStack::with_exact_deviation`] turns it off, in one call, for a
/// caller who knows their data is in that corner — which is a choice on record
/// rather than a precision quietly given up.
///
/// The result is clamped at zero before the square root: cancellation can make
/// the difference of two nearly equal moments *negative*, and `sqrt` of that is
/// a `NaN` in the output volume, which is neither a diagnosis nor a value.
fn separable_deviation(radius: [usize; 3]) -> Result<Chain> {
    Chain::parallel(
        vec![
            Chain::sequence(vec![
                Chain::op(VoxelwiseMapOp::new("morphology.square", |value| {
                    value * value
                })),
                separable_mean(radius)?,
            ]),
            separable_mean(radius)?,
        ],
        Box::new(DeviationCombine),
    )
}

/// `sqrt(max(0, mean_of_squares - mean^2))`, over exactly two branches in that
/// order.
///
/// A combine of its own rather than an `ArithmeticCombine` chain because the
/// clamp and the square root are the parts that matter, and expressing them as
/// three more voxelwise nodes would have cost two more block buffers to say the
/// same thing.
struct DeviationCombine;

impl Combine for DeviationCombine {
    fn name(&self) -> &'static str {
        "morphology.deviation"
    }

    /// Zero: it reads the voxel it writes, in each operand, and nothing else.
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn slicing(&self) -> Slicing {
        Slicing::Stencil
    }

    fn accepts(&self, inputs: &[Dtype]) -> bool {
        inputs.len() == 2 && inputs.iter().all(|&dtype| dtype == Dtype::F64)
    }

    fn produces(&self, _inputs: &[Dtype]) -> Dtype {
        Dtype::F64
    }

    fn output_shape(&self, inputs: &[[usize; 3]]) -> Result<[usize; 3]> {
        if inputs.len() != 2 || inputs[0] != inputs[1] {
            return Err(Error::InvalidArgument(format!(
                "morphology.deviation joins a mean of squares with a mean and was handed \
                 {inputs:?}"
            )));
        }
        Ok(inputs[0])
    }

    fn apply(&self, inputs: &[&Voxels], out: &mut Voxels, _at: &Anchor) -> Result<()> {
        let shapes: Vec<[usize; 3]> = inputs.iter().map(|input| input.shape()).collect();
        self.output_shape(&shapes)?;
        let squares = inputs[0].view::<f64>()?;
        let means = inputs[1].view::<f64>()?;
        let mut out = out.view_mut::<f64>()?;
        ndarray::Zip::from(&mut out)
            .and(&squares)
            .and(&means)
            .for_each(|slot, &square, &mean| {
                // Clamped before the root: cancellation can put the difference
                // just below zero, and a `NaN` voxel is neither a diagnosis nor
                // a value.
                *slot = (square - mean * mean).max(0.0).sqrt();
            });
        Ok(())
    }
}

/// The two integration scales Labkit takes the structure tensor at, as multiples
/// of the derivative scale. Six channels per scale in 3-D is `2 * 3`, and this
/// is the 2.
pub const INTEGRATION_SCALES: &[f64] = &[1.0, 3.0];
