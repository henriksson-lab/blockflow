// SPDX-License-Identifier: MIT
//
// Original work for this crate. Written from the definitions of the operations,
// not adapted from any implementation of them.
//
// The first ops in this crate that move real data. Until now `probes` held the
// only implementations of `BlockOp`, and they exist to prove the *framework* —
// an identity whose expected output is its input, a window sum that diverges
// when the halo is short. Those are still the right tools for that job. What
// they cannot do is be used, and this module is what a caller composes a chain
// out of.
//
// The families, and what each contributes beyond its arithmetic
// -------------------------------------------------------------
// Read down the right-hand column rather than the left: each row is here
// because it was the first thing that could not be said with what came before
// it, and the bottom two rows are where the `BlockOp` shape itself ran out.
// | module | ops | what it makes expressible |
// |---|---|---|
// | `voxelwise` | a general map, and the connectives over two inputs | the **sink of a diamond**: reach 0, two operands, which nothing here could express |
// | `rank` | an order statistic of a neighbourhood, the median included | a reach derived from a filter size, and a constant algebra that is exact rather than approximate |
// | `morphology` | erode, dilate, open, close | a reach that is **twice** the element, because two of the four are compositions |
// | `local` | windowed mean, deviation and rank on a sample lattice | a reach with **two terms**, and the globally anchored lattice that makes it decomposition-invariant at all |
// | `local` | thresholding against a local statistic | a threshold that varies with position, and inherits every property above |
// | `smooth` | a separable Gaussian | a cost that is **linear in the sum** of the kernel lengths rather than in their product, which the model had no reason to distinguish before |
// | `skeleton` | one thinning sub-iteration, and the sequence of them | an answer that depends on **where** the block is (the parity class is a fact about position), and a `Sequence` whose reach is the fold rather than a declaration |
// | `fill` | hole filling, as two `FragmentOp` phases | the first operation here that **no halo can express**: reachability is transitive over the whole volume, so it is a fragment-and-join rather than a `BlockOp` at all |
// | `regional` | the maxima of a greyscale volume, as the same two phases | the **second** op of that shape, which is what turned one op's internals into `components`: the same program with a different per-label fact, and a seam meeting that compares before it joins |
// | `components` | the union-find, the six-face geometry and the seam walk | nothing on its own — it is the part of `fill` and `regional` that is the *program* rather than the question |
// | `detect` | one point per connected region of a mask, at its centroid | the **producer** the point world had none of, and the first phase pair here that writes no level at all: a `fragments -> fragments` merge whose accumulators are integers, so a component split across four blocks totals *exactly* rather than nearly |
// | `voxelize` | scattered points into a dense volume | a `fragments -> volume` op whose reach is in **blocks** as well as voxels, and an accumulation order that has to be a function of the data rather than of the gather |
// | `sliding` | a windowed statistic over a histogram carried along a scan line | the first op whose kernel has **state between voxels**, so the answer depends on the order voxels are visited in — and the first with a stated *element type* constraint, since a histogram needs a bounded integer domain and refuses a float rather than binning it |
// | `reconstruct` | grey reconstruction, and the h-maxima transform over it | the first `IterativeOp` here: a **fixed point** whose substage count is a function of the data, reached at the external reach of *one* substage — the third answer to transitivity, beside a wide halo and a fragment-and-join |
//
// The shape every op in here has, and why
// ---------------------------------------
// Each operation is a **free function generic over the element type as far as
// the algorithm allows** — a rank filter over `Ord`, morphology over `bool`, a
// comparison over `PartialOrd`, a voxelwise map over nothing at all — with a
// thin `BlockOp` implementation on top that adapts the buffer it is handed to
// it.
//
// The shell is a `BlockOp` for most of them, a `FragmentOp` for the global two
// and an `IterativeOp` for `reconstruct`. The rule is the same in all three
// cases and it is the rule rather than the trait that matters: the algorithm is
// a free function over the narrowest bound it can be written under, and the
// implementation is an adapter that decides which buffer the free function is
// handed. `reconstruct` is the sharpest case, because its shell is a *step* and
// the loop around it belongs to the framework.
//
// That split is not stylistic, and it has now been cashed in. Change 5 of
// `docs/design/BLOCK_OPS.md` §"The combined pass" made the element type a tag
// (`voxels::Voxels`) and the rank 3; it rewrote every adapter in this module and
// **changed no kernel body**. What each adapter does now is one `match` on the
// tag: a `bool` volume reaches `erode_into` with no conversion and no copy,
// where before it was widened to `f64` and narrowed back at 8x the bytes.
//
// What an adapter still declares
// ------------------------------
// `accepts` and `produces` say which element types the shell can bridge — not
// which ones the *kernel* could take. `rank_filter_into` is generic over `Ord`,
// so its shell accepts every integer and both floats; `local_statistic_into` is
// generic over `Copy` with an `f64` accumulator, but `LocalStatistic::
// evaluate_into` is stated in `f64` and widening *that* is kernel work rather
// than shell work, so its shell accepts `f64` and says so here rather than
// pretending otherwise.
//
// Three rules this module holds itself to
// ---------------------------------------
// **`reach` is derived from the parameters and there is no field that sets
// it.** An element of size 7 reaches 3; an opening over it reaches 6; a
// statistic on a lattice of spacing `s` reaches `s` further than its window.
// Every one of those is computed from the parameter it follows from, in the same
// type that holds the parameter. The design's warning is explicit — a reach fed
// by the configured halo makes the guard compare a number against itself — and
// the way to be sure of that is to have nothing to feed it with.
//
// An element of *even* size has no centre voxel, so its reach is asymmetric —
// size 10 reads five below the anchor and four above — and the ops here state
// that per side in `reach_spec` while `reach` stays the wider of the two, which
// `Chain::reach_spec` checks remains a bound. Where a signature can hold only
// one integer per axis (`SubstageOperand`, `FragmentInput`) the wider side is
// declared and the over-fetch is written down at the declaration rather than
// left for a reader to discover.
//
// **`constant_maps_to` is declared only where it is exactly true.** The default
// is `None` and an op that says nothing is never skipped, so silence is safe and
// a wrong declaration is not. That is why the local *mean* declares nothing
// except at zero: `(v + v + ... + v) / m` is not `v` in binary floating point,
// and a block that was skipped would differ from a block that was computed in
// the last bit. A rank statistic selects a value that was read and therefore is
// exact, and says so.
//
// **Edge behaviour is defined, and it is defined at the volume boundary.** Every
// neighbourhood here is resolved against the array it is handed, by default by
// clamping to it. At a real volume boundary that is the whole story: there is
// nothing beyond to read, and the whole-volume reference resolves it identically.
// At a block seam it is *wrong*, deliberately, because a silent wrong answer is
// what the halo guard exists to convert into a loud one.
//
// The separable convolution is the one place where that rule is a **parameter**
// rather than a constant: `ridge::Boundary` names the convention and
// `smooth::Gaussian` and `ridge::ScaleSpace` carry it. Clamping is its default,
// so nothing that predates the choice moved. It is a parameter there and nowhere
// else because that is the only op whose neighbourhood is wide enough for the
// answer at the volume's face to be dominated by what the convention invents —
// a stencil that reaches one voxel gets the same index from both.
//
// Costs
// -----
// See `COST_MEASUREMENT`. Every `cost_per_voxel` in this module is a
// measurement, taken by `ops::cost::measure`, which is runnable.
//
// Four ops measure themselves instead, in their own files, and say so where
// their constants are: `ridge`, `skeleton`, `reconstruct` and `voxelwise`.
// `ops::cost::measure` builds one shared `f64` ramp for every case and consumes
// the result as `f64`, so it can neither feed nor read an op whose input and
// output are masks — and a thinning pass over a ramp does almost nothing, which
// is a measurement of the wrong program rather than a noisy measurement of the
// right one. `reconstruct` is out for a different reason and a harder one: the
// harness prices `Box<dyn BlockOp>` and an iterative op is not a `BlockOp` at
// all, so there is no signature to hand it through.
//
// `voxelwise` is out for a third reason, and it is a warning about this file's
// numbers rather than about that op. Its cases were first added to
// `cost::measure`'s list, and doing so moved the `gaussian smooth` row from 55
// to 73 ns/voxel with `smooth.rs` untouched; rebuilding both at
// `-C codegen-units=1` made them agree at 78. So the neighbourhood rows here
// swing by a third with codegen-unit partitioning, and *any* edit to
// `ops::cost` reshuffles the table four modules' constants were read off.
// `ops::voxelwise::cost_report` therefore has its own case list, and new cases
// anywhere should ask whether they are worth perturbing the old ones.

use crate::error::{Error, Result};

pub mod background;
pub mod components;
pub mod cost;
pub mod deconvolve;
pub mod detect;
pub mod element;
pub mod fill;
pub mod lattice;
pub mod local;
pub mod morphology;
pub mod normalise;
pub mod rank;
pub mod reconstruct;
pub mod regional;
pub mod resample;
pub mod ridge;
pub mod skeleton;
pub mod sliding;
pub mod smooth;
pub mod voxelize;
pub mod voxelwise;

pub use detect::{
    centroid_points, detect_phases, detect_regions, label_regions_into, merge_moments,
    moments_of_labels, owner_of, points_owned_by, LabelRegionsOp, Moments, RegionMoments,
    RegionPointsOp,
};
pub use element::{select_nth, ElementShape, Percentile, Rank, StructuringElement, Total};
pub use fill::{fill_phases, FillHolesOp, LabelBackgroundOp};
pub use lattice::{
    interpolate_block_edge, lattice_interpolate_into, lattice_interpolate_into_with,
    lattice_interpolate_phase, lattice_statistic_into, lattice_statistic_phase,
    statistic_block_edge, LatticeInterpolateOp, LatticeStatisticOp,
};
pub use local::{
    axis_max_distance, local_statistic_into, local_statistic_into_with,
    masked_local_statistic_into, masked_local_statistic_into_with, threshold_against_into,
    AdaptiveThresholdOp, Alignment, EmptyPopulation, Isodata, LocalStatistic, LocalStatisticOp,
    Population, SampleLattice, Sampling, Statistic,
};
pub use morphology::{close_into, dilate_into, erode_into, open_into, Morphology, MorphologyOp};
pub use normalise::{
    bounded_gain_into, bounded_gain_value, normalise_against_into, normalise_value,
    LevelCorrectionOp, LocalContrastOp, LocalGainOp, Removal,
};
pub use rank::{
    masked_rank_filter_into, masked_rank_filter_into_with, rank_filter_f64_into, rank_filter_into,
    ExcludedCentre, MaskedRankFilterOp, RankFilterOp,
};
pub use reconstruct::{
    flooding_bound, h_extrema, reconstruct_step_into, reconstruct_to_fixed_point, HExtremaOp,
    Reconstruction,
};
pub use regional::{
    ascending_neighbours, label_plateaux_into, maxima_from_labels_into, regional_maxima,
    regional_phases, LabelPlateauxOp, RegionalMaximaOp,
};
pub use resample::{
    resample_linear_into, resample_nearest_into, resample_phase, Interpolation, Ratio, Resample,
    ResampleOp,
};
pub use ridge::{
    gaussian_radius, gaussian_smooth_into, gaussian_smooth_into_with, gaussian_weights, hessian_at,
    ridge_response_into, symmetric_eigenvalues, Boundary, EigenResponse, Polarity, RatioResponse,
    Response, RidgeFilterOp, RidgeResponse, ScaleSpace,
};
pub use skeleton::{thin, thinning_pass, thinning_reach, ThinningOp};
pub use sliding::{
    sliding_histogram_into, sliding_histogram_with_plan, BinnedElement, Domain, HistogramQuery,
    RankQuery, ScanPlan, SlidingHistogramOp,
};
pub use smooth::{Gaussian, SmoothOp};
pub use voxelize::{decode_points, encode_points, Point, VoxelizeOp};
pub use voxelwise::{
    combine_into, from_set, is_set, logic_into, map_into, not_into, CombineOp, Compose, Identity,
    Logic, LogicCombine, MapFn, Not, Threshold, ThresholdTest, VoxelwiseMapOp, WidenOp,
    IDENTITY_COST, MAP_COST,
};

/// How the costs in this module were obtained, and what they are relative to.
///
/// `BlockOp::cost_per_voxel` says it must be measured rather than guessed, and
/// `docs/design/BLOCK_OPS.md` is blunt about why: the search returns the optimum
/// for whatever model it is given, and it is the model rather than the search
/// that has been wrong here before.
///
/// **Method.** `ops::cost::measure` runs each op over a fixed volume through the
/// same `BlockOp::apply` the executor calls, takes the best of several
/// repetitions — the contamination on a shared machine is one-sided, so a
/// minimum is the robust statistic and a mean is worthless — and divides by the
/// voxel count. The numbers are then expressed **relative to the voxelwise map**,
/// which is the cheapest thing in the module and is therefore the natural unit:
/// `1.0` means "one voxelwise map's worth of work per voxel", which is also what
/// the trait's default of `1.0` claims. `CostModel::read_cost_per_voxel` and
/// `write_cost_per_voxel` default to the same `1.0`, so a cost of `n` here says
/// this op costs about as much as `n` reads.
///
/// **What is measured and what is a shape.** The per-voxel figure for a
/// neighbourhood op depends on the element, so what is measured is the cost *per
/// element voxel* and the op multiplies by its own element size; a 27-voxel
/// median and a 343-voxel median are not one number. The same for a lattice: the
/// window cost is divided by the samples per voxel, and the interpolation is
/// charged flat.
///
/// **Where these were taken.** On the machine this crate was developed on, with
/// `--release`, one thread, over a 96 x 64 x 64 volume; `ops::cost::report`
/// prints the table and is the way to retake them somewhere else. They are
/// *ratios*, which is what the planner uses them for and what survives a change
/// of machine better than any absolute figure — `docs/design/BLOCK_OPS.md`
/// §"Simulating strategies": trust "A beats B", distrust "A takes 40 minutes".
///
/// **These constants are a seed, and nobody should try to make them precise.**
/// There may be no first run, so a cold planner has to start somewhere, and
/// this table is where it starts. It does not have to be accurate to do that
/// job — it has to have the *ordering* right, which it does. Its absolute scale
/// is known to be wrong: the `MAP_COST`-denominated family understates by about
/// 2.7x since the voxelwise map stopped going through a boxed closure, and the
/// neighbourhood rows swing by about a third on codegen-unit partitioning
/// alone, which is why the paragraph above this one exists. Rescaling them
/// would be chasing a number to a precision the measurement cannot support, and
/// it is unnecessary: [`crate::statistics`] accumulates *nanoseconds per unit of
/// declared cost* from real runs and calibrates the whole model at once, so a
/// systematic factor here is absorbed by evidence from the machine that will do
/// the work rather than corrected by a better guess on the machine that will
/// not. What a wrong constant here still costs is the *relative* weighting
/// between op families, since `CostModel` has one `compute_scale` for all of
/// them — see `statistics::Snapshot::family_spread`, which measures it.
pub const COST_MEASUREMENT: &str = "ops::cost::report";

pub(crate) fn shapes_agree(input: &[usize], out: &[usize], what: &str) -> Result<()> {
    if input != out {
        return Err(Error::ShapeMismatch {
            expected: input.to_vec(),
            got: out.to_vec(),
        })
        .map_err(|err: Error| Error::InvalidArgument(format!("{what}: {err}")));
    }
    Ok(())
}
