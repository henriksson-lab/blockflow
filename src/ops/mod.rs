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
// | `directional` | a whole thinning pass, as one op, and the sequence of them | the first op whose **sub-iteration cannot be a slot**: it reads a second array — the border set taken once per pass — that a `Sequence` has nowhere to thread, so the indivisible unit is the pass and the reach of twelve is a derivation the op has to state rather than a fold |
// | `fill` | hole filling, as two `FragmentOp` phases | the first operation here that **no halo can express**: reachability is transitive over the whole volume, so it is a fragment-and-join rather than a `BlockOp` at all |
// | `regional` | the maxima of a greyscale volume, as the same two phases | the **second** op of that shape, which is what turned one op's internals into `components`: the same program with a different per-label fact, and a seam meeting that compares before it joins |
// | `components` | the union-find, the six-face geometry and the seam walk | nothing on its own — it is the part of `fill` and `regional` that is the *program* rather than the question. Its one *choice* is `Connectivity`, re-exported here: `fill`, `regional` and `detect` each take it and each defaults to face connectivity, so nothing that predates it moved, and the wider ones are the same program over more seam pairs |
// | `detect` | one point per connected region of a mask, at its centroid | the **producer** the point world had none of, and the first phase pair here that writes no image at all: a `fragments -> fragments` merge whose accumulators are integers, so a component split across four blocks totals *exactly* rather than nearly |
// | `voxelize` | scattered points into a dense volume | a `fragments -> volume` op whose reach is in **blocks** as well as voxels, and an accumulation order that has to be a function of the data rather than of the gather |
// | `label` | scattered points into a volume as **names** rather than a sum | the same `fragments -> volume` shape as `voxelize` with a **stated collision rule** instead of an accumulation order: two points meeting on a voxel cannot be added, and `min` over the labels is invariant under every gather order by construction rather than by a sort |
// | `sliding` | a windowed statistic over a histogram carried along a scan line | the first op whose kernel has **state between voxels**, so the answer depends on the order voxels are visited in — and the first with a stated *element type* constraint, since a histogram needs a bounded integer domain and refuses a float rather than binning it |
// | `reconstruct` | grey reconstruction, and the h-maxima transform over it | the first `IterativeOp` here: a **fixed point** whose substage count is a function of the data, reached at the external reach of *one* substage — the third answer to transitivity, beside a wide halo and a fragment-and-join |
// | `configuration` | a mask rewritten by a table indexed on the 3x3x3 neighbourhood | the first op whose rule is **data rather than code** — 2^27 entries the caller supplies — and the first written as *both* shells over one kernel, so what a stated pass count and a fixed point cost differently is a comparison rather than an argument |
// | `watershed` | a cost volume partitioned into one basin per seed | the first op that **declares itself a planning barrier** rather than being one by arithmetic: its answer is a function of one global queue's pop order, so `AxisReach::All` is the honest reach and the cost of saying so is written down as memory per voxel rather than as an adjective |
// | `fft` | a real plane's Fourier transform, and a squared-difference landscape over integer lags through the correlation theorem | the first thing here that is **not an op at all**, and could not be: two inputs of different extents, an output indexed by *lag* rather than by position, and a complex intermediate `Voxels` cannot hold. `watershed` declares the barrier and still fits the shape; this one does not fit the shape, so it is free functions and a plan, and the absent `BlockOp` is the statement |
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

pub mod adjacency;
pub mod background;
pub mod components;
pub mod configuration;
pub mod coordinates;
pub mod cost;
pub mod deconvolve;
pub mod detect;
pub mod directional;
pub mod element;
/// **Not a `BlockOp`, and deliberately.** A Fourier coefficient is a sum over
/// every element of its input, so there is no halo that makes one and no
/// block-local form that approaches one. That module's header says what shape it
/// took instead and why the three obvious ways of wrapping it in this crate's
/// lattice do not exist.
pub mod fft;
pub mod fill;
pub mod label;
pub mod lattice;
pub mod local;
pub mod mixing;
pub mod morphology;
pub mod normalise;
pub mod rank;
pub mod reconstruct;
pub mod regional;
pub mod resample;
pub mod ridge;
pub mod rows;
/// **BSD-3-Clause, not this crate's MIT.** A translation of scikit-image's
/// seeded watershed, kept in a file of its own so the notice travels with it;
/// `watershed` is the MIT shell over it. See that file's header.
pub mod scikitimage_watershed;
pub mod skeleton;
pub mod sliding;
pub mod smooth;
pub mod tabulate;
pub mod voxelize;
pub mod voxelwise;
pub mod walk;
pub mod watershed;

pub use adjacency::{
    adjacent_pair_rows, adjacent_pairs, adjacent_pairs_into, adjacent_pairs_phase, collect_pairs,
    empty_pairs, encode_adjacent_pairs, forward_offsets, merge_pairs, pair_schema,
    walk_adjacent_pairs, AdjacentPairsOp, Pair, HIGHER_COLUMNS,
};
/// The only thing in `components` a *caller* chooses rather than a builder of
/// ops uses. The rest of that module stays behind its own path, because it is
/// machinery rather than surface.
///
/// `fill`, `regional` and `detect` each take one, through a `connecting` builder
/// on both of their phases and through their `append_connected` shorthand, and
/// each defaults to [`Connectivity::Faces`]. They are three separate choices and
/// not one: `fill`'s names the **background**'s adjacency and `detect`'s the
/// **foreground**'s, and the complementary-pair convention deliberately pairs a
/// narrow one with a wide one. `components`'s own header has the table.
pub use components::Connectivity;
pub use configuration::{
    configuration_bit, configuration_index_at, configuration_pass_into, configuration_passes_into,
    configuration_to_fixed_point, cost_report as configuration_cost_report,
    ConfigurationFixedPointOp, ConfigurationPassOp, ConfigurationTable, ConfigurationTemplate,
    CENTRE_BIT, CONFIGURATION_BITS, CONFIGURATION_COUNT, PASS_COST,
};
pub use coordinates::{
    block_base_indices, blocks_concatenate_in_order, collect_coordinates, coordinate_schema,
    empty_coordinates, encode_set_voxels, merge_coordinates, set_voxel_rows, set_voxels,
    set_voxels_into, set_voxels_phase, SetVoxelsOp,
};
pub use detect::{
    centroid_points, detect_phases, detect_regions, detect_regions_with, label_regions_into,
    label_regions_into_with, merge_moments, merge_moments_with, moments_of_labels, owner_of,
    points_owned_by, LabelRegionsOp, Moments, RegionMoments, RegionPointsOp,
};
pub use directional::{
    border_mask, clear_faces, directional_pass, directional_pass_into, directional_passes_into,
    directional_reach, directional_sub_iteration_into, directional_thin,
    directional_to_fixed_point, faces_are_clear, sub_iteration_sources, DirectionalPassOp,
    DIRECTIONAL_PASS_COST, SUB_ITERATIONS,
};
pub use element::{
    select_nth, ElementShape, Percentile, Rank, StepOrigin, StructuringElement, Total,
};
pub use fft::{
    correlate_direct, minimal_wrap_free_length, next_smooth_length, spectrum_width,
    squared_difference_direct, Complex, Correlation2, Landscape, Padding, RealTransform2,
    ShiftWindow, Spectrum, SquaredDifference, TransformBackend,
};
pub use fill::{
    agree_on_connectivity, fill_phases, label_background_into_with, merge_faces_with, FillHolesOp,
    LabelBackgroundOp,
};
pub use label::{
    label_ceiling, label_of, label_points_into, labelled_points, LabelPointsOp, MAX_EXACT_LABEL,
};
pub use lattice::{
    interpolate_block_edge, lattice_interpolate_into, lattice_interpolate_into_with,
    lattice_interpolate_phase, lattice_statistic_into, lattice_statistic_phase,
    statistic_block_edge, LatticeInterpolateOp, LatticeStatisticOp,
};
pub use local::{
    axis_max_distance, local_statistic_into, local_statistic_into_narrowed,
    local_statistic_into_with, masked_local_statistic_into, masked_local_statistic_into_narrowed,
    masked_local_statistic_into_with, threshold_against_into, AdaptiveThresholdOp, Alignment,
    EmptyPopulation, Isodata, LatticeNarrowing, LocalStatistic, LocalStatisticOp, Narrowing,
    Population, Rounding, SampleLattice, Sampling, Statistic,
};
pub use mixing::{LinearMap, TupleKernel, TupleOp};
pub use morphology::{
    close_into, close_into_at, dilate_into, dilate_into_at, erode_into, erode_into_at, open_into,
    open_into_at, Morphology, MorphologyOp,
};
pub use normalise::{
    bounded_gain_into, bounded_gain_value, normalise_against_into, normalise_value,
    LevelCorrectionOp, LocalContrastOp, LocalGainOp, Removal,
};
pub use rank::{
    masked_rank_filter_into, masked_rank_filter_into_at, masked_rank_filter_into_with,
    rank_filter_f64_into, rank_filter_f64_into_at, rank_filter_into, rank_filter_into_at,
    ExcludedCentre, MaskedRankFilterOp, RankFilterOp,
};
pub use reconstruct::{
    flooding_bound, h_extrema, reconstruct_step_into, reconstruct_step_into_at,
    reconstruct_to_fixed_point, HExtremaOp, Reconstruction,
};
pub use regional::{
    ascending_neighbours, ascending_neighbours_with, label_plateaux_into, label_plateaux_into_with,
    maxima_from_labels_into, merge_plateaux_with, regional_maxima, regional_maxima_with,
    regional_phases, LabelPlateauxOp, RegionalMaximaOp,
};
pub use resample::{
    resample_linear_into, resample_linear_into_with, resample_nearest_into,
    resample_nearest_into_with, resample_phase, Interpolation, OutputExtent, Ratio, Resample,
    ResampleOp,
};
pub use ridge::{
    gaussian_radius, gaussian_smooth_into, gaussian_smooth_into_with, gaussian_weights, hessian_at,
    ridge_response_into, symmetric_eigenvalues, Boundary, EigenResponse, Polarity, RatioResponse,
    Response, RidgeFilterOp, RidgeResponse, ScaleSpace,
};
pub use rows::{
    collect_rows, filter_blob, filter_into, gather_blob, gather_into, gathered_schema, merge_rows,
    scale_blob, scale_into, scaled_at, scaled_bound, scaled_index, value_at, walk_rows, ColumnTest,
    FilterRowsOp, GatherRowsOp, Limit, RowFilter, RowStreams, RowValues, ScaleRowsOp,
};
pub use skeleton::{thin, thinning_pass, thinning_reach, ThinningOp};
pub use sliding::{
    sliding_histogram_into, sliding_histogram_with_plan, BinnedElement, Domain, HistogramQuery,
    RankQuery, ScanPlan, SlidingHistogramOp,
};
pub use smooth::{Gaussian, SmoothOp};
pub use tabulate::{
    append_tabulate_phases, collect_tabulation, decode_partial, encode_partial, merge_tabulation,
    region_values, tabulate_phases, tabulation_schema, FixedPoint, MergeTabulationOp, RegionValues,
    TabulateValuesOp, Tally,
};
pub use voxelize::{decode_points, encode_points, Point, VoxelizeOp};
pub use voxelwise::{
    combine_into, from_set, is_set, logic_into, map_into, not_into, CombineOp, Compose, Identity,
    Logic, LogicCombine, MapFn, NarrowOp, Not, Threshold, ThresholdTest, VoxelwiseMapOp, WidenOp,
    IDENTITY_COST, MAP_COST,
};
pub use walk::{
    walk_blob, walk_from, walk_into, walk_schema, walked_distance, OffsetSequence, OffsetWalkOp,
};
pub use watershed::{
    cost_report as watershed_cost_report, seeded_watershed, seeded_watershed_into,
    seeded_watershed_into_reporting_peak, SeededWatershedOp, Separation, WATERSHED_COST,
    WATERSHED_LINE_COST,
};

// Appended rather than filed alphabetically, because this list is shared and an
// append is the edit that does not move anybody else's lines.
/// **The exact Euclidean distance transform**, as three separable whole-axis
/// sweeps and a pointwise finish. `ops::watershed`'s "a caller who wants a
/// distance-transform watershed passes the distance" is what this supplies.
pub mod distance;
pub use distance::{
    append_to as append_distance, brute_force_distance, chamfer_distance,
    cost_report as distance_cost_report, distance_transform, plan as distance_plan,
    seed as seed_distance_field, squared_distance_transform, sweep_axis as sweep_distance_axis,
    sweep_grid as distance_sweep_grid, working_set_bytes as distance_working_set_bytes,
    DistanceFinishOp, DistanceParams, DistanceSweepOp, Unbounded, DISTANCE_FINISH_COST,
    DISTANCE_SEED_SWEEP_COST, DISTANCE_SWEEP_COST,
};

// Appended, for the reason the block above it is: this list is shared today and
// an append is the edit that moves nobody else's lines.
/// **Convolution with a caller-supplied kernel**, in a caller-supplied sense
/// (correlation or convolution, named rather than assumed) and a caller-supplied
/// boundary convention. `ops::smooth` is the Gaussian; this is the general one.
pub mod convolve;
pub use convolve::{
    convolve_into, cost_report as convolve_cost_report, ConvolveOp, Kernel, Sense,
    CONVOLVE_COST_PER_TAP,
};
/// The arithmetic and selection sinks of a diamond, beside `voxelwise`'s
/// Boolean ones: add, subtract, multiply, divide, per-voxel minimum and maximum
/// between two images.
pub use voxelwise::{arithmetic_into, selection_into, Arithmetic, ArithmeticCombine, Arithmetical};

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

// The **second-moment half of `ops::tabulate`**, appended as its own `pub use`
// rather than folded into the `tabulate` list above: three workers are landing
// ops in this file today, and a line nobody else has to touch is a line nobody
// else can conflict with. `RegionShape`/`region_shape` are the shape reading of
// a tabulated row — the label volume's own measurement, over every voxel and at
// no scale — and `PrincipalAxes` is what its six `CENTRAL` columns decompose to.
pub use tabulate::{
    collect_shapes, from_signed_column, region_shape, signed_column, PrincipalAxes, RegionShape,
    AXIS_SEPARATION, CENTRAL, PAIRS,
};
