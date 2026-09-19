// SPDX-License-Identifier: MIT
//
// Original work for this crate. Written from the definitions of the operations,
// not adapted from any implementation of them.
//
// The ops in this crate that move real data. `probes` holds the other
// implementations of `BlockOp` and they exist to prove the *framework* — an
// identity whose expected output is its input, a window sum that diverges when
// the halo is short. This module is what a caller composes a chain out of.
//
// The families, and what each contributes beyond its arithmetic
// -------------------------------------------------------------
// Read down the right-hand column rather than the left: each row is here
// because it was the first thing that could not be said with what came before
// it, and the bottom two rows are where the `BlockOp` shape itself ran out.
//
// **It is a narrative of firsts and not an inventory, and the difference has
// cost something.** A family that added no new expressiveness is legitimately
// absent — but so, for a while, was one that added a great deal: the row world
// below had no entry at all while `coordinates`, `rows` and `tabulate` were
// three of the largest modules here, so an op belonging to it had nowhere to be
// announced and the missing *producer* went unnoticed. If a module here is a
// first, it needs a row; if it is not, its absence is the statement, and saying
// which is this list's job rather than a reader's.
// | module | ops | what it makes expressible |
// |---|---|---|
// | `voxelwise` | a general map, and the connectives over two inputs | the **sink of a diamond**: reach 0, two operands, which nothing here could express |
// | `rank` | an order statistic of a neighbourhood, the median included | a reach derived from a filter size, and a constant algebra that is exact rather than approximate |
// | `morphology` | erode, dilate, open, close | a reach that is **twice** the element, because two of the four are compositions |
// | `background` | a large-scale background estimate, and its removal | **not an op, and that is the entry.** There is no sweep, no window and no accumulator in the file: an opening large enough that no object survives it, subtracted from the original, is `morphology` and `voxelwise` already — so what this contributes is a `Chain` and a combine step, and its header says outright that *"every piece of the arithmetic already existed; what did not exist was the **shape**"*. The first row here whose answer to "what does it make expressible" is a composition rather than a kernel, which is why it returns a `Chain` and not a `BlockOp` |
// | `local` | windowed mean, deviation and rank on a sample lattice | a reach with **two terms**, and the globally anchored lattice that makes it decomposition-invariant at all |
// | `local` | thresholding against a local statistic | a threshold that varies with position, and inherits every property above |
// | `normalise` | removing a locally estimated level, and dividing by a locally estimated spread | the same estimator as `local`, called and not reimplemented, with a different combination step — and the first op that runs it **twice**, a centre and a spread on two independent lattices, which makes its reach a **maximum over two two-term reaches** rather than one. Nothing before it had a reach that was a fold over two derivations of the same kind |
// | `smooth` | a separable Gaussian | a cost that is **linear in the sum** of the kernel lengths rather than in their product, which the model had no reason to distinguish before |
// | `convolve` | a linear filter with a **caller-supplied** kernel | the first op whose kernel is a **parameter rather than a derivation**. `smooth` is the other linear filter and its weights come from a sigma, so a difference of Gaussians, a Laplacian, a Sobel and every hand-rolled stencil were one kernel each and unreachable. It is also the first op with a **parameter that has no default** — `Sense`, because correlation and convolution differ by a reflection, libraries disagree about which name computes which, and a crate that picked one silently would be wrong for half its readers on exactly the symmetric kernels everybody tests with |
// | `ridge` | a multi-scale second-derivative structure filter | the first op whose answer at one voxel is a **maximum over several scales**, and the first that needs an eigen-decomposition to produce one number. The scale-normalisation term is the part worth the row: without `sigma^gamma` the derivatives of a wider Gaussian are uniformly smaller and the maximum always chooses the narrowest scale, *"which is not a preference but an artefact"* — the first place here where an op has to correct for its own parameterisation before a comparison across parameters means anything |
// | `distance` | the exact Euclidean distance transform | the first op whose reach is [`crate::reach::AxisReach::All`] **on one axis at a time** — three independent 1-D sweeps, so it is a planning barrier per axis rather than for the whole op, which `watershed` below is not. It is also the only module here that ships a **deliberately wrong implementation as a control**: `chamfer_distance` is exact along the directions its offsets can represent and wrong in between, and `tests/distance_transform.rs` measures both halves on a fixture chosen so they differ. A parity test on a centred ball cannot tell the two apart, which is why the oblique slab is the fixture that ships |
// | `resample` | resizing by an exact rational factor per axis | the first op here whose **output image is a different shape from its input image** — the capability `docs/design/BLOCK_OPS.md` opened and left unused until this. Everything above it writes the extent it was handed |
// | `skeleton` | one thinning sub-iteration, and the sequence of them | an answer that depends on **where** the block is (the parity class is a fact about position), and a `Sequence` whose reach is the fold rather than a declaration |
// | `directional` | a whole thinning pass, as one op, and the sequence of them | the first op whose **sub-iteration cannot be a slot**: it reads a second array — the border set taken once per pass — that a `Sequence` has nowhere to thread, so the indivisible unit is the pass and the reach of twelve is a derivation the op has to state rather than a fold |
// | `fill` | hole filling, as two `FragmentOp` phases | the first operation here that **no halo can express**: reachability is transitive over the whole volume, so it is a fragment-and-join rather than a `BlockOp` at all |
// | `regional` | the maxima of a greyscale volume, as the same two phases | the **second** op of that shape, which is what turned one op's internals into `components`: the same program with a different per-label fact, and a seam meeting that compares before it joins |
// | `components` | the union-find, the six-face geometry and the seam walk | nothing on its own — it is the part of `fill` and `regional` that is the *program* rather than the question. Its one *choice* is `Connectivity`, re-exported here: `fill`, `regional` and `detect` each take it and each defaults to face connectivity, so nothing that predates it moved, and the wider ones are the same program over more seam pairs |
// | `detect` | one point per connected region of a mask, at its centroid | the **producer** the point world had none of, and the first phase pair here that writes no image at all: a `fragments -> fragments` merge whose accumulators are integers, so a component split across four blocks totals *exactly* rather than nearly |
// | `coordinates`, `rows`, `tabulate` | a coordinate list out of a mask; the maps and the grouped reduction over rows; a statistic per label read out of a volume | the first ops whose unit of decomposition is a **range of rows** rather than a neighbourhood of voxels — and where a halo is a **defect rather than a cost**, which is the reverse of everything above: an overlapping block recomputes, an overlapping row range emits a row *twice* and nothing downstream can tell that from a real row. With them the answer stops being an image, and the order stops being derivable from the assignment: a base index per block can only express a block-major order and a table's canonical order interleaves the blocks, so every row carries its own coordinate and the order is restored at the merge. `rows::RowSourceOp` and `crate::points::PointSourceOp` are the phase-0 producers that let a plan **start** from a table or a point set rather than derive one from an image — the half this family went without while three callers, one of them this crate's own test suite, wrote it for themselves; `docs/ops-survey/README.md` G17 |
// | `adjacency` | a mask in, every adjacent **pair** of set voxels out | the first op whose row is a **relation rather than a thing**. `coordinates` emits one row per set voxel; this emits one per touching pair, and the two compose over the same mask with nothing needed to relate them. What that costs is an ownership rule with no centroid to appeal to — a pair can straddle a seam, so the block whose core holds the lexicographically lower endpoint emits it, and `tests/adjacent_pairs.rs` pins a fixture that puts pairs across every seam *before* anything else asserts against it |
// | `walk` | rows in, the same rows with one more column: a distance read along a fixed path | the first row op that reads a **volume along a path** rather than at the row's own voxel. `ops::rows`' gather samples one voxel per row, so its reach is zero by construction; this walks an `OffsetSequence` until a `Limit` fails, so its reach is the **longest offset in the list** and is a property of a parameter rather than of the op |
// | `voxelize` | scattered points into a dense volume | a `fragments -> volume` op whose reach is in **blocks** as well as voxels, and an accumulation order that has to be a function of the data rather than of the gather |
// | `label` | scattered points into a volume as **names** rather than a sum | the same `fragments -> volume` shape as `voxelize` with a **stated collision rule** instead of an accumulation order: two points meeting on a voxel cannot be added, and `min` over the labels is invariant under every gather order by construction rather than by a sort |
// | `sliding` | a windowed statistic over a histogram carried along a scan line | the first op whose kernel has **state between voxels**, so the answer depends on the order voxels are visited in — and the first with a stated *element type* constraint, since a histogram needs a bounded integer domain and refuses a float rather than binning it |
// | `reconstruct` | grey reconstruction, and the h-maxima transform over it | the first `IterativeOp` here: a **fixed point** whose substage count is a function of the data, reached at the external reach of *one* substage — the third answer to transitivity, beside a wide halo and a fragment-and-join |
// | `deconvolve` | iterative deconvolution against a **known** kernel | the first op whose reach is a function of a **parameter the caller chooses**: `2r` per axis per iteration, `2rn` after `n` of them. That is the price of declining the frequency-domain family, whose reach spans the volume and is therefore `AxisReach::All` by type — a single block, in memory at once, on one worker. Not a second `reconstruct`: that one is an `IterativeOp` whose substage count is data-dependent and reached at the external reach of *one* substage; this is a `BlockOp` that folds a **stated** iteration count into its own halo |
// | `configuration` | a mask rewritten by a table indexed on the 3x3x3 neighbourhood | the first op whose rule is **data rather than code** — 2^27 entries the caller supplies — and the first written as *both* shells over one kernel, so what a stated pass count and a fixed point cost differently is a comparison rather than an argument |
// | `mixing` | a K-ary reach-0 shell (`TupleOp`), its kernel trait and the per-voxel matrix | the first op parameterised by **arity** rather than by a window — K co-located arrays in, K' out, every value read at the voxel it is written to. A window is what `Reach` describes and needs an axis to be stated over; arity needs only somewhere for the operands to come from, which is why this became reachable when a run could be handed images it did not compute. This is the register's G10 closed |
// | `watershed` | a cost volume partitioned into one basin per seed | the first op that **declares itself a planning barrier** rather than being one by arithmetic: its answer is a function of one global queue's pop order, so `AxisReach::All` is the honest reach and the cost of saying so is written down as memory per voxel rather than as an adjective |
// | `random_walker` | seeded random-walker probabilities from an intensity volume and seed image | the first volume op whose answer is a **sparse linear solve** assembled from block fragments: row IDs are a planner-visible image, sparse rows are emitted per block, and the barrier solve writes a dense probability image back through those row IDs |
// | `level_set` | signed-distance initialisation and geodesic active-contour evolution | the first op whose natural shape is an explicit PDE step over a running scalar field and a separate fixed image. The resident kernel is the reference; the one-step shell is a `BlockOp` with two-voxel reach, while the multi-step phase waits for `IterativeOp` to name fixed source images rather than only the phase input |
// | `fft` | a real plane's Fourier transform, and a squared-difference landscape over integer lags through the correlation theorem | the first thing here that is **not an op at all**, and could not be: two inputs of different extents, an output indexed by *lag* rather than by position, and a complex intermediate `Voxels` cannot hold. `watershed` declares the barrier and still fits the shape; this one does not fit the shape, so it is free functions and a plan, and the absent `BlockOp` is the statement |
// | `align` | block-reduced fitting of a 3-D coordinate map between two scalar volumes | the first op whose output is a small global state rather than a volume or row set. Each iteration maps blocks to local evidence, reduces it, and performs one global update before the next iteration. |
//
// Four modules here have no row, and this is the list saying which and why —
// under the rule above, an absence has to be a statement rather than a gap in
// somebody's attention. `element` is the neighbourhood and the order statistic
// that `rank`, `morphology` and `local` all share, so a row for it would be a
// row for a parameter three families already carry. `cost` is where the
// measured `cost_per_voxel` figures come from and makes nothing expressible.
// `lattice` ships two ops but they are `local`'s fused statistic **split into
// two phases** so its two dependencies can be priced separately — the same
// answer, made visible, which is a fact about planning rather than about what
// can be said. `scikitimage_watershed` is a translation carrying its own
// BSD-3-Clause notice and exists to be compared against, not composed with.
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

use ndarray::{Array3, ArrayView3, ArrayViewMut3};

use crate::assemble::ImageId;
use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::fragment::SourceBlocks;
use crate::op::{SourceInput, SourceInputs};
use crate::reach::Reach;
use crate::region::Region;
use crate::voxels::Voxels;

macro_rules! dispatch_f64_input {
    ($input:expr, $f16_error:expr, |$view:ident| $body:block) => {{
        let input = $input;
        match input.dtype() {
            Dtype::U8 => {
                let $view = input.view::<u8>()?;
                $body
            }
            Dtype::U16 => {
                let $view = input.view::<u16>()?;
                $body
            }
            Dtype::U32 => {
                let $view = input.view::<u32>()?;
                $body
            }
            Dtype::I8 => {
                let $view = input.view::<i8>()?;
                $body
            }
            Dtype::I16 => {
                let $view = input.view::<i16>()?;
                $body
            }
            Dtype::I32 => {
                let $view = input.view::<i32>()?;
                $body
            }
            Dtype::F32 => {
                let $view = input.view::<f32>()?;
                $body
            }
            Dtype::F64 => {
                let $view = input.view::<f64>()?;
                $body
            }
            Dtype::Bool | Dtype::U64 | Dtype::I64 => {
                let widened = input.widened();
                let $view = widened.view();
                $body
            }
            Dtype::F16 => Err($f16_error),
        }
    }};
}

pub(crate) fn apply_float_detour(
    input: &Voxels,
    out: &mut Voxels,
    run: impl FnOnce(ArrayView3<'_, f64>, ArrayViewMut3<'_, f64>) -> Result<()>,
) -> Result<()> {
    match input.dtype() {
        Dtype::F64 => run(input.view::<f64>()?, out.view_mut::<f64>()?),
        Dtype::F32 => {
            let widened = input.view::<f32>()?.mapv(f64::from);
            let mut result = Array3::zeros(widened.raw_dim());
            run(widened.view(), result.view_mut())?;
            let mut out = out.view_mut::<f32>()?;
            ndarray::Zip::from(&mut out)
                .and(&result)
                .for_each(|slot, &value| *slot = value as f32);
            Ok(())
        }
        dtype => Err(Error::InvalidArgument(format!(
            "expected float32 or float64 input, got {}",
            dtype.numpy_name()
        ))),
    }
}

pub(crate) fn accepts_mask_carrier(dtype: Dtype) -> bool {
    matches!(dtype, Dtype::Bool | Dtype::F64)
}

pub(crate) fn apply_mask_carrier(
    input: &Voxels,
    out: &mut Voxels,
    run: impl FnOnce(ArrayView3<'_, bool>, ArrayViewMut3<'_, bool>) -> Result<()>,
) -> Result<()> {
    match input.dtype() {
        Dtype::Bool => run(input.view::<bool>()?, out.view_mut::<bool>()?),
        Dtype::F64 => {
            let mask = input.view::<f64>()?.mapv(voxelwise::is_set);
            let mut result = Array3::from_elem(mask.raw_dim(), false);
            run(mask.view(), result.view_mut())?;
            let mut out = out.view_mut::<f64>()?;
            ndarray::Zip::from(&mut out)
                .and(&result)
                .for_each(|slot, &value| *slot = voxelwise::from_set(value));
            Ok(())
        }
        dtype => Err(Error::InvalidArgument(format!(
            "expected bool or float64 mask carrier, got {}",
            dtype.numpy_name()
        ))),
    }
}

pub(crate) fn expect_colocated(
    what: impl FnOnce() -> String,
    reference: &Voxels,
    operand: &Voxels,
) -> Result<()> {
    expect_extent(what, reference.shape(), operand.shape())
}

pub(crate) fn expect_extent(
    what: impl FnOnce() -> String,
    expected: [usize; 3],
    got: [usize; 3],
) -> Result<()> {
    if got == expected {
        return Ok(());
    }
    Err(Error::InvalidArgument(what()))
}

pub mod adjacency;
pub mod align;
pub mod background;
pub mod boundaries;
pub mod classify;
pub mod cleanup;
pub mod components;
pub mod configuration;
pub mod coordinates;
pub mod cost;
pub mod deconvolve;
pub mod detect;
pub mod directional;
pub mod edges;
pub mod element;
pub mod exposure;
pub mod features;
/// **Not a `BlockOp`, and deliberately.** A Fourier coefficient is a sum over
/// every element of its input, so there is no halo that makes one and no
/// block-local form that approaches one. That module's header says what shape it
/// took instead and why the three obvious ways of wrapping it in this crate's
/// lattice do not exist.
pub mod fft;
pub mod fill;
pub(crate) mod histogram;
pub mod interest;
pub mod label;
pub mod lattice;
pub mod level_set;
pub mod local;
pub mod measure;
pub mod mixing;
pub mod morphology;
pub mod normalise;
pub mod random_walker;
pub mod rank;
pub mod rasterise;
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
pub mod structure_tensor;
pub mod tabulate;
pub mod threshold;
pub mod transform;
pub mod voxelize;
pub mod voxelwise;
pub mod walk;
pub mod watershed;

pub use adjacency::{
    adjacent_pair_rows, adjacent_pairs, adjacent_pairs_into, adjacent_pairs_phase, collect_pairs,
    empty_pairs, encode_adjacent_pairs, forward_offsets, merge_pairs, pair_schema,
    walk_adjacent_pairs, AdjacentPairsOp, Pair, HIGHER_COLUMNS,
};
pub use boundaries::{
    boundary_points, find_boundaries_into, labelled_boundary_points, FindBoundariesOp,
    LabelBoundaryPoint,
};
pub use classify::{
    gather_samples, predict_workflow, sample_workflow, samples_from_rows, train_workflow, ClassMap,
    ForestPredictor, LabelIndex, Prediction, SampleCombine,
};
pub use cleanup::{
    append_clear_border_on_axes_phases, append_clear_border_phases, append_expand_labels_phase,
    append_fill_label_holes_2d_by_label_phase, append_filter_labels_by_size_phases,
    append_filter_labels_touching_border_on_axes_phase, append_object_distance_prune_phases,
    append_remove_small_holes_phases, append_remove_small_objects_phases, clear_border_into,
    clear_border_on_axes_into, expand_labels_into, fill_label_holes_2d_by_label_into,
    filter_labels_touching_border_on_axes_into, object_distance_prune_set,
    prune_by_object_distance_into, remove_small_holes_into, remove_small_objects_into,
    ApplyComponentMaskOp, ApplyLabelSizeFilterOp, ApplyObjectDistancePruneOp,
    ApplyRemoveSmallHolesOp, ExpandLabelsOp, FillLabelHoles2dByLabelOp,
    FilterLabelsTouchingBorderOnAxesOp, LabelBackgroundRegionsOp, LabelSizeCountsOp,
    ObjectDistanceSamplesOp,
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
pub use components::{decode_block_flags_for, encode_block_flags, Connectivity, Merge};
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
pub use edges::{
    append_canny_edges_phases, append_hysteresis_threshold_phases, canny_edges_into,
    canny_response_into, hysteresis_threshold_into, non_maximum_suppression_into,
    ApplyHysteresisOp, CannyResponseOp, HysteresisLabelsOp,
};
pub use element::{
    select_nth, ElementShape, Percentile, Rank, StepOrigin, StructuringElement, Total,
};
pub use exposure::{
    append_equalize_adapthist_phases, append_equalize_histogram_phases, equalize_adapthist_into,
    equalize_histogram, is_low_contrast, match_histogram, ClaheTileSamplesOp, EqualizeAdapthistOp,
    EqualizeHistogramOp,
};
pub use features::{Family, FeatureChannel, FeatureStack, Geometry};
pub use fft::{
    correlate_direct, minimal_wrap_free_length, next_smooth_length, spectrum_width,
    squared_difference_direct, Complex, Correlation2, Landscape, Padding, RealTransform2,
    ShiftWindow, Spectrum, SquaredDifference, TransformBackend,
};
pub use fill::{
    agree_on_connectivity, fill_phases, label_background_into_with, merge_faces_with, FillHolesOp,
    LabelBackgroundOp,
};
pub use interest::{
    append_response_peak_table_phases, difference_of_gaussians_response_into,
    hessian_determinant_response_into, laplacian_of_gaussian_response_into, response_peak_points,
    response_peak_schema, BlobDetection, BlobDetector, BlobResponse, BlobResponseOp, BlobScale,
    ResponsePeakRowsOp, RESPONSE_COLUMN,
};
pub use label::{
    label_ceiling, label_of, label_points_into, labelled_points, LabelPointsOp, MAX_EXACT_LABEL,
};
pub use lattice::{
    interpolate_block_edge, lattice_interpolate_into, lattice_interpolate_into_with,
    lattice_interpolate_phase, lattice_statistic_into, lattice_statistic_phase,
    statistic_block_edge, LatticeInterpolateOp, LatticeStatisticOp,
};
pub use level_set::{
    append_geodesic_level_set_phases, chan_vese_level_set_into, chan_vese_level_set_means,
    chan_vese_level_set_reporting_into, chan_vese_level_set_step_into, geodesic_level_set_into,
    geodesic_level_set_reporting_into, geodesic_level_set_step_into, level_set_mask_into,
    signed_distance_level_set, ChanVeseLevelSetConfig, ChanVeseMeans, GeodesicLevelSetConfig,
    GeodesicLevelSetStepOp, LevelSetReport, GEODESIC_LEVEL_SET_STEP_COST,
    MAX_EXPLICIT_LEVEL_SET_DT,
};
pub use local::{
    axis_max_distance, local_statistic_into, local_statistic_into_narrowed,
    local_statistic_into_with, masked_local_statistic_into, masked_local_statistic_into_narrowed,
    masked_local_statistic_into_with, threshold_against_into, AdaptiveThresholdOp, Alignment,
    EmptyPopulation, Isodata, LatticeNarrowing, LocalStatistic, LocalStatisticOp, Narrowing,
    Niblack, Population, Rounding, SampleLattice, Sampling, Sauvola, Statistic,
};
pub use measure::{
    assert_boundary_measurement_decomposition_invariant, assert_boundary_measurement_invariant,
    assert_custom_boundary_builder_invariant, assert_custom_region_builder_invariant,
    assert_object_measurement_invariant, assert_region_measurement_decomposition_invariant,
    assert_region_measurement_invariant, auto_distribution_measurement_schema,
    auto_distribution_measurement_schema_set, auto_distribution_measurements,
    boundary_distance_relationship_measurement_schema, boundary_measurement_schema,
    centroid_neighbor_measurement_schema, centroid_relationship_measurement_schema,
    collect_auto_distribution_measurements, collect_auto_distribution_measurements_set,
    collect_auto_distribution_rows_set, collect_auto_distribution_rows_with_set,
    collect_boundary_distance_relationship_measurements,
    collect_boundary_distance_relationship_rows,
    collect_boundary_distance_relationship_rows_with_contract, collect_boundary_measurements,
    collect_boundary_rows, collect_centroid_neighbor_measurements, collect_centroid_neighbor_rows,
    collect_centroid_neighbor_rows_with_contract, collect_centroid_relationship_measurements,
    collect_centroid_relationship_rows, collect_centroid_relationship_rows_with_contract,
    collect_class_a_shapes, collect_class_a_values, collect_colocalization_measurements,
    collect_colocalization_rows, collect_colocalization_rows_with_contract,
    collect_component_measurements, collect_component_rows, collect_contact_measurements,
    collect_contact_rows, collect_costes_colocalization_measurements,
    collect_costes_colocalization_rows, collect_costes_colocalization_rows_with_contract,
    collect_custom_boundary_rows, collect_custom_object_rows, collect_custom_region_rows,
    collect_distribution_measurements, collect_distribution_rows,
    collect_enclosing_sphere_measurements, collect_enclosing_sphere_rows,
    collect_enclosing_sphere_rows_with_contract, collect_exact_distribution_measurements,
    collect_exact_distribution_measurements_set, collect_exact_distribution_rows_set,
    collect_exact_distribution_rows_with_set, collect_exact_label_radius_measurements,
    collect_exact_label_radius_rows, collect_expansion_relationship_measurements,
    collect_expansion_relationship_rows, collect_expansion_relationship_rows_with_contract,
    collect_glcm_texture_measurements, collect_glcm_texture_rows,
    collect_glcm_texture_rows_with_contract, collect_granularity_measurements,
    collect_granularity_rows, collect_granularity_rows_with_set,
    collect_object_3d_moment_measurements, collect_object_3d_moment_measurements_set,
    collect_object_3d_moment_rows_set, collect_object_3d_moment_rows_with_set,
    collect_object_convex_hull_measurements, collect_object_convex_hull_rows,
    collect_object_convex_hull_rows_with_contract, collect_object_directional_feret_measurements,
    collect_object_directional_feret_rows, collect_object_directional_feret_rows_with_contract,
    collect_object_geometry_basic_measurements, collect_object_geometry_basic_rows,
    collect_object_geometry_basic_rows_with_contract, collect_object_geometry_measurements,
    collect_object_geometry_rows, collect_object_geometry_rows_with_contract,
    collect_object_hu_moment_measurements, collect_object_hu_moment_rows,
    collect_object_hu_moment_rows_with_contract, collect_object_projected_convex_measurements,
    collect_object_projected_convex_measurements_set, collect_object_projected_convex_rows_set,
    collect_object_projected_convex_rows_with_contract,
    collect_object_projected_convex_rows_with_set,
    collect_object_voxel_face_convex_hull_measurements, collect_object_voxel_face_convex_hull_rows,
    collect_object_voxel_face_convex_hull_rows_with_contract,
    collect_object_weighted_hu_moment_measurements, collect_object_weighted_hu_moment_rows,
    collect_object_weighted_hu_moment_rows_with_contract, collect_object_zernike3d_measurements,
    collect_object_zernike3d_measurements_set, collect_object_zernike3d_rows_set,
    collect_object_zernike3d_rows_with_contract, collect_object_zernike_moment_measurements,
    collect_object_zernike_moment_measurements_set, collect_object_zernike_moment_rows_set,
    collect_object_zernike_moment_rows_with_contract, collect_object_zernike_moment_rows_with_set,
    collect_overlap_assignment_measurements, collect_overlap_assignment_rows,
    collect_overlap_child_count_measurements, collect_overlap_child_count_rows,
    collect_overlap_relationship_measurements, collect_overlap_relationship_rows,
    collect_rank_weighted_colocalization_measurements, collect_rank_weighted_colocalization_rows,
    collect_rank_weighted_colocalization_rows_with_contract,
    collect_shared_boundary_radius_measurements, collect_shared_boundary_radius_rows,
    collect_topology_measurements, collect_topology_rows, collect_touching_neighbor_measurements,
    collect_touching_neighbor_rows, collect_touching_neighbor_rows_with_contract,
    colocalization_measurement_schema, colocalization_measurements, component_measurement_schema,
    contact_fraction_of_boundary, contact_measurement_schema,
    costes_colocalization_measurement_schema, costes_colocalization_measurements,
    decode_custom_measurement_rows, distribution_measurement_schema,
    enclosing_sphere_measurement_schema, encode_auto_distribution_measurements,
    encode_auto_distribution_measurements_set, encode_boundary_distance_relationship_measurements,
    encode_centroid_neighbor_measurements, encode_centroid_relationship_measurements,
    encode_colocalization_measurements, encode_component_measurements,
    encode_costes_colocalization_measurements, encode_enclosing_sphere_measurements,
    encode_exact_distribution_measurements, encode_exact_distribution_measurements_set,
    encode_exact_label_radius_measurements, encode_expansion_relationship_measurements,
    encode_granularity_measurements, encode_object_3d_moment_measurements,
    encode_object_3d_moment_measurements_set, encode_object_convex_hull_measurements,
    encode_object_directional_feret_measurements, encode_object_geometry_basic_measurements,
    encode_object_geometry_measurements, encode_object_hu_moment_measurements,
    encode_object_projected_convex_measurements, encode_object_projected_convex_measurements_set,
    encode_object_voxel_face_convex_hull_measurements,
    encode_object_weighted_hu_moment_measurements, encode_object_zernike3d_measurements,
    encode_object_zernike3d_measurements_set, encode_object_zernike_moment_measurements,
    encode_object_zernike_moment_measurements_set, encode_overlap_assignment_measurements,
    encode_overlap_child_count_measurements, encode_overlap_relationship_measurements,
    encode_rank_weighted_colocalization_measurements, encode_shared_boundary_radius_measurements,
    encode_topology_measurements, encode_touching_neighbor_measurements,
    equivalent_sphere_diameter, equivalent_sphere_diameter_for_voxels, equivalent_sphere_radius,
    equivalent_sphere_radius_for_voxels, exact_distribution_measurement_schema,
    exact_distribution_measurement_schema_set, exact_distribution_measurements,
    exact_label_radius_measurements, expansion_relationship_measurement_schema,
    expansion_until_adjacent_relationships_from_boundary_distances, fuse_glcm_texture_measurements,
    glcm_texture_measurement_schema, glcm_texture_measurements, granularity_measurement_schema,
    granularity_spectrum_measurements, object_3d_moment_measurement_schema,
    object_3d_moment_measurement_schema_set, object_3d_moment_measurements,
    object_3d_moment_measurements_set, object_boundary_distance_relationships,
    object_centroid_relationships, object_centroid_relationships_from_shapes,
    object_component_measurements, object_convex_hull_measurement_schema,
    object_convex_hull_measurements, object_directional_feret_measurement_schema,
    object_directional_feret_measurements, object_enclosing_sphere_measurements,
    object_expansion_until_adjacent_relationships, object_geometry_basic_measurement_schema,
    object_geometry_basic_measurements, object_geometry_basic_measurements_u32,
    object_geometry_measurement_schema, object_geometry_measurements,
    object_hu_moment_measurement_schema, object_hu_moments_measurements,
    object_projected_convex_measurement_schema, object_projected_convex_measurement_schema_set,
    object_projected_convex_measurements, object_projected_convex_measurements_set,
    object_topology_measurements, object_topology_measurements_with,
    object_voxel_face_convex_hull_measurement_schema, object_voxel_face_convex_hull_measurements,
    object_weighted_hu_moment_measurement_schema, object_weighted_hu_moments_measurements,
    object_zernike3d_measurement_schema, object_zernike3d_measurement_schema_set,
    object_zernike3d_measurements, object_zernike3d_measurements_set,
    object_zernike_moment_measurement_schema, object_zernike_moment_measurement_schema_set,
    object_zernike_moments_measurements, object_zernike_moments_measurements_set, orientation_yx,
    overlap_assignment_measurement_schema, overlap_child_count_measurement_schema,
    overlap_relationship_measurement_schema, principal_axes, principal_axis_lengths,
    radius_measurement_schema, rank_weighted_colocalization_measurement_schema,
    rank_weighted_colocalization_measurements, run_boundary_measure, run_object_measure,
    run_region_measure, shape_boundary_measurements, shared_boundary_distance_field,
    shared_boundary_radius_measurements, summarize_centroid_neighbors,
    summarize_centroid_neighbors_within, summarize_maximum_overlap_assignments,
    summarize_maximum_overlap_assignments_with_threshold,
    summarize_overlap_assignment_child_counts, summarize_touching_neighbors,
    topology_measurement_schema, touching_neighbor_measurement_schema, ApproxDistributionSet,
    ApproxMode, ApproxTolerance, AutoDistributionOp, AutoDistributionRows, AutoDistributionSet,
    BoundaryDistanceRelationshipRows, BoundaryDistanceRelationshipsOp, BoundaryFeature,
    BoundaryLabelsOp, BoundaryMeasure, BoundaryMeasureOp, BoundaryMeasurements,
    BoundaryPointTallyOp, BoundaryRows, CentroidNeighborRows, CentroidNeighborSummaryOp,
    CentroidRelationshipRows, CentroidRelationshipsOp, ClassARows, ColocalizationContract,
    ColocalizationFeature, ColocalizationMeasurements, ColocalizationPairsOp, ColocalizationRows,
    ColocalizationSumsOp, ContactFeature, ContactMeasurements, ContactRows,
    CostesColocalizationFeature, CostesColocalizationMeasurements, CostesColocalizationOp,
    CostesColocalizationRows, CustomBoundaryRows, CustomObjectRows, CustomRegionRows,
    DirectionSet3, DistributionFeature, DistributionMeasurements, DistributionPercentile,
    DistributionRows, DistributionSet, EnclosingSphereOp, ExactDistributionMeasurements,
    ExactDistributionOp, ExactDistributionRows, ExactDistributionSet, ExactDistributionTallyOp,
    ExactLabelRadiusMeasurements, ExactLabelRadiusOp, ExactLabelRadiusRows,
    ExpansionRelationshipRows, ExpansionUntilAdjacentRelationshipsOp, FeatureScalar, FoldLaw,
    FusedGlcmTextureMeasurements, GlcmOffset, GlcmQuantization, GlcmTextureContract,
    GlcmTextureFeature, GlcmTextureMeasurements, GlcmTextureOp, GranularityFeature,
    GranularityMeasurements, GranularityOp, GranularityRadius, GranularityRows, GranularitySet,
    HuMomentIndex, IntensityFeature, IntensityHistogramOp, IntensityImage, IntensityMeasurements,
    IntensitySet, LabelImage, MeasureSource, MeasureValues, MeasurementFrame, MeasurementFrameId,
    MeasurementKey, MeasurementPlan, MeasurementRows, MeasurementRowsWithContract,
    MeasurementSourceFact, MeasurementSourceFacts, Measurements,
    MergeBoundaryDistanceRelationshipsOp, MergeBoundaryLabelsOp, MergeColocalizationSumsOp,
    MergeCostesColocalizationOp, MergeEnclosingSphereOp, MergeExactDistributionOp,
    MergeExpansionUntilAdjacentRelationshipsOp, MergeGlcmTextureOp, MergeIntensityHistogramOp,
    MergeObjectConvexHullOp, MergeObjectDirectionalFeretOp, MergeObjectGeometryBasicOp,
    MergeObjectGeometryOp, MergeObjectHuMomentsOp, MergeObjectMoment3dOp,
    MergeObjectProjectedConvexOp, MergeObjectVoxelFaceConvexHullOp, MergeObjectWeightedHuMomentsOp,
    MergeObjectZernike3dOp, MergeObjectZernikeMomentsOp, MergeOverlapRelationshipsOp,
    MergeRankWeightedColocalizationOp, Moment3d, Moment3dKey, MultiGlcmTextureOp, NeighborSummary,
    ObjectBoundaryDistanceFeature, ObjectBoundaryDistanceMeasurements, ObjectComponentFeature,
    ObjectComponentMeasurements, ObjectComponentOp, ObjectConvexHullFeature,
    ObjectConvexHullMeasurements, ObjectConvexHullOp, ObjectConvexHullRows, ObjectConvexHullView,
    ObjectDirectionalFeretMeasurements, ObjectDirectionalFeretRows, ObjectDirectionalFeretTallyOp,
    ObjectEnclosingSphereFeature, ObjectEnclosingSphereMeasurements, ObjectEnclosingSphereRows,
    ObjectExpansionFeature, ObjectExpansionMeasurements, ObjectGeometryBasicMeasurements,
    ObjectGeometryBasicRows, ObjectGeometryBasicTallyOp, ObjectGeometryFeature,
    ObjectGeometryMeasurements, ObjectGeometryOp, ObjectGeometryRows, ObjectHuMomentFeature,
    ObjectHuMomentRows, ObjectHuMomentsMeasurements, ObjectHuMomentsOp, ObjectInputs,
    ObjectMeasure, ObjectMeasureMergeOp, ObjectMoment3dFeature, ObjectMoment3dMeasurements,
    ObjectMoment3dOp, ObjectMoment3dRows, ObjectMoment3dSet, ObjectNeighborFeature,
    ObjectNeighborMeasurements, ObjectOverlapAssignmentFeature,
    ObjectOverlapAssignmentMeasurements, ObjectOverlapChildCountFeature,
    ObjectOverlapChildCountMeasurements, ObjectOverlapFeature, ObjectOverlapMeasurements,
    ObjectPointTallyOp, ObjectProjectedConvexContract, ObjectProjectedConvexFeature,
    ObjectProjectedConvexMeasurements, ObjectProjectedConvexOp, ObjectProjectedConvexRows,
    ObjectProjectionContract, ObjectRelationshipFeature, ObjectRelationshipMeasurements,
    ObjectTopologyConvention, ObjectTopologyFeature, ObjectTopologyMeasurements, ObjectTopologyOp,
    ObjectView, ObjectVoxelFaceConvexHullFeature, ObjectVoxelFaceConvexHullMeasurements,
    ObjectVoxelFaceConvexHullRows, ObjectWeightedHuMomentFeature, ObjectWeightedHuMomentRows,
    ObjectWeightedHuMomentsMeasurements, ObjectWeightedHuMomentsOp, ObjectZernike3dContract,
    ObjectZernike3dDescriptor, ObjectZernike3dFeature, ObjectZernike3dMeasurements,
    ObjectZernike3dRows, ObjectZernike3dSet, ObjectZernikeMoment, ObjectZernikeMomentContract,
    ObjectZernikeMomentFeature, ObjectZernikeMomentRows, ObjectZernikeMomentSet,
    ObjectZernikeMomentsMeasurements, ObjectZernikeMomentsOp, OverlapAssignmentRows,
    OverlapAssignmentSummaryOp, OverlapAssignmentThreshold, OverlapChildCountRows,
    OverlapChildCountSummaryOp, OverlapRelationshipRows, OverlapRelationshipTallyOp,
    PhysicalSpacing, ProjectedConvexSet, ProjectionAxis, RadiusFeature,
    RankWeightedColocalizationFeature, RankWeightedColocalizationMeasurements,
    RankWeightedColocalizationOp, RankWeightedColocalizationRows, RegionMeasure, RegionMeasureOp,
    ShapeBoundaryFeature, ShapeBoundaryMeasurements, ShapeFeature, ShapeMeasurements, ShapeSet,
    SharedBoundaryRadiusMeasurements, SharedBoundaryRadiusOp, SharedBoundaryRadiusRows,
    TextureRows, TopologyRows, TouchingNeighborFeature, TouchingNeighborRows,
    TouchingNeighborSummaryOp, VoxelCount, WeightedHuPointTallyOp, WithinDistanceThreshold,
    Zernike3dKey, ZernikeMomentKey,
};
pub use mixing::{LinearMap, TupleKernel, TupleOp};
pub use morphology::{
    black_top_hat, close_into, close_into_at, dilate_into, dilate_into_at, dilate_placed_grey_into,
    dilate_placed_grey_into_at, dilate_placed_into, dilate_placed_into_at, erode_grey_into,
    erode_grey_into_at, erode_into, erode_into_at, grey_black_top_hat, grey_closing, grey_opening,
    grey_white_top_hat, hit_or_miss_into, hit_or_miss_into_at, morphological_gradient, open_into,
    open_into_at, white_top_hat, GreyDilateOp, GreyErodeOp, HitOrMiss, HitOrMissOp, Morphology,
    MorphologyOp,
};
pub use normalise::{
    bounded_gain_into, bounded_gain_value, normalise_against_into, normalise_value,
    LevelCorrectionOp, LocalContrastOp, LocalGainOp, Removal,
};
pub use random_walker::{
    append_grady_weight_phase, append_random_walker_binary_seed_image_phases,
    append_random_walker_row_id_phase, assemble_random_walker_system,
    assemble_random_walker_system_from_packed,
    assemble_random_walker_system_from_packed_seed_image,
    assemble_random_walker_system_from_seed_image, assemble_random_walker_system_from_sparse_table,
    collect_random_walker_rows, encode_random_walker_rows, grady_weight_phase,
    grady_weights_packed_into, packed_weight_shape, random_walker_binary_into,
    random_walker_binary_seed_image_into, random_walker_multilabel_seed_image_into,
    random_walker_row_id_phase, random_walker_row_ids_into, random_walker_sparse_columns,
    random_walker_sparse_schema, solve_random_walker_seed_image_system_into,
    solve_random_walker_sparse_table_seed_image_into, solve_random_walker_system_into,
    GradyWeightOp, GradyWeights, RandomWalkerConfig, RandomWalkerImages,
    RandomWalkerMultilabelSolve, RandomWalkerRowIdOp, RandomWalkerRowsOp, RandomWalkerSolve,
    RandomWalkerSolveImages, RandomWalkerSolveOp, RandomWalkerSparseColumns, RandomWalkerSystem,
    RandomWalkerWeights,
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
    ridge_response_into, symmetric_eigenvalues, Boundary, EigenResponse, HessianEigenvalueOp,
    Polarity, RatioResponse, Response, RidgeFilterOp, RidgeResponse, ScaleSpace,
};
pub use rows::{
    append_group_phases, collect_groups, collect_rows, decode_groups, encode_groups, filter_blob,
    filter_into, gather_blob, gather_into, gathered_schema, group_values, merge_rows, scale_blob,
    scale_into, scaled_at, scaled_bound, scaled_index, value_at, walk_rows, Aggregate, ColumnFold,
    ColumnTest, FilterRowsOp, GatherRowsOp, GroupFold, GroupRowsOp, GroupValues, Grouping, Limit,
    MergeGroupsOp, Reduction, RowFilter, RowStreams, RowValues, ScaleRowsOp, GROUP_ROWS,
    MAX_PACKED_COORDINATE,
};
pub use skeleton::{thin, thinning_pass, thinning_reach, ThinningOp};
pub use sliding::{
    sliding_histogram_into, sliding_histogram_with_plan, BinnedElement, Domain, HistogramQuery,
    RankQuery, ScanPlan, SlidingHistogramOp,
};
pub use smooth::{Gaussian, SmoothOp};
pub use structure_tensor::{
    corner_response_into, gaussian_gradient_into, gradient_at, CornerResponse, CornerResponseOp,
    Eigenvalue, GradientMagnitudeOp, StructureTensor, StructureTensorOp,
};
pub use tabulate::{
    append_tabulate_phases, collect_tabulation, decode_partial, encode_partial, merge_tabulation,
    region_values, tabulate_phases, tabulation_schema, FixedPoint, MergeTabulationOp, RegionValues,
    TabulateLabelsOp, TabulateValuesOp, Tally,
};
pub use threshold::{
    append_global_threshold_phases, li_threshold, mean_threshold, minimum_threshold,
    multi_otsu_thresholds, otsu_threshold, threshold_classes, threshold_mask, triangle_threshold,
    yen_threshold, ApplyGlobalThresholdOp, GlobalThreshold, GlobalThresholdOutput,
    GlobalThresholdSamplesOp, GlobalThresholdSelection,
};
pub use transform::{
    affine_transform_into, append_warp_phase, log_polar_transform_into, polar_transform_into,
    projective_transform_into, remap_into, rotate_into, warp_into, warp_phase, CoordinateMap,
    TransformBoundary, TransformInterpolation, WarpOp,
};
pub use voxelize::{decode_points, encode_points, Point, VoxelizeOp};
pub use voxelwise::{
    combine_into, from_set, is_set, logic_into, map_into, mask_logic_into, not_into, CarryOp,
    CombineOp, Compose, Gamma, Identity, LogCorrection, Logic, LogicCombine, MapFn, MaskElement,
    MaskFn, NarrowOp, Not, RescaleIntensity, Sigmoid, Threshold, ThresholdMask, ThresholdTest,
    VoxelwiseMapOp, VoxelwiseMaskOp, WidenOp, IDENTITY_COST, MAP_COST, MASK_COST,
};
pub use walk::{
    walk_blob, walk_from, walk_into, walk_schema, walked_distance, OffsetSequence, OffsetWalkOp,
};
pub use watershed::{
    cost_report as watershed_cost_report, seeded_watershed, seeded_watershed_into,
    seeded_watershed_into_reporting_peak, SeededWatershedOp, Separation, WATERSHED_COST,
    WATERSHED_LINE_COST,
};

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

/// **Convolution with a caller-supplied kernel**, in a caller-supplied sense
/// (correlation or convolution, named rather than assumed) and a caller-supplied
/// boundary convention. `ops::smooth` is the Gaussian; this is the general one.
pub mod convolve;
pub use convolve::{
    convolve_into, cost_report as convolve_cost_report, ConvolveOp, Kernel, RobertsDiagonal, Sense,
    CONVOLVE_COST_PER_TAP,
};
/// **The same filter through the Fourier transform**: overlap-save over a tile
/// grid anchored to the volume, so it is an ordinary bounded-reach `BlockOp` and
/// is byte-identical across lattices. Not bit-identical to [`ConvolveOp`] — the
/// two sum the same products in different orders — which is why it is a separate
/// op and not a flag on that one.
pub use convolve::{transform_convolve_cost_report, TransformConvolveOp, TRANSFORM_CONVOLVE_COST};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MaskSource {
    image: usize,
}

impl MaskSource {
    pub(crate) fn new(mask: impl Into<ImageId>) -> Self {
        Self {
            image: mask.into().index(),
        }
    }

    pub(crate) fn image(self) -> usize {
        self.image
    }

    pub(crate) fn source_input(self, reach: Reach) -> SourceInput {
        SourceInput::new(self.image, reach).holding(Dtype::Bool)
    }

    pub(crate) fn maybe_source_input(mask: Option<Self>, reach: Reach) -> Vec<SourceInput> {
        match mask {
            Some(mask) => vec![mask.source_input(reach)],
            None => Vec::new(),
        }
    }

    pub(crate) fn input_only_error(self, name: &str) -> Error {
        Error::InvalidArgument(format!(
            "{name}: the population comes from image {}, so this op has no answer from its input \
             alone. It is applied through `apply_with`.",
            self.image
        ))
    }

    pub(crate) fn bool_view<'a>(
        self,
        name: &str,
        sources: SourceInputs<'a>,
    ) -> Result<ArrayView3<'a, bool>> {
        let mask = sources.get(self.image)?;
        if mask.dtype() != Dtype::Bool {
            return Err(Error::InvalidArgument(format!(
                "{name}: the population is read from image {}, which holds {}. A population is a \
                 yes-or-no per voxel and is stored as one; a wider type would leave 'which \
                 non-zero values count' to be decided somewhere this op cannot see.",
                self.image,
                mask.dtype().numpy_name()
            )));
        }
        mask.view::<bool>()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LabelSeedSource {
    image: usize,
}

impl LabelSeedSource {
    pub(crate) fn new(seeds: impl Into<ImageId>) -> Self {
        Self {
            image: seeds.into().index(),
        }
    }

    pub(crate) fn image(self) -> usize {
        self.image
    }

    pub(crate) fn source_input(self, reach: Reach) -> SourceInput {
        SourceInput::new(self.image, reach).holding(Dtype::U32)
    }

    pub(crate) fn input_only_error(self, name: &str) -> Error {
        Error::InvalidArgument(format!(
            "{name}: the seeds come from image {}, so this op has no answer from its input alone \
             — it would flood nothing and write an empty volume. It is applied through \
             `apply_with`.",
            self.image
        ))
    }

    pub(crate) fn u32_view<'a>(
        self,
        name: &str,
        sources: SourceInputs<'a>,
    ) -> Result<ArrayView3<'a, u32>> {
        let seeds = sources.get(self.image)?;
        if seeds.dtype() != Dtype::U32 {
            return Err(Error::InvalidArgument(format!(
                "{name}: the seeds are read from image {}, which holds {}. A seed is a label and \
                 is stored as one; a float would leave 'which values are the same seed' to be \
                 decided somewhere this op cannot see.",
                self.image,
                seeds.dtype().numpy_name()
            )));
        }
        seeds.view::<u32>()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TypedSource {
    image: usize,
    dtype: Option<Dtype>,
}

impl TypedSource {
    pub(crate) fn new(image: impl Into<ImageId>) -> Self {
        Self {
            image: image.into().index(),
            dtype: None,
        }
    }

    pub(crate) fn dtype(self) -> Option<Dtype> {
        self.dtype
    }

    pub(crate) fn image_index(self) -> usize {
        self.image
    }

    pub(crate) fn holding(mut self, dtype: Dtype) -> Self {
        self.dtype = Some(dtype);
        self
    }

    pub(crate) fn voxelwise_input(self) -> SourceInput {
        self.source_input(Reach::none())
    }

    pub(crate) fn source_input(self, reach: Reach) -> SourceInput {
        let input = SourceInput::new(self.image, reach);
        match self.dtype {
            Some(dtype) => input.holding(dtype),
            None => input,
        }
    }

    pub(crate) fn block<'a>(self, sources: SourceBlocks<'a>) -> Result<&'a crate::env::BlockBuf> {
        sources.get(self.image)
    }

    pub(crate) fn block_at_extent<'a>(
        self,
        name: &str,
        role: &str,
        sources: SourceBlocks<'a>,
        read: &Region,
    ) -> Result<&'a crate::env::BlockBuf> {
        let block = self.block(sources)?;
        let crate::env::BlockBuf::Array(array) = block else {
            return Ok(block);
        };
        let expected = read.shape3();
        expect_extent(
            || {
                format!(
                    "{name}: the {role} arrived as {:?} for a block read extent of {expected:?}. \
                     The source operand is fetched at the block's own fetch region, so a \
                     disagreement here is the plan handing over two different geometries.",
                    array.shape()
                )
            },
            expected,
            array.shape(),
        )?;
        Ok(block)
    }
}

// The **second-moment half of `ops::tabulate`**. `RegionShape`/`region_shape`
// are the shape reading of a tabulated row — the label volume's own
// measurement, over every voxel and at no scale — and `PrincipalAxes` is what
// its six `CENTRAL` columns decompose to.
pub use tabulate::{
    collect_shapes, from_signed_column, region_shape, signed_column, PrincipalAxes, RegionShape,
    AXIS_SEPARATION, CENTRAL, PAIRS,
};
