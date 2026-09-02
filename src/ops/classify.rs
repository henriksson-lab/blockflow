// SPDX-License-Identifier: MIT
//
// Original work for this crate.

//! **The forest predictor**: 91 feature images in, one classification out.
//!
//! Step 3 of `docs/design/pixel-classification.md`. The forest itself, its
//! layout and the reason it is written here rather than taken from a crate are
//! in [`crate::forest`]; this file is only its shell as a
//! [`Combine`], plus the wrappers that assemble a whole
//! workflow out of `ops::features` and it.
//!
//! # Why a `Combine` and not a `BlockOp`
//!
//! A `BlockOp` reads one image. The predictor reads as many as the feature stack
//! has channels and writes one, which is exactly the shape of a fan-in's sink —
//! and `Chain::Parallel` already allocates one buffer per branch, folds the
//! reaches by a maximum and joins the results, so nothing new is needed.
//!
//! Two consequences, both of which the planner sees:
//!
//! * **Reach zero.** A voxel's class is a function of that voxel's features, so
//!   the predictor contributes nothing to any halo, and it is
//!   decomposition-invariant by construction — the existing invariance suites
//!   cover it without a special case.
//! * **Not a fold, and it says so.** [`Combine::fold_carrier`] exists so that an
//!   associative join can be accumulated branch by branch and hold three buffers
//!   whatever the arity. A tree walk needs every channel at a voxel
//!   *simultaneously*, which is the definition of not being a left fold over
//!   pairs, so this one answers `None` and its fan-in holds one buffer per arm.
//!   That is the open residency question step 2 of the design document recorded,
//!   and it is measured here rather than argued: see `tests/forest_predict.rs`.
//!
//! # Single-threaded per call, deliberately
//!
//! The design document names this as a constraint and it is worth restating
//! where the loop is. The block executor already parallelises across blocks, and
//! `simulate::Machine::contention` exists because nested parallelism is what
//! makes forty workers behave like 2.41. A predictor spawning its own threads
//! would fight the machinery this crate spent its measurements on.

use std::collections::BTreeMap;
use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayD, ArrayView1, ArrayView2};

use crate::geometry::BlockGrid;
use crate::op::{Output, SideBlock};

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::forest::{Forest, Samples, TrainingSpec};
use crate::op::{Anchor, Chain, Combine, Slicing};
use crate::ops::features::FeatureStack;
use crate::region::Region;
use crate::voxels::Voxels;

/// What the predictor writes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Prediction {
    /// The winning class, as a `u32` label volume — what a segmentation wants,
    /// and what `ops::components` and `ops::tabulate` consume.
    Label,
    /// The share of the vote one class received, in `[0, 1]`, as `f64`.
    ///
    /// **This is the output that feeds a watershed.** A boundary class's
    /// probability map is exactly the cost volume
    /// `ops::scikitimage_watershed` wants, which is the meeting point the design
    /// document notes between the random-forest workflow and ilastik's carving
    /// one.
    Probability { class: usize },
}

/// The forest, as the sink of a fan-in over a feature stack.
pub struct ForestPredictor {
    name: &'static str,
    forest: Arc<Forest>,
    prediction: Prediction,
    cost: f64,
}

impl ForestPredictor {
    pub fn new(name: &'static str, forest: Arc<Forest>, prediction: Prediction) -> Result<Self> {
        if let Prediction::Probability { class } = prediction {
            if class >= forest.classes() {
                return Err(Error::InvalidArgument(format!(
                    "{name}: asked for the probability of class {class}, but the forest has \
                     {} classes. The answer would be zero at every voxel.",
                    forest.classes()
                )));
            }
        }
        let cost = cost_for(&forest);
        Ok(Self {
            name,
            forest,
            prediction,
            cost,
        })
    }

    pub fn forest(&self) -> &Arc<Forest> {
        &self.forest
    }

    pub fn prediction(&self) -> Prediction {
        self.prediction
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }

    /// The arity check, which is also the **compatibility check between a
    /// trained forest and a feature stack**.
    ///
    /// A forest's splits name columns by index, so a stack with the right number
    /// of channels in the wrong order is the failure this exists to catch: it
    /// would run, produce a complete well-formed volume, and be wrong
    /// everywhere. The count is all a `Combine` can check from a `&[Dtype]`;
    /// [`predict_workflow`] checks the *names*, which is the real test, and does
    /// it at build time rather than at the first block.
    fn arity_agrees(&self, inputs: &[Dtype]) -> bool {
        inputs.len() == self.forest.channels().len()
    }
}

impl Combine for ForestPredictor {
    fn name(&self) -> &'static str {
        self.name
    }

    /// Zero, on every axis and at every volume size. A voxel's class is a
    /// function of that voxel's features.
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    /// **A stencil**, and the fan-in needs to hear it from here: a `Parallel`
    /// node is only as sliceable as its narrowest part, so a stack of stencil
    /// arms joined by a sink that says nothing is refused for intra-block
    /// slicing however sliceable its arms are.
    ///
    /// The claim is the loop's: each output voxel is written from the co-located
    /// voxel of each input, through one pass that reads no neighbour and carries
    /// nothing between voxels but a scratch tally it clears.
    fn slicing(&self) -> Slicing {
        Slicing::Stencil
    }

    /// Every channel must be `f64`, and there must be as many as the forest was
    /// trained on.
    ///
    /// `f64` rather than "anything numeric" because every feature op in
    /// `ops::features` produces `f64` and a mixed list would mean the forest's
    /// thresholds were compared against values of two precisions — which is the
    /// same class of quiet wrongness as the wrong column order.
    fn accepts(&self, inputs: &[Dtype]) -> bool {
        self.arity_agrees(inputs) && inputs.iter().all(|&dtype| dtype == Dtype::F64)
    }

    fn produces(&self, _inputs: &[Dtype]) -> Dtype {
        match self.prediction {
            Prediction::Label => Dtype::U32,
            Prediction::Probability { .. } => Dtype::F64,
        }
    }

    fn output_shape(&self, inputs: &[[usize; 3]]) -> Result<[usize; 3]> {
        let first = *inputs.first().ok_or_else(|| {
            Error::InvalidArgument(format!("{}: no branch results to classify", self.name))
        })?;
        if inputs.len() != self.forest.channels().len() {
            return Err(Error::InvalidArgument(format!(
                "{}: the forest was trained on {} channels and was handed {} branches. A \
                 split names a column by index, so a stack of the wrong width does not \
                 mean the wrong answer — it means reading past the end of one.",
                self.name,
                self.forest.channels().len(),
                inputs.len()
            )));
        }
        for (branch, shape) in inputs.iter().enumerate() {
            if shape != &first {
                return Err(Error::InvalidArgument(format!(
                    "{}: channel 0 produced {first:?} and channel {branch} produced {shape:?}. \
                     A voxel's feature vector is assembled from co-located voxels, and buffers \
                     of different extents have no such correspondence.",
                    self.name
                )));
            }
        }
        Ok(first)
    }

    fn apply(&self, inputs: &[&Voxels], out: &mut Voxels, _at: &Anchor) -> Result<()> {
        let shapes: Vec<[usize; 3]> = inputs.iter().map(|input| input.shape()).collect();
        let shape = self.output_shape(&shapes)?;
        if out.shape() != shape {
            return Err(Error::ShapeMismatch {
                expected: shape.to_vec(),
                got: out.shape().to_vec(),
            });
        }

        // The channels as flat slices, once, outside the voxel loop. Every
        // buffer here is a block the executor allocated contiguously, so this is
        // a borrow rather than a copy — and it turns the inner loop's
        // per-channel access from a strided `ndarray` index into an offset.
        //
        // The views are bound to a local before the slices are taken because a
        // slice borrows its view; collecting the two in one expression would
        // borrow from a temporary.
        let views = inputs
            .iter()
            .map(|input| input.view::<f64>())
            .collect::<Result<Vec<_>>>()?;
        let channels = views
            .iter()
            .map(|view| {
                view.to_slice().ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "{}: a channel buffer is not contiguous, so it cannot be read as a \
                         feature column",
                        self.name
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let voxels = shape[0] * shape[1] * shape[2];
        let mut features = vec![0.0f64; channels.len()];
        let mut tally = vec![0.0f32; self.forest.classes()];

        match self.prediction {
            Prediction::Label => {
                let mut out = out.view_mut::<u32>()?;
                let out = out.as_slice_mut().ok_or_else(|| {
                    Error::InvalidArgument(format!("{}: the output is not contiguous", self.name))
                })?;
                for at in 0..voxels {
                    for (slot, channel) in features.iter_mut().zip(channels.iter()) {
                        *slot = channel[at];
                    }
                    out[at] = self.forest.predict(&features, &mut tally);
                }
            }
            Prediction::Probability { class } => {
                let mut out = out.view_mut::<f64>()?;
                let out = out.as_slice_mut().ok_or_else(|| {
                    Error::InvalidArgument(format!("{}: the output is not contiguous", self.name))
                })?;
                for at in 0..voxels {
                    for (slot, channel) in features.iter_mut().zip(channels.iter()) {
                        *slot = channel[at];
                    }
                    out[at] = self.forest.probability(&features, class, &mut tally);
                }
            }
        }
        Ok(())
    }

    /// **`None`, and it means it.** See the module header: a tree walk needs
    /// every channel at a voxel at once.
    fn fold_carrier(&self, _inputs: &[Dtype]) -> Option<Dtype> {
        None
    }

    fn cost_per_voxel(&self, _branches: usize) -> f64 {
        self.cost
    }
}

/// Measured; see `super::COST_MEASUREMENT` for the method and
/// `tests/forest_predict.rs` for the run.
///
/// **Proportional to `trees x mean path`, which is the number of nodes visited
/// per voxel** — not to the node count, and not to the depth. A forest's cost is
/// a walk per tree, and what a walk costs is its length; the maximum depth would
/// misprice an unbalanced forest by the ratio between its deepest and its
/// average path, and the total node count would misprice every forest by its
/// breadth.
///
/// The per-visit constant is a dependent load, a compare and a branch, with the
/// node array too large for L2 at any interesting forest size — so it is a cache
/// miss more often than not, and that is why the figure is large beside a
/// voxelwise map's 1.0.
pub(super) fn cost_for(forest: &Forest) -> f64 {
    forest.trees() as f64 * forest.mean_path() * FOREST_COST_PER_NODE_VISIT
}

/// Measured; see [`COST_MEASUREMENT`]. One node visited, relative to a voxelwise
/// map.
pub const FOREST_COST_PER_NODE_VISIT: f64 = 6.56;

/// The measurement the constant above came from, kept as text so a re-run
/// elsewhere can be compared against it rather than merely replacing it.
/// `--release`, one thread, 91 channels over 64x64x32, best of 3, on the machine
/// this crate was developed on; the unit is the voxelwise map at 0.991 ns.
///
/// ```text
///   trees   depth   nodes  mean path visits/voxel     ns/voxel     ns/visit
///      10       8     820       6.87         68.7        447.3         6.51
///      10      20    2358       9.49         94.9        638.9         6.73
///      50      20   15156      10.44        521.8       3091.0         5.92
///     100      20   30594      10.35       1035.1       6527.5         6.31
///     200      20   57582      10.22       2044.0      13196.8         6.46
/// ```
///
/// **The model is right, which is the first thing the table says.** `ns/visit`
/// is flat — 5.92 to 6.73 across a twenty-fold range of node counts and a
/// seventy-fold range of visits — so cost really is `trees x mean path x a
/// constant`, and neither the node count nor the maximum depth would have
/// served. 6.56 is the mean, and the spread is ±6%.
///
/// **And it is flat despite the node array outgrowing cache**, which is worth
/// noting because the opposite was expected: 57582 nodes at 24 bytes is 1.4 MB,
/// past L2, and the walk's loads are dependent and effectively random. That it
/// does not degrade says the array stays resident in L3 across a block's worth
/// of voxels — every voxel walks the same trees — so the misses are amortised
/// over the block rather than paid per voxel. A forest large enough to leave L3
/// would break this, and the table is where that would show.
///
/// **What it means for the workload.** At Labkit's own default of 100 trees the
/// predictor costs **6528 ns per voxel, about 6590 times a voxelwise map**. The
/// whole 91-channel feature stack under it declares roughly 5000, so the
/// predictor is not merely the most expensive op in this crate by a wide margin
/// — it is *the majority of the entire workload*, and `docs/design/pixel-
/// classification.md`'s expectation that it would dominate is confirmed rather
/// than assumed. Two consequences: the planner's treatment of this chain follows
/// from this number rather than from the filters, and the contention term
/// measured for the multi-node work matters more here than on any fixture it was
/// taken against.
pub const COST_MEASUREMENT: &str = "tests/forest_predict.rs::print_the_predictor_cost";

// -------------------------------------------------------- the workflows --

/// **Predict**: a feature stack and a trained forest, as one chain.
///
/// The shape `ops::background::remove_background` uses — a function returning a
/// `Chain`, so the planner sees the whole thing and can cut it where it likes.
///
/// **The channel names are checked here, and this is the check that matters.**
/// A forest's splits name columns by index; a stack with the right *number* of
/// channels in a different order runs to completion and is wrong at every voxel.
/// `Combine::accepts` sees only a list of element types and cannot catch that.
/// So the names are compared, in order, before a chain is built at all — a
/// refusal at build time rather than a wrong volume at the end of a run.
pub fn predict_workflow(
    stack: &FeatureStack,
    forest: Arc<Forest>,
    prediction: Prediction,
) -> Result<Chain> {
    let names = stack.channel_names()?;
    if names != forest.channels() {
        let first = names
            .iter()
            .zip(forest.channels())
            .position(|(stack, trained)| stack != trained);
        return Err(Error::InvalidArgument(format!(
            "the forest was trained on {} channels and this stack has {}{}. A split names a \
             column by index, so running this would read a different feature than the one \
             each threshold was fitted against and produce a complete, well-formed, wrong \
             volume.",
            forest.channels().len(),
            names.len(),
            match first {
                Some(at) => format!(
                    ", first differing at {at}: trained on {:?}, stack has {:?}",
                    forest.channels()[at],
                    names[at]
                ),
                None => String::new(),
            }
        )));
    }
    Chain::parallel(
        stack.branches()?,
        Box::new(ForestPredictor::new("classify", forest, prediction)?),
    )
}

/// The labels found in a volume, in the class order the forest will use.
///
/// A forest's classes are `0..n`, and a label volume's values are whatever the
/// annotator drew — 1 and 2, or 3 and 7. So the distinct labels are collected,
/// sorted, and mapped to `0..n`, and **the mapping is returned** rather than
/// applied silently: a caller who is handed a `u32` volume of class indices has
/// to be able to get back to the labels they drew.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassMap {
    labels: Vec<u32>,
}

impl ClassMap {
    /// The original label of class `class`.
    pub fn label_of(&self, class: usize) -> Option<u32> {
        self.labels.get(class).copied()
    }

    pub fn classes(&self) -> usize {
        self.labels.len()
    }

    pub fn labels(&self) -> &[u32] {
        &self.labels
    }
}

/// **Gather training rows**: run the feature stack, and take one row at every
/// voxel whose label is not `unlabelled`.
///
/// # It computes the stack over the labels' neighbourhood, not the volume
///
/// Labels are sparse by construction — a few thousand brush-stroke voxels — and
/// a feature at a voxel depends only on the voxels within the chain's reach of
/// it. So the stack is computed over the **bounding box of the labelled voxels,
/// grown by that reach and clamped to the volume**, and nothing outside it is
/// touched.
///
/// **The rows are bit-identical to what the whole volume would have given**, and
/// that is a property rather than an approximation. Two cases and both are
/// exact: a labelled voxel at least `reach` inside the crop reads only voxels
/// the crop holds, and one whose neighbourhood runs past the crop has a crop
/// edge that *is* the volume edge — because the box was grown by the full reach
/// before clamping — so it meets the same boundary rule it would have met
/// anyway. `tests/forest_predict.rs` asserts the equality rather than arguing
/// it.
///
/// For a stroke drawn in one corner of a large volume this is the difference
/// between the work being proportional to the labels and proportional to the
/// array. For labels scattered to opposite corners the box is the volume again,
/// and the honest statement is that this is a crop, not a sparse traversal.
///
/// # What is still missing, stated accurately
///
/// A genuinely sparse path would compute the stack only in the blocks holding
/// labels, and skip the rest — which for a whole-volume annotation is nearly all
/// of them. It does **not** fall out of the fragment machinery the way an
/// earlier note in `docs/design/pixel-classification.md` claimed: a side output
/// is a [`BlockOp`](crate::op::BlockOp) feature, and the sink of the feature
/// stack is a [`Combine`], which has no `apply_side` and no `side_outputs`. The
/// fan-in's sink is the only place where all 91 channels exist at one voxel, so
/// the sampler has to live there, and giving `Combine` the same side-output pair
/// `BlockOp` has is the change that would allow it. That is a contained
/// extension and it is not this function.
pub fn gather_samples(
    stack: &FeatureStack,
    input: &Voxels,
    labels: &Voxels,
    unlabelled: u32,
) -> Result<(Samples, ClassMap)> {
    if labels.shape() != input.shape() {
        return Err(Error::ShapeMismatch {
            expected: input.shape().to_vec(),
            got: labels.shape().to_vec(),
        });
    }
    let volume = input.shape();
    let names = stack.channel_names()?;
    let channels = stack.branches()?;

    // The box the labels actually occupy, grown by what a feature reads. See
    // this function's header for why the rows this yields are the whole
    // volume's rows and not an approximation of them.
    let label_view = labels.view::<u32>()?;
    let reach = stack.reach(volume)?;
    let Some((start, extent)) = labelled_extent(label_view, unlabelled, reach, volume) else {
        return Err(Error::InvalidArgument(format!(
            "no voxel of the label volume differs from {unlabelled}, so there is nothing \
             to fit"
        )));
    };

    // `slice_region` and not a copy written here: it is the crate's own sub-box
    // copy, generic over the element type, and a second one would be a second
    // chance to get the striding wrong.
    let cropped_input = input.slice_region(&Region::new(&start, &extent))?;
    let mut columns: Vec<Vec<f64>> = Vec::with_capacity(channels.len());
    for arm in &channels {
        // One arm at a time, so the peak is one feature image and not
        // ninety-one. The same fusion argument the block executor makes, taken
        // here because a training crop is small enough to hold whole but a stack
        // over it is not.
        let mut out = Voxels::zeros(arm.produces(cropped_input.dtype())?, extent)?;
        arm.apply(&cropped_input, &mut out, &Anchor::whole(extent))?;
        let view = out.view::<f64>()?;
        columns.push(view.iter().copied().collect());
    }

    let mut distinct: Vec<u32> = label_view
        .iter()
        .copied()
        .filter(|&label| label != unlabelled)
        .collect();
    distinct.sort_unstable();
    distinct.dedup();
    if distinct.len() < 2 {
        return Err(Error::InvalidArgument(format!(
            "the label volume holds {} class{} besides {unlabelled}. A classifier needs two \
             to discriminate between; one is a constant image.",
            distinct.len(),
            if distinct.len() == 1 { "" } else { "es" }
        )));
    }

    let mut features = Vec::new();
    let mut rows = Vec::new();
    for i in 0..extent[0] {
        for j in 0..extent[1] {
            for k in 0..extent[2] {
                let label = label_view[[start[0] + i, start[1] + j, start[2] + k]];
                if label == unlabelled {
                    continue;
                }
                let at = (i * extent[1] + j) * extent[2] + k;
                for column in &columns {
                    features.push(column[at]);
                }
                rows.push(distinct.iter().position(|&found| found == label).unwrap() as u32);
            }
        }
    }
    Ok((
        Samples::new(features, rows, names)?,
        ClassMap { labels: distinct },
    ))
}

/// **Train**: a feature stack, a volume, its labels, and a fitting spec, to a
/// forest that [`predict_workflow`] will accept against the same stack.
///
/// The two halves of the workflow are deliberately separate functions rather
/// than one round trip: a caller trains once on a crop and predicts many times
/// on volumes, often in different processes, and the thing that travels between
/// them is the [`Forest`].
pub fn train_workflow(
    stack: &FeatureStack,
    input: &Voxels,
    labels: &Voxels,
    unlabelled: u32,
    spec: &TrainingSpec,
) -> Result<(Forest, ClassMap)> {
    let (samples, classes) = gather_samples(stack, input, labels, unlabelled)?;
    Ok((Forest::train(&samples, spec)?, classes))
}

/// The bounding box of the voxels whose label is not `unlabelled`, grown by
/// `reach` on every side and clamped to the volume.
///
/// `None` when nothing is labelled. The growth is what makes the crop's rows the
/// whole volume's rows; see [`gather_samples`].
fn labelled_extent(
    labels: ndarray::ArrayView3<'_, u32>,
    unlabelled: u32,
    reach: [usize; 3],
    volume: [usize; 3],
) -> Option<([usize; 3], [usize; 3])> {
    let mut low = [usize::MAX; 3];
    let mut high = [0usize; 3];
    let mut any = false;
    for ((i, j, k), &label) in labels.indexed_iter() {
        if label == unlabelled {
            continue;
        }
        any = true;
        for (axis, at) in [i, j, k].into_iter().enumerate() {
            low[axis] = low[axis].min(at);
            high[axis] = high[axis].max(at);
        }
    }
    if !any {
        return None;
    }
    let mut start = [0usize; 3];
    let mut extent = [0usize; 3];
    for axis in 0..3 {
        start[axis] = low[axis].saturating_sub(reach[axis]);
        let end = (high[axis] + reach[axis] + 1).min(volume[axis]);
        extent[axis] = end - start[axis];
    }
    Some((start, extent))
}

// ------------------------------------------------------- the row sampler --

/// Where every labelled voxel is, and which row of the gathered table it owns.
///
/// **Built once, from the label volume and the grid the run will use.** Both are
/// known before any feature is computed — labels are input data, not something
/// the run produces — which is what makes a fixed-shape side output possible for
/// a gather whose length looks data-dependent. It is not: the number of rows is
/// the number of labelled voxels, and that is countable up front.
///
/// # Why the grid is in here
///
/// A side output's per-block slice is a **box** — `Output::side_region` returns
/// one `Region` — so a block's rows have to be contiguous in the table. Labelled
/// voxels ordered by position in the volume are not contiguous within a block,
/// because a block is a sub-box and its voxels are interleaved with its
/// neighbours' in row-major order. So the rows are ordered **by block first**,
/// and that ordering is a function of the partition.
///
/// This makes a chain carrying a sampler valid only for the plan whose grid it
/// was built against, which is a real constraint and an existing one:
/// [`Chain::Source`] says the same of the image its
/// leaf names — "a chain carrying one constrains the plans it is valid for. That
/// is not a leak: which image is read is parity-visible". Here the grid is
/// checked rather than trusted: [`SampleCombine::side_region`] refuses a block
/// it was not built for, by name, rather than returning a plausible wrong range.
#[derive(Debug, Clone)]
pub struct LabelIndex {
    /// One per labelled voxel, ordered by block and then by position within it.
    positions: Vec<[usize; 3]>,
    /// The class of each, remapped to `0..n` in ascending order of the label
    /// the annotator drew.
    classes: Vec<u32>,
    /// `start` of each block's core, to the `(row offset, row count)` its
    /// labelled voxels own.
    by_block: BTreeMap<[usize; 3], (usize, usize)>,
    map: ClassMap,
}

impl LabelIndex {
    /// Index `labels` against `grid`, treating `unlabelled` as "no class".
    pub fn build(labels: &Voxels, unlabelled: u32, grid: &BlockGrid) -> Result<Self> {
        if labels.shape() != grid.volume() {
            return Err(Error::ShapeMismatch {
                expected: grid.volume().to_vec(),
                got: labels.shape().to_vec(),
            });
        }
        let view = labels.view::<u32>()?;
        let mut distinct: Vec<u32> = view
            .iter()
            .copied()
            .filter(|&label| label != unlabelled)
            .collect();
        distinct.sort_unstable();
        distinct.dedup();
        if distinct.len() < 2 {
            return Err(Error::InvalidArgument(format!(
                "the label volume holds {} class{} besides {unlabelled}. A classifier needs \
                 two to discriminate between; one is a constant image.",
                distinct.len(),
                if distinct.len() == 1 { "" } else { "es" }
            )));
        }

        let mut positions = Vec::new();
        let mut classes = Vec::new();
        let mut by_block = BTreeMap::new();
        // **Block by block, in the grid's own order**, so a block's rows are one
        // contiguous run and `side_region` is a box.
        for core in grid.cores() {
            let start = [core.core.start[0], core.core.start[1], core.core.start[2]];
            let shape = core.core.shape;
            let offset = positions.len();
            for i in start[0]..start[0] + shape[0] {
                for j in start[1]..start[1] + shape[1] {
                    for k in start[2]..start[2] + shape[2] {
                        let label = view[[i, j, k]];
                        if label == unlabelled {
                            continue;
                        }
                        positions.push([i, j, k]);
                        classes.push(
                            distinct
                                .iter()
                                .position(|&found| found == label)
                                .expect("collected from the same volume")
                                as u32,
                        );
                    }
                }
            }
            by_block.insert(start, (offset, positions.len() - offset));
        }
        Ok(Self {
            positions,
            classes,
            by_block,
            map: ClassMap { labels: distinct },
        })
    }

    pub fn rows(&self) -> usize {
        self.positions.len()
    }

    /// The voxel each row is for, in row order — which is block order, not
    /// volume order. This is what lets a caller join the gathered table back to
    /// the volume, and it is the only way to: the row index alone means nothing
    /// without the partition it was built against.
    pub fn positions(&self) -> &[[usize; 3]] {
        &self.positions
    }

    /// The class of each row, as an index into [`Self::classes`].
    pub fn row_classes(&self) -> &[u32] {
        &self.classes
    }

    pub fn classes(&self) -> &ClassMap {
        &self.map
    }

    /// The rows belonging to the block whose core begins at `start`.
    fn range(&self, start: [usize; 3]) -> Option<(usize, usize)> {
        self.by_block.get(&start).copied()
    }
}

/// **Gather one training row per labelled voxel, as a side output of the feature
/// stack's own fan-in.**
///
/// This is what `Combine::side_outputs` was added for, and the reason it had to
/// be a combine rather than a `BlockOp` is the whole of the design: a fan-in's
/// sink is the only place where every channel's value at a voxel exists at once.
/// A `BlockOp` after the stack sees one image; a branch inside it sees one
/// channel.
///
/// # What it buys over [`gather_samples`]
///
/// [`gather_samples`] computes the stack over a crop and keeps every channel of
/// every voxel of that crop in memory — 91 columns times the crop's voxels. At a
/// `512^3` crop that is 97 GB, so it is a function that works on an annotator's
/// working crop and falls over on a volume.
///
/// This runs inside an ordinary blocked phase. It holds one block's channels,
/// writes the rows for that block's labelled voxels, and moves on, so its
/// residency is the block's rather than the crop's and it works at whatever size
/// the planner can block.
///
/// # The primary image
///
/// A pixel phase writes an image, so this writes the **class of each voxel**,
/// `unlabelled` where there is none — the label volume as the run saw it. It
/// costs one image and it is the thing worth keeping: it is what a reader needs
/// to check which voxels a training run actually sampled.
pub struct SampleCombine {
    name: &'static str,
    channels: Vec<String>,
    index: Arc<LabelIndex>,
    unlabelled: u32,
}

impl SampleCombine {
    /// `channels` must be the stack's channel names, in branch order; the
    /// branches are the stack's arms and nothing else — the labels are not a
    /// branch, because they are indexed up front rather than read per block.
    pub fn new(
        name: &'static str,
        channels: Vec<String>,
        index: Arc<LabelIndex>,
        unlabelled: u32,
    ) -> Self {
        Self {
            name,
            channels,
            index,
            unlabelled,
        }
    }

    pub fn index(&self) -> &Arc<LabelIndex> {
        &self.index
    }

    /// The gathered table's name as a side output.
    pub const ROWS: &'static str = "samples.rows";

    /// And the class column's, kept as a second array rather than a column of
    /// the first: the features are `f64` and a class is not a number to be
    /// interpolated, so putting them in one array would be inviting exactly the
    /// arithmetic that must never be done to it.
    pub const CLASSES: &'static str = "samples.classes";
}

impl Combine for SampleCombine {
    fn name(&self) -> &'static str {
        self.name
    }

    /// Zero. Every channel is read at the voxel the row is for.
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    /// **Not a stencil, and this is the one place that matters.** The rows a
    /// block writes are a function of *which* voxels of it are labelled, and
    /// `side_region` answers for the block's whole valid region. Cutting that
    /// region into slabs for intra-block threading would give each slab the
    /// whole block's row range, and they would overwrite one another.
    fn slicing(&self) -> Slicing {
        Slicing::UNDECLARED
    }

    fn accepts(&self, inputs: &[Dtype]) -> bool {
        inputs.len() == self.channels.len() && inputs.iter().all(|&dtype| dtype == Dtype::F64)
    }

    fn produces(&self, _inputs: &[Dtype]) -> Dtype {
        Dtype::U32
    }

    fn output_shape(&self, inputs: &[[usize; 3]]) -> Result<[usize; 3]> {
        let first = *inputs.first().ok_or_else(|| {
            Error::InvalidArgument(format!("{}: no branch results to sample", self.name))
        })?;
        if inputs.len() != self.channels.len() {
            return Err(Error::InvalidArgument(format!(
                "{}: the sampler names {} channels and was handed {} branches",
                self.name,
                self.channels.len(),
                inputs.len()
            )));
        }
        Ok(first)
    }

    /// **The label each voxel was drawn with**, `unlabelled` where there is
    /// none — not the class *index*.
    ///
    /// The distinction is not cosmetic and it was a defect here first. Class
    /// indices run `0..n`, and the commonest sentinel for "no label" is `0`, so
    /// writing indices makes class 0 and unlabelled the same value in the one
    /// image whose purpose is to show which voxels were sampled. Writing the
    /// annotator's own labels is both unambiguous and the thing a reader wants:
    /// it is the label volume as the run saw it, and it round-trips through
    /// [`ClassMap`] to the indices the rows carry.
    fn apply(&self, inputs: &[&Voxels], out: &mut Voxels, at: &Anchor) -> Result<()> {
        let shapes: Vec<[usize; 3]> = inputs.iter().map(|input| input.shape()).collect();
        self.output_shape(&shapes)?;
        let mut out = out.view_mut::<u32>()?;
        out.fill(self.unlabelled);
        // `at.within` places the buffer in the volume, which is what the index
        // is stated in.
        let origin = at.offset;
        for (row, position) in self.index.positions.iter().enumerate() {
            let mut local = [0usize; 3];
            let mut inside = true;
            for axis in 0..3 {
                if position[axis] < origin[axis] {
                    inside = false;
                    break;
                }
                local[axis] = position[axis] - origin[axis];
                if local[axis] >= out.shape()[axis] {
                    inside = false;
                    break;
                }
            }
            if inside {
                out[local] = self.index.map.labels[self.index.classes[row] as usize];
            }
        }
        Ok(())
    }

    /// Two arrays: the features, and the class each row belongs to.
    fn side_outputs(&self, _volume: [usize; 3]) -> Vec<Output> {
        vec![
            Output::new(
                Self::ROWS,
                Dtype::F64,
                &[self.index.rows(), self.channels.len()],
            ),
            Output::new(Self::CLASSES, Dtype::F64, &[self.index.rows()]),
        ]
    }

    /// **Refuses a block the index was not built for**, rather than answering
    /// with a plausible wrong range. See [`LabelIndex`] for why the two are
    /// coupled at all.
    fn side_region(&self, which: usize, valid: &Region, _volume: [usize; 3]) -> Result<Region> {
        let start = [valid.start[0], valid.start[1], valid.start[2]];
        let (offset, count) = self.index.range(start).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "{}: no block of the grid this sampler was built against begins at {start:?}. \
                 A sampler's row layout is a function of the partition — see `LabelIndex` — \
                 so it must be built with the grid the plan actually uses.",
                self.name
            ))
        })?;
        match which {
            0 => Ok(Region::new(&[offset, 0], &[count, self.channels.len()])),
            1 => Ok(Region::new(&[offset], &[count])),
            _ => Err(Error::InvalidArgument(format!(
                "{}: side output {which}, and it declares two",
                self.name
            ))),
        }
    }

    fn apply_side(
        &self,
        inputs: &[&Voxels],
        _primary: &Voxels,
        block: &SideBlock<'_>,
    ) -> Result<Vec<ArrayD<f64>>> {
        let views = inputs
            .iter()
            .map(|input| input.view::<f64>())
            .collect::<Result<Vec<_>>>()?;
        let start = [
            block.within.start[0],
            block.within.start[1],
            block.within.start[2],
        ];
        // `within` is relative to the buffer; the index is in volume
        // coordinates, so the block's origin is the anchor's start plus it.
        let origin = block.at.offset;
        let core = [
            origin[0] + start[0],
            origin[1] + start[1],
            origin[2] + start[2],
        ];
        let (offset, count) = self.index.range(core).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "{}: no block of this sampler's grid begins at {core:?}",
                self.name
            ))
        })?;

        let mut rows = Array2::<f64>::zeros((count, self.channels.len()));
        let mut classes = Array1::<f64>::zeros(count);
        for row in 0..count {
            let position = self.index.positions[offset + row];
            let local = [
                position[0] - origin[0],
                position[1] - origin[1],
                position[2] - origin[2],
            ];
            for (column, view) in views.iter().enumerate() {
                rows[[row, column]] = view[local];
            }
            classes[row] = self.index.classes[offset + row] as f64;
        }
        Ok(vec![rows.into_dyn(), classes.into_dyn()])
    }

    fn cost_per_voxel(&self, _branches: usize) -> f64 {
        1.0
    }
}

/// **Sample**: a feature stack and a labelled volume, as one chain plus the
/// index that reads its output back.
///
/// The third wrapper beside [`train_workflow`] and [`predict_workflow`], and the
/// one that makes the blocked path usable without assembling a
/// [`SampleCombine`] by hand.
///
/// **It takes the grid**, which the other two do not, for the reason
/// [`LabelIndex`] gives: a block's rows have to be contiguous in the gathered
/// table, so the row order is a function of the partition. A caller therefore
/// plans first and builds this against the grid the plan chose — the same
/// sequence `PlanBuilder::regrid` exists for.
///
/// The returned index is what turns the run's output back into rows: it holds
/// the voxel each row is for and the [`ClassMap`] the classes were remapped
/// through. Hand it and the two arrays to [`samples_from_rows`].
pub fn sample_workflow(
    stack: &FeatureStack,
    labels: &Voxels,
    grid: &BlockGrid,
    unlabelled: u32,
) -> Result<(Chain, Arc<LabelIndex>)> {
    let channels = stack.channel_names()?;
    let index = Arc::new(LabelIndex::build(labels, unlabelled, grid)?);
    let chain = Chain::parallel(
        stack.branches()?,
        Box::new(SampleCombine::new(
            "sample",
            channels,
            Arc::clone(&index),
            unlabelled,
        )),
    )?;
    Ok((chain, index))
}

/// The two arrays [`SampleCombine`] wrote, as [`Samples`] a forest can be fitted
/// to.
///
/// `classes` is `f64` because a side output is — every one in this crate is an
/// `ArrayD<f64>` — and it holds class *indices*, which are whole numbers stored
/// in a float. They are checked on the way back rather than cast: a value that
/// is not a whole number, or not a class the index knows, means the arrays and
/// the index have come from different runs, and a silent `as u32` would turn
/// that into a forest fitted to the wrong labels.
pub fn samples_from_rows(
    rows: ArrayView2<'_, f64>,
    classes: ArrayView1<'_, f64>,
    index: &LabelIndex,
    channels: Vec<String>,
) -> Result<Samples> {
    if rows.shape()[0] != index.rows() || classes.len() != index.rows() {
        return Err(Error::InvalidArgument(format!(
            "the sampler's index holds {} rows and it was handed {} feature rows and {} \
             classes. The three come from one run or from none.",
            index.rows(),
            rows.shape()[0],
            classes.len()
        )));
    }
    if rows.shape()[1] != channels.len() {
        return Err(Error::InvalidArgument(format!(
            "{} columns of features against {} channel names",
            rows.shape()[1],
            channels.len()
        )));
    }
    let mut labels = Vec::with_capacity(classes.len());
    for (row, &value) in classes.iter().enumerate() {
        if value.fract() != 0.0 || value < 0.0 || value as usize >= index.classes().classes() {
            return Err(Error::InvalidArgument(format!(
                "row {row}'s class is {value}, which is not one of the {} the sampler's \
                 index knows. The arrays and the index are from different runs.",
                index.classes().classes()
            )));
        }
        labels.push(value as u32);
    }
    Samples::new(rows.iter().copied().collect(), labels, channels)
}
