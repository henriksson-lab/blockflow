//! **Model-based instance segmentation as a `blockflow` op.**
//!
//! A network that finds objects in a tile — Cellpose, StarDist, anything with
//! the same shape — is a per-tile function. Running one over an image too large
//! to hold is a block-decomposition problem, and that is the part this crate is
//! about. The network itself belongs to whichever crate implements
//! [`SegmentBackend`].
//!
//! # Why the op is not a `BlockOp`
//!
//! The obvious design is a `BlockOp` mapping an intensity block to a label
//! block. It is wrong, and the reason decides everything else here.
//!
//! Blocks overlap by a halo; **cores do not**. An object detected inside block
//! A's halo is also detected by block B, so something must decide which block
//! owns it — and whatever that rule is, a *label image* is clipped at the core
//! boundaries: A writes an object's pixels only within A's core, and B writes
//! background there because B discarded it. Per-object measurements taken from
//! such an image are measurements of fragments.
//!
//! So the measurement happens **inside the op**, where the whole object is
//! present in the halo'd buffer, and what comes out is rows rather than pixels.
//! That is `blockflow`'s [`FragmentOp`](blockflow::FragmentOp), and it is the
//! same program `blockflow::ops::detect` runs.
//!
//! # The two rules that make the answer independent of the cut
//!
//! **Ownership is by centroid.** A detected object is kept by a block if and
//! only if its centroid lies in that block's core, half-open on every axis.
//! Cores tile the volume exactly, so every object is claimed by exactly one
//! block — no union-find, no IoU matching between neighbours.
//!
//! **The id is a function of the data, not of the schedule.** An object's id is
//! derived from the position of its lowest voxel, so two runs at different
//! block sizes give the same object the same id, no counter is shared between
//! blocks, and two objects cannot collide because they cannot share a voxel.
//! Ids are `u64` because a whole-slide image has more voxels than a `u32` can
//! address. See [`identify`] for why the lowest voxel rather than the centroid.
//!
//! What neither rule covers is stated where it is handled rather than left
//! implicit: an object straddling a core boundary that *both* blocks detect,
//! each with a centroid on its own side, is emitted twice. See
//! [`InstanceSegment`] for the halo rule that makes it rare and
//! [`schema`] for the column a consumer dedups on.

use std::sync::Arc;

use crate::error::{Error, Result};
use crate::op::SourceInput;
use crate::table::{Column, ColumnType, RowBuilder, Schema, Value};
use crate::{
    Anchor, BlockBuf, BlockOutput, BlockView, Coverage, Dtype, FragmentOp, FragmentOutput, ImageId,
    Lifecycle, Reach, Region, SourceBlocks, Voxels,
};
use ndarray::{Array3, ArrayView3};

pub mod flow;
pub mod stub;

#[cfg(feature = "cellpose")]
pub mod cellpose;
#[cfg(feature = "stardist")]
pub mod stardist;

// ------------------------------------------------------------- backend --

/// One tile in, local labels out.
///
/// The whole of what a segmentation model has to supply. Everything about
/// blocks, ownership, ids, measurement and merging is [`InstanceSegment`]'s,
/// and an implementor of this trait cannot get any of it wrong because it is
/// never told about it.
///
/// Rank 3 with a degenerate axis for a 2-D image, matching `blockflow`'s own
/// convention: 2-D is a volume one voxel deep. A backend that only does 2-D
/// says so by refusing a tile deeper than one, and says it by name.
pub trait SegmentBackend: Send + Sync {
    fn name(&self) -> &'static str;

    /// Label the objects in `tile`. `0` is background; every other value names
    /// one object within this tile.
    ///
    /// The labels need not be contiguous, need not start at 1, and mean nothing
    /// outside this call — [`InstanceSegment`] renumbers by position.
    ///
    /// `at` says where `tile` sits in the volume. **Most backends ignore it**,
    /// and a position-independent one should. It is here for the one that
    /// cannot be: a backend whose own internal tiling must be anchored to
    /// absolute coordinates rather than to the buffer it happens to be handed,
    /// which is what makes its answer a function of the pixels instead of a
    /// function of the block. See [`crate::model_segment::flow`].
    fn segment(&self, tile: ArrayView3<'_, f32>, at: &Anchor) -> Result<Array3<u32>>;

    /// Relative compute cost per voxel, in `blockflow`'s units, where `1.0` is
    /// an ordinary voxelwise pass.
    ///
    /// A network is not an ordinary voxelwise pass and the default does not
    /// pretend otherwise by staying at `1.0`: it is stated by each backend from
    /// what that backend was measured at, because a planner that prices a
    /// three-second tile as a memcpy will schedule it as one.
    fn cost_per_voxel(&self) -> f64;
}

// -------------------------------------------------------------- schema --

/// One row per object: where it is, how big it is, and a reduction of every
/// measured image over its voxels.
///
/// Every column is `u64` and every column merges by an **associative and
/// commutative** operation — `+` for the sums, `min` and `max` for the extremes.
/// That is not decoration: it is what lets a consumer fold two rows for the same
/// object without the answer depending on which arrived first. The discipline is
/// `blockflow::ops::tabulate`'s and the argument for it is in that module's
/// header.
///
/// | column | what it is | merges by |
/// |---|---|---|
/// | `id` | the object's identity, derived from its centroid — see the crate header | it is the key |
/// | `count` | voxels in the object | `+` |
/// | `sum_0`, `sum_1`, `sum_2` | per-axis sum of the object's voxel coordinates, in **volume** coordinates | `+` |
/// | `sum_c{i}`, `min_c{i}`, `max_c{i}` | over image `i` of the measured set | `+`, `min`, `max` |
///
/// The coordinate sums are carried rather than only the rounded centroid the row
/// is positioned at, because `sum / count` is the exact centroid and the
/// position is that to a voxel. A consumer that needs sub-voxel positions — a
/// dedup by distance, say — has them; one that does not can ignore three
/// columns.
///
/// **Image `0` of the measured set is the image being segmented.** A run that
/// segments on a nuclear stain and measures three markers gets four `sum_c`
/// columns, and the first is the stain. That is deliberate: the segmented
/// channel's own intensity is a measurement like any other, and leaving it out
/// would make a caller declare it twice to get it.
pub fn schema(measured: usize) -> Result<Schema> {
    let mut columns = vec![
        Column::u64("id"),
        Column::u64("count"),
        Column::u64("sum_0"),
        Column::u64("sum_1"),
        Column::u64("sum_2"),
    ];
    for image in 0..measured {
        columns.push(Column::new(format!("sum_c{image}"), ColumnType::U64));
        columns.push(Column::new(format!("min_c{image}"), ColumnType::U64));
        columns.push(Column::new(format!("max_c{image}"), ColumnType::U64));
    }
    Schema::new(columns)
}

/// The fixed columns every row carries, before the measured ones.
pub const FIXED_COLUMNS: usize = 5;

/// The column index of `sum_c{image}`. `min` and `max` follow it.
pub fn measured_column(image: usize) -> usize {
    FIXED_COLUMNS + image * 3
}

/// **Combine two rows describing one object.**
///
/// The operation a consumer performs when it finds the same object twice — the
/// duplicate the crate header names, where an object straddling a core boundary
/// is detected on both sides and each half's centroid lands in its own block's
/// core. It is here rather than in the application because the fold is a
/// property of the schema, and a consumer writing its own would be writing a
/// second definition of what these columns mean.
///
/// | column | folded by |
/// |---|---|
/// | `id` | the key — the two must agree, and a disagreement is refused |
/// | `count`, `sum_0..2`, `sum_c{i}` | `+` |
/// | `min_c{i}` | `min` |
/// | `max_c{i}` | `max` |
///
/// Every one of those is associative and commutative **in `u64`**, which is why
/// every column is a `u64`: the folded answer does not depend on which
/// duplicate arrived first, or on how many there were, or on the order a
/// consumer happened to visit them in. `tests/rows.rs` asserts that rather than
/// asserting the table above.
///
/// What it deliberately does *not* do is decide **whether** two rows are one
/// object. Two rows with one id are one object by construction — ids come from
/// a voxel position and objects cannot share a voxel — but the duplicate case
/// produces two rows with *different* ids, because the two halves have
/// different lowest voxels. Matching those is a positional question with a
/// tolerance in it, and a tolerance is the caller's to choose; see the plan's
/// dedup pass. This function is what to call once the caller has decided.
pub fn fold(measured: usize, left: &[u64], right: &[u64]) -> Result<Vec<u64>> {
    let width = FIXED_COLUMNS + measured * 3;
    if left.len() != width || right.len() != width {
        return Err(Error::InvalidArgument(format!(
            "fold over {measured} measured image(s) wants rows of {width} columns and was given \
             {} and {}",
            left.len(),
            right.len()
        )));
    }
    if left[0] != right[0] {
        return Err(Error::InvalidArgument(format!(
            "fold: rows for objects {} and {} are not two readings of one object. An id is \
             derived from a voxel position and two objects cannot share a voxel, so folding \
             across ids would add together two things that are genuinely different.",
            left[0], right[0]
        )));
    }

    let mut folded = Vec::with_capacity(width);
    folded.push(left[0]);
    // count, sum_0, sum_1, sum_2
    for column in 1..FIXED_COLUMNS {
        folded.push(left[column] + right[column]);
    }
    for image in 0..measured {
        let base = measured_column(image);
        folded.push(left[base] + right[base]);
        folded.push(left[base + 1].min(right[base + 1]));
        folded.push(left[base + 2].max(right[base + 2]));
    }
    Ok(folded)
}

/// The object's centroid, exactly, as `(sum, count)` per axis.
///
/// Returned as the two integers rather than as a float because that is what the
/// row holds and because the division is the caller's to make at whatever
/// precision they need: `sum / count` is the exact centre, and the position the
/// row is filed under is that rounded to a voxel.
pub fn centroid(row: &[u64]) -> ([u64; 3], u64) {
    ([row[2], row[3], row[4]], row[1])
}

// ------------------------------------------------------------- the op --

/// Segment each block, keep the objects whose centroids it owns, measure them,
/// and emit a row each.
///
/// # The halo, which is the one parameter that can be silently wrong
///
/// `halo` must be **at least the largest object diameter expected**. It is what
/// makes an object owned by a block wholly present in that block's buffer, and
/// therefore what makes its measurement a measurement of the object rather than
/// of the part that fell inside the core.
///
/// Too small does not fail; it under-measures the objects near seams, and it
/// makes duplicates more likely — an object straddling a core boundary that
/// each block sees only half of gets a centroid on each side and is emitted
/// twice. Neither shows up as an error. What does show up is the duplicate
/// count from a positional dedup pass over the sealed table, which is why that
/// pass should report what it merged rather than merging quietly.
///
/// Too large costs reads and inference: the buffer is
/// `(edge + 2 * halo)^2` and every voxel of it goes through the network.
///
/// # What it declares
///
/// * `reads_pixels` — the block's own image, over the read extent;
/// * `source_inputs` — one per measured image, each at the same reach, so each
///   arrives at exactly the extent the pixels do and indices line up;
/// * one fragment stream, `Coverage::EveryBlock` — a block that owns no object
///   still writes an empty row blob, because a phase writing no image is not
///   constrained by the tiling check and the coverage declaration is the only
///   thing that can fail.
pub struct InstanceSegment {
    name: &'static str,
    backend: Arc<dyn SegmentBackend>,
    halo: [usize; 3],
    stream: String,
    lifecycle: Lifecycle,
    measured: Vec<(ImageId, Dtype)>,
    scale: u32,
    /// Skip the backend for a block whose core holds nothing above this.
    ///
    /// **The largest optimisation available on a whole slide, and it is data
    /// and not scheduling.** A scanned slide is mostly empty glass: `2079_R1`
    /// is 66048 x 157440 at level 0 and its scanned tissue — OpenSlide's
    /// `bounds-*` — is 45312 x 50944, so about 78% of the image contains no
    /// cell at all. Inference over it costs the same as over tissue and finds
    /// nothing.
    ///
    /// The test is on the **core**, not the buffer, and that is what makes it
    /// safe rather than merely cheap: an object is kept by this block only if
    /// its centroid lies in the core (see the crate header), and an object with
    /// no signal in the core cannot have its centroid there. A block whose core
    /// is empty therefore owns nothing, whatever is in its halo — so skipping
    /// it drops no object that any block would have kept.
    ///
    /// `None` runs the backend everywhere, which is what this did before the
    /// parameter existed.
    empty_below: Option<f64>,
}

impl InstanceSegment {
    /// `measured` names the images reduced over each object, **besides** the
    /// one being segmented, which is always measured and is always image `0` of
    /// the row's measured set.
    ///
    /// Each is given with what it holds, because a supplied input has no
    /// producing phase and therefore no element type the plan can fold — the
    /// readers are the only declaration there is.
    pub fn new(
        name: &'static str,
        backend: Arc<dyn SegmentBackend>,
        halo: [usize; 3],
        stream: impl Into<String>,
        lifecycle: Lifecycle,
        measured: Vec<(ImageId, Dtype)>,
    ) -> Self {
        Self {
            name,
            backend,
            halo,
            stream: stream.into(),
            lifecycle,
            measured,
            scale: 0,
            empty_below: None,
        }
    }

    /// Skip blocks whose core holds nothing above `level`. See [`Self::empty_below`].
    #[must_use]
    pub fn skipping_empty(mut self, level: Option<f64>) -> Self {
        self.empty_below = level;
        self
    }

    /// Accumulate measured values as fixed point, `round(value * 2^scale)`.
    ///
    /// `0` — the default — is exact for an integer acquisition: a `u8` or `u16`
    /// sample is a whole number and its sum over an object is a whole number.
    ///
    /// It is a parameter and not an inference from the element type because the
    /// caller is the only one who knows the *range*: `2^scale` times the largest
    /// sample times the largest object must fit in a `u64`, and this crate knows
    /// none of those three. A float image with values in `[0, 1]` wants a scale
    /// of about 20; the same image scaled to `[0, 65535]` wants 0.
    ///
    /// Why fixed point at all, rather than an `f64` column: floating-point
    /// addition does not associate, so a sum folded across a seam would be a
    /// function of which partial arrived first — the decomposition showing
    /// through in the answer, which is the one defect this whole arrangement
    /// exists to prevent.
    #[must_use]
    pub fn with_scale(mut self, scale: u32) -> Self {
        self.scale = scale;
        self
    }

    /// How many images a row measures: the segmented one, plus the declared.
    pub fn measured_images(&self) -> usize {
        1 + self.measured.len()
    }

    pub fn schema(&self) -> Result<Schema> {
        schema(self.measured_images())
    }

    pub fn stream(&self) -> &str {
        &self.stream
    }

    fn quantise(&self, value: f64, image: usize) -> Result<u64> {
        let scaled = value * f64::from(1u32 << self.scale.min(52));
        if !scaled.is_finite() || scaled < 0.0 {
            return Err(Error::InvalidArgument(format!(
                "op {:?}: measured image {image} holds {value}, and the row columns are `u64` \
                 accumulators merged by addition. A negative or non-finite sample has no \
                 representation there, and widening the column to `f64` would make the seam \
                 merge order-dependent — see `with_scale`.",
                self.name
            )));
        }
        Ok(scaled.round() as u64)
    }
}

/// One object, accumulated over the voxels the backend gave it.
struct Object {
    count: u64,
    /// The object's lowest voxel in raster order, in volume coordinates. Set
    /// when the object is first seen, which — because the accumulation walks
    /// the block in raster order — **is** the lowest one, at no cost.
    first: [usize; 3],
    /// Per-axis sum of volume coordinates.
    position: [u64; 3],
    /// Per measured image: sum, min, max.
    sum: Vec<u64>,
    low: Vec<u64>,
    high: Vec<u64>,
}

impl Object {
    fn new(measured: usize, first: [usize; 3]) -> Self {
        Self {
            count: 0,
            first,
            position: [0; 3],
            sum: vec![0; measured],
            low: vec![u64::MAX; measured],
            high: vec![0; measured],
        }
    }

    /// `sums / count`, rounded half up, in integer arithmetic.
    ///
    /// Half up towards the far end of the axis, and computed as
    /// `(2 * sum + count) / (2 * count)` so that no floating point is involved
    /// at any step — two different cuts compute the same two integers and take
    /// the same quotient, which is what makes "does a centroid at exactly `x.5`
    /// round the same way under every cut" not a question about a last bit.
    /// This is `blockflow::ops::detect`'s rule and it is the same rule here for
    /// the same reason.
    fn centroid(&self) -> [usize; 3] {
        let mut at = [0usize; 3];
        for axis in 0..3 {
            at[axis] = ((2 * self.position[axis] + self.count) / (2 * self.count)) as usize;
        }
        at
    }
}

/// The id of an object whose **lowest voxel in raster order** is `first`.
///
/// # Why the lowest voxel and not the centroid
///
/// Both are functions of the data, which is the property that matters: two runs
/// at different block sizes give one object one id, and nothing is shared
/// between blocks to make it so. The centroid was the obvious choice and it has
/// a collision the lowest voxel does not.
///
/// Two *disjoint* objects can share a rounded centroid — an L and a dot at the
/// L's concave corner, a ring and something inside it — and would then share an
/// id. They cannot share a voxel, so they cannot share a lowest voxel, and the
/// ids are distinct **by construction** rather than by an argument about how
/// unlikely the arrangement is. No check is needed and none is written, which
/// is the right amount of code for a collision that cannot happen.
///
/// It is as stable as the centroid under retiling and for the same reason: with
/// a halo at least the object's diameter the owning block holds the whole
/// object, so it sees the true lowest voxel. Under a halo too short the object
/// fragments, and the fragments get their own ids — which is what a fragment
/// is.
///
/// The row is still *positioned* at the centroid, because position is what a
/// spatial query is over and "where is this object" is answered by its centre.
/// Identity and position are different questions and this is the one place they
/// are allowed different answers.
///
/// `+ 1` so that `0` is never an id, leaving it free to mean "no object" in
/// anything that renders these back into a label volume.
pub fn identify(first: [usize; 3], volume: [usize; 3]) -> u64 {
    1 + (first[0] as u64) * (volume[1] as u64) * (volume[2] as u64)
        + (first[1] as u64) * (volume[2] as u64)
        + first[2] as u64
}

/// Is `at` inside `region`, half-open on every axis?
///
/// Half-open is what makes cores tile exactly: a voxel on a shared face belongs
/// to the block whose core starts there and to no other. A closed test would
/// give it to both, and every object centred on a seam would be emitted twice.
fn owns(region: &Region, at: [usize; 3]) -> bool {
    (0..3).all(|axis| {
        at[axis] >= region.start[axis] && at[axis] < region.start[axis] + region.shape[axis]
    })
}

impl FragmentOp for InstanceSegment {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, axis: usize, _volume_len: usize) -> usize {
        self.halo[axis]
    }

    fn cost_per_voxel(&self) -> f64 {
        self.backend.cost_per_voxel()
    }

    fn reads_pixels(&self) -> bool {
        true
    }

    fn writes_pixels(&self) -> bool {
        false
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        self.measured
            .iter()
            .map(|(image, dtype)| {
                SourceInput::new(*image, Reach::symmetric(self.halo)).holding(*dtype)
            })
            .collect()
    }

    /// **Nothing crosses a seam here**, which is the whole point of measuring
    /// inside the op.
    ///
    /// A row is computed entirely from one block's buffers: the owning block
    /// holds the whole object, because the halo carried it in, so no partial
    /// sum is ever combined with another block's. This op declares no fragment
    /// input and folds nothing, so there is no order for an answer to depend
    /// on. `blockflow` checks the first half of that — an op declaring
    /// `PerBlock` may not declare a fragment input with a non-zero reach.
    ///
    /// The failure this rules out is the one `ops::detect` deferred a weighted
    /// centroid over: a per-object reduction summed in pieces and combined in
    /// `f64` gives a different last bit depending on which block merged first,
    /// so the same slide cut two ways gives two numbers and neither looks
    /// wrong. Here there are no pieces.
    ///
    /// Separately — and it is worth saying because a consumer *may* fold two
    /// rows, for the duplicate case the crate header names — every column is an
    /// integer merged by `+`, `min` or `max`, so that fold is order-independent
    /// too. That is the consumer's property, not this declaration's.
    fn seam_fold(&self) -> Option<crate::SeamFold> {
        Some(crate::SeamFold::PerBlock)
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            self.stream.clone(),
            self.lifecycle,
            // Every block, always. A block that owns no object still writes an
            // empty blob, and the difference between "owned nothing" and "never
            // ran" is exactly what the coverage check is for.
            Coverage::EveryBlock,
        )]
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        self.apply_with(at, SourceBlocks::none())
    }

    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let measured = self.measured_images();
        let schema = Arc::new(self.schema()?);
        let mut rows = RowBuilder::new(Arc::clone(&schema));

        let BlockBuf::Array(pixels) = at.pixels()? else {
            // An accounting run has no data. It still writes the blob, because
            // what such a run measures is the IO, and a phase that silently
            // produced nothing would be measuring a different program.
            return Ok(BlockOutput::fragment(self.stream.clone(), rows.encode()));
        };

        // The values, widened once. `widened` is `f64` for every element type a
        // block can hold, which is the one conversion that works without this
        // op knowing what the image is.
        let intensity = pixels.widened();

        // Nothing in the core, nothing to own. Checked before the backend runs,
        // because the backend is the expensive part.
        if let Some(level) = self.empty_below {
            let core = at.core;
            let offset = at.at.offset;
            let mut highest = f64::NEG_INFINITY;
            for i in 0..core.shape[0] {
                for j in 0..core.shape[1] {
                    for k in 0..core.shape[2] {
                        let value = intensity[[
                            core.start[0] + i - offset[0],
                            core.start[1] + j - offset[1],
                            core.start[2] + k - offset[2],
                        ]];
                        if value > highest {
                            highest = value;
                        }
                    }
                }
            }
            if highest <= level {
                // An empty blob, not an absent one: the stream declares
                // `Coverage::EveryBlock`, and "owned nothing" must stay
                // distinguishable from "never ran".
                return Ok(BlockOutput::fragment(self.stream.clone(), rows.encode()));
            }
        }
        let tile = intensity.mapv(|value| value as f32);
        let labels = self.backend.segment(tile.view(), &at.at)?;
        if labels.shape() != tile.shape() {
            return Err(Error::InvalidArgument(format!(
                "backend {:?} was handed a tile of {:?} and returned labels of {:?}. The labels \
                 are read at the tile's own indices, so a different shape has no reading.",
                self.backend.name(),
                tile.shape(),
                labels.shape()
            )));
        }

        // Every measured image at the same extent, so one index reads them all.
        // Image 0 of the set is the segmented image itself.
        let mut values: Vec<ndarray::Array3<f64>> = Vec::with_capacity(measured);
        values.push(intensity);
        for (image, _) in &self.measured {
            let BlockBuf::Array(buf) = sources.get(image.index())? else {
                return Err(Error::InvalidArgument(format!(
                    "op {:?}: {} arrived without values, and a row measuring it would be a \
                     plausible number for a reduction that never happened.",
                    self.name,
                    crate::assemble::describe_image(image.index())
                )));
            };
            if buf.shape() != tile.shape() {
                return Err(Error::InvalidArgument(format!(
                    "op {:?}: {} arrived at {:?} and the block is {:?}. A measured image is read \
                     at the block's own fetch region and is indexed voxel for voxel with it.",
                    self.name,
                    crate::assemble::describe_image(image.index()),
                    buf.shape(),
                    tile.shape()
                )));
            }
            values.push(buf.widened());
        }

        // One pass over the block, accumulating per local label.
        let Anchor { offset, volume } = at.at;
        let mut objects: std::collections::BTreeMap<u32, Object> = Default::default();
        let shape = labels.dim();
        for i in 0..shape.0 {
            for j in 0..shape.1 {
                for k in 0..shape.2 {
                    let label = labels[[i, j, k]];
                    if label == 0 {
                        continue;
                    }
                    let here = [offset[0] + i, offset[1] + j, offset[2] + k];
                    let object = objects
                        .entry(label)
                        .or_insert_with(|| Object::new(measured, here));
                    object.count += 1;
                    for axis in 0..3 {
                        object.position[axis] += here[axis] as u64;
                    }
                    for (image, array) in values.iter().enumerate() {
                        let value = self.quantise(array[[i, j, k]], image)?;
                        object.sum[image] += value;
                        object.low[image] = object.low[image].min(value);
                        object.high[image] = object.high[image].max(value);
                    }
                }
            }
        }

        // Keep what this block owns, and number it by where it is.
        for object in objects.values() {
            let centroid = object.centroid();
            if !owns(at.core, centroid) {
                continue;
            }
            let mut values = Vec::with_capacity(schema.width());
            values.push(Value::U64(identify(object.first, volume)));
            values.push(Value::U64(object.count));
            for axis in 0..3 {
                values.push(Value::U64(object.position[axis]));
            }
            for image in 0..measured {
                values.push(Value::U64(object.sum[image]));
                values.push(Value::U64(object.low[image]));
                values.push(Value::U64(object.high[image]));
            }
            rows.push(centroid, &values)?;
        }

        Ok(BlockOutput::fragment(self.stream.clone(), rows.encode()))
    }
}

/// Convert a `Voxels` block to `f32`, the element type a backend is handed.
///
/// Public because a backend that wants to look at the raw block — to check a
/// range, say — should use the same conversion the op does rather than a second
/// one that could differ.
pub fn as_f32(block: &Voxels) -> Array3<f32> {
    block.widened().mapv(|value| value as f32)
}
