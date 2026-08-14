// SPDX-License-Identifier: MIT
//
// Original work for this crate. Written from the definition of the operation —
// deposit a kernel at every point and add — not adapted from any implementation
// of it.
//
// **Scattered points in, dense pixels out.** A stream of point coordinates
// arrives as sidecar fragments, one per block; this renders them into a volume.
// The pure count is the `radius = 0` case and the ball is the general one, and
// they are the same code because the kernel is a [`StructuringElement`] — the
// same type morphology takes its reach from.
//
// Why this is a `FragmentOp` and not a `BlockOp`
// ---------------------------------------------
// `BlockOp::apply(input, out, at)` is `region -> region`. This op's input is not
// a region: it is a variable-length list of coordinates whose size has nothing
// to do with any block's voxel count, and whose *positions* are what decides
// which block reads it. `fragment.rs` names this shape `fragments -> volume`
// and it is the third of the three that `apply` cannot express.
//
// The reach, and why the op is plannable at all
// ---------------------------------------------
// A point deposits into the voxels its kernel covers and nowhere else, so a
// point in a neighbouring block matters here **iff** its kernel overlaps this
// block. Both halves of the declaration follow from the one parameter:
//
// * the **voxel** reach is the kernel's radius on that axis, clamped to the
//   volume. That is what `Decomposition::check` shrinks the read extent by, and
//   with the halo below it leaves `valid == core`;
// * the **block** reach — `FragmentInput::reach`, which is in block units and is
//   the number the executor builds the gather neighbourhood from — is
//   `ceil(radius / smallest core on that axis)`, clamped to the lattice.
//
// Both come from `StructuringElement::radius`, and there is no field that sets
// either. This module's header states the rule and `element.rs` states the
// reason: a reach fed by anything a caller can set makes the guard compare a
// number against itself.
//
// **Boundedness is the plannability condition, and it is worth saying plainly.**
// A kernel with a finite radius makes the set of blocks that can reach this one
// finite and computable before the run; a kernel whose support depended on the
// data — a splat whose radius is a per-point field, a deposit that falls off
// forever — would make it unknowable, and the only honest declarations left
// would be "the whole lattice" (a full-reach phase, which `fragment.rs` explains
// is a reboot) or a number that is silently short. A `StructuringElement` is
// finite by construction, so this op cannot be handed the unbounded case; if it
// ever grows a kernel that could be, [`VoxelizeOp::new`] is where it must refuse
// rather than under-declare.
//
// Why the divisor is the nominal block edge, and what that rests on
// ------------------------------------------------------------------
// A block reach is a count of **block indices**, so the conversion asks how far
// one index step is worth in voxels. `ceil(radius / edge)` is not merely safe
// here, it is *exact*, and it is exact because of one property of the lattice:
//
// > **`BlockGrid` starts every core at `index * edge`, so the only short block
// > on an axis is the last one.**
//
// That is the premise, and everything below is arithmetic on it. Reaching
// *down* from block `b`, whose core starts at `b * edge`, crosses only blocks
// below `b`, and every one of those is full — so `R` steps are worth `R * edge`
// and `R = ceil(radius / edge)` covers `[lo - radius, lo)` exactly. Reaching
// *up*, a step that lands on the short block has already reached the end of the
// volume, and there is nothing beyond it to be short of.
//
// The obvious conservative alternative is to divide by the **smallest** core on
// the axis — `volume - (blocks - 1) * edge` — which is what a lattice whose
// short block sat in the middle would need. It is not used, because under the
// premise above it buys nothing and costs a wider halo at every block: on a
// `[20]` axis of `[8]` blocks a radius of 8 would fetch two blocks where one
// suffices, and that is paid by every task of the phase, on every run.
//
// [`smallest_core`] computes it anyway, and the test
// `the_two_divisors_are_both_sufficient_today_and_do_not_agree` sweeps both
// against the geometry — a core dilated by the radius, intersected with another
// core — and asserts that the one this op uses is sufficient. **The day
// `BlockGrid` grows a layout whose short block is not last, that test fails and
// names the premise it was protecting.**
//
// Where the points must be, and what that rules out
// -------------------------------------------------
// **A point in block B's fragment must lie in B's core.** Cores tile the volume,
// so this makes "which block owns this point" a total function of the
// coordinate, which is what makes a point on a seam land in exactly one
// fragment, and it is the premise the reach derivation rests on: if a producer
// could put a point anywhere, no finite block reach would be correct. A fragment
// that breaks it is refused, naming the point and the core — not dropped, and
// not silently rendered in the wrong place. The same refusal covers a point
// outside the volume, which is outside every core.
//
// What that rules out is real and is stated rather than discovered: a producer
// whose points may land outside its own core — the centroid of an object that
// straddles a seam, for instance — cannot feed this op directly. Its points have
// to be re-keyed to the block that owns them first. Expressing it here would
// need a second radius, "how far a point may sit from the block that emitted
// it", and that is a parameter about the *producer*, not about this op.
//
// Accumulation order is part of the answer
// ----------------------------------------
// Two points whose kernels overlap add into the same voxel, and floating-point
// `+` is not associative, so the order the contributions are summed in is
// observable in the result. It is fixed here rather than left to the gather:
//
// * every contribution is accumulated in **`f64`**, whatever the level's element
//   type. An `f32` accumulator would put the disagreement at around `1e-7` of
//   the total, which is small enough to look like noise and large enough to make
//   two decompositions of the same data differ;
// * contributions are summed in the order **(point position, then source block,
//   then position within that block's fragment)**, and the kernel's offsets are
//   walked in `StructuringElement`'s canonical order. So the answer is a
//   function of the point set in *volume* coordinates and of nothing else.
//
// The first key is the position rather than the source block, and that is the
// load-bearing choice. Sorting by source block first would be deterministic for
// one lattice and **not decomposition-invariant**: two points that share a block
// under one cut sit in different blocks under another, and their relative order
// flips with the cut. Points that share a *position* share a block under every
// cut, so the tie-break falls through to the producer's own listing and stays
// put. Decomposition invariance is the property this crate exists to defend, so
// it wins over the cheaper key.
//
// At the volume boundary
// ----------------------
// A kernel that overhangs the edge **contributes its overlapping part**: the
// voxels beyond the edge do not exist, so there is nothing to deposit into and
// nothing to redistribute. The total mass of the output is then
// `sum over points of weight * |kernel ∩ volume|`, which is exactly computable
// from the definition — that is what the tests assert against. Normalising the
// kernel to unit mass is a caller's decision and is expressed as a weight of
// `1 / |kernel|`, not as a mode of this op.
//
// Coverage
// --------
// This op declares **no output stream**. `Coverage` is a decision about a
// *fragment* output, and this phase's output is a pixel level: the tiling guard
// over its valid regions is a real guard here, not the vacuous one
// `fragment.rs` warns about, because the valid regions are what the pixels are
// written from. Inventing a stream to have something to declare would add a
// second thing to keep consistent and no second check.
//
// On the input side the op tolerates either coverage, and that is deliberate: to
// a sum, "the block wrote an empty list" and "the block wrote nothing" are the
// same fact. A producer should still declare `Coverage::EveryBlock` — the
// distinction is free to write and the guard is worth more than the bytes — but
// a `Sparse` stream is not wrong *here*, and this op is one of the few readers
// for which that is true.
//
// What this deliberately does not do
// ----------------------------------
// **Sub-voxel positions.** A point names a voxel. A caller holding fractional
// coordinates rounds before writing the fragment, because which way to round —
// and whether to split a point's mass across the neighbouring voxels instead —
// is a decision about their data, and a splitting kernel is a different
// operation with a different reach.

use ndarray::{Array3, ArrayViewMut3, Axis, Slice};

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::fragment::{BlockOutput, BlockView, FragmentInput, FragmentOp};
use crate::geometry::BlockGrid;
use crate::region::Region;
use crate::sidecar::check_stream_name;
use crate::voxels::{VoxelElement, Voxels};

use super::element::{ElementShape, StructuringElement};

// ----------------------------------------------------------------- points --

/// The point type and its encoding live in [`crate::points`], with the store
/// that keeps point sets, and are re-exported here so that this op's callers
/// read one module rather than two.
///
/// They moved because a point is not a fact about this op: it is what a set of
/// positions is made of, and `points` is where a set of positions is sorted,
/// indexed and queried. Leaving a second definition here would have been two
/// encodings of the same four words, which is the shape of a bug that only
/// appears once the two have drifted.
pub use crate::points::{decode_points, encode_points, Point, WORDS_PER_POINT};

// ---------------------------------------------------------------- kernels --

/// The kernel of the pure count: one voxel, the point's own.
///
/// A named constructor rather than a note in the documentation, because "render
/// the density" is the commonest thing this op is asked for and
/// `from_radius(Box, [0, 0, 0])` does not read as it.
pub fn single_voxel() -> StructuringElement {
    StructuringElement::from_radius(ElementShape::Box, [0, 0, 0])
}

/// A ball of the given per-axis radius: the ellipsoid inscribed in the box,
/// which is `element.rs`'s [`ElementShape::Ellipsoid`].
///
/// Per-axis rather than one number because a volume's voxels are not required to
/// be cubes, and which radius belongs on which axis is the caller's knowledge.
pub fn ball(radius: [usize; 3]) -> StructuringElement {
    StructuringElement::from_radius(ElementShape::Ellipsoid, radius)
}

// ----------------------------------------------------------------- kernel --

/// One point's contribution, with everything the summation order is defined by.
#[derive(Debug, Clone, Copy)]
struct Contribution {
    at: [usize; 3],
    block: [usize; 3],
    index: usize,
    weight: f64,
}

/// Render `fragments` into `window`.
///
/// `out` covers `window` of the volume `grid` is cut from; every contribution
/// that falls outside `window` is dropped, which is how a block renders its own
/// part of the answer without knowing anything about the others. Points are
/// **not** required to lie inside `window` — a point outside it whose kernel
/// reaches in is exactly the case the block reach exists for — but every point
/// must lie inside the core of the block whose fragment carries it, and this is
/// where that is checked.
///
/// Free function first, `FragmentOp` shell on top, for this module's usual
/// reason: the order the sum is taken in is a property of *this* function, and a
/// test can permute its argument. A test cannot permute what the executor
/// gathers.
pub fn voxelize_into(
    fragments: &[([usize; 3], Vec<Point>)],
    grid: &BlockGrid,
    element: &StructuringElement,
    window: &Region,
    mut out: ArrayViewMut3<'_, f64>,
) -> Result<()> {
    let volume = grid.volume();
    if window.ndim() != 3 {
        return Err(Error::invalid(format!(
            "voxelize: a window is 3-D, got rank {}",
            window.ndim()
        )));
    }
    window.check_within(&volume, "voxelize window")?;
    let shape = [out.shape()[0], out.shape()[1], out.shape()[2]];
    if shape.to_vec() != window.shape {
        return Err(Error::ShapeMismatch {
            expected: window.shape.clone(),
            got: shape.to_vec(),
        });
    }

    let mut contributions = Vec::new();
    for (block, points) in fragments {
        let core = core_of(grid, *block).ok_or_else(|| {
            Error::invalid(format!(
                "voxelize: a fragment is keyed to block {block:?}, which is outside this \
                 phase's lattice of {:?} blocks",
                grid.blocks_per_axis()
            ))
        })?;
        for (index, point) in points.iter().enumerate() {
            if !contains(&core, point.at) {
                return Err(Error::invalid(format!(
                    "voxelize: point {index} of block {block:?} is at {:?}, which is outside \
                     that block's core {:?}..{:?}. A point must lie in the core of the block \
                     whose fragment carries it: cores tile the volume, so that is what makes \
                     every point owned by exactly one block, and it is the premise the block \
                     reach is derived from. A producer whose points may land outside its own \
                     core must re-key them to the block that owns them.",
                    point.at,
                    core.start,
                    core.end()
                )));
            }
            contributions.push(Contribution {
                at: point.at,
                block: *block,
                index,
                weight: point.weight,
            });
        }
    }

    // The stated order, and the reason it is this one is in the module header:
    // position first, so that the answer does not depend on how the volume was
    // cut. The key is total — no two contributions share (block, index) — so
    // the sort needs no stability of its own.
    contributions.sort_unstable_by(|left, right| {
        left.at
            .cmp(&right.at)
            .then(left.block.cmp(&right.block))
            .then(left.index.cmp(&right.index))
    });

    for contribution in &contributions {
        for offset in element.offsets() {
            let mut inside = true;
            let mut local = [0usize; 3];
            for axis in 0..3 {
                let position = contribution.at[axis] as isize + offset[axis];
                // Beyond the volume there is no voxel to deposit into: the
                // kernel is clipped, not redistributed.
                if position < 0 || position as usize >= volume[axis] {
                    inside = false;
                    break;
                }
                let position = position as usize;
                if position < window.start[axis] || position >= window.start[axis] + shape[axis] {
                    inside = false;
                    break;
                }
                local[axis] = position - window.start[axis];
            }
            if inside {
                out[local] += contribution.weight;
            }
        }
    }
    Ok(())
}

/// The core of `index` on `grid`, or `None` for an index off the lattice.
///
/// Recomputed from `volume` and `block` rather than searched for in
/// `BlockGrid::cores`, which allocates every core of the lattice to answer one
/// question about one of them.
fn core_of(grid: &BlockGrid, index: [usize; 3]) -> Option<Region> {
    let volume = grid.volume();
    let edge = grid.block();
    let mut start = [0usize; 3];
    let mut shape = [0usize; 3];
    for axis in 0..3 {
        start[axis] = index[axis].checked_mul(edge[axis])?;
        if start[axis] >= volume[axis] {
            return None;
        }
        shape[axis] = (start[axis] + edge[axis]).min(volume[axis]) - start[axis];
    }
    Some(Region::new(&start, &shape))
}

fn contains(region: &Region, at: [usize; 3]) -> bool {
    (0..3).all(|axis| {
        at[axis] >= region.start[axis] && at[axis] < region.start[axis] + region.shape[axis]
    })
}

/// The smallest core extent on `axis` — `volume - (blocks - 1) * edge`, which
/// is the last block's, because `BlockGrid` starts every core at `index * edge`.
///
/// **This is the divisor the block reach is deliberately *not* derived with.**
/// It is the conservative one: correct for a lattice whose short block could sit
/// anywhere, and wider than necessary for the one `BlockGrid` produces. It is
/// here because a premise that nothing computes is a premise nothing can test,
/// and `the_two_divisors_are_both_sufficient_today_and_do_not_agree` is what
/// stands between the derivation and a change of layout. See the module header.
pub fn smallest_core(grid: &BlockGrid, axis: usize) -> usize {
    let volume = grid.volume()[axis];
    let edge = grid.block()[axis];
    let blocks = volume.div_ceil(edge);
    edge.min(volume - (blocks - 1) * edge)
}

/// The block reach a kernel of `radius` needs on `axis`, on this lattice:
/// `ceil(radius / edge)`, which is exact under the layout the module header
/// states.
///
/// Clamped to `blocks - 1`, which is the whole lattice: reaching further is not
/// wider, it is only a wider halo, and `neighbourhood` clamps there anyway.
pub fn block_reach(grid: &BlockGrid, radius: usize, axis: usize) -> usize {
    let blocks = grid.blocks_per_axis()[axis];
    radius.div_ceil(grid.block()[axis]).min(blocks - 1)
}

// ------------------------------------------------------------------ shell --

/// Render a stream of points into a volume.
///
/// Reads one fragment stream, writes pixels, and declares no fragment stream of
/// its own. Both reaches are derived from the kernel and the lattice; see the
/// module header.
pub struct VoxelizeOp {
    name: &'static str,
    stream: String,
    stream_phase: usize,
    element: StructuringElement,
    grid: BlockGrid,
    block_reach: [usize; 3],
}

impl VoxelizeOp {
    /// The lattice is an argument because the block reach is a statement in
    /// block indices and cannot be derived without one — the same reason
    /// `probes::FragmentReduceOp` takes one. It is kept, not just measured: the
    /// phase this op runs on must be cut the same way, and [`Self::apply`]
    /// refuses a `BlockView` whose grid disagrees rather than reaching with a
    /// number derived for a different lattice.
    pub fn new(
        name: &'static str,
        stream: impl Into<String>,
        stream_phase: usize,
        element: StructuringElement,
        grid: &BlockGrid,
    ) -> Result<Self> {
        let stream = stream.into();
        check_stream_name(&stream)?;
        if element.is_empty() {
            return Err(Error::invalid(format!(
                "voxelize op {name:?} was given a kernel with no voxels in it, so every point \
                 would deposit nothing and the output would be indistinguishable from one with \
                 no points at all"
            )));
        }
        let mut reach = [0usize; 3];
        for (axis, value) in reach.iter_mut().enumerate() {
            *value = block_reach(grid, element.reach(axis), axis);
        }
        Ok(Self {
            name,
            stream,
            stream_phase,
            element,
            grid: grid.clone(),
            block_reach: reach,
        })
    }

    pub fn element(&self) -> &StructuringElement {
        &self.element
    }

    /// The declared reach in **blocks**, which is what the executor builds the
    /// gather neighbourhood from. Public so a test can check it against the
    /// geometry rather than against itself.
    pub fn block_reach(&self) -> [usize; 3] {
        self.block_reach
    }

    pub fn stream(&self) -> &str {
        &self.stream
    }

    /// The lattice this op's reach was derived for is the lattice it may run on.
    ///
    /// A phase cut differently would hand this op a `BlockView` whose block
    /// indices mean something else, and the gather neighbourhood would be a
    /// number computed for a grid that is not the one being walked. Nothing else
    /// checks it: `check_phase_work` compares the *input stream's* lattice with
    /// the phase's, which is a different pair.
    pub fn check_grid(&self, grid: &BlockGrid) -> Result<()> {
        if grid.volume() != self.grid.volume() || grid.block() != self.grid.block() {
            return Err(Error::invalid(format!(
                "voxelize op {:?} derived its block reach {:?} from a lattice of {:?}/{:?} and \
                 is running on {:?}/{:?}. The reach is a count of block indices, so it means \
                 something else on another lattice; build the op with the grid the phase runs \
                 on.",
                self.name,
                self.block_reach,
                self.grid.volume(),
                self.grid.block(),
                grid.volume(),
                grid.block()
            )));
        }
        Ok(())
    }
}

impl FragmentOp for VoxelizeOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// The kernel's radius, clamped to the volume — beyond the volume there is
    /// nothing to read, so claiming more would only make the halo wider.
    /// Derived from the element, and there is no field that could set it.
    fn reach(&self, axis: usize, volume_len: usize) -> usize {
        self.element.reach(axis).min(volume_len)
    }

    /// No pixel IO on the way in: the input is the fragment stream, and a phase
    /// running this op reads not one voxel of the level below it.
    fn reads_pixels(&self) -> bool {
        false
    }

    fn writes_pixels(&self) -> bool {
        true
    }

    /// Gathered rather than streamed. The reach is bounded by the kernel radius
    /// and is therefore a handful of neighbours, not the lattice; gathering is
    /// what makes the fetch count an observable property of the declaration —
    /// `fragment.rs` on [`FragmentOp::gathers`] — and there is no residency
    /// argument for streaming a neighbourhood this size.
    fn gathers(&self) -> bool {
        true
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own(self.stream.clone(), self.stream_phase).with_reach(self.block_reach)]
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        self.check_grid(at.grid)?;

        let mut fragments = Vec::new();
        for (key, bytes) in at.fragments(&self.stream) {
            fragments.push((key.block, decode_points(bytes)?));
        }

        let mut pixels = at.output_buffer(0.0)?;
        // A simulated environment carries no array, and the honest thing is to
        // allocate no accumulator for it either: those runs exist for volumes
        // that could not be held, so a block-sized `f64` buffer is exactly what
        // must not be built. `BlockBuf::as_array_mut` is documented for this.
        if pixels.as_array_mut().is_none() {
            return Ok(BlockOutput::nothing().with_pixels(pixels));
        }

        // Only the valid region is filled. The executor slices it out of the
        // read extent and writes that; the halo margin around it is a region
        // this block cannot answer for — its own contributors may lie beyond
        // the gathered neighbourhood — so it is left at zero rather than filled
        // with a number nobody may trust.
        let valid = at.valid;
        let mut accumulated =
            Array3::<f64>::zeros((valid.shape[0], valid.shape[1], valid.shape[2]));
        voxelize_into(
            &fragments,
            at.grid,
            &self.element,
            valid,
            accumulated.view_mut(),
        )?;

        let mut offset = [0usize; 3];
        for axis in 0..3 {
            offset[axis] = valid.start[axis] - at.read.start[axis];
        }
        if let Some(array) = pixels.as_array_mut() {
            store(&accumulated, offset, array)?;
        }
        Ok(BlockOutput::nothing().with_pixels(pixels))
    }
}

/// Write an `f64` accumulator into `at` of a buffer of the level's own type.
///
/// Rounded to nearest for the integer types, for `resample.rs`'s reason:
/// truncating biases every deposit down by up to a whole unit, and a count that
/// is systematically low is the kind of wrongness nobody notices. `bool` is not
/// rounded — it takes `VoxelElement::from_f64`'s stated convention, non-zero is
/// true, so a `bool` level answers "did any point's kernel cover this voxel".
fn store(accumulated: &Array3<f64>, at: [usize; 3], into: &mut Voxels) -> Result<()> {
    let shape = [
        accumulated.shape()[0],
        accumulated.shape()[1],
        accumulated.shape()[2],
    ];
    let held = into.shape();
    for axis in 0..3 {
        if at[axis] + shape[axis] > held[axis] {
            return Err(Error::invalid(format!(
                "voxelize: writing {shape:?} at {at:?} of a block of {held:?} would run off \
                 axis {axis}"
            )));
        }
    }
    macro_rules! arm {
        ($type:ty, $round:expr) => {{
            let mut window = into.view_mut::<$type>()?;
            for axis in 0..3 {
                window
                    .slice_axis_inplace(Axis(axis), Slice::from(at[axis]..at[axis] + shape[axis]));
            }
            for (target, &value) in window.iter_mut().zip(accumulated.iter()) {
                *target = <$type>::from_f64(if $round { value.round() } else { value });
            }
        }};
    }
    match into.dtype() {
        Dtype::Bool => arm!(bool, false),
        Dtype::U8 => arm!(u8, true),
        Dtype::U16 => arm!(u16, true),
        Dtype::U32 => arm!(u32, true),
        Dtype::U64 => arm!(u64, true),
        Dtype::I8 => arm!(i8, true),
        Dtype::I16 => arm!(i16, true),
        Dtype::I32 => arm!(i32, true),
        Dtype::I64 => arm!(i64, true),
        Dtype::F32 => arm!(f32, false),
        Dtype::F64 => arm!(f64, false),
        Dtype::F16 => {
            return Err(Error::invalid(
                "voxelize: no buffer holds half-precision; a level of it could not have been \
                 allocated in the first place",
            ))
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fragment::{fragment_phase, neighbourhood};

    fn window_of(volume: [usize; 3]) -> Region {
        Region::whole(&volume)
    }

    /// Render a point set over one block, which is the reference every
    /// decomposition is compared against.
    fn whole(
        volume: [usize; 3],
        element: &StructuringElement,
        points: &[Point],
    ) -> Result<Array3<f64>> {
        let grid = BlockGrid::whole(volume)?;
        let mut out = Array3::<f64>::zeros((volume[0], volume[1], volume[2]));
        voxelize_into(
            &[([0, 0, 0], points.to_vec())],
            &grid,
            element,
            &window_of(volume),
            out.view_mut(),
        )?;
        Ok(out)
    }

    /// Which block of `grid` owns `at` — the total function the "a point lies in
    /// its writer's core" rule makes possible, and what a producer keys by.
    fn owner(grid: &BlockGrid, at: [usize; 3]) -> [usize; 3] {
        let edge = grid.block();
        [at[0] / edge[0], at[1] / edge[1], at[2] / edge[2]]
    }

    /// A point set split into per-block fragments the way a producer must.
    fn split(grid: &BlockGrid, points: &[Point]) -> Vec<([usize; 3], Vec<Point>)> {
        let mut out: std::collections::BTreeMap<[usize; 3], Vec<Point>> = Default::default();
        for point in points {
            out.entry(owner(grid, point.at)).or_default().push(*point);
        }
        out.into_iter().collect()
    }

    #[test]
    fn a_single_voxel_kernel_deposits_the_weight_at_the_point_and_nowhere_else() {
        let element = single_voxel();
        assert_eq!(element.len(), 1);
        let rendered = whole(
            [4, 4, 4],
            &element,
            &[Point::unit([1, 2, 3]), Point::weighted([0, 0, 0], 2.5)],
        )
        .unwrap();
        assert_eq!(rendered[[1, 2, 3]], 1.0);
        assert_eq!(rendered[[0, 0, 0]], 2.5);
        assert_eq!(rendered.iter().sum::<f64>(), 3.5);
    }

    /// The mass of a ball is its member count, per point, when nothing clips it.
    /// Derived from the element rather than from a run of this code.
    #[test]
    fn a_ball_deposits_the_weight_into_every_voxel_of_the_element() {
        let element = ball([1, 1, 1]);
        let rendered = whole([9, 9, 9], &element, &[Point::weighted([4, 4, 4], 3.0)]).unwrap();
        assert_eq!(rendered.iter().sum::<f64>(), 3.0 * element.len() as f64);
        for offset in element.offsets() {
            let at = [
                (4 + offset[0]) as usize,
                (4 + offset[1]) as usize,
                (4 + offset[2]) as usize,
            ];
            assert_eq!(rendered[at], 3.0, "member {offset:?} got nothing");
        }
        assert_eq!(
            rendered.iter().filter(|&&value| value != 0.0).count(),
            element.len()
        );
    }

    /// The stated edge rule, against a number that comes from the definition:
    /// the mass left is the members whose offset stays inside the volume.
    #[test]
    fn a_kernel_that_overhangs_the_volume_contributes_its_overlapping_part() {
        let volume = [6usize, 6, 6];
        let element = ball([2, 2, 2]);
        let at = [0usize, 3, 3];
        let expected = element
            .offsets()
            .iter()
            .filter(|offset| {
                (0..3).all(|axis| {
                    let position = at[axis] as isize + offset[axis];
                    position >= 0 && (position as usize) < volume[axis]
                })
            })
            .count();
        assert!(
            expected < element.len(),
            "this point must actually overhang"
        );
        let rendered = whole(volume, &element, &[Point::unit(at)]).unwrap();
        assert_eq!(rendered.iter().sum::<f64>(), expected as f64);
    }

    #[test]
    fn a_point_outside_its_writers_core_is_refused_and_so_is_one_outside_the_volume() {
        let grid = BlockGrid::new([8, 4, 4], [4, 4, 4]).unwrap();
        let mut out = Array3::<f64>::zeros((8, 4, 4));
        // in the volume, but in block 1's core rather than block 0's
        let error = voxelize_into(
            &[([0, 0, 0], vec![Point::unit([5, 0, 0])])],
            &grid,
            &single_voxel(),
            &window_of([8, 4, 4]),
            out.view_mut(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("outside that block's core"), "{error}");

        // outside the volume altogether: the same refusal, because cores tile
        let error = voxelize_into(
            &[([1, 0, 0], vec![Point::unit([9, 0, 0])])],
            &grid,
            &single_voxel(),
            &window_of([8, 4, 4]),
            out.view_mut(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("outside that block's core"), "{error}");
        assert_eq!(out.iter().sum::<f64>(), 0.0, "a refused run deposited");
    }

    #[test]
    fn a_fragment_keyed_off_the_lattice_is_refused() {
        let grid = BlockGrid::new([8, 4, 4], [4, 4, 4]).unwrap();
        let mut out = Array3::<f64>::zeros((8, 4, 4));
        let error = voxelize_into(
            &[([7, 0, 0], vec![Point::unit([0, 0, 0])])],
            &grid,
            &single_voxel(),
            &window_of([8, 4, 4]),
            out.view_mut(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("outside this phase's lattice"), "{error}");
    }

    /// The order the module header states, and the demonstration that it is
    /// load-bearing: the same three weights summed the other way round differ in
    /// the last bits, so a run that accumulated in gather order would produce a
    /// different volume from one that did not.
    #[test]
    fn permuting_the_fragments_does_not_change_one_bit() {
        let volume = [16usize, 4, 4];
        let grid = BlockGrid::new(volume, [2, 4, 4]).unwrap();
        let element = ball([2, 0, 0]);
        // three points whose kernels all cover voxel 7, in three blocks
        let points = [
            Point::weighted([5, 1, 1], 1e-16),
            Point::weighted([7, 1, 1], 1e-16),
            Point::weighted([9, 1, 1], 1.0),
        ];
        // Not vacuous: summed from zero, these three do not associate. Two
        // half-ulp weights added to each other survive; added to `1.0` one at a
        // time they vanish.
        let in_position_order = ((0.0 + points[0].weight) + points[1].weight) + points[2].weight;
        let reversed = ((0.0 + points[2].weight) + points[1].weight) + points[0].weight;
        assert_ne!(
            in_position_order.to_bits(),
            reversed.to_bits(),
            "the weights chosen are order-insensitive, so this test proves nothing"
        );

        let mut fragments = split(&grid, &points);
        let mut reference = Array3::<f64>::zeros((volume[0], volume[1], volume[2]));
        voxelize_into(
            &fragments,
            &grid,
            &element,
            &window_of(volume),
            reference.view_mut(),
        )
        .unwrap();
        assert_eq!(fragments.len(), 3, "the points must land in three blocks");

        fragments.reverse();
        let mut permuted = Array3::<f64>::zeros((volume[0], volume[1], volume[2]));
        voxelize_into(
            &fragments,
            &grid,
            &element,
            &window_of(volume),
            permuted.view_mut(),
        )
        .unwrap();
        for (left, right) in reference.iter().zip(permuted.iter()) {
            assert_eq!(left.to_bits(), right.to_bits());
        }
        // and the voxel all three cover holds the sum taken in position order,
        // which is the order the module header states
        assert_eq!(reference[[7, 1, 1]].to_bits(), in_position_order.to_bits());
    }

    /// Sorting by source block first would be deterministic and *not*
    /// decomposition-invariant. This is the case that shows the difference: the
    /// same three points, cut two ways, and the answer must not move.
    /// The listing order is deliberately *not* the position order: under a cut
    /// that puts all three in one fragment, a sum taken in listing order gives
    /// `1.0` and one taken in position order gives `1.0` plus an ulp. So a
    /// source-block-first key would make the answer depend on the cut, and that
    /// is what this pins.
    #[test]
    fn two_cuts_of_the_same_points_sum_in_the_same_order() {
        let volume = [16usize, 16, 1];
        let element = ball([3, 3, 0]);
        let points = [
            Point::weighted([8, 8, 0], 1.0),
            Point::weighted([7, 7, 0], 1e-16),
            Point::weighted([7, 8, 0], 1e-16),
        ];
        // voxel [7, 8, 0] is covered by all three; in position order the two
        // small weights meet each other before they meet the large one
        let in_position_order: f64 = ((0.0 + 1e-16) + 1e-16) + 1.0;
        let in_listing_order: f64 = ((0.0 + 1.0) + 1e-16) + 1e-16;
        assert_ne!(in_position_order.to_bits(), in_listing_order.to_bits());

        let mut rendered = Vec::new();
        for block in [[16usize, 16, 1], [8, 16, 1], [8, 8, 1], [5, 5, 1]] {
            let grid = BlockGrid::new(volume, block).unwrap();
            let mut out = Array3::<f64>::zeros((volume[0], volume[1], volume[2]));
            voxelize_into(
                &split(&grid, &points),
                &grid,
                &element,
                &window_of(volume),
                out.view_mut(),
            )
            .unwrap();
            assert_eq!(out[[7, 8, 0]].to_bits(), in_position_order.to_bits());
            rendered.push(out);
        }
        for other in &rendered[1..] {
            for (left, right) in rendered[0].iter().zip(other.iter()) {
                assert_eq!(left.to_bits(), right.to_bits());
            }
        }
    }

    // The encoding's own round trip and its three refusals moved to
    // `crate::points`, with the type they are about.

    // ------------------------------------------------------------- reaches --

    /// The premise the derivation rests on, checked directly: the only core
    /// shorter than the nominal edge is the last one on its axis.
    #[test]
    fn only_the_last_core_on_an_axis_is_ever_short() {
        for volume in [[20usize, 9, 5], [32, 8, 4], [17, 17, 1]] {
            for edge in [[8usize, 4, 2], [5, 3, 1], [3, 9, 5]] {
                let grid = BlockGrid::new(volume, edge).unwrap();
                let counts = grid.blocks_per_axis();
                for core in grid.cores() {
                    for axis in 0..3 {
                        if core.core.shape[axis] < grid.block()[axis] {
                            assert_eq!(
                                core.index[axis],
                                counts[axis] - 1,
                                "block {:?} of {volume:?}/{edge:?} is short on axis {axis} \
                                 without being last: the block reach divides by the nominal \
                                 edge and that is only exact while this holds",
                                core.index
                            );
                        }
                        assert_eq!(core.core.start[axis], core.index[axis] * grid.block()[axis]);
                    }
                }
            }
        }
    }

    #[test]
    fn the_smallest_core_is_the_partial_block_where_there_is_one() {
        let even = BlockGrid::new([32, 4, 4], [8, 4, 4]).unwrap();
        assert_eq!(smallest_core(&even, 0), 8);
        // 20 = 8 + 8 + 4, so the last core on the axis is half an edge
        let ragged = BlockGrid::new([20, 4, 4], [8, 4, 4]).unwrap();
        assert_eq!(ragged.blocks_per_axis(), [3, 1, 1]);
        assert_eq!(smallest_core(&ragged, 0), 4);
        assert_eq!(smallest_core(&ragged, 1), 4);
        // one block: the core is the volume
        let single = BlockGrid::whole([20, 4, 4]).unwrap();
        assert_eq!(smallest_core(&single, 0), 20);
    }

    #[test]
    fn the_block_reach_is_the_radius_over_the_block_edge_clamped_to_the_lattice() {
        let ragged = BlockGrid::new([20, 4, 4], [8, 4, 4]).unwrap();
        assert_eq!(block_reach(&ragged, 0, 0), 0);
        assert_eq!(block_reach(&ragged, 1, 0), 1, "ceil(1 / 8)");
        assert_eq!(block_reach(&ragged, 8, 0), 1, "ceil(8 / 8)");
        assert_eq!(block_reach(&ragged, 9, 0), 2, "ceil(9 / 8)");
        // the partial last core does not shorten the step: 20 = 8 + 8 + 4, and
        // a radius of 8 from any core still lands within one block index
        assert_eq!(smallest_core(&ragged, 0), 4);
        assert_eq!(block_reach(&ragged, 8, 0), 1);
        // 3 blocks: reaching two is reaching all of them, and no further
        assert_eq!(block_reach(&ragged, 40, 0), 2);
        // one block on the axis: nothing to reach for
        assert_eq!(block_reach(&BlockGrid::whole([20, 4, 4]).unwrap(), 9, 0), 0);
    }

    /// Every block whose core can reach this one must be in the gathered
    /// neighbourhood. **This is the reach test**: it is stated in geometry — a
    /// core dilated by the radius, intersected with another core — and it fails
    /// for any under-declaration, on any lattice in the sweep.
    fn reach_covers_everything(divisor: fn(&BlockGrid, usize) -> usize) -> Vec<String> {
        let mut short = Vec::new();
        for volume in [[20usize, 9, 5], [32, 8, 4], [17, 17, 1], [8, 3, 3]] {
            for edge in [[8usize, 4, 2], [5, 3, 1], [3, 9, 5], [32, 8, 4]] {
                let grid = BlockGrid::new(volume, edge).unwrap();
                let counts = grid.blocks_per_axis();
                for radius in [[0usize, 0, 0], [1, 1, 1], [4, 2, 1], [9, 5, 3], [12, 0, 0]] {
                    let mut reach = [0usize; 3];
                    for axis in 0..3 {
                        reach[axis] = radius[axis]
                            .div_ceil(divisor(&grid, axis).max(1))
                            .min(counts[axis] - 1);
                    }
                    for target in grid.cores() {
                        let gathered = neighbourhood(target.index, reach, counts);
                        for source in grid.cores() {
                            // does any point of `source`'s core have a kernel
                            // that touches `target`'s core?
                            let touches = (0..3).all(|axis| {
                                let (a_lo, a_hi) = (
                                    source.core.start[axis].saturating_sub(radius[axis]),
                                    source.core.start[axis]
                                        + source.core.shape[axis]
                                        + radius[axis],
                                );
                                let b_lo = target.core.start[axis];
                                let b_hi = b_lo + target.core.shape[axis];
                                a_lo < b_hi && b_lo < a_hi
                            });
                            if touches && !gathered.contains(&source.index) {
                                short.push(format!(
                                    "volume {volume:?} block {edge:?} radius {radius:?}: block \
                                     {:?} does not gather {:?}",
                                    target.index, source.index
                                ));
                            }
                        }
                    }
                }
            }
        }
        short
    }

    /// The divisor the op actually uses.
    fn nominal(grid: &BlockGrid, axis: usize) -> usize {
        grid.block()[axis]
    }

    #[test]
    fn the_declared_neighbourhood_holds_every_block_that_can_reach_this_one() {
        let short = reach_covers_everything(nominal);
        assert!(
            short.is_empty(),
            "the derived block reach is short in {} case(s); first: {}",
            short.len(),
            short[0]
        );
    }

    /// What the derivation chose, and what it declined, measured rather than
    /// asserted in prose. The op divides by the **nominal edge**, which is exact
    /// because `BlockGrid` puts the short block last on every axis and a step
    /// that lands on it has already reached the end of the volume. The
    /// conservative alternative is the smallest core; it is also sufficient, and
    /// wider wherever an edge block is partial, which is the cost the op
    /// declines to pay at every block.
    ///
    /// **This is the test protecting the layout premise.** If `BlockGrid` ever
    /// produces a lattice whose short block is not the last one, the first
    /// assertion fails and names the case.
    #[test]
    fn the_two_divisors_are_both_sufficient_today_and_do_not_agree() {
        let short = reach_covers_everything(nominal);
        assert!(
            short.is_empty(),
            "dividing by the nominal edge is short in {} case(s): `BlockGrid` no longer puts \
             the only short core last on its axis, which is the premise the block reach \
             derivation rests on. Divide by `smallest_core` instead. First: {}",
            short.len(),
            short[0]
        );
        let conservative = reach_covers_everything(smallest_core);
        assert!(conservative.is_empty(), "first: {}", conservative[0]);

        // and they are not the same number: on a ragged axis the conservative
        // divisor fetches a second block for a radius that fits in one
        let ragged = BlockGrid::new([20, 4, 4], [8, 4, 4]).unwrap();
        assert_eq!(nominal(&ragged, 0), 8);
        assert_eq!(smallest_core(&ragged, 0), 4);
        assert_eq!(block_reach(&ragged, 8, 0), 1);
        assert_eq!(8usize.div_ceil(smallest_core(&ragged, 0)), 2);
    }

    /// The two reaches are the same parameter said twice, and the phase built
    /// from them keeps every block's core trustworthy.
    #[test]
    fn the_phase_a_voxelize_op_builds_keeps_every_core_valid() {
        let grid = BlockGrid::new([20, 4, 4], [8, 4, 4]).unwrap();
        let op = VoxelizeOp::new("voxelize", "points", 0, ball([5, 1, 1]), &grid).unwrap();
        assert_eq!(op.block_reach(), [1, 0, 0], "ceil(5 / 8)");
        assert_eq!(op.reach(0, 20), 5);
        assert_eq!(op.reach(1, 4), 1);
        let phase = fragment_phase(&op, grid).unwrap();
        assert_eq!(phase.reach, [5, 1, 1]);
        // the halo is the wider of the two statements, in voxels: one block of
        // 8 on the split axis, the radius itself on the two that are not cut
        assert_eq!(phase.halo, [8, 1, 1]);
        for block in &phase.blocks {
            assert_eq!(block.valid, block.core, "block {:?}", block.index);
        }
    }

    #[test]
    fn an_op_refuses_a_lattice_its_reach_was_not_derived_for() {
        let grid = BlockGrid::new([20, 4, 4], [8, 4, 4]).unwrap();
        let op = VoxelizeOp::new("voxelize", "points", 0, ball([5, 1, 1]), &grid).unwrap();
        op.check_grid(&grid).unwrap();
        let error = op
            .check_grid(&BlockGrid::new([20, 4, 4], [4, 4, 4]).unwrap())
            .unwrap_err()
            .to_string();
        assert!(error.contains("derived its block reach"), "{error}");
    }

    #[test]
    fn an_unusable_stream_name_is_refused_at_construction() {
        let grid = BlockGrid::new([20, 4, 4], [8, 4, 4]).unwrap();
        assert!(VoxelizeOp::new("voxelize", "", 0, single_voxel(), &grid).is_err());
        assert!(VoxelizeOp::new("voxelize", "a/b", 0, single_voxel(), &grid).is_err());
    }
}
