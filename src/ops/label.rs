// SPDX-License-Identifier: MIT
//
// Original work for this crate. Written from the definition of the operation —
// stamp one integer per point into a volume — not adapted from any
// implementation of it.
//
// **Scattered points in, a label volume out.** `ops::voxelize` is the other op
// of this shape and it is the wrong one for this job, twice over: it renders
// `f64`, and where two contributions land on one voxel it **adds**. A sum is the
// right answer for a density and it is a nonsense answer for a label — labels
// 3 and 5 meeting on a voxel do not make an 8.
//
// What a label volume is, here
// ----------------------------
// > **zero is unmarked, and every other value is a label.**
//
// One convention, and it is what the consumers of such a volume already assume:
// a flood that grows regions out of marked voxels reads zero as "nothing here"
// and every distinct positive integer as a distinct starting region. So:
//
// * the output is an **integer** image. `Dtype::Bool` and the float types are
//   refused by name — see [`label_ceiling`] — because a two-valued type cannot
//   hold a label at all (`ops::voxelize` into a `bool` image is the op that
//   answers "did any point's kernel cover this voxel") and a float one invites
//   labels to be compared with `==` on values that arrived through a rounding;
// * a label of **zero is refused**, for `VoxelizeOp::new`'s reason about an
//   empty kernel: a point that stamps zero is indistinguishable from a point
//   that was never there, so it is a request the op cannot carry out rather than
//   one it should carry out silently;
// * a **negative or fractional** label is refused. Rounding one would be
//   inventing a label the caller did not name, and two points a tenth apart
//   would silently become one.
//
// Where the label comes from
// --------------------------
// From the point's own [`Point::weight`], read as an integer by [`label_of`].
// There is no other place it could come from, and the alternative is worth
// naming because it is the obvious one: **the point's rank in the whole set**.
// That is not available to a block. A block gathers a bounded neighbourhood of
// fragments — that boundedness is what makes the op plannable at all — so it
// cannot know how many points precede one of its own in the volume without
// reading the entire set, which is a full-reach phase and a different program. A
// label that is a *field of the point* travels with the point, is the same
// number in every block that sees it, and needs nothing gathered to be known.
//
// The weight is an `f64` and a label is an integer, so the range is the one an
// `f64` names exactly: `1 ..= 2^53`, further clamped by what the destination
// type holds. [`label_of`] refuses everything else and says which rule it broke.
//
// The collision rule: **the lowest label wins**
// ---------------------------------------------
// Two points can claim one voxel — two points on the same voxel with any kernel,
// or two points a few voxels apart with a kernel wide enough to overlap, which
// with any radius above zero is the ordinary case and not an error. One of the
// labels has to win, and this op's rule is that the **smaller** one does.
//
// The reason is not that the smaller label is more deserving; it is that `min`
// is **associative, commutative and idempotent**, so the answer does not depend
// on the order the contributions are visited in — *by construction*, with no
// ordering to state and no sort to get right. That is the whole difference from
// `ops::voxelize`, which needs a stated summation order and a sort at every
// block because floating-point `+` is not associative. Here there is nothing to
// order: a block may walk its gathered fragments in any order, and two blocks of
// two different lattices may walk different subsets of them, and every one of
// them lands on the same value at every voxel.
//
// It also happens to be **"first wins" under the order this crate already has**.
// `crate::points` sorts by the coordinate triple and then by the weight's bits;
// at one voxel the coordinates are equal, so the tiebreak is the weight, and for
// the positive finite weights a label may be, bit order is numeric order. The
// first point at a voxel in the canonical order is the one with the smallest
// label. So the rule agrees with the crate's own ordering rather than
// introducing a second one — but it does not *depend* on that ordering, which is
// why it is stated as `min` and implemented as `min`.
//
// **What was rejected, and why.** *Last wins* is the same argument with `max`
// and is equally invariant; it is not chosen because it disagrees with the
// canonical order for no gain. *Refuse on collision* is invariant too and is
// wrong for a different reason: with a kernel of any radius, two points close
// enough to overlap are a normal input — a caller stamping balls of radius two
// around points three apart has asked for overlap — so a refusal would make the
// op unusable in exactly the case a collision rule is for.
//
// Where the points must be
// ------------------------
// The same rule `ops::voxelize` states, enforced by the same check and refused
// with the same kind of message: **a point in block B's fragment must lie in B's
// core**. Cores tile the volume, so this makes ownership a total function of the
// coordinate, and it is the premise the block reach is derived from.
//
// A point **outside the volume** is therefore refused, because it is outside
// every core. That is a deliberate agreement with `ops::voxelize` rather than
// with the tools this crate is compared against, several of which drop such a
// point quietly. The argument is that both ops read *the same stream*: a point
// set that one of them accepts and the other refuses would make a producer's
// output legal or illegal depending on which consumer it was wired to, which is
// a property no producer can check. A caller with points outside the volume
// filters or clamps them before writing the fragment, where the decision belongs.
//
// The reaches
// -----------
// Both are `ops::voxelize`'s, derived from the same kernel by the same
// functions, and this module does not restate the derivation — see that module's
// header for why the divisor is the nominal block edge and what test protects
// the premise. The voxel reach is the kernel's radius clamped to the volume; the
// block reach is [`super::voxelize::block_reach`].
//
// Coverage, and the halo
// ----------------------
// No output stream is declared: this phase's output is a pixel image, exactly as
// in `ops::voxelize`. Only the **valid** region of a block is filled; the halo
// margin around it is left at zero, because a voxel out there may have a
// contributor beyond the gathered neighbourhood and an unmarked voxel is a
// smaller lie than a wrong label.
//
// A stamp is a gather's transpose, so it asks the element the same question
// -------------------------------------------------------------------------
// The kernel is a [`StructuringElement`], and one of those can be built from
// [`super::element::StepOrigin::ClippedStart`], whose members are **not one
// set**: a decimation counted from the clipped start of the window re-phases
// wherever the window meets a low face of the volume. Every op that *gathers* a
// window asks [`StructuringElement::offsets_at`] for the members at the voxel it
// is evaluating. This op does not gather — it **stamps**, placing the element
// around a point and writing — and it asks the same method for a reason worth
// writing down rather than inheriting.
//
// A gather is `out[c] = f({ in[c + o] : o in K(c) })`, so its incidence matrix
// has `M[c, c + o] != 0` for `o` in `K(c)`. The transpose of that is a
// **scatter**: for each source index `p`, write into `p + o` for every `o` in
// `K(p)` — the kernel read at the *source*, which here is the point. So the
// stamp below is exactly `M^T` when, and only when, it evaluates the element at
// the point's own position. Asking at any other position would be a different
// operator, not a different spelling of this one.
//
// That is the structural half. The substantive half is what `ClippedStart`
// actually names: it is a property of a **(position, volume)** pair, stated by
// the array expression `a[max(0, c - lo) : min(c + hi + 1, n) : step]`, and that
// expression says nothing about reading or writing. A gather puts it on the
// right of the assignment and a stamp puts it on the left; the slice is the same
// slice. The step that makes the two agree exactly is that this op's source
// space and destination space are **the same volume** — a point must lie in a
// core and cores tile the volume, so `n` is `grid.volume()` on both sides. Were
// a stamp ever to deposit into a differently-sized array, "the window at the
// point" and "the window at the voxel" would clip against different extents and
// the two questions would come apart; here they cannot.
//
// So this op **honours** the origin, at the point's position in the volume. What
// it costs is one regeneration per *point* rather than per voxel, which is the
// cheapest place in the crate this question is asked; what it costs an anchored
// element is nothing, because `offsets_at` hands that element its own slice back
// and the lift below keeps even that out of the loop.
//
// **A re-phased window can hold a count the element does not.** Fewer where a
// face truncates it, and for a shaped element occasionally *more* — a ball of
// radius two stepped by two keeps seven offsets in the interior and eight at the
// phase reached from `[1, 1, 1]`. Nothing here is sized from
// [`StructuringElement::len`]: the loop walks the slice it was handed, `min` has
// no arity, and the collision rule does not count. So a surprising count changes
// which voxels are marked and nothing else, which is what it should change.

use ndarray::{Array3, ArrayView3, ArrayViewMut3, Axis, Slice};

use std::collections::BTreeMap;

use crate::decomposition::Decomposition;
use crate::dtype::Dtype;
use crate::env::{BlockBuf, Environment};
use crate::error::{Error, Result};
use crate::fragment::{BlockOutput, BlockView, FragmentInput, FragmentOp, PhaseView, SeamFold};
use crate::geometry::BlockGrid;
use crate::op::{Chain, Placement};
use crate::points::{decode_points, Point};
use crate::region::Region;
use crate::sidecar::{check_stream_name, Lifecycle};
use crate::voxels::Voxels;

use super::components::{
    bytes_to_words, core_within_read, empty_planes, expect_end, label_members_into_with, planes_of,
    push_planes, read_header, take_planes, walk_seams_with, words_to_bytes, Connectivity,
    FacePlanes, LabelIndex, Union, UNLABELLED,
};
use super::element::{StepOrigin, StructuringElement};
use super::voxelize::block_reach;

// ----------------------------------------------------------------- labels --

/// The largest integer an `f64` names without also naming the one beside it.
///
/// Above this the spacing between representable values is two or more, so a
/// weight of `2^53 + 1` arrives as `2^53` and two labels a caller believes are
/// distinct are one label. [`label_of`] refuses anything larger rather than
/// letting two objects merge in the last bit of a float.
pub const MAX_EXACT_LABEL: u64 = 1 << 53;

/// The largest label an image of `dtype` can hold, or `None` for a type that
/// cannot hold labels at all.
///
/// **Integer types only, and the two refusals each have a reason.** `Dtype::Bool`
/// holds one bit, which is enough to say "marked" and not enough to say *which*
/// mark — the op that writes that volume is `ops::voxelize` into a `bool` image,
/// which answers exactly "did any point's kernel cover this voxel". The float
/// types are refused because a label is compared for equality by every consumer
/// of a label volume, and equality on a value that has passed through a rounding
/// is the kind of comparison that works until it does not; a caller who really
/// wants labels in an `f64` buffer writes this to an integer image and puts
/// `ops::voxelwise::WidenOp` after it, where the widening is exact and is
/// something they asked for.
///
/// The ceiling is the smaller of the type's own maximum and [`MAX_EXACT_LABEL`],
/// so a `u64` image does not admit labels the `f64` weight could not have
/// carried.
pub fn label_ceiling(dtype: Dtype) -> Option<u64> {
    let own = match dtype {
        Dtype::U8 => u8::MAX as u64,
        Dtype::U16 => u16::MAX as u64,
        Dtype::U32 => u32::MAX as u64,
        Dtype::U64 => u64::MAX,
        Dtype::I8 => i8::MAX as u64,
        Dtype::I16 => i16::MAX as u64,
        Dtype::I32 => i32::MAX as u64,
        Dtype::I64 => i64::MAX as u64,
        Dtype::Bool | Dtype::F16 | Dtype::F32 | Dtype::F64 => return None,
    };
    Some(own.min(MAX_EXACT_LABEL))
}

/// The label a point carries: its [`weight`](Point::weight), read as an integer.
///
/// Four refusals, and each one is a value that would otherwise become a label
/// the caller did not name:
///
/// * **zero** — indistinguishable from an unmarked voxel, so a point stamping it
///   would vanish;
/// * **negative** — a label volume's convention has no negative half; zero is
///   the one reserved value and the rest count upwards;
/// * **fractional** — rounding would merge two labels a fraction apart;
/// * **above [`MAX_EXACT_LABEL`]** — beyond it the `f64` no longer names every
///   integer, so two distinct labels can arrive as one.
///
/// A non-finite weight cannot reach here: `points::decode_points` refuses it
/// when the fragment is read.
pub fn label_of(point: &Point) -> Result<u64> {
    let weight = point.weight;
    if !weight.is_finite() {
        return Err(Error::invalid(format!(
            "label: the point at {:?} carries {weight}, which is not a finite number and cannot \
             be a label",
            point.at
        )));
    }
    if weight < 1.0 {
        return Err(Error::invalid(format!(
            "label: the point at {:?} carries {weight}, and a label is a whole number of at \
             least 1. Zero is reserved for an unmarked voxel, so a point stamping it would be \
             indistinguishable from a point that was never written; there is no negative half.",
            point.at
        )));
    }
    if weight.fract() != 0.0 {
        return Err(Error::invalid(format!(
            "label: the point at {:?} carries {weight}, which is not a whole number. Rounding it \
             would be inventing a label the caller did not name, and two points a fraction apart \
             would silently become one.",
            point.at
        )));
    }
    if weight > MAX_EXACT_LABEL as f64 {
        return Err(Error::invalid(format!(
            "label: the point at {:?} carries {weight}, which is above {MAX_EXACT_LABEL}. Beyond \
             that an f64 no longer names every integer, so two labels a caller believes are \
             distinct would arrive as one.",
            point.at
        )));
    }
    Ok(weight as u64)
}

/// Every point of `fragments`, with its label, checked.
///
/// Both halves of the check are here and nowhere else, so that a run which
/// allocates no buffer refuses exactly what a run which allocates one refuses:
/// the point lies in the core of the block whose fragment carries it, and its
/// weight is a label this destination can hold. `ceiling` comes from
/// [`label_ceiling`] applied to the image's element type.
///
/// The result is in the order the fragments were handed over, which is
/// deliberately *not* stated as part of the answer — see the module header on
/// why `min` needs no order.
pub fn labelled_points(
    fragments: &[([usize; 3], Vec<Point>)],
    grid: &BlockGrid,
    ceiling: u64,
) -> Result<Vec<(u64, [usize; 3])>> {
    let mut out = Vec::new();
    for (block, points) in fragments {
        let core = core_of(grid, *block).ok_or_else(|| {
            Error::invalid(format!(
                "label: a fragment is keyed to block {block:?}, which is outside this phase's \
                 lattice of {:?} blocks",
                grid.blocks_per_axis()
            ))
        })?;
        for (index, point) in points.iter().enumerate() {
            if !contains(&core, point.at) {
                return Err(Error::invalid(format!(
                    "label: point {index} of block {block:?} is at {:?}, which is outside that \
                     block's core {:?}..{:?}. A point must lie in the core of the block whose \
                     fragment carries it: cores tile the volume, so that is what makes every \
                     point owned by exactly one block, and it is the premise the block reach is \
                     derived from. A point outside the volume is outside every core and is \
                     refused here too, which is the same rule `ops::voxelize` states — the two \
                     ops read the same stream and cannot disagree about which streams are legal.",
                    point.at,
                    core.start,
                    core.end()
                )));
            }
            let label = label_of(point)?;
            if label > ceiling {
                return Err(Error::invalid(format!(
                    "label: point {index} of block {block:?} carries the label {label}, and the \
                     largest this destination holds is {ceiling}. A label written past the end of \
                     its type saturates onto another label, which is two objects becoming one \
                     with nothing to see afterwards."
                )));
            }
            out.push((label, point.at));
        }
    }
    Ok(out)
}

/// Stamp `fragments` into `window` as labels.
///
/// `out` covers `window` of the volume `grid` is cut from and **must arrive
/// zero-filled**, zero being the unmarked value; every contribution outside
/// `window` is dropped, which is how a block writes its own part of the answer
/// without knowing anything about the others. Points are not required to lie
/// inside `window` — a point outside it whose kernel reaches in is what the
/// block reach exists for — but every point must lie in the core of the block
/// whose fragment carries it.
///
/// Where two labels reach one voxel, **the smaller wins**; see the module
/// header. Because `min` does not care what order it is applied in, this
/// function needs no sort and its answer does not depend on the order of
/// `fragments`, of the points inside one fragment, or of the kernel's offsets.
///
/// The kernel is placed at each point's position **in the volume**, through
/// [`StructuringElement::offsets_at`], which is what makes an element whose
/// members re-phase near a low face stamp the window it names there. The point's
/// coordinate is already a volume coordinate and `volume` is `grid`'s, so the
/// window a block stamps is the window the whole-volume run stamps — a block
/// seam is not a face and nothing here can mistake one for the other.
///
/// Free function first, `FragmentOp` shell on top, for this module's usual
/// reason: a test can permute this function's argument and cannot permute what
/// the executor gathers.
pub fn label_points_into(
    fragments: &[([usize; 3], Vec<Point>)],
    grid: &BlockGrid,
    element: &StructuringElement,
    window: &Region,
    ceiling: u64,
    mut out: ArrayViewMut3<'_, u64>,
) -> Result<()> {
    let volume = grid.volume();
    if window.ndim() != 3 {
        return Err(Error::invalid(format!(
            "label: a window is 3-D, got rank {}",
            window.ndim()
        )));
    }
    window.check_within(&volume, "label window")?;
    let shape = [out.shape()[0], out.shape()[1], out.shape()[2]];
    if shape.to_vec() != window.shape {
        return Err(Error::ShapeMismatch {
            expected: window.shape.clone(),
            got: shape.to_vec(),
        });
    }

    // The element's members **at one point**, for the one element that has more
    // than one set of them. Owned out here so that a point pays no allocation for
    // it, and untouched by every other element.
    let mut scratch: Vec<[isize; 3]> = Vec::new();
    // **The one member set, where there is one**, lifted out of the loop rather
    // than asked for at every point. `offsets_at` hands back exactly this slice
    // for an anchored element, so the two are the same answer; the lift keeps the
    // path that has nothing to ask about — every element without a step, which is
    // every kernel this op has ever been given — a slice walk and nothing else.
    let fixed = (element.origin() == StepOrigin::Anchor).then(|| element.offsets());
    for (label, at) in labelled_points(fragments, grid, ceiling)? {
        // The transpose of a gather reads the kernel at the **source**, and the
        // source of a stamp is the point. See the module header for why that is
        // the same question a gathering op asks and not merely a similar one.
        let stamped = match fixed {
            Some(offsets) => offsets,
            None => element.offsets_at(
                [at[0] as isize, at[1] as isize, at[2] as isize],
                volume,
                &mut scratch,
            ),
        };
        for offset in stamped {
            let mut inside = true;
            let mut local = [0usize; 3];
            for axis in 0..3 {
                let position = at[axis] as isize + offset[axis];
                // Beyond the volume there is no voxel to stamp: the kernel is
                // clipped, exactly as `ops::voxelize` clips it.
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
                let slot = &mut out[local];
                // The collision rule, in one line: unmarked takes the label,
                // marked keeps the smaller of the two.
                if *slot == 0 || label < *slot {
                    *slot = label;
                }
            }
        }
    }
    Ok(())
}

/// The core of `index` on `grid`, or `None` for an index off the lattice.
///
/// The same arithmetic `ops::voxelize` does, and for the same reason it is
/// recomputed rather than searched for in `BlockGrid::cores`: that allocates
/// every core of the lattice to answer one question about one of them.
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

// ------------------------------------------------------------------ shell --

/// Stamp a stream of points into a volume as labels, one label per point.
///
/// Reads one fragment stream, writes pixels, declares no fragment stream of its
/// own. Both reaches are derived from the kernel and the lattice, by
/// `ops::voxelize`'s functions; see this module's header.
pub struct LabelPointsOp {
    name: &'static str,
    stream: String,
    stream_phase: usize,
    element: StructuringElement,
    grid: BlockGrid,
    block_reach: [usize; 3],
}

impl LabelPointsOp {
    /// The lattice is an argument because the block reach is a statement in
    /// block indices and cannot be derived without one; it is kept rather than
    /// only measured, and [`Self::check_grid`] refuses a `BlockView` whose grid
    /// disagrees. `ops::voxelize::VoxelizeOp::new` takes it for the same reason.
    ///
    /// A kernel with no voxels in it is refused: every point would stamp nothing
    /// and the output would be indistinguishable from one with no points at all.
    /// `ops::voxelize::single_voxel` is the kernel that stamps the point's own
    /// voxel and nothing else, which is the commonest thing to want here.
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
                "label op {name:?} was given a kernel with no voxels in it, so every point would \
                 stamp nothing and the output would be indistinguishable from one with no points \
                 at all"
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
    pub fn check_grid(&self, grid: &BlockGrid) -> Result<()> {
        if grid.volume() != self.grid.volume() || grid.block() != self.grid.block() {
            return Err(Error::invalid(format!(
                "label op {:?} derived its block reach {:?} from a lattice of {:?}/{:?} and is \
                 running on {:?}/{:?}. The reach is a count of block indices, so it means \
                 something else on another lattice; build the op with the grid the phase runs on.",
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

impl FragmentOp for LabelPointsOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// The kernel's radius, clamped to the volume — `ops::voxelize::reach`'s
    /// derivation exactly, including the reversed direction: this op *deposits*,
    /// so the points a window needs are the ones the reflected kernel reaches,
    /// and the wider side is declared because the signature holds one integer
    /// per axis.
    fn reach(&self, axis: usize, volume_len: usize) -> usize {
        self.element.reach(axis).min(volume_len)
    }

    /// No pixel IO on the way in: the input is the fragment stream, and a phase
    /// running this op reads not one voxel of the image below it.
    fn reads_pixels(&self) -> bool {
        false
    }

    fn writes_pixels(&self) -> bool {
        true
    }

    /// Gathered rather than streamed, for `ops::voxelize`'s reason: the reach is
    /// bounded by the kernel radius and is therefore a handful of neighbours.
    fn gathers(&self) -> bool {
        true
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own(self.stream.clone(), self.stream_phase).with_reach(self.block_reach)]
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        self.check_grid(at.grid)?;

        let mut pixels = at.output_buffer(0.0)?;
        // `BlockBuf::dtype` answers for a simulated block too, so the element
        // type is checked before anything is allocated and a run over a volume
        // too large to hold still refuses a destination that cannot carry
        // labels.
        let dtype = pixels.dtype();
        let ceiling = label_ceiling(dtype).ok_or_else(|| {
            Error::invalid(format!(
                "label op {:?} writes labels — zero is unmarked and every other value names a \
                 point — and an image of {} holds no such value. Write an integer image; a \
                 two-valued image is `ops::voxelize` into a bool image, and a float one is this \
                 op followed by `ops::voxelwise::WidenOp`.",
                self.name,
                dtype.numpy_name()
            ))
        })?;

        let mut fragments = Vec::new();
        for (key, bytes) in at.fragments(&self.stream) {
            fragments.push((key.block, decode_points(bytes)?));
        }

        if pixels.as_array_mut().is_none() {
            // A simulated run allocates no accumulator — those runs exist for
            // volumes that could not be held — but it still **checks**, because
            // every refusal above and in `labelled_points` is a fact about the
            // point set rather than about the buffer, and a simulated run that
            // accepted a stream the real one refuses would be worth less than
            // one that did not run at all.
            labelled_points(&fragments, at.grid, ceiling)?;
            return Ok(BlockOutput::nothing().with_pixels(pixels));
        }

        // Only the valid region is filled; the halo margin around it may have
        // contributors beyond the gathered neighbourhood, so it is left
        // unmarked rather than labelled with a number nobody may trust.
        let valid = at.valid;
        let mut labels = Array3::<u64>::zeros((valid.shape[0], valid.shape[1], valid.shape[2]));
        label_points_into(
            &fragments,
            at.grid,
            &self.element,
            valid,
            ceiling,
            labels.view_mut(),
        )?;

        let mut offset = [0usize; 3];
        for axis in 0..3 {
            offset[axis] = valid.start[axis] - at.read.start[axis];
        }
        if let Some(array) = pixels.as_array_mut() {
            store(&labels, offset, array)?;
        }
        Ok(BlockOutput::nothing().with_pixels(pixels))
    }
}

/// Write a `u64` label buffer into `at` of a buffer of the image's own type.
///
/// No rounding and no saturation, unlike `ops::voxelize::store`: every label has
/// already been checked against [`label_ceiling`] for this element type, so the
/// cast is exact and a value that would have saturated was refused with a
/// message naming it. That is the difference between a quantity and a name — a
/// count clipped to a type's maximum is a bad number, a label clipped to it is a
/// different object.
fn store(labels: &Array3<u64>, at: [usize; 3], into: &mut Voxels) -> Result<()> {
    let shape = [labels.shape()[0], labels.shape()[1], labels.shape()[2]];
    let held = into.shape();
    for axis in 0..3 {
        if at[axis] + shape[axis] > held[axis] {
            return Err(Error::invalid(format!(
                "label: writing {shape:?} at {at:?} of a block of {held:?} would run off axis \
                 {axis}"
            )));
        }
    }
    macro_rules! arm {
        ($type:ty) => {{
            let mut window = into.view_mut::<$type>()?;
            for axis in 0..3 {
                window
                    .slice_axis_inplace(Axis(axis), Slice::from(at[axis]..at[axis] + shape[axis]));
            }
            for (target, &value) in window.iter_mut().zip(labels.iter()) {
                *target = value as $type;
            }
        }};
    }
    match into.dtype() {
        Dtype::U8 => arm!(u8),
        Dtype::U16 => arm!(u16),
        Dtype::U32 => arm!(u32),
        Dtype::U64 => arm!(u64),
        Dtype::I8 => arm!(i8),
        Dtype::I16 => arm!(i16),
        Dtype::I32 => arm!(i32),
        Dtype::I64 => arm!(i64),
        // `label_ceiling` returns `None` for each of these and `apply` refuses
        // before reaching here, so this arm is unreachable through the op.
        other => {
            return Err(Error::invalid(format!(
                "label: an image of {} holds no labels",
                other.numpy_name()
            )))
        }
    }
    Ok(())
}

// ============================================================================
// A mask in, a **globally consistent** label volume out
// ============================================================================
//
// Everything above this line is the op that stamps labels a caller already has.
// Below it is the op that *derives* them: connected-component labelling over a
// mask, decomposed, with one numbering that is a function of the volume and not
// of where the volume was cut.
//
// Why this is here and not in `ops::fill`
// ----------------------------------------
// `ops::fill` and `ops::regional` already do nine tenths of it. Both label each
// block locally, write the labels as a `u32` image, and close them across the
// seams with the union-find in `ops::components`. Neither hands the closed
// labelling back: `fill` collapses it to a mask of holes, `regional` to a mask
// of maxima, and `detect` to a handful of points. The ops-survey index records
// the consequence — *"no op under `src/ops/` produces a label volume, while
// `ops::tabulate`'s header opens 'One row per region of a **label volume**', so
// the crate's most complete per-object measurement cannot be driven by the
// crate's own segmentation."* This module's subject is exactly "a label volume",
// which is why the op lands beside [`LabelPointsOp`] rather than beside the two
// ops whose machinery it borrows.
//
// The numbering, and why it is not the union-find's
// --------------------------------------------------
// A union-find hands back a **root node**, and a root node is
// `(block, local label)` flattened — which is to say it is a function of the
// decomposition, twice over: which block, and which label within it. Writing
// `find(node) + 1` into the volume produces a correct *partition* whose *labels*
// change when the block size changes. That is not a globally consistent label
// volume; it is a globally consistent equivalence relation with a
// decomposition-dependent name for each class, and every consumer that stores a
// label — a table of regions, a graph whose vertices are labels, anything
// written to disk and read back beside another run — is then wrong in a way no
// per-voxel comparison catches.
//
// So the numbering is stated as a rule about the **volume**:
//
// > components are numbered from 1, in the order their lowest voxel is met in a
// > row-major scan of the whole volume.
//
// That is `label_members_into_with`'s own rule — the one it applies inside a
// block — lifted to the volume, and it is why the blocked answer here is
// **byte-identical** to the whole-volume reference rather than merely a
// relabelling of it. It costs one extra `u64` per block-local label in the
// fragment: the least row-major index, in volume coordinates, of any voxel
// carrying that label. `min` over that is associative, commutative and
// idempotent, so folding it onto a component's root is order-independent for
// exactly [`Union::fold_or`]'s reason, and [`Union::fold_min`] is that fold.
//
// Sorting the roots by their component's least voxel then gives a dense
// numbering, and the sort has no ties to break: two distinct components cannot
// share a least voxel, because a voxel belongs to one component.
//
// Two ways to spend the map, and this module ships both
// ------------------------------------------------------
// The merge's whole answer is a **table** — one `u32` per `(block, local
// label)`. Its size is the number of block-local labels, which is the component
// count plus whatever the cut splits, so it grows with the **lattice** and not
// with the voxel count: it is the one quantity here that does not scale with the
// volume. That is the premise the second design below rests on, so it is
// asserted as a ratio at every grid rather than described —
// `the_reconciliation_table_is_far_smaller_than_the_volume_it_reconciles` — and
// [`GlobalLabels::table_bytes`] is measured rather than derived, because the
// per-block overhead stops being negligible at a fine cut.
//
// What to *do* with that table is a design question with two answers, and they
// differ in more than taste:
//
// | | what it is | what a consumer sees |
// |---|---|---|
// | **materialise** | [`RelabelComponentsOp`], a `fragments -> volume` phase that rewrites the local labels into global ones and writes a second `u32` image | an ordinary image |
// | **decorate** | [`RelabelledEnvironment`], which applies the table to a read of the local-label image as the read is served | an ordinary image |
//
// The second subsumes the first: an identity op over a decorated environment
// writes the materialised volume, so there is one mechanism and a trivial
// materialiser rather than two mechanisms. What it costs is stated with the
// type, and it is not lines of code — see [`RelabelledEnvironment`].
//
// **The materialising phase is still not three phases**, and that is a framework
// fact rather than a choice made here. The natural shape is label, merge,
// relabel; `fragment::check_phase_work` refuses a pixel phase after a
// fragment-only one, because the image that phase would read went unwritten. So
// the merge belongs to the relabelling phase.
//
// **What has changed is where in that phase it sits, and what the phase fetches.**
// This paragraph used to end: *every block re-runs the whole union-find — and,
// because a whole-lattice fragment reach is also the halo, every block of that
// phase also reads the entire label image.* Neither half is true any more, and
// the two halves were two separate changes:
//
// * [`RelabelComponentsOp`] declares `barrier() == true`, so the dependency on
//   the labelling phase is stated as an edge rather than bought with a
//   whole-volume fetch. The halo drops to the reach, which is zero, and each
//   block reads its own core;
// * it declares [`FragmentOp::reduce`](crate::fragment::FragmentOp::reduce), so
//   the merge runs **once for the phase** rather than once per block, and the
//   fragment set is transmitted a number of times that does not grow with the
//   lattice.
//
// The second is the larger of the two by a wide margin and the reason is not
// traffic: one merge is small and there were `blocks` of them.
// `docs/design/barriers.md` §7.2 is where that is timed and §7.4 is why the two
// had to land in that order — without a barrier there is no moment at which the
// fragment set is complete, so a hoisted reduction is not well defined.
//
// **The decorated design's advantage was G7's barrier, obtained by not being in
// the plan. The plan can state one now**, so the two designs no longer differ in
// kind — see the recommendation below.
//
// **What the decorated design costs the caller is unchanged, and it is an
// invariant no type enforces.** Its merge must run between two `execute_phases`
// calls, because phases pipeline: a consumer block may begin before every
// producer block has written its fragment, so a table built lazily on first read
// would be built from an incomplete fragment set and the run would fail — or not
// — depending on the schedule. That arm is correct by the caller's discipline.
// The materialising arm is the one that no longer is: the barrier is the
// framework stating the same thing, and `strategy::reduce_phase` is where it is
// enforced.
//
// What each design costs in invariants
// -------------------------------------
// The second axis, and the one that is not a number. Neither list is about how
// much code there is; both are about what a *future* op on this machinery has to
// keep true, and what happens when it does not.
//
// **Both designs share three**, and they come from the merge rather than from
// either design:
//
// 1. the flood's connectivity and the seam walk's are one equivalence relation
//    split in half, so a plan whose halves disagree answers differently
//    depending on where the volume was cut. Checked at plan time by
//    [`component_label_phases`], and *not* checkable for a caller driving the
//    kernels by hand;
// 2. every per-label fact the merge folds must be associative, commutative and
//    idempotent, or two blocks fold it to two different answers. `fold_or` and
//    [`Union::fold_min`] are; a running mean would not be, and nothing in the
//    types says so;
// 3. the local-label image is **decomposition-dependent** and must never be
//    published, compared between runs or read by anything but the merge's own
//    consumer.
//
// **Materialising adds two, both of them shaped like a wrong answer rather than
// an error, and the barrier shrank both:**
//
// 4. the relabelling phase's `apply` must write only its own core. Reading block
//    3's label 2 as this block's label 2 produces a complete, well-formed, wrong
//    volume; `components::core_within_read` exists because this is the third op
//    on the shape and it is the third op that had to remember. **The read extent
//    used to be the whole volume, and with the barrier it is the core**, so the
//    slice is now the identity and the hazard is latent rather than live — which
//    is exactly why the slice stays written out. An op that dropped it would be
//    correct today and wrong the moment anything gave this phase a halo again;
// 5. it holds a **second** image of the label width alive with the first. That
//    used to be two *whole volumes* per concurrent block, because the halo was
//    the volume; it is now two *blocks*. It is still a constraint on what else
//    the plan wants alive at that phase, and it is no longer the dominant one.
//
// **Decorating adds three, and the first is the expensive one to keep true:**
//
// 6. an `Environment` decorator's forwarding is total or it is wrong. Thirty-odd
//    methods, most of them defaulted; a method left unforwarded silently reverts
//    an inner environment's override to the trait default, and a method *added*
//    to `Environment` later must be added here too. Nothing checks that;
// 7. the table is a function of the labelling lattice, and the decorator applies
//    it by coordinate. A table from another cut of the same volume is a
//    well-formed wrong answer — `tests/global_labels.rs` asserts that it is
//    wrong rather than pretending it is caught;
// 8. the remap is paid **per reader**, so the cost of a decorated image is a
//    property of the plan around it rather than of the image. That is not a
//    hazard, it is the term the recommendation turns on, and
//    `tests/label_materialisation_cost.rs` measures both sides of it.
//
// Five against six is not the comparison; **4 and 7 are the same kind of thing**
// — an addressing scheme applied under the wrong numbering — and 6 is the one
// with no analogue on the other side.
//
// Which to prefer, and what moved it
// -----------------------------------
// The recommendation used to rest on a factor of **25.4x** in bytes moved,
// measured by `tests/label_materialisation_cost.rs` on a recorded volume at 256
// blocks, and it was not close. `docs/design/barriers.md` §6.1 and §6.2 both
// name themselves as the conditions under which that stops being the answer,
// and both have now fired: the materialising phase declares a barrier and hoists
// its reduction, so the gap it was losing by is the gap the design note projects
// at **1.13x**, which is one extra read and one extra write of the label volume
// and is the price of materialising at all rather than of anything in this
// module.
//
// **That projection has not been re-measured on the recorded volume**, and the
// honest statement of where this stands is that it has not.
// `tests/label_materialisation_cost.rs` reads its fixture from
// `BLOCKFLOW_LABEL_FIXTURE` and the recording is not on every machine; its four
// arms are still the out-of-plan *simulations* they were written as, so they
// price the shapes rather than these ops. What has been measured, on the shipped
// ops and on a synthetic volume, is `tests/barrier_migration.rs`, and it agrees
// with the projection's structure and not with its constants — as §8.8 says it
// should, because a fragment there is a block face and the constants are a
// property of what a fragment weighs.
//
// So: **materialising is now the better default**, for §6.2's reason rather than
// for a number — a materialised volume is remapped once and a decorated one is
// remapped per reader, and the write the decorated design avoids was never its
// advantage. The decorator remains the answer for a label volume with no reader,
// or one reader that reads a fraction of it.

/// What one block tells the component merge about itself.
///
/// The six face planes, plus one `u64` per local label. Deliberately not the
/// block's label volume: what the merge needs is which labels meet across a
/// seam, and only the faces can say.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComponentFaces {
    /// How many components this block found on its own.
    pub labels: u32,
    /// Per label, the least row-major index **in volume coordinates** of a voxel
    /// carrying it. This is what makes the global numbering a function of the
    /// volume; see the section header.
    pub first: Vec<u64>,
    /// The six faces, ordered `axis * 2 + side` with side 0 low and 1 high.
    pub faces: FacePlanes,
}

impl ComponentFaces {
    /// Read a block's faces off its label volume, given the per-label first
    /// voxels [`first_voxels`] computed.
    pub fn of(labels: ArrayView3<'_, u32>, count: u32, first: Vec<u64>) -> Result<Self> {
        if first.len() != count as usize {
            return Err(Error::InvalidArgument(format!(
                "{count} labels but {} first-voxel entries",
                first.len()
            )));
        }
        Ok(Self {
            labels: count,
            first,
            faces: planes_of(labels),
        })
    }

    /// The empty report: a block with nothing to say, which is what an
    /// accounting run produces and is a different fact from no fragment at all.
    pub fn empty() -> Self {
        Self {
            labels: 0,
            first: Vec::new(),
            faces: empty_planes(),
        }
    }

    /// A self-describing byte form: little-endian `u32` throughout, with a magic
    /// and a version in front, and each `u64` as low word then high word. See
    /// `components::read_header` for why the magic is not decoration.
    pub fn encode(&self) -> Vec<u8> {
        let mut words: Vec<u32> = vec![COMPONENT_MAGIC, COMPONENT_VERSION, self.labels];
        for &first in &self.first {
            words.push(first as u32);
            words.push((first >> 32) as u32);
        }
        push_planes(&self.faces, &mut words);
        words_to_bytes(&words)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        const NOUN: &str = "a component-faces fragment";
        let words = bytes_to_words(bytes, NOUN)?;
        let labels = read_header(&words, COMPONENT_MAGIC, COMPONENT_VERSION, NOUN)?;
        let mut at = 3usize;
        let end = at + 2 * labels as usize;
        if words.len() < end {
            return Err(Error::InvalidArgument(format!(
                "{NOUN} ends inside its first-voxel table"
            )));
        }
        let first = words[at..end]
            .chunks_exact(2)
            .map(|pair| pair[0] as u64 | ((pair[1] as u64) << 32))
            .collect();
        at = end;
        let faces = take_planes(&words, &mut at, NOUN)?;
        expect_end(&words, at, NOUN)?;
        Ok(Self {
            labels,
            first,
            faces,
        })
    }
}

/// `"CMPN"` little-endian.
const COMPONENT_MAGIC: u32 = 0x4e50_4d43;
const COMPONENT_VERSION: u32 = 1;

/// `"GLBL"` little-endian, and distinct from [`COMPONENT_MAGIC`] on purpose:
/// the reduction blob and the fragment are both `u32` words written by one op,
/// so the only thing that stops one being decoded as the other is that they say
/// which they are.
const TABLE_MAGIC: u32 = 0x4c42_4c47;
const TABLE_VERSION: u32 = 1;

fn push_usize(words: &mut Vec<u32>, value: usize) {
    let value = value as u64;
    words.push(value as u32);
    words.push((value >> 32) as u32);
}

fn take_usize(words: &[u32], at: &mut usize, noun: &str, what: &str) -> Result<usize> {
    let end = *at + 2;
    if words.len() < end {
        return Err(Error::InvalidArgument(format!("{noun} ends inside {what}")));
    }
    let value = words[*at] as u64 | ((words[*at + 1] as u64) << 32);
    *at = end;
    usize::try_from(value).map_err(|_| {
        Error::InvalidArgument(format!(
            "{noun} carries {value} as {what}, which this machine's `usize` does not hold"
        ))
    })
}

/// Per local label, the least row-major index in **volume** coordinates of a
/// voxel carrying it.
///
/// `offset` is where the block's labels sit in the volume and `volume` is the
/// whole extent, so the index computed is the one a whole-volume scan would
/// have reached that voxel at. Unlabelled voxels take no part.
///
/// The `min` is written out rather than relying on the scan order agreeing with
/// the volume's. It does agree — row-major over a box is the volume's row-major
/// restricted to that box, both being lexicographic on the coordinate triple —
/// but the fold below is a `min` and this being one too means the two cannot
/// come apart if the traversal here is ever reordered.
pub fn first_voxels(
    labels: ArrayView3<'_, u32>,
    count: u32,
    offset: [usize; 3],
    volume: [usize; 3],
) -> Vec<u64> {
    let mut first = vec![u64::MAX; count as usize];
    let shape = [labels.shape()[0], labels.shape()[1], labels.shape()[2]];
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let label = labels[[i, j, k]];
                if label == UNLABELLED {
                    continue;
                }
                let at = ((offset[0] + i) as u64 * volume[1] as u64 + (offset[1] + j) as u64)
                    * volume[2] as u64
                    + (offset[2] + k) as u64;
                let slot = &mut first[label as usize - 1];
                if at < *slot {
                    *slot = at;
                }
            }
        }
    }
    first
}

// ------------------------------------------------------------ the merge --

/// The whole answer of the merge: what every block-local label is called in the
/// volume.
///
/// One `u32` per `(block, local label)`, and nothing else. That is what makes
/// the second design below possible at all — the reconciliation between a
/// decomposed labelling and a global one is a **table**, not a volume, and it is
/// smaller than the volume by the ratio of voxels to components.
pub struct GlobalLabels {
    /// Per block, per local label minus one, the global label.
    per_block: BTreeMap<[usize; 3], Vec<u32>>,
    components: u32,
    /// The labelling lattice: how the volume was cut when the local labels were
    /// written. A voxel's block is `coordinate / block`, so this is what turns a
    /// read region back into the numbering it was written under.
    block: [usize; 3],
    lattice: [usize; 3],
    volume: [usize; 3],
}

impl GlobalLabels {
    /// Close every block's local labels into components and number the
    /// components by where their lowest voxel sits in the volume.
    ///
    /// `connectivity` **must** be the one the labelling ran under, for
    /// `ops::components`' reason: the flood inside a block and the walk across a
    /// seam generate one equivalence relation between them.
    pub fn merge(
        reports: &BTreeMap<[usize; 3], ComponentFaces>,
        grid: &BlockGrid,
        connectivity: Connectivity,
    ) -> Result<Self> {
        let counts = grid.blocks_per_axis();
        let index = LabelIndex::build(reports, counts, |report| report.labels)?;
        let firsts = index.gather(reports, |report| &report.first[..], u64::MAX);
        let mut sets = Union::new(index.total());
        walk_seams_with(
            reports,
            counts,
            &index,
            connectivity,
            |report| &report.faces,
            |a, b| sets.union(a, b),
        )?;

        // Every component's least voxel, then the components in the order a
        // whole-volume scan would have met them. `sort_unstable` is enough
        // because the keys are distinct: two components cannot share a voxel.
        let least = sets.fold_min(&firsts);
        let mut roots: Vec<(u64, usize)> = (0..index.total())
            .filter(|&node| sets.find(node) == node)
            .map(|node| (least[node], node))
            .collect();
        roots.sort_unstable();
        // Ascending, so a component with no voxel of its own — `u64::MAX` from
        // `gather`'s `missing` — sorts to the **end** and is checked there.
        if let Some(&(first, node)) = roots.last() {
            if first == u64::MAX {
                return Err(Error::InvalidArgument(format!(
                    "component {node} has no voxel of its own, so the fragments carry more \
                     labels than the label volume does. The two are written by one call and a \
                     mismatch means they came from different runs."
                )));
            }
        }
        if roots.len() > u32::MAX as usize {
            return Err(Error::InvalidArgument(format!(
                "{} components, which is more than a `u32` label volume holds. The local \
                 labels this closes are themselves `u32` per block, so this is reachable only \
                 on a lattice with more blocks than a single block has labels.",
                roots.len()
            )));
        }
        let mut named = vec![0u32; index.total()];
        for (rank, &(_, root)) in roots.iter().enumerate() {
            named[root] = rank as u32 + 1;
        }

        Ok(Self {
            per_block: index.per_block_of(&mut sets, &named),
            components: roots.len() as u32,
            block: grid.block(),
            lattice: counts,
            volume: grid.volume(),
        })
    }

    /// How many components the volume has.
    pub fn components(&self) -> u32 {
        self.components
    }

    /// The lattice the local labels were written on.
    pub fn lattice(&self) -> [usize; 3] {
        self.lattice
    }

    /// The block edge the local labels were written on.
    pub fn block(&self) -> [usize; 3] {
        self.block
    }

    /// The bytes this table occupies, which is the figure the two designs are
    /// compared on and the one a decorated read has to hold resident.
    ///
    /// The payload only — the `u32` per `(block, label)` — plus the `BTreeMap`'s
    /// own key and `Vec` header per block, which at a fine lattice is not
    /// negligible and is the reason it is counted rather than derived from the
    /// component count.
    pub fn table_bytes(&self) -> usize {
        let entries: usize = self.per_block.values().map(|labels| labels.len()).sum();
        entries * std::mem::size_of::<u32>()
            + self.per_block.len()
                * (std::mem::size_of::<[usize; 3]>() + std::mem::size_of::<Vec<u32>>())
    }

    /// The whole table as bytes, which is what [`RelabelComponentsOp::reduce`]
    /// hands the phase.
    ///
    /// **A magic and a version, for the reason `barriers.md` §7.7 gives.** A
    /// reduction that computes the wrong thing answers plausibly in every block
    /// and no guard outside the op could catch it; the blob being the op's own
    /// encoding makes the op's own [`Self::decode`] the one place a mismatch can
    /// surface, so it is given something to refuse.
    ///
    /// **The lattice travels with the table** — the block edge, the lattice
    /// counts and the volume — because the table is a function of the cut the
    /// local labels were written under, and applying one cut's table to another
    /// cut's labels is a complete, well-formed, wrong volume. That is hazard 7
    /// of the module header, and carrying the three lets a reader refuse it
    /// rather than reproduce it.
    ///
    /// Little-endian `u32` throughout, each `usize` as low word then high word,
    /// which is `ComponentFaces::encode`'s convention and not a second one.
    pub fn encode(&self) -> Vec<u8> {
        let mut words: Vec<u32> = vec![TABLE_MAGIC, TABLE_VERSION, self.components];
        for value in self.block.iter().chain(&self.lattice).chain(&self.volume) {
            push_usize(&mut words, *value);
        }
        push_usize(&mut words, self.per_block.len());
        for (block, labels) in &self.per_block {
            for axis in 0..3 {
                push_usize(&mut words, block[axis]);
            }
            push_usize(&mut words, labels.len());
            words.extend_from_slice(labels);
        }
        words_to_bytes(&words)
    }

    /// The inverse of [`Self::encode`].
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        const NOUN: &str = "a global-label table";
        let words = bytes_to_words(bytes, NOUN)?;
        let components = read_header(&words, TABLE_MAGIC, TABLE_VERSION, NOUN)?;
        let mut at = 3usize;
        let mut geometry = [0usize; 9];
        for slot in geometry.iter_mut() {
            *slot = take_usize(&words, &mut at, NOUN, "its geometry")?;
        }
        let block = [geometry[0], geometry[1], geometry[2]];
        let lattice = [geometry[3], geometry[4], geometry[5]];
        let volume = [geometry[6], geometry[7], geometry[8]];
        let blocks = take_usize(&words, &mut at, NOUN, "its block count")?;
        let mut per_block = BTreeMap::new();
        for _ in 0..blocks {
            let index = [
                take_usize(&words, &mut at, NOUN, "a block index")?,
                take_usize(&words, &mut at, NOUN, "a block index")?,
                take_usize(&words, &mut at, NOUN, "a block index")?,
            ];
            let len = take_usize(&words, &mut at, NOUN, "a block's label count")?;
            let end = at.checked_add(len).ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "{NOUN} declares more labels for block {index:?} than the address space holds"
                ))
            })?;
            if words.len() < end {
                return Err(Error::InvalidArgument(format!(
                    "{NOUN} ends inside block {index:?}'s labels"
                )));
            }
            per_block.insert(index, words[at..end].to_vec());
            at = end;
        }
        expect_end(&words, at, NOUN)?;
        Ok(Self {
            per_block,
            components,
            block,
            lattice,
            volume,
        })
    }

    /// Refuse a table built on a different cut of the volume from the one whose
    /// labels it is about to be applied to.
    ///
    /// Hazard 7 of the module header, made checkable. The remap reads a voxel's
    /// block off its coordinate, so a table from another lattice produces a
    /// complete, well-formed, wrong volume; a reduction is exactly where that
    /// could arrive, because the blob a block is handed is bytes and bytes carry
    /// no provenance beyond what they say.
    pub fn check_lattice(&self, grid: &BlockGrid) -> Result<()> {
        if self.block == grid.block()
            && self.lattice == grid.blocks_per_axis()
            && self.volume == grid.volume()
        {
            return Ok(());
        }
        Err(Error::InvalidArgument(format!(
            "a global-label table built on a {:?} lattice of {:?} blocks over a {:?} volume is \
             being applied to a {:?} lattice of {:?} blocks over a {:?} volume. The table names \
             a voxel's block by dividing its coordinate, so one built on another cut is a \
             complete, well-formed, wrong answer rather than an error.",
            self.lattice,
            self.block,
            self.volume,
            grid.blocks_per_axis(),
            grid.block(),
            grid.volume()
        )))
    }

    /// What block `block`'s local labels are called in the volume, indexed
    /// `[local - 1]`.
    pub fn labels_of(&self, block: [usize; 3]) -> Result<&[u32]> {
        self.per_block
            .get(&block)
            .map(|labels| &labels[..])
            .ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "the merge produced no answer for block {block:?} of a {:?} lattice. The \
                     table is built from the same fragments the labels were written with, so a \
                     missing block means the two came from different runs.",
                    self.lattice
                ))
            })
    }

    /// Rewrite a block's local labels into global ones.
    pub fn remap_block(
        &self,
        block: [usize; 3],
        labels: ArrayView3<'_, u32>,
        mut out: ArrayViewMut3<'_, u32>,
    ) -> Result<()> {
        super::shapes_agree(labels.shape(), out.shape(), "GlobalLabels::remap_block")?;
        let table = self.labels_of(block)?;
        for (slot, &label) in out.iter_mut().zip(labels.iter()) {
            *slot = self.lookup(table, label, block)?;
        }
        Ok(())
    }

    /// Rewrite a **read region** of the label volume in place.
    ///
    /// This is the decorator's kernel and it is where the design earns or loses:
    /// the region is whatever a consumer asked for and need not be one block, so
    /// every voxel is looked up under *its own* block's numbering. A halo, a
    /// differently cut consumer lattice and a whole-volume read are all the same
    /// case.
    ///
    /// One table lookup per `(row, block)` rather than per voxel: within a row
    /// the block index changes only at a lattice plane, so the run between two
    /// planes shares one slice.
    pub fn remap_region(&self, region: &Region, mut labels: ArrayViewMut3<'_, u32>) -> Result<()> {
        let shape = [labels.shape()[0], labels.shape()[1], labels.shape()[2]];
        for axis in 0..3 {
            let end = region.start[axis] + region.shape[axis];
            if region.shape[axis] != shape[axis] || end > self.volume[axis] {
                return Err(Error::InvalidArgument(format!(
                    "a read of {:?}+{:?} arrived as a {shape:?} buffer over a {:?} volume. A \
                     decorated read is rewritten by position, so the region and the buffer it \
                     holds have to be the same box.",
                    region.start, region.shape, self.volume
                )));
            }
        }
        for i in 0..shape[0] {
            let bi = (region.start[0] + i) / self.block[0];
            for j in 0..shape[1] {
                let bj = (region.start[1] + j) / self.block[1];
                let mut k = 0usize;
                while k < shape[2] {
                    let global = region.start[2] + k;
                    let bk = global / self.block[2];
                    let stop = (((bk + 1) * self.block[2]) - region.start[2]).min(shape[2]);
                    let block = [bi, bj, bk];
                    let table = self.labels_of(block)?;
                    for slot in k..stop {
                        let label = labels[[i, j, slot]];
                        if label != UNLABELLED {
                            labels[[i, j, slot]] = self.lookup(table, label, block)?;
                        }
                    }
                    k = stop;
                }
            }
        }
        Ok(())
    }

    fn lookup(&self, table: &[u32], label: u32, block: [usize; 3]) -> Result<u32> {
        if label == UNLABELLED {
            return Ok(UNLABELLED);
        }
        table.get(label as usize - 1).copied().ok_or_else(|| {
            Error::InvalidArgument(format!(
                "block {block:?} carries label {label} and the merge gave it {} labels. The \
                 table and the label volume are written by one call, so a gap means the two \
                 came from different runs.",
                table.len()
            ))
        })
    }
}

// ------------------------------------------------------------- the phases --

/// Phase 0: label each block's components and say what crosses its faces.
///
/// **Reach zero.** A block-local labelling reads nothing outside its own core;
/// everything that would need a neighbour is in the fragment instead.
///
/// It writes the `u32` local labels as an image, and that image is
/// **decomposition-dependent on purpose** — the same voxel is label 4 under one
/// cut and label 11 under another. `ops::fill`'s header makes the argument and
/// it holds unchanged here: the local labels are an addressing scheme, and the
/// only things that read them are the merge's table and the rewrite that
/// consumes it, both of which are built from the same fragments.
pub struct LabelComponentsOp {
    name: &'static str,
    stream: String,
    lifecycle: Lifecycle,
    connectivity: Connectivity,
}

impl LabelComponentsOp {
    pub fn new(name: &'static str, stream: impl Into<String>, lifecycle: Lifecycle) -> Self {
        Self {
            name,
            stream: stream.into(),
            lifecycle,
            connectivity: Connectivity::Faces,
        }
    }

    /// The same op, flooding under a stated [`Connectivity`].
    pub fn connecting(mut self, connectivity: Connectivity) -> Self {
        self.connectivity = connectivity;
        self
    }

    pub fn connectivity(&self) -> Connectivity {
        self.connectivity
    }

    pub fn stream(&self) -> &str {
        &self.stream
    }
}

impl FragmentOp for LabelComponentsOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// Nothing crosses as **pixels**. Components are found in this block alone,
    /// and the seam is stitched afterwards from the face fragments on
    /// `self.stream`, whose reach belongs to the op that reads them.
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn reads_pixels(&self) -> bool {
        true
    }

    fn writes_pixels(&self) -> bool {
        true
    }

    /// Labels, whatever the mask arrived as. The same statement
    /// `ops::fill::LabelBackgroundOp` makes and for the same reason: a label is
    /// an integer and the width is this op's to name.
    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::U32
    }

    fn outputs(&self) -> Vec<crate::fragment::FragmentOutput> {
        vec![crate::fragment::FragmentOutput::new(
            self.stream.clone(),
            self.lifecycle,
            crate::fragment::Coverage::EveryBlock,
            // The six-faces shape, as `fill` and `regional` write.
        )
        .sized(crate::fragment::SidecarSize::block_faces())]
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let mut buffer = at.output_buffer(0.0)?;
        let BlockBuf::Array(pixels) = at.pixels()? else {
            // An accounting run has no data. "Present and empty" is a different
            // fact from "absent", which is what `Coverage::EveryBlock` checks.
            return Ok(BlockOutput::fragment(
                self.stream.clone(),
                ComponentFaces::empty().encode(),
            )
            .with_pixels(buffer));
        };
        let mask = super::fill::as_mask(pixels)?;
        let shape = [mask.shape()[0], mask.shape()[1], mask.shape()[2]];
        let BlockBuf::Array(out) = &mut buffer else {
            unreachable!("the environment gave data for the input and none for the output");
        };
        let mut view = out.view_mut::<u32>()?;
        let count =
            label_members_into_with(shape, self.connectivity, |at| mask[at], view.view_mut())?;
        let first = first_voxels(view.view(), count, at.at.offset, at.at.volume);
        let faces = ComponentFaces::of(view.view(), count, first)?;
        Ok(BlockOutput::fragment(self.stream.clone(), faces.encode()).with_pixels(buffer))
    }
}

/// Phase 1: close the components and write the **global** label volume.
///
/// This is the *materialising* half of the pair, and it is the shape the
/// framework admits **now that it can state a barrier**. It declares three
/// things and each of them replaces something this op used to pay:
///
/// | it declares | it no longer pays |
/// |---|---|
/// | [`Self::barrier`] | a whole-volume halo, and therefore a whole re-read of the `u32` local-label image in every block |
/// | [`Self::reduce`] | the union-find once per block, and the fragment set transmitted once per block |
/// | [`Self::seam_fold`] | nothing — this one is a cost, and a small one; see the method |
///
/// **The merge still cannot be its own phase.** The three-phase shape — label,
/// merge, relabel — is refused by `fragment::check_phase_work`, because the
/// fragment-only middle phase would leave its image unwritten. What changed is
/// that a phase's reduction no longer has to be a per-block quantity, so the
/// merge lives in this phase without being re-derived in each of its blocks.
pub struct RelabelComponentsOp {
    name: &'static str,
    stream: String,
    faces_phase: usize,
    lattice: [usize; 3],
    connectivity: Connectivity,
}

impl RelabelComponentsOp {
    /// `faces_phase` is the phase whose blocks wrote the faces — part of the
    /// address rather than a default, for `FragmentInput`'s reason.
    pub fn new(
        name: &'static str,
        stream: impl Into<String>,
        faces_phase: usize,
        grid: &BlockGrid,
    ) -> Self {
        Self {
            name,
            stream: stream.into(),
            faces_phase,
            lattice: grid.blocks_per_axis(),
            connectivity: Connectivity::Faces,
        }
    }

    /// The same op, addressed by a [`crate::assemble::Phase`] handle.
    pub fn reading(
        name: &'static str,
        stream: impl Into<String>,
        faces: crate::assemble::Phase,
        grid: &BlockGrid,
    ) -> Self {
        Self::new(name, stream, faces.index(), grid)
    }

    /// The same op, closing the components under a stated [`Connectivity`]. It
    /// must be the labelling's; [`component_label_phases`] checks the pair.
    pub fn connecting(mut self, connectivity: Connectivity) -> Self {
        self.connectivity = connectivity;
        self
    }

    pub fn connectivity(&self) -> Connectivity {
        self.connectivity
    }

    pub fn lattice(&self) -> [usize; 3] {
        self.lattice
    }
}

impl FragmentOp for RelabelComponentsOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// Nothing this block is authoritative for reaches past its core. The
    /// whole-lattice halo is the **fragment** reach in [`Self::inputs`], not a
    /// pixel one; the global numbering that makes two blocks agree arrives on
    /// `self.stream` and `apply` rewrites the core only.
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn reads_pixels(&self) -> bool {
        true
    }

    fn writes_pixels(&self) -> bool {
        true
    }

    /// A label volume in, a label volume out — the same width, and this is the
    /// one op in the pair for which "unchanged" is the honest answer.
    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::U32
    }

    /// **Yes**, and it is the declaration this phase existed to be unable to
    /// make. See the type's own documentation for what it buys and what it gives
    /// up, which here is nothing: a whole-lattice fragment reach already waited
    /// for every block of the phase below, and the barrier only changes how it
    /// says so.
    fn barrier(&self) -> bool {
        true
    }

    /// **Nothing per block.** The merge is [`Self::reduce`]'s and the table
    /// arrives in the blob, so a block that gathered anything would be holding a
    /// fragment set it has no use for.
    fn gathers(&self) -> bool {
        false
    }

    /// The stream, at **reach zero**.
    ///
    /// It is still declared, because that is what makes it resolvable in
    /// [`Self::reduce`] — `PhaseView` offers the streams the plan records and no
    /// others, for a block's reason: an undeclared stream is one the plan
    /// neither orders nor prices. What changed is the reach. With the merge in
    /// `apply` every block needed every fragment and said so, which is the
    /// `(1 + blocks) x F` multiplier `barriers.md` §7.6 measures; with the merge
    /// in `reduce` the *phase* needs them and no block does, so the set is
    /// transmitted twice — written once, read once — at every lattice.
    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own(self.stream.clone(), self.faces_phase).with_reach([0, 0, 0])]
    }

    /// The answer is a function of the **set** of fragments and not of the order
    /// they arrive in, and this is the declaration that says so and gets it
    /// checked.
    ///
    /// It matters more here than the variant's usual case. `PhaseView` walks the
    /// lattice row-major, and **two lattices walk two different orders**, so a
    /// reduction whose answer depended on the order would make the global
    /// numbering a property of how the volume was cut — which is the one
    /// property this op exists to not have. The executor reduces a second time
    /// with the lattice reversed and requires the same bytes.
    ///
    /// It is true by construction here — [`GlobalLabels::merge`] is handed a
    /// `BTreeMap`, the union-find folds with `min`, and the components are
    /// numbered by their least voxel — and the declaration is the statement that
    /// it must stay true.
    ///
    /// **It costs nothing per block**, and that is a consequence of the hoisting
    /// rather than a coincidence: the same declaration makes the executor apply
    /// each block a second time with its neighbourhood reversed, and it skips
    /// that when the neighbourhood holds at most one fragment. [`Self::inputs`]
    /// reaches zero, so it holds one.
    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::Unordered)
    }

    /// The merge, **once for the phase**.
    ///
    /// This is where the union-find went, and `barriers.md` §7.2 is the argument
    /// for moving it: one merge is small at every lattice, there was one per
    /// block, and at a fine cut their sum exceeded the whole rest of the
    /// pipeline. The table it produces is the same table every block used to
    /// build for itself.
    fn reduce(&self, at: &PhaseView<'_>) -> Result<Vec<u8>> {
        let mut reports = BTreeMap::new();
        for (key, bytes) in at.fragments(&self.stream)? {
            reports.insert(key.block, ComponentFaces::decode(&bytes)?);
        }
        Ok(GlobalLabels::merge(&reports, at.grid, self.connectivity)?.encode())
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let mut buffer = at.output_buffer(0.0)?;
        let BlockBuf::Array(pixels) = at.pixels()? else {
            return Ok(BlockOutput::nothing().with_pixels(buffer));
        };
        let labels = pixels.view::<u32>()?;

        // The phase's own reduction, decoded. `check_lattice` is the guard
        // hazard 7 of the module header asks for and the blob is where it can
        // finally be applied: a table is bytes, and bytes carry no provenance
        // beyond what they say.
        let global = GlobalLabels::decode(at.reduced)?;
        global.check_lattice(at.grid)?;

        // **Only the core.** With the halo relieved this is the whole buffer,
        // and it stays written this way rather than being simplified away:
        // `components::core_within_read` is the third op's worth of remembering
        // that a per-block answer decodes that block's numbering and no other.
        let (offset, extent) = core_within_read(at)?;
        let window = ndarray::s![
            offset[0]..offset[0] + extent[0],
            offset[1]..offset[1] + extent[1],
            offset[2]..offset[2] + extent[2],
        ];
        let BlockBuf::Array(out) = &mut buffer else {
            unreachable!("the environment gave data for the input and none for the output");
        };
        let mut view = out.view_mut::<u32>()?;
        global.remap_block(at.index, labels.slice(window), view.slice_mut(window))?;
        Ok(BlockOutput::nothing().with_pixels(buffer))
    }
}

/// The two phases, on one lattice: a mask on image 0, local labels on image 1,
/// the global label volume on image 2.
///
/// **The two connectivities are checked here**, which is the only place both ops
/// are in one hand. See `ops::fill::agree_on_connectivity`.
pub fn component_label_phases(
    grid: BlockGrid,
    mask_dtype: Dtype,
    label: &LabelComponentsOp,
    relabel: &RelabelComponentsOp,
) -> Result<Decomposition> {
    super::fill::agree_on_connectivity(label.connectivity(), relabel.connectivity())?;
    let volume = grid.volume();
    let mut labelling = crate::fragment::fragment_phase(label, grid.clone())?;
    labelling.dtype = Some(label.produces(mask_dtype));
    let mut relabelling = crate::fragment::fragment_phase(relabel, grid)?;
    relabelling.dtype = Some(relabel.produces(Dtype::U32));
    let plan = Decomposition {
        volume,
        dtype: mask_dtype,
        phases: vec![labelling, relabelling],
        chain_reach: [0, 0, 0],
    };
    plan.check()?;
    Ok(plan)
}

/// One phase: label each block and stop. What the **decorated** design plans,
/// and the whole of what it plans.
///
/// The merge that closes the labels is not a phase here — it is
/// [`GlobalLabels::merge`], run once over the fragments this phase left behind,
/// by the caller, between two `execute_phases` calls.
///
/// **That used to be the whole of the difference the measurement was about**: a
/// merge outside the plan is not a halo either, so nothing re-read the label
/// image and nothing re-ran the union-find per block. Both of those are now true
/// of [`component_label_phases`] as well, stated by the plan rather than by the
/// caller's discipline. What is left of the difference is the extra read and
/// write of the label volume that materialising *is*, and the fact that a
/// decorated image is remapped once per reader.
pub fn component_labelling_phase(
    grid: BlockGrid,
    mask_dtype: Dtype,
    label: &LabelComponentsOp,
) -> Result<Decomposition> {
    let volume = grid.volume();
    let mut labelling = crate::fragment::fragment_phase(label, grid)?;
    labelling.dtype = Some(label.produces(mask_dtype));
    let plan = Decomposition {
        volume,
        dtype: mask_dtype,
        phases: vec![labelling],
        chain_reach: [0, 0, 0],
    };
    plan.check()?;
    Ok(plan)
}

/// Gather every block's component fragment out of a finished run's sidecars.
///
/// The counterpart of the executor's own gather, for the case where the merge is
/// **not** a phase. `Coverage::EveryBlock` is checked by the executor at the end
/// of the phase, so a stream that gets here has one fragment per block; a
/// missing one is still refused rather than assumed empty, because
/// `LabelIndex::build` cannot tell the two apart and would be wrong either way.
pub fn gather_component_faces(
    env: &dyn Environment,
    stream: &str,
    phase: usize,
    grid: &BlockGrid,
) -> Result<BTreeMap<[usize; 3], ComponentFaces>> {
    let counts = grid.blocks_per_axis();
    let mut reports = BTreeMap::new();
    for i in 0..counts[0] {
        for j in 0..counts[1] {
            for k in 0..counts[2] {
                let block = [i, j, k];
                let bytes = env.read_sidecar(stream, phase, block)?.ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "block {block:?} wrote no {stream:?} fragment in phase {phase}. The \
                         stream is declared every-block, so a missing one is a block that did \
                         not run rather than a block with nothing to say."
                    ))
                })?;
                reports.insert(block, ComponentFaces::decode(&bytes)?);
            }
        }
    }
    Ok(reports)
}

// ---------------------------------------------------------- the decorator --

/// An [`Environment`] that applies a [`GlobalLabels`] table to reads of one
/// image, so that a consumer of the **local** label volume sees the **global**
/// one without a global label volume existing anywhere.
///
/// What this is, mechanically
/// --------------------------
/// Every read the executor performs goes through `Environment::read(image,
/// region)`. This forwards all of them and rewrites the buffer for one of them.
/// The region is whatever the consumer asked for — a core, a core with a halo, a
/// whole volume — and [`GlobalLabels::remap_region`] handles all of those by
/// looking each voxel up under the block it was *written* by, which is a
/// function of its coordinate and the labelling lattice. So the consumer's
/// lattice need not be the labelling's, and nothing about the consumer changes.
///
/// **A trivial materialiser over this is the other design.** A one-op identity
/// plan reading the decorated image writes the global label volume as an
/// ordinary image. That is the whole of why this is offered as the decorator and
/// not as a pair of unrelated features: there is one mechanism, and
/// materialisation is a use of it.
///
/// What it costs, and it is not lines
/// -----------------------------------
/// Three things, and every one of them is an invariant some future op has to
/// respect rather than a quantity that can be measured once:
///
/// 1. **The forwarding is total or it is wrong.** `Environment` has thirty-odd
///    methods and most of them are defaulted. A decorator that forwards only the
///    required ones silently reverts every overridden default to the trait's —
///    an inner environment that overrode `slice` to avoid a copy, or
///    `read_sidecar` to go to a network store, would be bypassed and the run
///    would still produce a well-formed answer. So every method is forwarded
///    here, including the ones whose defaults are currently what the inner
///    environment uses, and a method added to `Environment` later is a method
///    that must be added here too. Nothing checks that.
/// 2. **The table has to be right about the lattice.** The remap reads a voxel's
///    block off its coordinate, so a table built on a different lattice from the
///    one the labels were written on produces a complete, well-formed, wrong
///    volume. [`GlobalLabels`] carries the lattice it was built on and
///    [`GlobalLabels::remap_region`] refuses a region outside the volume, but it
///    cannot check that the image it is being applied to is the image the labels
///    were written into — the `image` number is the caller's statement.
/// 3. **It is a read-time cost, so it is paid per reader.** Two consumers of the
///    same label volume remap it twice; a materialised volume is remapped once.
///    That is the axis the measurement is about, and it is the one the
///    complexity argument cannot settle.
///
/// What it does **not** cost is the one thing the "avoid a write" framing
/// suggests: see the acceptance suite for what the write is actually worth
/// against what the phase that would have written it costs.
pub struct RelabelledEnvironment<'e> {
    inner: &'e dyn Environment,
    image: usize,
    table: std::sync::Arc<GlobalLabels>,
    /// Reads intercepted and rewritten, so a run can assert the decoration
    /// happened rather than assume it. A decoration that silently matched no
    /// read would leave local labels in place and answer plausibly.
    remapped: std::sync::atomic::AtomicU64,
}

impl<'e> RelabelledEnvironment<'e> {
    /// Decorate reads of `image` with `table`.
    pub fn new(
        inner: &'e dyn Environment,
        image: impl Into<crate::assemble::ImageId>,
        table: std::sync::Arc<GlobalLabels>,
    ) -> Self {
        Self {
            inner,
            image: image.into().index(),
            table,
            remapped: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// The table this applies.
    pub fn table(&self) -> &GlobalLabels {
        &self.table
    }

    /// How many reads have been intercepted and rewritten.
    pub fn remapped_reads(&self) -> u64 {
        self.remapped.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Environment for RelabelledEnvironment<'_> {
    fn volume(&self) -> [usize; 3] {
        self.inner.volume()
    }

    fn prepare(&self, decomposition: &Decomposition) -> Result<()> {
        self.inner.prepare(decomposition)
    }

    /// The one method that is not a forward.
    ///
    /// The rewrite is in place on the buffer the inner environment just
    /// produced, so nothing is allocated here and the bytes the counters
    /// recorded are the bytes that moved. An `Accounted` buffer — a costing run
    /// — carries no data and is passed through: there is nothing to rewrite and
    /// the cost of the read is unchanged by whether it is decorated.
    fn read(&self, image: usize, region: &Region) -> Result<BlockBuf> {
        let mut buf = self.inner.read(image, region)?;
        if image != self.image {
            return Ok(buf);
        }
        if let BlockBuf::Array(voxels) = &mut buf {
            if voxels.dtype() != Dtype::U32 {
                return Err(Error::InvalidArgument(format!(
                    "image {image} is decorated with a global label table and holds {:?}. The \
                     table is a map from the `u32` labels `ops::label`'s labelling phase writes, \
                     so a different width is a different image.",
                    voxels.dtype()
                )));
            }
            self.table.remap_region(region, voxels.view_mut::<u32>()?)?;
            self.remapped
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(buf)
    }

    fn apply(
        &self,
        slot: &Chain,
        input: &BlockBuf,
        sources: &[(usize, BlockBuf)],
        at: &Placement,
    ) -> Result<BlockBuf> {
        self.inner.apply(slot, input, sources, at)
    }

    /// Forwarded, because a wrapper that does not forward this declines every
    /// cut the planner offers and does so silently. See
    /// [`Environment::apply_sliced`].
    fn apply_sliced(
        &self,
        slot: &Chain,
        input: &BlockBuf,
        sources: &[(usize, BlockBuf)],
        at: &Placement,
        slabs: usize,
    ) -> Result<(BlockBuf, usize)> {
        self.inner.apply_sliced(slot, input, sources, at, slabs)
    }

    fn write(&self, image: usize, within: &Region, valid: &Region, buf: &BlockBuf) -> Result<()> {
        self.inner.write(image, within, valid, buf)
    }

    fn declare_side_output(&self, output: &crate::op::Output) -> Result<()> {
        self.inner.declare_side_output(output)
    }

    fn apply_side(
        &self,
        slot: &Chain,
        input: &BlockBuf,
        sources: &[(usize, BlockBuf)],
        primary: &BlockBuf,
        block: &crate::op::SideBlock<'_>,
    ) -> Result<Vec<crate::voxels::SideBuf>> {
        self.inner.apply_side(slot, input, sources, primary, block)
    }

    fn write_side(
        &self,
        output: &crate::op::Output,
        phase: usize,
        region: &Region,
        buf: &crate::voxels::SideBuf,
    ) -> Result<()> {
        self.inner.write_side(output, phase, region, buf)
    }

    fn put_side(
        &self,
        output: &crate::op::Output,
        phase: usize,
        region: &Region,
        buf: &crate::voxels::SideBuf,
    ) -> Result<()> {
        self.inner.put_side(output, phase, region, buf)
    }

    fn side_constant(&self, region: &Region) -> crate::voxels::SideBuf {
        self.inner.side_constant(region)
    }

    fn release_side(&self, buf: &crate::voxels::SideBuf) {
        self.inner.release_side(buf)
    }

    fn uniform(&self, buf: &BlockBuf) -> Option<f64> {
        self.inner.uniform(buf)
    }

    fn constant(&self, dtype: Dtype, region: &Region, value: f64) -> Result<BlockBuf> {
        self.inner.constant(dtype, region, value)
    }

    fn release(&self, buf: &BlockBuf) {
        self.inner.release(buf)
    }

    fn slice(&self, buf: &BlockBuf, holds: &Region, region: &Region) -> Result<BlockBuf> {
        self.inner.slice(buf, holds, region)
    }

    fn place(
        &self,
        target: &mut BlockBuf,
        holds: &Region,
        region: &Region,
        source: &BlockBuf,
    ) -> Result<()> {
        self.inner.place(target, holds, region, source)
    }

    fn same(&self, left: &BlockBuf, right: &BlockBuf) -> Option<bool> {
        self.inner.same(left, right)
    }

    fn apply_substage(
        &self,
        op: &dyn crate::iterate::IterativeOp,
        index: usize,
        operands: &[BlockBuf],
        at: &crate::op::Anchor,
    ) -> Result<BlockBuf> {
        self.inner.apply_substage(op, index, operands, at)
    }

    fn finish(&self, image: usize) -> Result<()> {
        self.inner.finish(image)
    }

    fn discard_image(&self, image: usize) -> Result<()> {
        self.inner.discard_image(image)
    }

    fn discard_image_after(&self, image: usize, phase: usize) -> Result<()> {
        self.inner.discard_image_after(image, phase)
    }

    fn counters(&self) -> &crate::env::EnvCounters {
        self.inner.counters()
    }

    fn chunk_shape(&self) -> [usize; 3] {
        self.inner.chunk_shape()
    }

    fn sidecars(&self) -> Option<&crate::sidecar::Sidecars> {
        self.inner.sidecars()
    }

    fn require_sidecars(&self) -> Result<&crate::sidecar::Sidecars> {
        self.inner.require_sidecars()
    }

    fn declare_sidecar(&self, stream: &str, lifecycle: Lifecycle) -> Result<()> {
        self.inner.declare_sidecar(stream, lifecycle)
    }

    fn write_sidecar(
        &self,
        stream: &str,
        phase: usize,
        block: [usize; 3],
        bytes: &[u8],
    ) -> Result<()> {
        self.inner.write_sidecar(stream, phase, block, bytes)
    }

    fn read_sidecar(
        &self,
        stream: &str,
        phase: usize,
        block: [usize; 3],
    ) -> Result<Option<Vec<u8>>> {
        self.inner.read_sidecar(stream, phase, block)
    }

    fn sidecar_keys(&self, stream: &str) -> Result<Vec<crate::sidecar::FragmentKey>> {
        self.inner.sidecar_keys(stream)
    }

    fn sidecar_fragments(
        &self,
        stream: &str,
    ) -> Result<Vec<(crate::sidecar::FragmentKey, Vec<u8>)>> {
        self.inner.sidecar_fragments(stream)
    }

    fn discard_sidecars(&self) -> Result<crate::sidecar::Discarded> {
        self.inner.discard_sidecars()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fragment::fragment_phase;
    use crate::ops::element::ElementShape;
    use crate::ops::voxelize::{ball, single_voxel};

    /// **The reduction blob round-trips, and every way of getting it wrong is
    /// refused by name.**
    ///
    /// A phase reduction is the one thing in this crate with no external guard:
    /// `barriers.md` §7.7 is explicit that the executor cannot know what the
    /// reduction was supposed to be, so the op's own decode is the only place a
    /// mismatch can surface. That makes these four refusals the whole of the
    /// mitigation rather than defensive extras.
    #[test]
    fn the_reduction_blob_round_trips_and_a_wrong_one_is_refused() {
        let grid = BlockGrid::new([4, 4, 4], [2, 4, 4]).expect("a lattice");
        let mut per_block = BTreeMap::new();
        per_block.insert([0usize, 0, 0], vec![1u32, 2]);
        per_block.insert([1usize, 0, 0], vec![2u32]);
        let table = GlobalLabels {
            per_block,
            components: 2,
            block: grid.block(),
            lattice: grid.blocks_per_axis(),
            volume: grid.volume(),
        };
        let bytes = table.encode();
        let back = GlobalLabels::decode(&bytes).expect("a round trip");
        assert_eq!(back.components(), 2);
        assert_eq!(back.block(), grid.block());
        assert_eq!(back.lattice(), grid.blocks_per_axis());
        assert_eq!(back.labels_of([0, 0, 0]).expect("block 0"), &[1, 2]);
        assert_eq!(back.labels_of([1, 0, 0]).expect("block 1"), &[2]);
        back.check_lattice(&grid).expect("the same lattice");

        // A fragment of the *other* kind this module writes, decoded as this
        // one. The magic is what stops it, and it is the reason there are two.
        let faces = ComponentFaces::empty().encode();
        let message = GlobalLabels::decode(&faces)
            .err()
            .expect("a component fragment is not a table")
            .to_string();
        assert!(message.contains("magic"), "{message}");

        // Truncated inside the payload rather than at a field boundary.
        let message = GlobalLabels::decode(&bytes[..bytes.len() - 4])
            .err()
            .expect("a short blob")
            .to_string();
        assert!(message.contains("global-label table"), "{message}");

        // A byte count that is not a whole number of words.
        let mut ragged = bytes.clone();
        ragged.push(0);
        assert!(
            GlobalLabels::decode(&ragged).is_err(),
            "a byte count that is not a whole number of words is a truncated blob"
        );

        // The lattice guard: hazard 7 of the module header, and the one a blob
        // makes reachable because bytes carry no provenance.
        let other = BlockGrid::new([4, 4, 4], [4, 4, 4]).expect("a lattice");
        let message = back
            .check_lattice(&other)
            .expect_err("a table from another cut")
            .to_string();
        assert!(message.contains("wrong answer"), "{message}");
    }

    fn window_of(volume: [usize; 3]) -> Region {
        Region::whole(&volume)
    }

    /// Stamp a point set over one block, which is the reference every
    /// decomposition is compared against.
    fn whole(
        volume: [usize; 3],
        element: &StructuringElement,
        points: &[Point],
    ) -> Result<Array3<u64>> {
        let grid = BlockGrid::whole(volume)?;
        let mut out = Array3::<u64>::zeros((volume[0], volume[1], volume[2]));
        label_points_into(
            &[([0, 0, 0], points.to_vec())],
            &grid,
            element,
            &window_of(volume),
            MAX_EXACT_LABEL,
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

    fn split(grid: &BlockGrid, points: &[Point]) -> Vec<([usize; 3], Vec<Point>)> {
        let mut out: std::collections::BTreeMap<[usize; 3], Vec<Point>> = Default::default();
        for point in points {
            out.entry(owner(grid, point.at)).or_default().push(*point);
        }
        out.into_iter().collect()
    }

    #[test]
    fn a_single_voxel_kernel_stamps_the_label_at_the_point_and_nowhere_else() {
        let stamped = whole(
            [4, 4, 4],
            &single_voxel(),
            &[
                Point::weighted([1, 2, 3], 7.0),
                Point::weighted([0, 0, 0], 3.0),
            ],
        )
        .unwrap();
        assert_eq!(stamped[[1, 2, 3]], 7);
        assert_eq!(stamped[[0, 0, 0]], 3);
        assert_eq!(stamped.iter().filter(|&&value| value != 0).count(), 2);
        // and the label is the label, not a count: two points, and no voxel
        // holds a sum of them
        assert_eq!(stamped.iter().copied().max(), Some(7));
    }

    /// The whole reason this op is not `ops::voxelize`: two labels on one voxel
    /// must not add. `3 + 5` is `8`, which is a label neither point carried and
    /// which this fixture would report by name.
    #[test]
    fn two_labels_on_one_voxel_do_not_add() {
        let stamped = whole(
            [3, 3, 3],
            &single_voxel(),
            &[
                Point::weighted([1, 1, 1], 5.0),
                Point::weighted([1, 1, 1], 3.0),
            ],
        )
        .unwrap();
        assert_eq!(
            stamped[[1, 1, 1]],
            3,
            "the lowest label must win; 8 would be a sum and 5 would be the other rule"
        );
        assert_eq!(stamped.iter().filter(|&&value| value != 0).count(), 1);
    }

    /// The collision rule is `min` and not "whichever came last", so the same
    /// two points listed the other way round give the same answer — and the
    /// fixture is asymmetric, so a rule of "last wins" would give `3` for one
    /// order and `5` for the other.
    #[test]
    fn the_collision_rule_does_not_depend_on_the_order_the_points_arrive_in() {
        let low = Point::weighted([1, 1, 1], 3.0);
        let high = Point::weighted([1, 1, 1], 5.0);
        let forwards = whole([3, 3, 3], &single_voxel(), &[low, high]).unwrap();
        let backwards = whole([3, 3, 3], &single_voxel(), &[high, low]).unwrap();
        assert_eq!(forwards, backwards);
        assert_eq!(forwards[[1, 1, 1]], 3);

        // and across fragments, which is the shape the executor hands over
        let grid = BlockGrid::new([4, 4, 4], [2, 4, 4]).unwrap();
        let mut out = Array3::<u64>::zeros((4, 4, 4));
        label_points_into(
            &[
                ([0, 0, 0], vec![Point::weighted([1, 1, 1], 9.0)]),
                ([1, 0, 0], vec![Point::weighted([2, 1, 1], 2.0)]),
            ],
            &grid,
            &ball([1, 0, 0]),
            &window_of([4, 4, 4]),
            MAX_EXACT_LABEL,
            out.view_mut(),
        )
        .unwrap();
        // voxel [1,1,1] is claimed by 9 (its own point) and by 2 (the
        // neighbouring point's kernel), and 2 wins
        assert_eq!(out[[1, 1, 1]], 2);
        assert_eq!(out[[2, 1, 1]], 2);
        assert_eq!(out[[0, 1, 1]], 9);
        assert_eq!(out[[3, 1, 1]], 2);
    }

    /// A ball stamps the label into every voxel of the element, not a count of
    /// them. Derived from the element rather than from a run of this code.
    #[test]
    fn a_ball_stamps_the_label_into_every_voxel_of_the_element() {
        let element = ball([1, 1, 1]);
        let stamped = whole([9, 9, 9], &element, &[Point::weighted([4, 4, 4], 12.0)]).unwrap();
        for offset in element.offsets() {
            let at = [
                (4 + offset[0]) as usize,
                (4 + offset[1]) as usize,
                (4 + offset[2]) as usize,
            ];
            assert_eq!(stamped[at], 12, "member {offset:?} was not stamped");
        }
        assert_eq!(
            stamped.iter().filter(|&&value| value != 0).count(),
            element.len()
        );
    }

    #[test]
    fn a_kernel_that_overhangs_the_volume_stamps_its_overlapping_part() {
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
        let stamped = whole(volume, &element, &[Point::weighted(at, 4.0)]).unwrap();
        assert_eq!(
            stamped.iter().filter(|&&value| value == 4).count(),
            expected
        );
    }

    /// No points at all: an unmarked volume, and not an error.
    #[test]
    fn an_empty_point_set_stamps_nothing() {
        let stamped = whole([3, 3, 3], &ball([1, 1, 1]), &[]).unwrap();
        assert!(stamped.iter().all(|&value| value == 0));

        // and a fragment that is present and empty is the same answer as no
        // fragment at all, which is what `Coverage::EveryBlock` costs a reader
        let grid = BlockGrid::new([4, 4, 4], [2, 4, 4]).unwrap();
        let mut out = Array3::<u64>::zeros((4, 4, 4));
        label_points_into(
            &[([0, 0, 0], Vec::new()), ([1, 0, 0], Vec::new())],
            &grid,
            &single_voxel(),
            &window_of([4, 4, 4]),
            MAX_EXACT_LABEL,
            out.view_mut(),
        )
        .unwrap();
        assert!(out.iter().all(|&value| value == 0));
    }

    /// Every point on one voxel: one label, the lowest, and no arithmetic on
    /// the others.
    #[test]
    fn every_point_on_one_voxel_leaves_one_label() {
        let points: Vec<Point> = [11.0, 4.0, 9.0, 4.0, 250.0]
            .into_iter()
            .map(|label| Point::weighted([2, 2, 2], label))
            .collect();
        let stamped = whole([5, 5, 5], &single_voxel(), &points).unwrap();
        assert_eq!(stamped[[2, 2, 2]], 4);
        assert_eq!(stamped.iter().filter(|&&value| value != 0).count(), 1);
        // 11 + 4 + 9 + 4 + 250 is 278, which is what a sum would have left here
        assert_ne!(stamped[[2, 2, 2]], 278);
    }

    /// The same point set, split into fragments four ways, stamped over the
    /// whole window each time: one answer, every time.
    ///
    /// This is the kernel-level half of decomposition invariance — the executor
    /// half is in `tests/point_labels.rs`, where each block also writes only its
    /// own window. What it discriminates is a rule that depended on **which
    /// fragment a contribution arrived in**: the labels below are deliberately
    /// not in listing order and not in position order, and two of them collide
    /// across a seam under three of the four cuts.
    #[test]
    fn every_cut_of_the_same_points_stamps_the_same_volume() {
        let volume = [16usize, 12, 4];
        let element = ball([2, 2, 1]);
        let points = [
            Point::weighted([3, 3, 1], 40.0),
            Point::weighted([8, 6, 2], 5.0),
            // this pair straddles the seam of a [8, .., ..] cut and their
            // kernels overlap, so the collision is genuinely across a seam
            Point::weighted([7, 6, 2], 31.0),
            Point::weighted([12, 9, 1], 2.0),
            Point::weighted([15, 11, 3], 900.0),
        ];
        let reference = whole(volume, &element, &points).unwrap();
        assert!(
            reference.iter().any(|&value| value == 5),
            "the low label of the straddling pair must survive somewhere"
        );

        for block in [[16usize, 12, 4], [8, 6, 4], [5, 5, 2], [3, 12, 1]] {
            let grid = BlockGrid::new(volume, block).unwrap();
            let mut out = Array3::<u64>::zeros((volume[0], volume[1], volume[2]));
            label_points_into(
                &split(&grid, &points),
                &grid,
                &element,
                &window_of(volume),
                MAX_EXACT_LABEL,
                out.view_mut(),
            )
            .unwrap();
            assert_eq!(out, reference, "cut into {block:?}");
        }
    }

    /// Two points whose kernels collide **across a seam**, arranged so that
    /// "the block that owns the voxel wins" is a different answer from "the
    /// lowest label wins".
    ///
    /// Under the `[4, .., ..]` cut, voxel `[3, 1, 1]` is in block 0's core and
    /// its own point carries label 8; the point that reaches it from block 1
    /// carries 2. A rule that let the owning block's point win would leave 8
    /// there, and would leave 2 under a cut that put both points in one block.
    #[test]
    fn a_collision_across_a_seam_resolves_the_same_way_from_either_side() {
        let volume = [8usize, 4, 4];
        let element = ball([2, 0, 0]);
        let points = [
            Point::weighted([3, 1, 1], 8.0),
            Point::weighted([4, 1, 1], 2.0),
        ];
        let reference = whole(volume, &element, &points).unwrap();
        assert_eq!(reference[[3, 1, 1]], 2, "the lower label must win");
        assert_eq!(reference[[1, 1, 1]], 8, "out of the other point's reach");

        for block in [[8usize, 4, 4], [4, 4, 4], [2, 4, 4], [1, 4, 4], [3, 2, 2]] {
            let grid = BlockGrid::new(volume, block).unwrap();
            let mut out = Array3::<u64>::zeros((volume[0], volume[1], volume[2]));
            label_points_into(
                &split(&grid, &points),
                &grid,
                &element,
                &window_of(volume),
                MAX_EXACT_LABEL,
                out.view_mut(),
            )
            .unwrap();
            assert_eq!(out, reference, "cut into {block:?}");
        }
    }

    #[test]
    fn a_point_outside_its_writers_core_is_refused_and_so_is_one_outside_the_volume() {
        let grid = BlockGrid::new([8, 4, 4], [4, 4, 4]).unwrap();
        let mut out = Array3::<u64>::zeros((8, 4, 4));
        // in the volume, but in block 1's core rather than block 0's
        let error = label_points_into(
            &[([0, 0, 0], vec![Point::weighted([5, 0, 0], 1.0)])],
            &grid,
            &single_voxel(),
            &window_of([8, 4, 4]),
            MAX_EXACT_LABEL,
            out.view_mut(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("outside that block's core"), "{error}");

        // outside the volume altogether: the same refusal, because cores tile.
        // This is the live disagreement with the tools this is compared
        // against, which drop such a point quietly.
        let error = label_points_into(
            &[([1, 0, 0], vec![Point::weighted([9, 0, 0], 1.0)])],
            &grid,
            &single_voxel(),
            &window_of([8, 4, 4]),
            MAX_EXACT_LABEL,
            out.view_mut(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("outside that block's core"), "{error}");
        assert!(out.iter().all(|&value| value == 0), "a refused run stamped");
    }

    #[test]
    fn a_fragment_keyed_off_the_lattice_is_refused() {
        let grid = BlockGrid::new([8, 4, 4], [4, 4, 4]).unwrap();
        let mut out = Array3::<u64>::zeros((8, 4, 4));
        let error = label_points_into(
            &[([7, 0, 0], vec![Point::weighted([0, 0, 0], 1.0)])],
            &grid,
            &single_voxel(),
            &window_of([8, 4, 4]),
            MAX_EXACT_LABEL,
            out.view_mut(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("outside this phase's lattice"), "{error}");
    }

    // ------------------------------------------------------------- labels --

    #[test]
    fn a_label_is_a_whole_number_of_at_least_one_that_an_f64_names_exactly() {
        assert_eq!(label_of(&Point::weighted([0, 0, 0], 1.0)).unwrap(), 1);
        assert_eq!(
            label_of(&Point::weighted([0, 0, 0], MAX_EXACT_LABEL as f64)).unwrap(),
            MAX_EXACT_LABEL
        );
        for (weight, expected) in [
            (0.0, "at least 1"),
            (-3.0, "at least 1"),
            (2.5, "not a whole number"),
            (1e300, "no longer names every integer"),
        ] {
            let error = label_of(&Point::weighted([1, 1, 1], weight))
                .unwrap_err()
                .to_string();
            assert!(error.contains(expected), "for {weight}: {error}");
        }
        // `Point::unit` is a label of 1, which is what the counting case means
        // when it is read as a label
        assert_eq!(label_of(&Point::unit([2, 2, 2])).unwrap(), 1);
    }

    #[test]
    fn the_ceiling_is_the_smaller_of_the_types_maximum_and_what_an_f64_names() {
        assert_eq!(label_ceiling(Dtype::U8), Some(255));
        assert_eq!(label_ceiling(Dtype::U16), Some(65_535));
        assert_eq!(label_ceiling(Dtype::U32), Some(u32::MAX as u64));
        assert_eq!(label_ceiling(Dtype::I32), Some(i32::MAX as u64));
        // a `u64` image does not admit labels the `f64` weight could not carry
        assert_eq!(label_ceiling(Dtype::U64), Some(MAX_EXACT_LABEL));
        assert_eq!(label_ceiling(Dtype::I64), Some(MAX_EXACT_LABEL));
        for refused in [Dtype::Bool, Dtype::F16, Dtype::F32, Dtype::F64] {
            assert_eq!(label_ceiling(refused), None, "{refused:?}");
        }
    }

    /// A label past the destination's end is refused rather than saturated: two
    /// objects becoming one is not something a caller can see afterwards.
    #[test]
    fn a_label_the_destination_cannot_hold_is_refused_by_name() {
        let grid = BlockGrid::whole([2, 2, 2]).unwrap();
        let error = labelled_points(
            &[([0, 0, 0], vec![Point::weighted([0, 0, 0], 300.0)])],
            &grid,
            label_ceiling(Dtype::U8).unwrap(),
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("the largest this destination holds"),
            "{error}"
        );
        // and the same label is fine one type up
        assert!(labelled_points(
            &[([0, 0, 0], vec![Point::weighted([0, 0, 0], 300.0)])],
            &grid,
            label_ceiling(Dtype::U16).unwrap(),
        )
        .is_ok());
    }

    // ------------------------------------------------------------ reaches --

    /// The two reaches are the same parameter said twice, and the phase built
    /// from them keeps every block's core trustworthy. The numbers are
    /// `ops::voxelize`'s, which is the point — one derivation, two ops.
    #[test]
    fn the_phase_a_label_op_builds_keeps_every_core_valid() {
        let grid = BlockGrid::new([20, 4, 4], [8, 4, 4]).unwrap();
        let op = LabelPointsOp::new("label", "points", 0, ball([5, 1, 1]), &grid).unwrap();
        assert_eq!(op.block_reach(), [1, 0, 0], "ceil(5 / 8)");
        assert_eq!(op.reach(0, 20), 5);
        assert_eq!(op.reach(1, 4), 1);
        let phase = fragment_phase(&op, grid).unwrap();
        assert_eq!(phase.reach, [5, 1, 1]);
        assert_eq!(phase.halo, [8, 1, 1]);
        for block in &phase.blocks {
            assert_eq!(block.valid, block.core, "block {:?}", block.index);
        }
    }

    #[test]
    fn an_op_refuses_a_lattice_its_reach_was_not_derived_for() {
        let grid = BlockGrid::new([20, 4, 4], [8, 4, 4]).unwrap();
        let op = LabelPointsOp::new("label", "points", 0, ball([5, 1, 1]), &grid).unwrap();
        op.check_grid(&grid).unwrap();
        let error = op
            .check_grid(&BlockGrid::new([20, 4, 4], [4, 4, 4]).unwrap())
            .unwrap_err()
            .to_string();
        assert!(error.contains("derived its block reach"), "{error}");
    }

    #[test]
    fn a_kernel_with_no_voxels_and_an_unusable_stream_name_are_refused_at_construction() {
        let grid = BlockGrid::new([20, 4, 4], [8, 4, 4]).unwrap();
        assert!(LabelPointsOp::new("label", "", 0, single_voxel(), &grid).is_err());
        assert!(LabelPointsOp::new("label", "a/b", 0, single_voxel(), &grid).is_err());
        if let Ok(empty) = StructuringElement::from_offsets(Vec::new()) {
            assert!(empty.is_empty(), "this fixture needs an empty element");
            match LabelPointsOp::new("label", "points", 0, empty, &grid) {
                Ok(_) => panic!("an empty kernel was accepted"),
                Err(error) => assert!(error.to_string().contains("no voxels in it"), "{error}"),
            }
        }
    }

    /// The rule stated as a property rather than as a fixture: whatever the
    /// order and whatever the split, the value at a voxel is the smallest label
    /// whose kernel covers it. Written out from the definition, and compared
    /// against the code, so the two cannot drift.
    #[test]
    fn every_voxel_holds_the_smallest_label_that_reaches_it() {
        let volume = [10usize, 8, 4];
        let element = ball([2, 2, 1]);
        let points = [
            Point::weighted([2, 2, 1], 40.0),
            Point::weighted([3, 3, 2], 7.0),
            Point::weighted([7, 5, 2], 19.0),
            Point::weighted([8, 5, 2], 19.0),
            Point::weighted([0, 0, 0], 100.0),
        ];
        let stamped = whole(volume, &element, &points).unwrap();
        for i in 0..volume[0] {
            for j in 0..volume[1] {
                for k in 0..volume[2] {
                    let mut want = 0u64;
                    for point in &points {
                        let covers = element.offsets().iter().any(|offset| {
                            [i, j, k].iter().enumerate().all(|(axis, &position)| {
                                point.at[axis] as isize + offset[axis] == position as isize
                            })
                        });
                        if covers {
                            let label = point.weight as u64;
                            if want == 0 || label < want {
                                want = label;
                            }
                        }
                    }
                    assert_eq!(stamped[[i, j, k]], want, "at {:?}", [i, j, k]);
                }
            }
        }
    }

    // ------------------------------------------- a kernel that re-phases --

    /// The reference rule, written out in the arithmetic it is stated in:
    /// `a[max(0, c - lo) : min(c + hi + 1, n) : step]`, as a list of coordinates.
    ///
    /// Independent of `StructuringElement` on purpose — every assertion below
    /// that says which voxels a point stamps is compared against this and not
    /// against `offsets_at`, so the op and the element cannot agree with each
    /// other and both be wrong.
    fn reference_window(centre: usize, lo: usize, hi: usize, step: usize, n: usize) -> Vec<usize> {
        (centre.saturating_sub(lo)..(centre + hi + 1).min(n))
            .step_by(step)
            .collect()
    }

    /// A flat `size`-wide box on axis 0, decimated by `step`, at either origin.
    fn flat(size: usize, step: usize, origin: StepOrigin) -> StructuringElement {
        StructuringElement::from_size_stepped_at(
            ElementShape::Box,
            [size, 1, 1],
            [step, 1, 1],
            origin,
        )
        .unwrap()
    }

    fn marked(stamped: &Array3<u64>) -> Vec<usize> {
        (0..stamped.shape()[0])
            .filter(|&i| stamped[[i, 0, 0]] != 0)
            .collect()
    }

    /// **The stamp places the window the origin names at the point's own
    /// position**, which is the transpose of the gather the same element gives —
    /// see the module header for why those are one question and not two.
    ///
    /// The point sits inside `lo` of the low face, which is the only place the
    /// two origins differ, and the expected set is the reference array expression
    /// written out rather than `offsets_at` asked twice.
    #[test]
    fn a_re_phasing_kernel_stamps_the_window_at_the_points_own_position() {
        let volume = [24usize, 1, 1];
        let element = flat(11, 2, StepOrigin::ClippedStart);
        let (lo, hi) = (5usize, 5usize);
        for centre in [0usize, 1, 2, 3, 4, 5, 12] {
            let stamped = whole(volume, &element, &[Point::weighted([centre, 0, 0], 7.0)]).unwrap();
            assert_eq!(
                marked(&stamped),
                reference_window(centre, lo, hi, 2, volume[0]),
                "a point at {centre} did not stamp the window the reference expression names"
            );
        }
    }

    /// **The negative control**: the same program with the origin changed stamps
    /// a different volume, so the assertion above is about the origin rather than
    /// about an element whose two readings happen to agree.
    ///
    /// At `centre = 2` the two sets are not merely different, they are
    /// **disjoint** — `{0, 2, 4, 6}` against `{1, 3, 5, 7}` — which is as far
    /// apart as two windows of one element can be.
    #[test]
    fn the_two_origins_stamp_different_volumes() {
        let volume = [24usize, 1, 1];
        let clipped = flat(11, 2, StepOrigin::ClippedStart);
        let anchored = flat(11, 2, StepOrigin::Anchor);
        assert_ne!(clipped, anchored);

        let point = [Point::weighted([2, 0, 0], 7.0)];
        let left = marked(&whole(volume, &clipped, &point).unwrap());
        let right = marked(&whole(volume, &anchored, &point).unwrap());
        assert_eq!(left, vec![0, 2, 4, 6]);
        assert_eq!(right, vec![1, 3, 5, 7]);
        assert!(
            left.iter().all(|at| !right.contains(at)),
            "the two origins must be telling apart here"
        );

        // and deep in the interior they agree, which is why this is only ever
        // visible near a low face
        let deep = [Point::weighted([12, 0, 0], 7.0)];
        assert_eq!(
            marked(&whole(volume, &clipped, &deep).unwrap()),
            marked(&whole(volume, &anchored, &deep).unwrap())
        );
    }

    /// **The anchored kernel cannot see where it is.** Its members are one set,
    /// so what it stamps at a point is that set translated to the point and
    /// clipped at the volume — the same picture everywhere, and never a function
    /// of how near a face the point sits. Swept over *every* position of a
    /// volume, against the translate-and-clip model written out here rather than
    /// asked of the element.
    ///
    /// This is what every existing caller depends on, because an unstepped
    /// element normalises to [`StepOrigin::Anchor`]: every kernel this op has
    /// been given until now is on the unchanged side of this assertion.
    ///
    /// The second half is what keeps the first from being a tautology: the
    /// re-phasing element is swept the same way and **fails** that model, at a
    /// position the assertion names.
    #[test]
    fn an_anchored_kernel_stamps_its_own_members_translated_at_every_position() {
        let volume = [24usize, 1, 1];
        // the model: the element's own offsets, moved to the point and clipped
        let translated = |element: &StructuringElement, centre: usize| -> Vec<usize> {
            element
                .offsets()
                .iter()
                .filter_map(|offset| {
                    let at = centre as isize + offset[0];
                    (at >= 0 && (at as usize) < volume[0]).then_some(at as usize)
                })
                .collect()
        };

        for element in [
            flat(11, 2, StepOrigin::Anchor),
            flat(8, 3, StepOrigin::Anchor),
            ball([2, 0, 0]),
            single_voxel(),
        ] {
            for centre in 0..volume[0] {
                let stamped =
                    whole(volume, &element, &[Point::weighted([centre, 0, 0], 7.0)]).unwrap();
                assert_eq!(
                    marked(&stamped),
                    translated(&element, centre),
                    "an anchored kernel of {:?} saw where it was placed, at {centre}",
                    element.size()
                );
            }
        }

        // and the re-phasing one does not satisfy that model — near a low face it
        // stamps a set the translation does not name, which is the whole content
        // of the assertion above
        let clipped = flat(11, 2, StepOrigin::ClippedStart);
        let disagreeing: Vec<usize> = (0..volume[0])
            .filter(|&centre| {
                let stamped =
                    whole(volume, &clipped, &[Point::weighted([centre, 0, 0], 7.0)]).unwrap();
                marked(&stamped) != translated(&clipped, centre)
            })
            .collect();
        // `lo` is 5 and the stride is 2, so the anchored lattice sits on the odd
        // residue class of the offsets. A point at an odd `c` inside the face has
        // its clipped start at `-c`, which is that same class, and the two rules
        // land on one another; at an even `c` they land on opposite classes and
        // the sets are disjoint. So the disagreement is `{0, 2, 4}` and not all of
        // `0..5` — a fact about residues, and the reason this is written out.
        assert_eq!(
            disagreeing,
            vec![0, 2, 4],
            "the re-phasing kernel must leave the translate-and-clip model where the window meets \
             the low face on a different residue class, and nowhere else"
        );
    }

    /// **A re-phased window can hold more members than the element does**, and
    /// the stamp writes every one of them: nothing here is sized from
    /// [`StructuringElement::len`].
    ///
    /// A ball of radius two stepped by two keeps seven offsets in the interior —
    /// the centre and the six poles — and eight at the phase reached from
    /// `[1, 1, 1]`, where the stride lands on `-1` and `+1` on every axis and all
    /// eight corners of that cube are inside the ball's surface. The counts are
    /// derived here from the shape's own rule rather than taken from a run.
    #[test]
    fn a_re_phased_window_can_hold_more_members_than_the_element_does() {
        let volume = [9usize, 9, 9];
        let element = StructuringElement::from_sides_stepped_at(
            ElementShape::Ellipsoid,
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            StepOrigin::ClippedStart,
        )
        .unwrap();
        // the interior set: the centre and the six poles, `sum (d / 2)^2 <= 1`
        assert_eq!(element.len(), 7);

        let mut scratch = Vec::new();
        let window = element.offsets_at([1, 1, 1], volume, &mut scratch).to_vec();
        // the phase from `[1, 1, 1]`: `-1` and `+1` on each axis, `3 * (1/2)^2`
        // being `0.75`, so every corner of that cube is a member
        assert_eq!(window.len(), 8);
        assert!(window.len() > element.len());

        let stamped = whole(volume, &element, &[Point::weighted([1, 1, 1], 5.0)]).unwrap();
        assert_eq!(
            stamped.iter().filter(|&&value| value == 5).count(),
            8,
            "the stamp wrote a count taken from the element rather than from the window"
        );
        for offset in &window {
            let at = [
                (1 + offset[0]) as usize,
                (1 + offset[1]) as usize,
                (1 + offset[2]) as usize,
            ];
            assert_eq!(stamped[at], 5, "member {offset:?} was not stamped");
        }

        // and the interior point stamps the seven the element names, so the two
        // counts are genuinely both live in one run
        let deep = whole(volume, &element, &[Point::weighted([4, 4, 4], 5.0)]).unwrap();
        assert_eq!(deep.iter().filter(|&&value| value == 5).count(), 7);
    }

    /// **Decomposition invariance for a kernel whose window depends on where it
    /// is placed**: every cut of the same point set stamps the same volume, byte
    /// for byte.
    ///
    /// It holds by construction here — a point's coordinate is a volume
    /// coordinate and `grid.volume()` is the volume under every cut — and it is
    /// measured anyway, because "by construction" is an argument about the code
    /// as it stands and this is the property the crate exists to defend. The
    /// volume is **narrower on axis 2 than the window**, so every point re-phases
    /// there and no block holds only interior voxels.
    #[test]
    fn every_cut_of_the_same_points_stamps_the_same_volume_for_a_re_phasing_kernel() {
        let volume = [16usize, 12, 4];
        let element = StructuringElement::from_size_stepped_at(
            ElementShape::Box,
            [11, 3, 9],
            [2, 1, 3],
            StepOrigin::ClippedStart,
        )
        .unwrap();
        assert!(
            element.sides(2).0 >= volume[2],
            "the window must be wider than axis 2 of the volume, so every point re-phases there"
        );
        let points = [
            Point::weighted([3, 3, 1], 40.0),
            Point::weighted([8, 6, 2], 5.0),
            Point::weighted([7, 6, 2], 31.0),
            Point::weighted([12, 9, 1], 2.0),
            Point::weighted([15, 11, 3], 900.0),
            Point::weighted([0, 0, 0], 60.0),
        ];
        let reference = whole(volume, &element, &points).unwrap();
        assert!(reference.iter().any(|&value| value == 5));
        assert!(reference.iter().any(|&value| value == 0));

        for block in [
            [16usize, 12, 4],
            [8, 6, 4],
            [5, 5, 2],
            [3, 12, 1],
            [16, 5, 3],
        ] {
            let grid = BlockGrid::new(volume, block).unwrap();
            let mut out = Array3::<u64>::zeros((volume[0], volume[1], volume[2]));
            label_points_into(
                &split(&grid, &points),
                &grid,
                &element,
                &window_of(volume),
                MAX_EXACT_LABEL,
                out.view_mut(),
            )
            .unwrap();
            assert_eq!(out, reference, "cut into {block:?}");
        }

        // the negative control, through the same sweep: the anchored element is a
        // different answer, so the invariance above is invariance of *this* rule
        let anchored = StructuringElement::from_size_stepped_at(
            ElementShape::Box,
            [11, 3, 9],
            [2, 1, 3],
            StepOrigin::Anchor,
        )
        .unwrap();
        assert_ne!(whole(volume, &anchored, &points).unwrap(), reference);
    }
}
