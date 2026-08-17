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
// * the output is an **integer** level. `Dtype::Bool` and the float types are
//   refused by name — see [`label_ceiling`] — because a two-valued type cannot
//   hold a label at all (`ops::voxelize` into a `bool` level is the op that
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
// No output stream is declared: this phase's output is a pixel level, exactly as
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

use ndarray::{Array3, ArrayViewMut3, Axis, Slice};

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::fragment::{BlockOutput, BlockView, FragmentInput, FragmentOp};
use crate::geometry::BlockGrid;
use crate::points::{decode_points, Point};
use crate::region::Region;
use crate::sidecar::check_stream_name;
use crate::voxels::Voxels;

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

/// The largest label a level of `dtype` can hold, or `None` for a type that
/// cannot hold labels at all.
///
/// **Integer types only, and the two refusals each have a reason.** `Dtype::Bool`
/// holds one bit, which is enough to say "marked" and not enough to say *which*
/// mark — the op that writes that volume is `ops::voxelize` into a `bool` level,
/// which answers exactly "did any point's kernel cover this voxel". The float
/// types are refused because a label is compared for equality by every consumer
/// of a label volume, and equality on a value that has passed through a rounding
/// is the kind of comparison that works until it does not; a caller who really
/// wants labels in an `f64` buffer writes this to an integer level and puts
/// `ops::voxelwise::WidenOp` after it, where the widening is exact and is
/// something they asked for.
///
/// The ceiling is the smaller of the type's own maximum and [`MAX_EXACT_LABEL`],
/// so a `u64` level does not admit labels the `f64` weight could not have
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
/// [`label_ceiling`] applied to the level's element type.
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
    /// running this op reads not one voxel of the level below it.
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
                 point — and a level of {} holds no such value. Write an integer level; a \
                 two-valued level is `ops::voxelize` into a bool level, and a float one is this \
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

/// Write a `u64` label buffer into `at` of a buffer of the level's own type.
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
                "label: a level of {} holds no labels",
                other.numpy_name()
            )))
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fragment::fragment_phase;
    use crate::ops::element::ElementShape;
    use crate::ops::voxelize::{ball, single_voxel};

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
        // a `u64` level does not admit labels the `f64` weight could not carry
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
