// SPDX-License-Identifier: MIT
//
// Original work for this crate. Written from the definition of the operation,
// not adapted from any implementation of it.
//
// Filling the holes in a binary mask: every background component that does not
// reach the outside of the volume becomes foreground.
//
// The first op here that is genuinely global
// ------------------------------------------
// Every other op in `ops/` answers a voxel from a bounded neighbourhood of it,
// and the whole framework is arranged around that: a reach, a halo, a guard that
// compares the two. This one cannot be written that way at any halo, and the
// reason is not that the halo would be large — it is that there is no halo at
// all that works. Whether a background voxel is inside a hole depends on whether
// a path exists from it to the volume's outside, and a path can be arbitrarily
// long and arbitrarily crooked. A cavity three voxels across can drain through a
// channel that wanders the length of the volume. No local window can see that.
//
// So this is a **fragment-and-join**, in two phases:
//
// | phase | shape | what it does |
// |---|---|---|
// | 0 | `volume -> fragments`, and writes pixels | label each block's background locally; write the labels as an image; emit the block's six faces and which of its labels touch the volume's outside |
// | 1 | `fragments -> volume`, **behind a barrier** | close every block's faces into global components **once for the phase**, then rewrite each block's labels into the filled mask from that one answer |
//
// Phase 0 is embarrassingly parallel with **reach zero** — a block-local
// component labelling reads nothing outside its own core. Everything global is
// paid for exactly once, in phase 1, and what crosses between them is six
// planes of labels per block rather than any volume of pixels.
//
// **"Exactly once" is now literally true**, and for most of this file's life it
// was not. Phase 1 declares `FragmentOp::barrier` — its dependency on phase 0 is
// completion rather than region coverage — and computes the union-find in
// `FragmentOp::reduce`, which runs at the one moment a barrier creates. Before
// that declaration existed the only way to say "after all of phase 0" was to
// fetch the whole volume in every block, and the merge had nowhere to live but
// `apply`, which is per block. "What this costs" below is the measurement of
// the difference, and `ops::components::Merge` is the switch the three arms of it are
// built from.
//
// The intermediate image is decomposition-dependent, and the output is not
// -----------------------------------------------------------------------------
// Worth stating plainly because it looks at first like the defect this crate
// exists to prevent. Block-local labels *are* a function of the block: the same
// background voxel gets label 4 under one decomposition and label 11 under
// another, and the image phase 0 writes therefore differs between two runs that
// must agree. That is fine, and it is fine for a reason rather than by luck: the
// labels are an addressing scheme, phase 1 consumes them only through the
// `(block, label) -> outside?` map it builds from the same fragments, and
// nothing outside these two phases ever reads them. The **output** is a function
// of the mask alone, and the acceptance suite asserts it byte-identically across
// decompositions — including one where the block count is 1, which is the case
// with no seams at all.
//
// Connectivity: one choice, and it is the **background's**
// ---------------------------------------------------------
// A hole is a background component. Two decisions hide in that sentence, and
// they are not the same kind of decision:
//
// * **Background rather than foreground** is fixed here and is not a parameter.
//   Hole filling is a statement about the complement, so the labelling runs on
//   the complement, and an op that labelled the foreground would be a different
//   operation. `ops::detect` is that operation.
// * **Which neighbours join** is a [`Connectivity`], stated by the caller, and
//   [`Connectivity::Faces`] unless one is. It is genuinely a choice — a caller
//   who wants a cavity that leaks only diagonally treated as closed asks for
//   `Faces`, and one who wants it drained asks for `FacesEdgesAndCorners` — and
//   the two are both right about different questions.
//
// **There is exactly one connectivity here, and it is the background's**, because
// the background is the only thing this op labels. That is worth saying because
// the literature carries a second number beside it: the *complementary pair*
// convention, under which a 6-connected foreground is analysed against a
// 26-connected background and vice versa, so that a surface separates what it
// looks like it separates. This op cannot honour that pairing on a caller's
// behalf, because it never sees a foreground connectivity — nothing here labels
// the foreground. A plan that fills holes and then runs `ops::detect` over the
// result is where the pair becomes visible, and choosing `Faces` for the fill and
// `FacesEdgesAndCorners` for the detection is the caller's to do. Both ops take
// the parameter separately for exactly that reason.
//
// **The fragment did not have to change.** The obvious guess is that a wider
// connectivity needs the twelve edge lines and the eight corner voxels beside the
// six face planes. It does not: a voxel with any neighbour outside its block lies
// on a *face* of that block, so the six planes already are the whole boundary
// shell, and the edge lines and corner voxels are rows and single entries of
// them. `components`'s header is where that is argued; what widened is the seam
// walk's inputs and nothing about the bytes.
//
// **Both phases carry the choice and [`fill_phases`] refuses a pair that
// disagrees.** Phase 0's flood and phase 1's seam walk are two halves of one
// equivalence relation, so a plan that labelled at twenty-six and merged at six
// would join inside a block what it kept apart across a seam — a
// decomposition-dependent answer, which is the one defect this crate exists to
// prevent, and it is refused at planning time rather than discovered.
//
// Two walls this op ran into, and what was done about each
// --------------------------------------------------------
// **A fragment-only phase is terminal as far as images go, and that is still
// true.** The natural three-phase shape — label, merge, relabel — puts a
// `fragments -> fragments` merge between two pixel phases, and that cannot be
// planned: `fragment.rs`'s `check_phase_work` refuses it in as many words,
// because image `p+1` would go unwritten and phase `p+1` would read an image
// nobody produced. The merge is therefore folded into phase 1, which reads the
// fragments *and* the labels and writes the answer.
//
// **What has changed is what the fold costs.** The sentence that stood here said
// the cost of it is that every block re-runs the same global union-find, and that
// was right; it is no longer the case, and the wall it names is not the reason.
// `FragmentOp::reduce` gives the phase somewhere to put an answer that belongs to
// the phase rather than to a block, so the merge runs once inside a two-phase
// plan and Rule A is not in the way of anything. `docs/design/barriers.md` §7.5
// is the argument for why the whole-phase blob is specifically the shape that
// does not need the third phase — and §7.3 is the record of why the merge was
// ever per block, which was never a choice anybody made.
//
// **A fragment op could not change the element type of the image it writes.**
// Phase 0 reads a `bool` mask and writes `u32` labels, and `check_dtypes` folds
// a plan's element types over the *chain* — which a fragment phase owns no slot
// of. So the fold saw nothing, compared `bool` against the `u32` the plan
// allocated, and refused the plan with a message about ops the phase does not
// have. Fixed rather than worked around: `FragmentOp::produces` now exists,
// defaulted to "unchanged" so nothing that shipped before it changed, and
// `check_dtypes` consults it. That is the same shape `BlockOp::produces` has and
// the same reason it exists.
//
// What this costs, stated rather than discovered
// ------------------------------------------------
// **Read in the past tense down to "What the declarations bought".** Everything
// below is the analysis that produced `docs/design/barriers.md`, and it is the
// analysis of the shape this op *had*: a whole-lattice fragment reach, a
// whole-volume halo, and a merge in every block. That shape is still buildable —
// it is `Merge::PerBlock` — and it is what the measurement at the end compares
// against. Nothing here has been deleted, because the argument is what makes the
// numbers at the end mean anything.
//
// Under `Merge::PerBlock` phase 1 declares a whole-lattice fragment reach, so on
// a lattice of `N` blocks it transfers every block's fragment to each of `N`
// blocks, and each fragment is the block's six face planes. **Hold the block edge fixed and grow the volume**
// and that is quadratic in the block count, because each block contributes a
// fixed amount of face: a lattice of a thousand 256-cube blocks moves on the
// order of a terabyte of face planes to do a merge whose answer is one boolean
// per label.
//
// **Hold the volume fixed and cut it more finely and it is not quadratic**, and
// the difference is worth stating because the two regimes look like a
// contradiction and are not. The traffic is `(1 + N) x F`, where `F` is what all
// the fragments weigh together — the **total face area of the cut** — and `F` is
// constant per block only in the first regime. In the second it obeys
//
// > `F = sum over axes of (cuts on that axis) x (area of the face perpendicular
// > to it)`
//
// so it grows with the cut, but sub-linearly in `N`, and *which axis is cut*
// matters more than how many blocks result: `F` is dominated by the largest
// face, which is the one perpendicular to the volume's **shortest** axis. Cutting
// that axis is doubly expensive, because it raises both factors of `(1 + N) x F`.
// Swept in `tests/label_materialisation_cost.rs`, which found a cubic cut
// carrying three times the fragments of a slab cut *at the same block count* on
// an anisotropic volume — and the prediction that motivated the sweep was the
// other way round.
//
// **Three sentences of this header were a false reassurance, and the shape of the
// mistake is worth more than the correction.** They said the union-find is over
// face labels rather than voxels, so it is small next to the pixels, and that the
// fragment is six planes against a whole block of pixels. Every clause is true
// **per invocation** — which is exactly why it read as safe — and the paragraph
// never multiplied by the number of invocations, of which there is one per block.
// Measured in `tests/label_materialisation_cost.rs`, on a recorded `bool` volume
// of `[16, 1304, 3369]`, timing the merge alone — `ops::label`'s, which is this
// program with a different per-label fact, so the shape is this one's and the
// constants are not claimed for `fill` exactly. **The CPU half:**
//
// | lattice | every block's merge, together, serial |
// |---|---|
// | `[1, 1, 1]` | 0.10 s |
// | `[1, 2, 2]` | 0.14 s |
// | `[2, 4, 4]` | 2.47 s |
// | `[4, 8, 8]` | **33.67 s** |
//
// One invocation stays between about three hundredths and a seventh of a second
// across that whole sweep — it really is small, at every lattice, which is
// exactly why the sentence read as safe — and there is one per block. The same
// pipeline with the merge hoisted out of the per-block loop runs end to end in
// under four seconds. So at a fine cut the **redundant merge alone is the largest
// single cost of the op**, in a currency no byte counter shows, and it grows
// faster than the block count, because there are `N` invocations and each decodes
// a fragment set that is itself growing.
//
// **The byte half fails too**, past the cut where the total face area exceeds the
// volume, and that point is reachable: measured on the same volume, the fragment
// set against the label image is 12.8% at one block, 25.5% at `[2, 2, 2]`, 51.0%
// at `[4, 4, 4]` and **101.9%** at `[8, 8, 8]` — where one transmission of the
// fragments costs more than one transmission of the whole label image, before any
// per-block multiplier. "The fragments are small beside the pixels" is a
// statement about coarse lattices and nobody had swept it.
//
// **And that reach was also a halo, which cost pixels as well as fragments.**
// `fragment_phase` sets `halo = max(reach, fragment reach * block edge)` for a
// phase with no barrier, so phase 1's halo was the whole volume, so its *read
// extent* was the whole volume, so every block of it read the entire label
// image. That looked at first like a convenience constructor conflating two
// quantities — the executor gathers fragments from `neighbourhood(index,
// input.reach, counts)` and never consults the halo — and building the phases by
// hand with a zero halo was the obvious fix. The guard refused it, and was right
// to: **the halo was also the dependency edge.** Phases here are pipelined, so
// what made block `b` of phase 1 wait for the blocks of phase 0 whose fragments
// it reads was precisely the halo, and a zero halo would have had block `b` read
// fragments nobody had written yet. The coupling was load-bearing and the two
// costs were one cost.
//
// What the declarations bought, measured
// ----------------------------------------
// Both declarations landed and `tests/fragment_join_barrier.rs` is the evidence.
// It builds three arms out of **one op with one word changed** — `Merge` — so
// every other line is the same line and a difference in the counters is
// attributable to the declaration and to nothing else, and it runs `ops::regional`
// beside this op, which is the same program with a different per-label fact.
// Every byte column is predicted from a formula and then compared; the answer is
// checked byte-for-byte against a whole-volume reference in all three arms at
// every lattice, which is what makes the arm with the barrier *withheld* the
// liveness control rather than a second implementation.
//
// On `[16, 32, 32]` of `bool`, in bytes, with the merge count beside them:
//
// | blocks | arm | pixel reads | fragments | total | merges |
// |---|---|---|---|---|---|
// | 32 | in every block | 2 113 536 | 1 691 316 | 3 886 772 | 64 |
// | 32 | + a barrier | 81 920 | 1 691 316 | 1 855 156 | 64 |
// | 32 | + hoisted | 81 920 | 153 756 | **317 596** | **2** |
// | 256 | in every block | 16 793 600 | 29 508 740 | 46 384 260 | 512 |
// | 256 | + a barrier | 81 920 | 29 508 740 | 29 672 580 | 512 |
// | 256 | + hoisted | 81 920 | 344 460 | **508 300** | **2** |
//
// **91.3x the traffic at 256 blocks, and 256 times the merges**, to compute a
// table of one boolean per label. Two things in that table are worth more than
// the ratio:
//
// * **The barrier alone is worth 1.56x here and the specification's own example
//   is worth 2.7x.** Not a contradiction and not a constant: a barrier removes
//   the *pixel* half and leaves the fragment half, so what it is worth is
//   whatever fraction of the traffic the pixels were. At 256 blocks of this
//   volume the fragment set is 175% of the label image — the regime "The byte
//   half fails too" above predicts and this measurement lands inside — so the
//   pixel half it removes is the smaller half. **The fraction is a property of
//   the cut and must be measured per op and per lattice, never quoted.**
// * **The merge column is `2 x blocks`, not `blocks`.** Phase 1 declares
//   `SeamFold::Unordered`, which is true of it and is checked rather than
//   believed — the executor applies each block a second time with the
//   neighbourhood reversed. Under `Merge::PerBlock` that doubles the union-find,
//   which is a cost this header never named because the declaration was never
//   made. Hoisted it costs one extra reduction for the phase and no extra
//   application at all, because the per-block fragment reach is then nothing and
//   a one-element sequence has no order to check. So the check is a per-block
//   cost that hoisting turns into a per-phase one, by the same multiplier it
//   removes from everything else.
//
// The wall clock moves with the merges rather than with the bytes, which is
// §7.2's whole point: at 256 blocks and one thread the in-plan arm takes on the
// order of ninety times the hoisted one, tracking the merge column rather than
// the byte column. That figure is **reported by the test and not asserted**, and
// no absolute seconds are quoted here, because it moves with what else the
// machine is doing while the byte columns reproduce to the digit.
//
// **What is left, and it is geometry.** The floor is `F` itself — the total face
// area of the cut, transmitted a fixed small number of times — and no
// declaration removes it: telling one block about another's boundary costs the
// boundary. `barriers.md` §7.6 swept that and found the growth intrinsic and the
// *multiplier* an artefact, and it is the multiplier that has gone.
//
// The classic alternative would reduce nothing further and is recorded because
// it used to be the way out: propagate labels to immediate neighbours only,
// `fragments -> fragments` at block reach 1, and iterate until nothing changes.
// It is expressible here *except* for the stopping rule — the number of rounds
// depends on how crooked the components are, which is a property of the data, and
// a `Decomposition` is data-blind on purpose. It was the answer while the merge
// was per block, because it would have shrunk what each block had to see; with
// the merge hoisted there is one merge over the whole set and nothing for the
// rounds to save.
//
// Placement is still worth what it was worth: the reduction's input is a set of
// small objects somewhere, and `distributed::placement` runs a phase where the
// data is. In a distributed run the blob is not shipped at all — every worker
// derives it from the same fragment set, which is `barriers.md` §9.
//
// What is here and what is in `components`
// ----------------------------------------
// This was the only op with this shape until `ops::regional` arrived, which asks
// a different question with the same program: label locally, exchange six face
// planes, close the labels with a union-find, fold one boolean per label over
// each component. Everything in that sentence that is not the question now lives
// in `super::components` and is shared — the disjoint sets, the face geometry,
// the plane encoding, the flat `(block, label)` numbering and the seam walk.
//
// What stayed here is what makes this op *this* op: a component is a run of
// **background**, the per-label fact is "does this component reach the outside of
// the volume", and a seam meeting always unions because two background labels
// that touch are one background component. The fragment type, its magic, its
// public functions and its error messages are where they were.
//
// **Two more things joined it while this op was migrated onto the barrier.**
// `components::Merge` — where the global merge runs — and the reduction blob's
// codec, `components::encode_block_flags` and `decode_block_flags_for`. Both
// were written here, because this is the op that needed them first, and both
// were moved: every fragment-and-join op merges to exactly one flag per label
// per block, so the declaration and the encoding are the family's and not this
// op's. `agree_on_connectivity` is the one that did **not** move, and the
// asymmetry is deliberate — it is a refusal about a pair of *this* file's ops
// that two other files happen to reuse, not a shared representation.

use std::collections::BTreeMap;

use ndarray::{Array3, ArrayView3, ArrayViewMut3};

use crate::assemble::{Phase, PlanBuilder};
use crate::decomposition::Decomposition;
use crate::dtype::Dtype;
use crate::env::BlockBuf;
use crate::error::{Error, Result};
use crate::fragment::{
    fragment_phase, BlockOutput, BlockView, Coverage, FragmentInput, FragmentOp, FragmentOutput,
    PhaseView, SeamFold, SidecarSize,
};
use crate::geometry::BlockGrid;
use crate::sidecar::Lifecycle;
use crate::voxels::Voxels;

use super::components::{
    bytes_to_words, core_within_read, decode_block_flags_for, empty_planes, encode_block_flags,
    expect_end, face_axes, label_members_into, label_members_into_with, planes_of, push_planes,
    read_header, take_planes, walk_seams_with, words_to_bytes, Connectivity, FacePlanes,
    LabelIndex, Merge, Union, UNLABELLED,
};
use super::shapes_agree;
use super::voxelwise::is_set;

pub use super::components::face_index;

// ------------------------------------------------------------- labelling --

/// The label reserved for the foreground. Background components are numbered
/// from one, so a zero in the label volume means "this voxel was already set"
/// and needs no lookup at all.
///
/// The same number `components::UNLABELLED` is, and named separately because
/// what "no component" means is this op's: here it is a foreground voxel, which
/// is not part of the background being labelled at all.
pub const FOREGROUND: u32 = UNLABELLED;

/// Label the **background** of `mask` into `out`, six-connected, and return how
/// many components were found.
///
/// The traversal is `components::label_members_into` and the whole of what is
/// said here is the membership test: a voxel belongs to the background exactly
/// when the mask leaves it clear. Everything that makes the labelling
/// deterministic, iterative and six-connected is stated there, once, for the two
/// ops that share it.
///
/// Six-connected because that is [`Connectivity`]'s default;
/// [`label_background_into_with`] is the form that says which.
pub fn label_background_into(
    mask: ArrayView3<'_, bool>,
    out: ArrayViewMut3<'_, u32>,
) -> Result<u32> {
    shapes_agree(mask.shape(), out.shape(), "label_background_into")?;
    let shape = [mask.shape()[0], mask.shape()[1], mask.shape()[2]];
    label_members_into(shape, |at| !mask[at], out)
}

/// [`label_background_into`] under a stated [`Connectivity`], which is the
/// **background's** — see the module header for why there is only one here.
///
/// Everything that function promises holds: the membership test, the scan-order
/// numbering, the iterative traversal. What widens is which background voxels one
/// flood reaches, and a wider one leaves fewer, larger components.
pub fn label_background_into_with(
    mask: ArrayView3<'_, bool>,
    connectivity: Connectivity,
    out: ArrayViewMut3<'_, u32>,
) -> Result<u32> {
    shapes_agree(mask.shape(), out.shape(), "label_background_into_with")?;
    let shape = [mask.shape()[0], mask.shape()[1], mask.shape()[2]];
    label_members_into_with(shape, connectivity, |at| !mask[at], out)
}

/// Rewrite a label volume into the filled mask.
///
/// `outside[label - 1]` says whether the component reaches the volume's outside.
/// A foreground voxel stays set; a background voxel is set **unless** its
/// component drains outside, which is the whole of the operation.
pub fn fill_from_labels_into(
    labels: ArrayView3<'_, u32>,
    outside: &[bool],
    mut out: ArrayViewMut3<'_, bool>,
) -> Result<()> {
    shapes_agree(labels.shape(), out.shape(), "fill_from_labels_into")?;
    for (slot, &label) in out.iter_mut().zip(labels.iter()) {
        *slot = match label {
            FOREGROUND => true,
            _ => {
                let index = label as usize - 1;
                let escapes = *outside.get(index).ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "label {label} has no entry in the outside map, which holds {}. The \
                         map is built from the same fragment the labels were written with, so \
                         a gap means the two came from different runs.",
                        outside.len()
                    ))
                })?;
                !escapes
            }
        };
    }
    Ok(())
}

// -------------------------------------------------------------- fragment --

/// What one block tells the merge about itself.
///
/// Six planes of labels and one flag per label, and nothing else. Deliberately
/// not the block's whole label volume: what the merge needs is which labels meet
/// across a seam, and only the faces can.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockFaces {
    /// How many background components this block found.
    pub labels: u32,
    /// Per label, whether any of its voxels lies on an outer face of the
    /// **volume** — which is what makes a component drain rather than be a hole.
    pub touches_outside: Vec<bool>,
    /// The six faces, ordered `axis * 2 + side` with side 0 low and 1 high, each
    /// as `(shape, labels)` in row-major order over the two axes that are not
    /// this face's.
    pub faces: FacePlanes,
}

impl BlockFaces {
    /// Read a block's faces off its label volume.
    pub fn of(labels: ArrayView3<'_, u32>, count: u32, touches_outside: Vec<bool>) -> Result<Self> {
        if touches_outside.len() != count as usize {
            return Err(Error::InvalidArgument(format!(
                "{count} labels but {} outside flags",
                touches_outside.len()
            )));
        }
        Ok(Self {
            labels: count,
            touches_outside,
            faces: planes_of(labels),
        })
    }

    /// The empty report: a block with nothing to say, which is what an
    /// accounting run produces and is a different fact from no fragment at all.
    pub fn empty() -> Self {
        Self {
            labels: 0,
            touches_outside: Vec::new(),
            faces: empty_planes(),
        }
    }

    /// A self-describing byte form: little-endian `u32` throughout, with a magic
    /// and a version in front. See `components::read_header` for why the magic
    /// is not decoration.
    pub fn encode(&self) -> Vec<u8> {
        let mut words: Vec<u32> = vec![MAGIC, VERSION, self.labels];
        words.extend(
            self.touches_outside
                .iter()
                .map(|&flag| if flag { 1 } else { 0 }),
        );
        push_planes(&self.faces, &mut words);
        words_to_bytes(&words)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        const NOUN: &str = "a block-faces fragment";
        let words = bytes_to_words(bytes, NOUN)?;
        let labels = read_header(&words, MAGIC, VERSION, NOUN)?;
        let mut at = 3usize;
        let end = at + labels as usize;
        if words.len() < end {
            return Err(Error::InvalidArgument(format!(
                "{NOUN} ends inside its outside flags"
            )));
        }
        let touches_outside = words[at..end].iter().map(|&word| word != 0).collect();
        at = end;
        let faces = take_planes(&words, &mut at, NOUN)?;
        expect_end(&words, at, NOUN)?;
        Ok(Self {
            labels,
            touches_outside,
            faces,
        })
    }
}

/// `"FILL"` little-endian.
const MAGIC: u32 = 0x4c4c_4946;
const VERSION: u32 = 1;

/// `"FILR"` little-endian — the **reduction** blob, not the fragment.
///
/// A separate magic rather than a second version of `MAGIC`, for the reason
/// `components::read_header` gives about the fragment magic: neither a
/// fragment's address nor a reduction blob's says what its bytes mean, and these
/// two travel through the same op. A blob decoded as a fragment, or the reverse,
/// is refused on the first word.
const REDUCTION_MAGIC: u32 = 0x524c_4946;

// ------------------------------------------------------------- the merge --

/// Close every block's local labels into global components and report, per
/// block, which of its labels drain to the volume's outside.
///
/// `reports` is keyed by block index and must hold every block of `counts`; a
/// missing block is refused rather than assumed empty, because "absent" and
/// "present with nothing to say" are different facts and only one of them is a
/// block that ran.
///
/// The result is keyed the same way and each entry has one flag per label of
/// that block, in label order, so `outside[label - 1]` is the lookup
/// [`fill_from_labels_into`] wants.
pub fn merge_faces(
    reports: &BTreeMap<[usize; 3], BlockFaces>,
    counts: [usize; 3],
) -> Result<BTreeMap<[usize; 3], Vec<bool>>> {
    merge_faces_with(reports, counts, Connectivity::Faces)
}

/// [`merge_faces`] under a stated [`Connectivity`].
///
/// **It must be the one the labelling ran under.** The flood inside a block and
/// the walk across a seam generate one equivalence relation between them, so two
/// different choices would join within a block what they keep apart across a
/// seam — an answer that depends on where the volume was cut. [`fill_phases`] and
/// [`append_connected`] refuse a mismatched pair at planning time; a caller
/// driving the kernels by hand is the one place the pairing is not checked.
pub fn merge_faces_with(
    reports: &BTreeMap<[usize; 3], BlockFaces>,
    counts: [usize; 3],
    connectivity: Connectivity,
) -> Result<BTreeMap<[usize; 3], Vec<bool>>> {
    let index = LabelIndex::build(reports, counts, |report| report.labels)?;
    let escapes = index.gather(reports, |report| &report.touches_outside[..], false);
    let mut sets = Union::new(index.total());

    // **Two background labels that meet across a seam are one component,
    // unconditionally.** There is nothing to compare: the labelling ran on the
    // complement of the mask, so both sides being labelled at all is already the
    // whole of the condition.
    walk_seams_with(
        reports,
        counts,
        &index,
        connectivity,
        |report| &report.faces,
        |a, b| sets.union(a, b),
    )?;

    // A component drains if any of its members does.
    let root_escapes = sets.fold_or(&escapes);
    Ok(index.per_block(&mut sets, &root_escapes))
}

// ---------------------------------------------------------------- phases --

/// Phase 0: label each block's background and say what crosses its faces.
///
/// **Reach zero.** A block-local labelling reads nothing outside its own core;
/// everything that would need a neighbour is in the fragment instead. That is
/// the point of the split — the expensive, per-voxel half is fully parallel and
/// halo-free, and only the cheap, global half is a reduction.
pub struct LabelBackgroundOp {
    name: &'static str,
    stream: String,
    lifecycle: Lifecycle,
    /// Which background voxels count as adjacent. [`Connectivity::Faces`] unless
    /// a caller said otherwise, and that default is the whole of the
    /// compatibility story: every existing constructor leaves it alone, so every
    /// existing caller gets the labels it always got.
    connectivity: Connectivity,
}

impl LabelBackgroundOp {
    pub fn new(name: &'static str, stream: impl Into<String>, lifecycle: Lifecycle) -> Self {
        Self {
            name,
            stream: stream.into(),
            lifecycle,
            connectivity: Connectivity::Faces,
        }
    }

    /// The same op, labelling the background under a stated [`Connectivity`].
    ///
    /// A consuming builder rather than a fourth argument to [`Self::new`], for
    /// `detect::RegionPointsOp::emitting`'s reason: every call site that does not
    /// say this word keeps its signature and its answer, and a mechanical edit
    /// across every caller is exactly the change that gets one call site wrong.
    ///
    /// **The merge has to be told the same thing.** See
    /// [`FillHolesOp::connecting`], and [`fill_phases`], which refuses a pair
    /// that disagrees.
    pub fn connecting(mut self, connectivity: Connectivity) -> Self {
        self.connectivity = connectivity;
        self
    }

    /// Which background voxels this op counts as adjacent.
    pub fn connectivity(&self) -> Connectivity {
        self.connectivity
    }

    pub fn stream(&self) -> &str {
        &self.stream
    }
}

impl FragmentOp for LabelBackgroundOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// Nothing crosses as **pixels**. The background is labelled inside this
    /// block from this block's mask alone; the six faces and the label count a
    /// neighbour needs leave on `self.stream`, whose reach is declared in blocks
    /// by whoever reads it.
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn reads_pixels(&self) -> bool {
        true
    }

    fn writes_pixels(&self) -> bool {
        true
    }

    /// **`u32` labels, whatever width the mask arrived in.** A mask may be one
    /// bit or eight bytes; a component index is neither, and saying so here is
    /// what lets the plan allocate the label image at the width it is actually
    /// written at rather than at the mask's.
    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::U32
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            self.stream.clone(),
            self.lifecycle,
            // Every block, always. A block whose background is empty still has
            // six faces of zeros and a label count of nought, and the merge
            // needs to see that rather than infer it from an absence.
            Coverage::EveryBlock,
            // The six-faces shape: a header, a word per label, and the block's six
            //         // boundary planes.
        )
        .sized(SidecarSize::block_faces())]
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let BlockBuf::Array(pixels) = at.pixels()? else {
            // An accounting run has no data. It still writes a fragment and a
            // block, because what it is measuring is the IO, and a phase that
            // silently produced nothing would measure a different program.
            return Ok(
                BlockOutput::fragment(self.stream.clone(), BlockFaces::empty().encode())
                    .with_pixels(at.output_buffer(0.0)?),
            );
        };
        let mask = as_mask(pixels)?;
        let shape = [mask.shape()[0], mask.shape()[1], mask.shape()[2]];

        let mut buffer = at.output_buffer(0.0)?;
        let BlockBuf::Array(out) = &mut buffer else {
            unreachable!("the environment gave data for the input and none for the output");
        };
        let count = {
            let labels = out.view_mut::<u32>()?;
            label_background_into_with(mask.view(), self.connectivity, labels)?
        };

        let labels = out.view::<u32>()?;
        let touches_outside = outside_flags(labels, count, at.at.offset, at.at.volume, shape);
        let faces = BlockFaces::of(labels, count, touches_outside)?;
        let bytes = faces.encode();
        Ok(BlockOutput::fragment(self.stream.clone(), bytes).with_pixels(buffer))
    }
}

/// Which labels have a voxel on an outer face of the whole volume.
///
/// The test is on the **volume**, not on the block: a block's own faces are
/// seams almost everywhere, and treating a seam as the outside would drain
/// every hole that happened to be cut by one — which is precisely the
/// decomposition-dependent answer this op exists to avoid.
pub fn outside_flags(
    labels: ArrayView3<'_, u32>,
    count: u32,
    offset: [usize; 3],
    volume: [usize; 3],
    shape: [usize; 3],
) -> Vec<bool> {
    let mut flags = vec![false; count as usize];
    for axis in 0..3 {
        for side in 0..2 {
            let local = if side == 0 { 0 } else { shape[axis] - 1 };
            let global = offset[axis] + local;
            let is_volume_face = if side == 0 {
                global == 0
            } else {
                global + 1 == volume[axis]
            };
            if !is_volume_face {
                continue;
            }
            let [u, v] = face_axes(axis);
            for a in 0..shape[u] {
                for b in 0..shape[v] {
                    let mut position = [0usize; 3];
                    position[axis] = local;
                    position[u] = a;
                    position[v] = b;
                    let label = labels[position];
                    if label != FOREGROUND {
                        flags[label as usize - 1] = true;
                    }
                }
            }
        }
    }
    flags
}

/// Phase 1: close the components and write the filled mask.
///
/// **Declares a [barrier], and runs its merge once for the phase.** Its answer
/// depends on every block, and that is now stated as the dependency it is rather
/// than paid for as a whole-volume fetch: the phase waits for all of phase 0,
/// fetches only its own core, and computes the union-find in
/// [`FragmentOp::reduce`] at the moment the barrier creates. [`Merge`] is the
/// declaration and the module header measures what each of its three shapes
/// costs.
///
/// [barrier]: crate::fragment::FragmentOp::barrier
pub struct FillHolesOp {
    name: &'static str,
    stream: String,
    faces_phase: usize,
    filled: Dtype,
    lattice: [usize; 3],
    /// Which background voxels count as adjacent **across a seam**, which has to
    /// be what the labelling used within a block. [`Connectivity::Faces`] unless
    /// a caller said otherwise.
    connectivity: Connectivity,
    /// Where the union-find runs. [`Merge::OnceForThePhase`] unless a
    /// measurement asked for one of the other two.
    merge: Merge,
}

impl FillHolesOp {
    /// `faces_phase` is the phase whose blocks wrote the faces — part of the
    /// address rather than a default, for `FragmentInput`'s reason: a stream
    /// written by two phases holds two generations and "the fragments of stream
    /// s" is not a well-formed request.
    /// `filled` is the element type the answer is written in. Stated rather
    /// than inherited: this op reads a `u32` label image and hands back a mask,
    /// so "unchanged" is exactly the wrong default and there is no width it
    /// could infer.
    pub fn new(
        name: &'static str,
        stream: impl Into<String>,
        faces_phase: usize,
        filled: Dtype,
        grid: &BlockGrid,
    ) -> Self {
        Self {
            name,
            stream: stream.into(),
            faces_phase,
            filled,
            lattice: grid.blocks_per_axis(),
            connectivity: Connectivity::Faces,
            merge: Merge::default(),
        }
    }

    /// The same op with the merge somewhere else. See [`Merge`].
    ///
    /// **Nothing in a shipping plan should call this.** The default is the shape
    /// that costs least in all three currencies; the other two exist so that the
    /// difference can be measured out of one op, and so that the barrier has a
    /// liveness control — the same op with it withheld must answer identically.
    pub fn merging(mut self, merge: Merge) -> Self {
        self.merge = merge;
        self
    }

    /// Where this op's merge runs.
    pub fn merge(&self) -> Merge {
        self.merge
    }

    /// The same op, closing the components under a stated [`Connectivity`].
    ///
    /// **It must be the labelling's.** The two phases are one equivalence
    /// relation split in half — the flood inside a block and the walk across a
    /// seam — so a mismatched pair joins within a block what it keeps apart
    /// across a seam, and the answer becomes a function of where the volume was
    /// cut. [`fill_phases`] and [`append_connected`] check the pair; this builder
    /// on its own cannot, because it can only see one of the two ops.
    pub fn connecting(mut self, connectivity: Connectivity) -> Self {
        self.connectivity = connectivity;
        self
    }

    /// Which background voxels this op counts as adjacent.
    pub fn connectivity(&self) -> Connectivity {
        self.connectivity
    }

    /// The same op, addressed by a [`Phase`] handle instead of a number.
    ///
    /// See `crate::assemble::Phase`: a phase index written as a literal is not
    /// refused when it is wrong, it reads a different generation of the stream.
    pub fn reading(
        name: &'static str,
        stream: impl Into<String>,
        faces: Phase,
        filled: Dtype,
        grid: &BlockGrid,
    ) -> Self {
        Self::new(name, stream, faces.index(), filled, grid)
    }

    /// The lattice this op was built for, which is also the reach it declares.
    pub fn lattice(&self) -> [usize; 3] {
        self.lattice
    }
}

impl FragmentOp for FillHolesOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// Nothing this block is authoritative for reaches past its core, and since
    /// the barrier the read extent is that core: `fragment_phase` no longer
    /// widens the halo to cover a fragment reach the barrier states directly.
    /// The question that crosses the seam — *is this cavity connected to the
    /// outside* — is answered on the stream.
    ///
    /// Under [`Merge::PerBlock`] the whole-lattice fragment reach in
    /// [`Self::inputs`] is still a whole-volume halo, and `apply` slices back to
    /// the core before it writes. That arm exists to be measured.
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    /// **The answer depends on every block, and this is where that is said.**
    ///
    /// Before it could be said, the only spelling was a whole-lattice fragment
    /// reach, which `fragment_phase` turned into a whole-volume halo, a
    /// whole-volume fetch and an edge to every task below — three costs to
    /// express one ordering. The module header measures what dropping the first
    /// two is worth on this op.
    fn barrier(&self) -> bool {
        self.merge.is_barriered()
    }

    /// **The union-find is a function of the set of fragments, not of their
    /// order**, and this is the claim being checked rather than asserted.
    ///
    /// It is true here structurally rather than by luck: the merge is driven off
    /// a `BTreeMap` keyed by block index, so the fragments are put back into the
    /// lattice's own order whatever order they arrived in; the seam walk unions
    /// pairs, and a union-find's partition does not depend on the order the
    /// unions arrive in; and the per-label fact is a boolean folded with `or`,
    /// which associates exactly. Nothing here accumulates in `f64`.
    ///
    /// What the declaration buys is that a future edit which broke any of those
    /// three is caught: the executor reduces a second time with the lattice
    /// walked backwards and requires the same bytes. What it costs is one extra
    /// reduction per phase — against the one extra `apply` **per block** the same
    /// declaration costs under [`Merge::PerBlock`], which is the multiplier
    /// hoisting removes from this check as well as from everything else.
    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::Unordered)
    }

    fn reads_pixels(&self) -> bool {
        true
    }

    fn writes_pixels(&self) -> bool {
        true
    }

    /// The mask width the caller asked for; see [`FillHolesOp::new`].
    fn produces(&self, _input: Dtype) -> Dtype {
        self.filled
    }

    /// **Nothing per block once the merge is hoisted**, and the whole lattice
    /// when it is not.
    ///
    /// The stream is declared either way, because that is what makes it
    /// resolvable — a `reduce` may read the streams the plan records and no
    /// others, for the same reason a block may. What the reach says is how much
    /// of the set *this block* needs, and with the merge in
    /// [`FragmentOp::reduce`] the answer is none of it: the phase needs the set,
    /// the block needs the phase's answer. That is where the `(1 + blocks) x F`
    /// multiplier goes.
    ///
    /// Under the two per-block shapes the reach is the lattice, stated as the
    /// lattice rather than as a large number — which is why the constructor
    /// takes a grid. "Everything" is a different integer on every lattice, and a
    /// saturating sentinel is not a way out: the reach is multiplied by the block
    /// edge to get a halo, and a sentinel overflows the geometry rather than
    /// clamping. `FragmentInput::whole` takes a grid for the same reason.
    fn inputs(&self) -> Vec<FragmentInput> {
        let reach = if self.merge.is_hoisted() {
            [0, 0, 0]
        } else {
            self.lattice
        };
        vec![FragmentInput::own(self.stream.clone(), self.faces_phase).with_reach(reach)]
    }

    /// Nothing to gather once the merge is hoisted: `apply` reads the phase's
    /// answer, not the fragments it was derived from. Gathering a reach of
    /// nothing would still fetch this block's own fragment, which is one whole
    /// transmission of the set across the phase for bytes no block reads.
    fn gathers(&self) -> bool {
        !self.merge.is_hoisted()
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        Vec::new()
    }

    /// **The merge, once for the phase.** Every block then reads its own row out
    /// of the blob instead of re-deriving the whole table.
    ///
    /// Empty under the two per-block shapes, which is what makes them plannable
    /// at all: `check_phase_work` probes `reduce` on every phase without a
    /// barrier and refuses one that answers, and [`Merge::PerBlock`] has no
    /// barrier.
    fn reduce(&self, at: &PhaseView<'_>) -> Result<Vec<u8>> {
        if !self.merge.is_hoisted() {
            return Ok(Vec::new());
        }
        let mut reports = BTreeMap::new();
        at.stream_fragments(&self.stream, &mut |key, bytes| {
            reports.insert(key.block, BlockFaces::decode(bytes)?);
            Ok(())
        })?;
        let counts = at.grid.blocks_per_axis();
        let outside = merge_faces_with(&reports, counts, self.connectivity)?;
        encode_block_flags(&outside, counts, REDUCTION_MAGIC)
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let BlockBuf::Array(pixels) = at.pixels()? else {
            return Ok(BlockOutput::nothing().with_pixels(at.output_buffer(0.0)?));
        };
        let labels = pixels.view::<u32>()?;

        let counts = at.grid.blocks_per_axis();
        let mine = if self.merge.is_hoisted() {
            decode_block_flags_for(
                at.reduced,
                counts,
                at.index,
                REDUCTION_MAGIC,
                "a hole-filling reduction",
            )?
        } else {
            let mut reports = BTreeMap::new();
            for (key, bytes) in at.fragments(&self.stream) {
                reports.insert(key.block, BlockFaces::decode(bytes)?);
            }
            let outside = merge_faces_with(&reports, counts, self.connectivity)?;
            outside.get(&at.index).cloned().ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "the merge produced no answer for block {:?}, which is the block asking",
                    at.index
                ))
            })?
        };
        let mine = &mine[..];

        // **Only the core, and this is not an optimisation.** Under
        // [`Merge::PerBlock`] the read extent is the whole volume — the
        // whole-lattice fragment reach is also the halo — so `labels` holds every
        // block's labels, numbered per block, while `mine` answers for this
        // block's numbering and no other. With a barrier the read extent *is* the
        // core and this slice is the identity; it stays because the op is one op
        // and the geometry is read rather than assumed. See
        // `components::core_within_read`, which states the whole argument.
        let (offset, extent) = core_within_read(at)?;
        let window = ndarray::s![
            offset[0]..offset[0] + extent[0],
            offset[1]..offset[1] + extent[1],
            offset[2]..offset[2] + extent[2],
        ];
        let core_labels = labels.slice(window);

        let mut buffer = at.output_buffer(0.0)?;
        let BlockBuf::Array(out) = &mut buffer else {
            unreachable!("the environment gave data for the input and none for the output");
        };
        match out.dtype() {
            Dtype::Bool => {
                let mut view = out.view_mut::<bool>()?;
                fill_from_labels_into(core_labels, mine, view.slice_mut(window))?;
            }
            _ => {
                let mut flags = Array3::from_elem((extent[0], extent[1], extent[2]), false);
                fill_from_labels_into(core_labels, mine, flags.view_mut())?;
                let mut view = out.view_mut::<f64>()?;
                ndarray::Zip::from(&mut view.slice_mut(window))
                    .and(&flags)
                    .for_each(|slot, &flag| *slot = if flag { 1.0 } else { 0.0 });
            }
        }
        Ok(BlockOutput::nothing().with_pixels(buffer))
    }
}

/// The two phases, on one lattice.
///
/// Both are built with `fragment_phase`, so both halos come from the ops'
/// declarations rather than from this function, and both are now zero: the
/// labelling needs nothing outside its core, and the filling states its
/// dependency on the whole of the labelling as a barrier rather than buying it
/// with a halo. The preamble explains what that halo used to cost and what
/// removing it was worth.
///
/// `mask_dtype` is the element type of the image the mask arrives in; the width
/// of the answer comes from `fill`, which is where a caller states it.
///
/// **The two connectivities are checked here**, which is the only place both ops
/// are in one hand. See [`agree_on_connectivity`].
pub fn fill_phases(
    grid: BlockGrid,
    mask_dtype: Dtype,
    label: &LabelBackgroundOp,
    fill: &FillHolesOp,
) -> Result<Decomposition> {
    agree_on_connectivity(label.connectivity(), fill.connectivity())?;
    let volume = grid.volume();
    let mut labelling = fragment_phase(label, grid.clone())?;
    labelling.dtype = Some(label.produces(mask_dtype));
    let mut filling = fragment_phase(fill, grid)?;
    filling.dtype = Some(fill.produces(Dtype::U32));
    let plan = Decomposition {
        volume,
        dtype: mask_dtype,
        phases: vec![labelling, filling],
        chain_reach: [0, 0, 0],
    };
    plan.check()?;
    Ok(plan)
}

/// The same two phases, **appended to a plan that already has some**.
///
/// [`fill_phases`] builds a whole `Decomposition`, so it cannot be part of one;
/// this is its body against a builder, with the two `dtype` lines gone because
/// [`PlanBuilder::fragments`] asks the ops what they write.
///
/// `filled` is the element type the answer is written in — stated rather than
/// inherited, for [`FillHolesOp::new`]'s reason. Returns the filling's phase,
/// which is the one that writes the mask.
pub fn append_to(
    plan: &mut PlanBuilder,
    stream: impl Into<String>,
    lifecycle: Lifecycle,
    filled: Dtype,
) -> Result<Phase> {
    append_connected(plan, stream, lifecycle, filled, Connectivity::Faces)
}

/// [`append_to`], with the background's [`Connectivity`] said out loud.
///
/// The general one; `append_to` is this at [`Connectivity::Faces`], which is what
/// the two phases have always been. Kept as two functions rather than one with a
/// fifth argument so that no existing call site has to be edited to say what it
/// was already doing — `detect::append_emitting`'s reason.
///
/// **The choice goes to both phases from here**, which is what makes this the
/// safe way to ask for a wider one: a caller building the two ops by hand has to
/// remember to say it twice, and [`fill_phases`] is where that is caught.
pub fn append_connected(
    plan: &mut PlanBuilder,
    stream: impl Into<String>,
    lifecycle: Lifecycle,
    filled: Dtype,
    connectivity: Connectivity,
) -> Result<Phase> {
    let stream = stream.into();
    let grid = plan.grid().clone();
    let faces = plan.fragments(
        LabelBackgroundOp::new("background labelling", stream.clone(), lifecycle)
            .connecting(connectivity),
    )?;
    plan.fragments(
        FillHolesOp::reading("hole filling", stream, faces, filled, &grid).connecting(connectivity),
    )
}

/// The two phases' connectivities, refused unless they are one.
///
/// A block-local flood and a seam walk generate **one** equivalence relation
/// between them. Two different choices make the relation depend on where the
/// block boundary fell — voxels joined inside a block and kept apart across a
/// seam — which is the one defect this crate exists to prevent, and it produces
/// a plausible-looking answer rather than an error. So it is refused at planning
/// time, before anything is scheduled.
///
/// `pub` because `ops::regional` and `ops::detect` have the same two halves and
/// the same hazard, and one message is better than three.
pub fn agree_on_connectivity(labelling: Connectivity, merge: Connectivity) -> Result<()> {
    if labelling == merge {
        return Ok(());
    }
    Err(Error::InvalidArgument(format!(
        "the labelling phase is {labelling:?}-connected and the merge phase is \
         {merge:?}-connected. The flood inside a block and the walk across a seam are two \
         halves of one adjacency relation, so a pair that disagrees joins voxels within a \
         block that it keeps apart across a seam — which makes the answer depend on where the \
         volume was cut. Give both phases the same connectivity."
    )))
}

/// A block's pixels as a mask, whatever width they arrived in.
///
/// `pub` because every op here that takes a mask needs the same bridge — this
/// one and `ops::detect` — and two spellings of "what counts as set" is one too
/// many. The rule is `voxelwise::is_set`'s and is stated there.
pub fn as_mask(pixels: &Voxels) -> Result<Array3<bool>> {
    match pixels.dtype() {
        Dtype::Bool => Ok(pixels.view::<bool>()?.to_owned()),
        _ => Ok(pixels.widened().mapv(is_set)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole-volume oracle: label the background of the entire mask at once
    /// and clear every component that meets an outer face. The same kernel the
    /// blocked path uses, called once — so a disagreement is a decomposition
    /// bug and not a modelling difference.
    pub(crate) fn fill_whole(mask: &Array3<bool>) -> Array3<bool> {
        let shape = mask.raw_dim();
        let mut labels = Array3::<u32>::zeros(shape);
        let count = label_background_into(mask.view(), labels.view_mut()).unwrap();
        let volume = [mask.shape()[0], mask.shape()[1], mask.shape()[2]];
        let flags = outside_flags(labels.view(), count, [0, 0, 0], volume, volume);
        let mut out = Array3::from_elem(shape, false);
        fill_from_labels_into(labels.view(), &flags, out.view_mut()).unwrap();
        out
    }

    /// A shell with a cavity in it. The cavity must fill; the shell must stay;
    /// the outside must stay clear.
    fn shell(extent: usize) -> Array3<bool> {
        let mut mask = Array3::from_elem((extent, extent, extent), false);
        let (low, high) = (2, extent - 3);
        for i in low..=high {
            for j in low..=high {
                for k in low..=high {
                    let on_shell =
                        i == low || i == high || j == low || j == high || k == low || k == high;
                    mask[[i, j, k]] = on_shell;
                }
            }
        }
        mask
    }

    #[test]
    fn a_cavity_fills_and_the_outside_does_not() {
        let mask = shell(11);
        let filled = fill_whole(&mask);
        // the interior, which was clear and enclosed
        assert!(!mask[[5, 5, 5]]);
        assert!(filled[[5, 5, 5]], "an enclosed cavity must fill");
        // the shell itself
        assert!(filled[[2, 5, 5]]);
        // outside the shell
        assert!(!filled[[0, 0, 0]], "the outside must not fill");
        assert!(!filled[[1, 5, 5]]);
    }

    /// The one that separates hole filling from "set everything": a cavity with
    /// a channel to the outside is *not* a hole, however narrow the channel.
    #[test]
    fn a_cavity_with_a_one_voxel_drain_does_not_fill() {
        let mut mask = shell(11);
        let filled_before = fill_whole(&mask);
        assert!(filled_before[[5, 5, 5]]);

        // punch a single voxel out of the shell
        mask[[2, 5, 5]] = false;
        let filled = fill_whole(&mask);
        assert!(
            !filled[[5, 5, 5]],
            "a component that drains through one voxel is not a hole"
        );
        // and the drain is face-connected all the way, so nothing inside fills
        assert!(!filled[[3, 5, 5]]);
    }

    /// Six-connectivity, asserted rather than assumed: a diagonal-only gap does
    /// not drain. This is the decision the preamble states, and a test that did
    /// not distinguish it would pass under either connectivity.
    #[test]
    fn a_diagonal_only_gap_does_not_drain_because_the_background_is_six_connected() {
        let mut mask = Array3::from_elem((7, 7, 7), false);
        // a 3x3x3 box of shell around a single interior voxel
        for i in 2..=4 {
            for j in 2..=4 {
                for k in 2..=4 {
                    mask[[i, j, k]] = !(i == 3 && j == 3 && k == 3);
                }
            }
        }
        assert!(fill_whole(&mask)[[3, 3, 3]]);

        // remove a corner of the shell: the cavity now touches the outside
        // diagonally, and only diagonally
        mask[[2, 2, 2]] = false;
        assert!(
            fill_whole(&mask)[[3, 3, 3]],
            "under six-connectivity a corner opening is not a drain"
        );

        // remove a face voxel instead, and it drains
        let mut leaky = mask.clone();
        leaky[[2, 3, 3]] = false;
        assert!(!fill_whole(&leaky)[[3, 3, 3]]);
    }

    #[test]
    fn a_mask_with_no_holes_is_returned_unchanged() {
        let mut mask = Array3::from_elem((9, 8, 7), false);
        for i in 1..4 {
            for j in 1..4 {
                for k in 1..4 {
                    mask[[i, j, k]] = true;
                }
            }
        }
        assert_eq!(fill_whole(&mask), mask);
        // and it is idempotent
        assert_eq!(fill_whole(&fill_whole(&mask)), mask);
    }

    #[test]
    fn an_all_set_mask_and_an_all_clear_mask_are_both_fixed_points() {
        let set = Array3::from_elem((5, 5, 5), true);
        assert_eq!(fill_whole(&set), set);
        let clear = Array3::from_elem((5, 5, 5), false);
        assert_eq!(
            fill_whole(&clear),
            clear,
            "an all-clear volume is one component that touches every face"
        );
    }

    /// Labels are numbered in the order their lowest voxel is met, which is what
    /// makes the intermediate image reproducible between runs of one
    /// decomposition.
    #[test]
    fn labels_are_numbered_in_scan_order_and_are_reproducible() {
        let mut mask = Array3::from_elem((5, 5, 5), true);
        mask[[1, 1, 1]] = false;
        mask[[3, 3, 3]] = false;
        let mut labels = Array3::<u32>::zeros((5, 5, 5));
        let count = label_background_into(mask.view(), labels.view_mut()).unwrap();
        assert_eq!(count, 2);
        assert_eq!(labels[[1, 1, 1]], 1);
        assert_eq!(labels[[3, 3, 3]], 2);
        assert_eq!(labels[[0, 0, 0]], FOREGROUND);

        let mut again = Array3::<u32>::zeros((5, 5, 5));
        label_background_into(mask.view(), again.view_mut()).unwrap();
        assert_eq!(labels, again);
    }

    /// A component that spans the whole block, which is what an implementation
    /// written as a depth-first recursion overflows the stack on.
    #[test]
    fn a_component_spanning_the_whole_block_is_labelled_without_recursing() {
        let mask = Array3::from_elem((64, 64, 64), false);
        let mut labels = Array3::<u32>::zeros((64, 64, 64));
        let count = label_background_into(mask.view(), labels.view_mut()).unwrap();
        assert_eq!(count, 1);
        assert!(labels.iter().all(|&label| label == 1));
    }

    #[test]
    fn a_faces_fragment_survives_a_round_trip_and_a_wrong_one_is_refused() {
        let mask = shell(9);
        let mut labels = Array3::<u32>::zeros(mask.raw_dim());
        let count = label_background_into(mask.view(), labels.view_mut()).unwrap();
        let flags = outside_flags(labels.view(), count, [0, 0, 0], [9, 9, 9], [9, 9, 9]);
        let faces = BlockFaces::of(labels.view(), count, flags).unwrap();

        let bytes = faces.encode();
        assert_eq!(BlockFaces::decode(&bytes).unwrap(), faces);

        // a stream written by something else
        let mut foreign = bytes.clone();
        foreign[0] ^= 0xff;
        assert!(BlockFaces::decode(&foreign).is_err());
        // a truncated one
        assert!(BlockFaces::decode(&bytes[..bytes.len() - 4]).is_err());
        // one with something appended
        let mut extra = bytes;
        extra.extend_from_slice(&[0, 0, 0, 0]);
        assert!(BlockFaces::decode(&extra).is_err());
        // and something that is not words at all
        assert!(BlockFaces::decode(&[1, 2, 3]).is_err());
    }

    /// The merge refuses a lattice it was not given every block of. "Absent" and
    /// "present and empty" are different facts, and only one of them is a block
    /// that ran.
    #[test]
    fn the_merge_refuses_a_lattice_with_a_block_missing() {
        let mut reports = BTreeMap::new();
        reports.insert([0, 0, 0], BlockFaces::empty());
        assert!(merge_faces(&reports, [2, 1, 1]).is_err());
        reports.insert([1, 0, 0], BlockFaces::empty());
        assert!(merge_faces(&reports, [2, 1, 1]).is_ok());
    }

    /// The parameter reaches the labelling, asked in the way that can tell.
    ///
    /// A cavity that opens on a corner only: face-connected it is a hole and
    /// fills, corner-connected it drains and does not. A fixture whose opening
    /// were a face would answer the same under every connectivity and would pass
    /// with the parameter dropped on the floor.
    #[test]
    fn the_background_connectivity_reaches_the_answer_and_the_bare_form_is_faces() {
        let mut mask = Array3::from_elem((7, 7, 7), false);
        for i in 2..=4 {
            for j in 2..=4 {
                for k in 2..=4 {
                    mask[[i, j, k]] = !(i == 3 && j == 3 && k == 3);
                }
            }
        }
        // remove one corner of the shell: the cavity now touches the outside
        // diagonally, and only diagonally
        mask[[2, 2, 2]] = false;

        let count = |connectivity| {
            let mut labels = Array3::<u32>::zeros(mask.raw_dim());
            let found =
                label_background_into_with(mask.view(), connectivity, labels.view_mut()).unwrap();
            let flags = outside_flags(labels.view(), found, [0, 0, 0], [7, 7, 7], [7, 7, 7]);
            let mut out = Array3::from_elem(mask.raw_dim(), false);
            fill_from_labels_into(labels.view(), &flags, out.view_mut()).unwrap();
            out[[3, 3, 3]]
        };
        assert!(
            count(Connectivity::Faces),
            "a corner opening is not a drain under six"
        );
        assert!(
            count(Connectivity::FacesAndEdges),
            "nor under eighteen: the opening is a corner step"
        );
        assert!(
            !count(Connectivity::FacesEdgesAndCorners),
            "under twenty-six it drains, so the cavity is not a hole"
        );

        // and the bare form is the face-connected one, byte for byte
        let mut bare = Array3::<u32>::zeros(mask.raw_dim());
        let mut stated = Array3::<u32>::zeros(mask.raw_dim());
        assert_eq!(
            label_background_into(mask.view(), bare.view_mut()).unwrap(),
            label_background_into_with(mask.view(), Connectivity::Faces, stated.view_mut())
                .unwrap()
        );
        assert_eq!(bare, stated);
    }

    /// The two phases are two halves of one relation, and a plan whose halves
    /// disagree is refused **before** it is scheduled rather than answering
    /// something that depends on where the volume was cut.
    #[test]
    fn a_plan_whose_two_phases_disagree_about_connectivity_is_refused() {
        let grid = BlockGrid::new([8, 8, 8], [4, 4, 4]).unwrap();
        let label = LabelBackgroundOp::new("label", "s", Lifecycle::DeleteOnExit)
            .connecting(Connectivity::FacesEdgesAndCorners);
        let merge = FillHolesOp::new("fill", "s", 0, Dtype::Bool, &grid);
        let message = fill_phases(grid.clone(), Dtype::Bool, &label, &merge)
            .unwrap_err()
            .to_string();
        assert!(message.contains("same connectivity"), "{message}");

        // and the matched pair plans, at both ends of the range
        for connectivity in [Connectivity::Faces, Connectivity::FacesEdgesAndCorners] {
            let label = LabelBackgroundOp::new("label", "s", Lifecycle::DeleteOnExit)
                .connecting(connectivity);
            let merge =
                FillHolesOp::new("fill", "s", 0, Dtype::Bool, &grid).connecting(connectivity);
            assert_eq!(label.connectivity(), connectivity);
            assert_eq!(merge.connectivity(), connectivity);
            assert!(fill_phases(grid.clone(), Dtype::Bool, &label, &merge).is_ok());
        }
    }

    /// Two blocks whose faces are different shapes came from two different
    /// lattices, and the merge says so rather than silently zipping the shorter.
    #[test]
    fn the_merge_refuses_faces_from_two_different_lattices() {
        let mut small = BlockFaces::empty();
        small.faces[face_index(0, 1)] = ([2, 2], vec![0; 4]);
        let mut large = BlockFaces::empty();
        large.faces[face_index(0, 0)] = ([3, 3], vec![0; 9]);
        let mut reports = BTreeMap::new();
        reports.insert([0, 0, 0], small);
        reports.insert([1, 0, 0], large);
        let message = merge_faces(&reports, [2, 1, 1]).unwrap_err().to_string();
        assert!(message.contains("two different lattices"), "{message}");
    }
}
