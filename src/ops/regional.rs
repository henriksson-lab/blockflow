// SPDX-License-Identifier: MIT
//
// Original work for this crate. Written from the definition of the operation,
// not adapted from any implementation of it.
//
// The **regional maxima** of a greyscale volume: the connected plateaus from
// which one cannot ascend.
//
// A *regional maximum* is a connected component of voxels of equal value, every
// voxel adjacent to which has a **strictly lower** value. The output is a mask,
// set where a voxel belongs to one.
//
// Why this is not a `BlockOp`, and what it is instead
// ---------------------------------------------------
// The same wall `ops::fill` hit, for the same reason and with a different
// payload. A plateau is a connected set of equal values, so it can be
// arbitrarily long and arbitrarily crooked: a flat ridge one voxel thick can run
// the length of the volume, and whether it is a maximum depends on every voxel
// touching it — including the one at the far end, which may be any distance
// away. No halo answers that, and the reason is not that the halo would be large
// but that there is no halo at all that works. A block that decided on its own
// would call each cut piece of the ridge a maximum, and would be wrong on every
// piece except the one holding the higher voxel.
//
// So this is a **fragment-and-join**, in two phases, and it is `fill`'s two
// phases with a different fact per label:
//
// | phase | shape | what it does |
// |---|---|---|
// | 0 | `volume -> fragments`, and writes pixels | label the block's plateaus locally; write the labels as a `u32` level; emit the block's six face planes, one value per label and one flag per label |
// | 1 | `fragments -> volume` | read every block's faces, close the local labels into global components, OR the flag over each component, and rewrite this block's labels into the mask |
//
// The per-label fact is "**does any voxel adjacent to this component have a
// strictly greater value**", and it merges by OR across the component exactly as
// `fill`'s outside flag does. The component labelling is over voxels of *equal
// value* rather than over the background, which is the only real difference in
// phase 0 — and one addition to the fragment, which is the per-label **value**,
// because a seam meeting cannot be resolved without it. See "The seam" below.
//
// Read `ops::fill`'s header for the two walls this shape ran into. Both apply
// here unchanged and neither had to be re-solved: a fragment-only phase is
// terminal as far as levels go, so the merge is folded into phase 1 and every
// block re-runs it; and the whole-lattice fragment reach of phase 1 is also its
// halo, which is load-bearing because the halo is the dependency edge between
// pipelined phases. The costs are `fill`'s costs, stated there.
//
// The two adjacencies, which are one relation and must stay one
// --------------------------------------------------------------
// The definition names two adjacency relations, and the first thing to say about
// them is that they are the same one:
//
// * **the plateau connectivity** — which equal-valued voxels count as one
//   component;
// * **the ascent adjacency** — which neighbours are consulted for "is anything
//   here strictly greater".
//
// **They must agree, so this op takes one [`Connectivity`] and not two.** The
// definition is a statement about one neighbourhood graph `G`: a regional maximum
// is a connected component of `G` restricted to equal values, out of which no
// edge of `G` goes upward. Using two different graphs breaks it in a way that is
// easy to make concrete. Suppose the plateau were 26-connected and the ascent
// 6-connected. Take `v(0,0,0) = 5`, `v(1,1,1) = 5` — one plateau, joined by a
// corner step — and `v(2,2,2) = 6`, which is a corner step from `(1,1,1)` and
// 6-adjacent to nothing in the plateau. The plateau would be reported as a
// maximum, and yet one can walk `(0,0,0) -> (1,1,1) -> (2,2,2)` using exactly the
// steps the plateau connectivity itself permits. "A plateau from which one cannot
// ascend" would be false of it. The mirrored mistake — plateau narrower than
// ascent — is less dramatic but no better: a plateau would be disqualified by a
// voxel it cannot itself step to.
//
// So the parameter is one parameter. Offering the pair separately would be
// offering a way to state something the definition does not admit, and the
// argument above is the reason the API does not have that shape. It is the one
// place in this crate where a caller might reasonably expect two knobs and gets
// one on purpose.
//
// Which one it is, is the caller's, and [`Connectivity::Faces`] unless a caller
// says. What widening costs is *maxima*: a higher voxel that touches a plateau
// diagonally disqualifies it under twenty-six and does not under six, so six
// reports **more** maxima, never fewer. Neither is a correction of the other.
//
// **The fragment did not have to change**, which is the thing that looks least
// likely and is worth saying. A voxel with any neighbour outside its block lies
// on a *face* of that block, so the six planes already are the block's whole
// boundary shell, and the edge lines and corner voxels a wider connectivity meets
// across are rows and single entries of them. `components`'s header argues it;
// what widened is the seam walk's inputs and nothing about the bytes.
//
// **Both phases carry the choice and [`regional_phases`] refuses a pair that
// disagrees**, for `ops::fill`'s reason: the flood inside a block and the walk
// across a seam are two halves of one relation, and a mismatched pair makes the
// answer depend on where the volume was cut.
//
// Equality of values, and what to do about NaN
// ---------------------------------------------
// Plateaus are components of **equal** value, and the test is exact equality
// with no tolerance. A tolerance is not merely imprecise here, it is
// ill-defined: `|a - b| < e` is not transitive, so which voxels end up in one
// component would depend on the order the flood fill visited them, and the
// answer would stop being a function of the volume. The whole point of this
// crate is that it is one.
//
// Exact means `==` on the element type — **not** equality of bit patterns. The
// two differ for `-0.0` and `+0.0`, and `==` is the right one, because equality
// and the strict order have to be the same relation's two halves. IEEE 754 gives
// trichotomy on non-NaN values: exactly one of `<`, `==`, `>` holds. Comparing
// bit patterns would split `-0.0` from `+0.0` into two "different" plateaus that
// are nevertheless neither above nor below one another, so neither would
// disqualify the other and both would be reported as maxima — two maxima where
// the order relation says there is one plateau.
//
// **NaN is unordered with everything, itself included, and this op takes that
// literally.** A NaN voxel satisfies neither `==` nor `>` with any value, so:
//
// * it joins no plateau — not even another NaN, whatever the payload bits — and
//   is left [`UNLABELLED`], so it is **never** in the output mask;
// * it disqualifies nothing, because `neighbour > mine` is false when either
//   side is NaN.
//
// Both facts fall out of the two relations the operation is defined in, which is
// why this is the rule rather than a special case bolted on. The alternative
// considered was to let a NaN neighbour disqualify a plateau, on the reading
// that NaN is missing data and a maximum next to missing data is a claim one
// cannot support. It was not taken: it needs an explicit NaN test at every
// comparison — precisely the kind of hand-written special case that ends up
// applied in one of the two places it is needed — and it makes the answer depend
// on data that has no order, which is the thing this operation is about. A
// caller who wants the conservative reading gets it by replacing NaN with
// `+inf` before the run, which is one voxelwise map and says exactly what it
// means.
//
// The element types this accepts, and the two it refuses
// ------------------------------------------------------
// The kernels are generic over `T: Copy + PartialOrd`, which is the whole of
// what the algorithm needs. The **shell** is narrower, and `super`'s rules say
// to state which: it accepts every element type except `u64` and `i64`, and
// [`accepts`] is where that is written down.
//
// The reason is the fragment. A seam meeting compares two plateaus' values, so
// the value has to travel, and the fragment carries it as an `f64`. Widening a
// `u64` to `f64` is lossy above `2^53` — two distinct integers can round to one
// double — and the damage is not a rounding error but a **decomposition
// dependency**: inside a block the two values would be compared exactly and be
// two plateaus, across a seam they would compare equal and be one. The same
// volume would give two answers depending on where the block boundary fell,
// which is the one defect this crate exists to prevent. Refusing is the honest
// answer until the fragment carries the element's own bits and the merge
// compares in the element's own type; that is a real extension and it is not
// pretended here.
//
// Every other type widens to `f64` exactly: `bool`, `u8`, `u16`, `u32`, `i8`,
// `i16`, `i32` all fit in 53 bits, and `f32 -> f64` preserves equality, order
// and NaN. So the whole pipeline runs in `f64` after the adapter, and the
// answer is the answer for the original values.

use std::collections::BTreeMap;

use ndarray::{Array3, ArrayView3, ArrayViewMut3};

use crate::assemble::{Phase, PlanBuilder};
use crate::decomposition::Decomposition;
use crate::dtype::Dtype;
use crate::env::BlockBuf;
use crate::error::{Error, Result};
use crate::fragment::{
    fragment_phase, BlockOutput, BlockView, Coverage, FragmentInput, FragmentOp, FragmentOutput,
};
use crate::geometry::BlockGrid;
use crate::sidecar::Lifecycle;
use crate::voxels::Voxels;

use super::components::{
    bytes_to_words, core_within_read, empty_planes, expect_end, offset_by, planes_of, push_planes,
    read_header, take_planes, walk_seams_with, words_to_bytes, Connectivity, FacePlanes,
    LabelIndex, Union, UNLABELLED,
};
use super::fill::agree_on_connectivity;
use super::shapes_agree;

// ------------------------------------------------------------- labelling --

/// Is `value` unordered with everything, itself included?
///
/// The one NaN test in this file, and it is a `partial_cmp` against itself
/// rather than `f64::is_nan` for two reasons. It stays available to the generic
/// kernels — `T: PartialOrd` is exactly the bound under which a value can fail
/// to compare with itself, and for every integer type it is a comparison the
/// optimiser deletes — and it says what the operation actually cares about,
/// which is not "this is a NaN" but "this value stands in neither of the two
/// relations the definition is written in".
fn unordered<T: PartialOrd>(value: &T) -> bool {
    value.partial_cmp(value).is_none()
}

/// Label the **plateaus** of `values` into `out`, six-connected, and return how
/// many were found together with the value each one holds.
///
/// Six-connected because that is [`Connectivity`]'s default;
/// [`label_plateaux_into_with`] is the form that says which — and whichever it
/// is, [`ascending_neighbours`] has to be asked the same thing.
///
/// A plateau is a face-connected component of voxels of equal value. Every voxel
/// belongs to exactly one, *except* a voxel that is unordered with itself — a
/// NaN — which belongs to none and is left [`UNLABELLED`]; see the module header
/// for why that is the definition rather than an exception to it.
///
/// Deterministic, and deterministic in a way that matters: plateaus are numbered
/// in the order their lowest voxel is met in row-major order, so the same block
/// always produces the same labels. Two runs of the same decomposition are then
/// byte-identical in the intermediate level as well as in the output, which is
/// what makes the intermediate worth looking at when something is wrong.
///
/// Iterative rather than recursive. A plateau can span the whole block — a
/// constant volume is one — and a depth-first recursion over a 256-cube is a
/// stack overflow rather than a slow answer.
pub fn label_plateaux_into<T: Copy + PartialOrd>(
    values: ArrayView3<'_, T>,
    out: ArrayViewMut3<'_, u32>,
) -> Result<(u32, Vec<T>)> {
    shapes_agree(values.shape(), out.shape(), "label_plateaux_into")?;
    label_plateaux(values, Connectivity::Faces, out)
}

/// [`label_plateaux_into`] under a stated [`Connectivity`], which is **also the
/// ascent's** — see the module header for why the two cannot differ.
///
/// Everything that function promises holds: the exact equality, the NaN rule, the
/// scan-order numbering, the iterative traversal. What widens is which
/// equal-valued voxels one flood reaches.
///
/// [`ascending_neighbours_with`] has to be given the same choice, and so does the
/// merge; the ops in this file carry one field between them for exactly that
/// reason.
pub fn label_plateaux_into_with<T: Copy + PartialOrd>(
    values: ArrayView3<'_, T>,
    connectivity: Connectivity,
    out: ArrayViewMut3<'_, u32>,
) -> Result<(u32, Vec<T>)> {
    shapes_agree(values.shape(), out.shape(), "label_plateaux_into_with")?;
    label_plateaux(values, connectivity, out)
}

/// The one traversal both forms above are, with the shape already checked.
fn label_plateaux<T: Copy + PartialOrd>(
    values: ArrayView3<'_, T>,
    connectivity: Connectivity,
    mut out: ArrayViewMut3<'_, u32>,
) -> Result<(u32, Vec<T>)> {
    let shape = [values.shape()[0], values.shape()[1], values.shape()[2]];
    out.fill(UNLABELLED);

    let mut next = UNLABELLED;
    let mut held: Vec<T> = Vec::new();
    let mut stack: Vec<[usize; 3]> = Vec::new();
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let seed = [i, j, k];
                if out[seed] != UNLABELLED || unordered(&values[seed]) {
                    continue;
                }
                let level = values[seed];
                next += 1;
                held.push(level);
                out[seed] = next;
                stack.push(seed);
                while let Some(at) = stack.pop() {
                    for &by in connectivity.offsets() {
                        let Some(to) = offset_by(at, by, shape) else {
                            continue;
                        };
                        // Exact equality against the seed's value. Not a
                        // tolerance, and not a comparison against the voxel we
                        // stepped from: both would let a component drift, and
                        // where it drifted to would depend on the traversal.
                        // A NaN neighbour fails this and is left unlabelled,
                        // which is the same rule the seed test applies.
                        if out[to] != UNLABELLED || values[to] != level {
                            continue;
                        }
                        out[to] = next;
                        stack.push(to);
                    }
                }
            }
        }
    }
    Ok((next, held))
}

/// Per label, whether any voxel adjacent to that plateau — **within this
/// array** — holds a strictly greater value.
///
/// This is the whole of the operation's local half, and it is deliberately
/// block-local: a neighbour across a seam is not visible here and is not meant
/// to be. The merge sees it, through the face planes and the per-label values,
/// and that is the only place a seam is resolved. A block that peeked past its
/// own edge would be reading a halo, and the module header is about why there
/// isn't one.
///
/// Unlabelled voxels are neither sources nor obstacles: a NaN has no plateau to
/// speak for, and `neighbour > mine` is false when either side is NaN, so it
/// disqualifies nothing.
pub fn ascending_neighbours<T: Copy + PartialOrd>(
    values: ArrayView3<'_, T>,
    labels: ArrayView3<'_, u32>,
    count: u32,
) -> Result<Vec<bool>> {
    shapes_agree(values.shape(), labels.shape(), "ascending_neighbours")?;
    ascending(values, labels, count, Connectivity::Faces)
}

/// [`ascending_neighbours`] under a stated [`Connectivity`], which must be the
/// one the plateaus were labelled under. The module header is where that is
/// argued, and it is the reason this takes the parameter at all rather than
/// staying face-connected while the labelling widened.
pub fn ascending_neighbours_with<T: Copy + PartialOrd>(
    values: ArrayView3<'_, T>,
    labels: ArrayView3<'_, u32>,
    count: u32,
    connectivity: Connectivity,
) -> Result<Vec<bool>> {
    shapes_agree(values.shape(), labels.shape(), "ascending_neighbours_with")?;
    ascending(values, labels, count, connectivity)
}

/// The one pass both forms above are, with the shape already checked.
fn ascending<T: Copy + PartialOrd>(
    values: ArrayView3<'_, T>,
    labels: ArrayView3<'_, u32>,
    count: u32,
    connectivity: Connectivity,
) -> Result<Vec<bool>> {
    let shape = [values.shape()[0], values.shape()[1], values.shape()[2]];
    let mut flags = vec![false; count as usize];
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let at = [i, j, k];
                let label = labels[at];
                if label == UNLABELLED {
                    continue;
                }
                let slot = *flags.get(label as usize - 1).ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "label {label} is outside the {count} label(s) this array was said to \
                         hold, so the labels and the count came from two different runs."
                    ))
                })?;
                if slot {
                    continue;
                }
                let mine = values[at];
                for &by in connectivity.offsets() {
                    let Some(to) = offset_by(at, by, shape) else {
                        continue;
                    };
                    if values[to] > mine {
                        flags[label as usize - 1] = true;
                        break;
                    }
                }
            }
        }
    }
    Ok(flags)
}

/// Rewrite a label volume into the regional-maxima mask.
///
/// `ascends[label - 1]` says whether anything adjacent to that component stands
/// strictly higher. A labelled voxel is set **unless** its component has such a
/// neighbour, which is the whole of the operation; an [`UNLABELLED`] voxel is
/// clear, because it is in no component and therefore in no maximum.
pub fn maxima_from_labels_into(
    labels: ArrayView3<'_, u32>,
    ascends: &[bool],
    mut out: ArrayViewMut3<'_, bool>,
) -> Result<()> {
    shapes_agree(labels.shape(), out.shape(), "maxima_from_labels_into")?;
    for (slot, &label) in out.iter_mut().zip(labels.iter()) {
        *slot = match label {
            UNLABELLED => false,
            _ => {
                let index = label as usize - 1;
                let higher = *ascends.get(index).ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "label {label} has no entry in the ascent map, which holds {}. The map \
                         is built from the same fragment the labels were written with, so a \
                         gap means the two came from different runs.",
                        ascends.len()
                    ))
                })?;
                !higher
            }
        };
    }
    Ok(())
}

// -------------------------------------------------------------- fragment --

/// What one block tells the merge about itself.
///
/// Six planes of labels, one value per label and one flag per label, and nothing
/// else. Deliberately not the block's whole label volume: what the merge needs is
/// which plateaus meet across a seam and how they compare, and only the faces can
/// meet.
///
/// The value is the field `fill`'s fragment has no counterpart for, and it is
/// what a seam meeting cannot be resolved without — see [`merge_plateaux`].
#[derive(Debug, Clone, PartialEq)]
pub struct PlateauFaces {
    /// How many plateaus this block found.
    pub labels: u32,
    /// Per label, the value every voxel of that plateau holds. Carried as `f64`;
    /// see the module header for the two element types that costs.
    pub values: Vec<f64>,
    /// Per label, whether any voxel adjacent to that plateau *inside this block*
    /// holds a strictly greater value. The merge ORs the seam's verdict into
    /// this and then over the whole component.
    pub ascends: Vec<bool>,
    /// The six faces, ordered `axis * 2 + side` with side 0 low and 1 high, each
    /// as `(shape, labels)` in row-major order over the two axes that are not
    /// this face's.
    pub faces: FacePlanes,
}

impl PlateauFaces {
    /// Read a block's faces off its label volume.
    pub fn of(
        labels: ArrayView3<'_, u32>,
        count: u32,
        values: Vec<f64>,
        ascends: Vec<bool>,
    ) -> Result<Self> {
        if values.len() != count as usize || ascends.len() != count as usize {
            return Err(Error::InvalidArgument(format!(
                "{count} labels but {} value(s) and {} ascent flag(s)",
                values.len(),
                ascends.len()
            )));
        }
        Ok(Self {
            labels: count,
            values,
            ascends,
            faces: planes_of(labels),
        })
    }

    /// The empty report: a block with nothing to say, which is what an
    /// accounting run produces and is a different fact from no fragment at all.
    pub fn empty() -> Self {
        Self {
            labels: 0,
            values: Vec::new(),
            ascends: Vec::new(),
            faces: empty_planes(),
        }
    }

    /// A self-describing byte form: little-endian `u32` throughout, with a magic
    /// and a version in front, and each value as its two `f64` bit halves low
    /// word first.
    ///
    /// The bits rather than a decimal rendering, because the merge compares
    /// these values for exact equality and a round trip through text is a
    /// different number. `f64::to_bits` is exact both ways for every value
    /// including NaN, which is what "self-describing" has to mean here.
    pub fn encode(&self) -> Vec<u8> {
        let mut words: Vec<u32> = vec![MAGIC, VERSION, self.labels];
        words.extend(self.ascends.iter().map(|&flag| if flag { 1 } else { 0 }));
        for value in &self.values {
            let bits = value.to_bits();
            words.push(bits as u32);
            words.push((bits >> 32) as u32);
        }
        push_planes(&self.faces, &mut words);
        words_to_bytes(&words)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        const NOUN: &str = "a plateau-faces fragment";
        let words = bytes_to_words(bytes, NOUN)?;
        let labels = read_header(&words, MAGIC, VERSION, NOUN)?;
        let count = labels as usize;

        let mut at = 3usize;
        if words.len() < at + count {
            return Err(Error::InvalidArgument(format!(
                "{NOUN} ends inside its ascent flags"
            )));
        }
        let ascends = words[at..at + count]
            .iter()
            .map(|&word| word != 0)
            .collect();
        at += count;

        if words.len() < at + 2 * count {
            return Err(Error::InvalidArgument(format!(
                "{NOUN} ends inside its plateau values"
            )));
        }
        let mut values = Vec::with_capacity(count);
        for pair in words[at..at + 2 * count].chunks_exact(2) {
            values.push(f64::from_bits((pair[1] as u64) << 32 | pair[0] as u64));
        }
        at += 2 * count;

        let faces = take_planes(&words, &mut at, NOUN)?;
        expect_end(&words, at, NOUN)?;
        Ok(Self {
            labels,
            values,
            ascends,
            faces,
        })
    }
}

/// `"RMAX"` little-endian. Distinct from `fill`'s, which is the point: the two
/// fragments differ only in their per-label payload, so a stream name used by
/// both would otherwise decode one as the other.
const MAGIC: u32 = 0x5841_4d52;
const VERSION: u32 = 1;

// ------------------------------------------------------------- the merge --

/// Close every block's local plateaus into global ones and report, per block,
/// which of its labels have something strictly higher adjacent to them
/// **anywhere**.
///
/// `reports` is keyed by block index and must hold every block of `counts`; a
/// missing block is refused rather than assumed empty, because "absent" and
/// "present with nothing to say" are different facts and only one of them is a
/// block that ran.
///
/// The result is keyed the same way and each entry has one flag per label of
/// that block, in label order, so `ascends[label - 1]` is the lookup
/// [`maxima_from_labels_into`] wants.
///
/// **What a seam meeting does, which is where this differs from `fill`.** Two
/// labels that touch across a seam are not automatically one component — they
/// are one only if their values are equal. So each meeting is a three-way
/// comparison, and every branch of it is load-bearing:
///
/// * equal — the two labels are the same plateau, and are unioned;
/// * one strictly greater — the lower one has just been shown to have a higher
///   neighbour, so its flag is set. This is the case a block cannot see for
///   itself and the only reason the values travel in the fragment;
/// * neither — one side is NaN, and the module header says why nothing happens.
pub fn merge_plateaux(
    reports: &BTreeMap<[usize; 3], PlateauFaces>,
    counts: [usize; 3],
) -> Result<BTreeMap<[usize; 3], Vec<bool>>> {
    merge_plateaux_with(reports, counts, Connectivity::Faces)
}

/// [`merge_plateaux`] under a stated [`Connectivity`], which must be the one the
/// plateaus were labelled and the ascents taken under.
///
/// All three are the same relation, and the seam is where the third one becomes
/// visible: a widened walk both joins more plateaus *and* finds more ascents,
/// because a meeting whose values differ is an ascent for the lower side. Both
/// come from the same pair set, so there is nothing here that could widen by
/// halves.
pub fn merge_plateaux_with(
    reports: &BTreeMap<[usize; 3], PlateauFaces>,
    counts: [usize; 3],
    connectivity: Connectivity,
) -> Result<BTreeMap<[usize; 3], Vec<bool>>> {
    let index = LabelIndex::build(reports, counts, |report| report.labels)?;
    let values = index.gather(reports, |report| &report.values[..], f64::NAN);
    let mut ascends = index.gather(reports, |report| &report.ascends[..], false);
    let mut sets = Union::new(index.total());

    walk_seams_with(
        reports,
        counts,
        &index,
        connectivity,
        |report| &report.faces,
        |a, b| {
            let (here, there) = (values[a], values[b]);
            if here == there {
                sets.union(a, b);
            } else if here > there {
                ascends[b] = true;
            } else if there > here {
                ascends[a] = true;
            }
        },
    )?;

    // A component has something higher next to it if any of its members does.
    let root_ascends = sets.fold_or(&ascends);
    Ok(index.per_block(&mut sets, &root_ascends))
}

// ---------------------------------------------------------------- phases --

/// Whether this op's shell can bridge `dtype`.
///
/// Everything but the 64-bit integers, and the module header is where the reason
/// is written down: the fragment carries a plateau's value as an `f64`, `u64`
/// and `i64` do not fit in one exactly, and the loss would show up as a
/// *decomposition dependency* rather than as a rounding error.
pub fn accepts(dtype: Dtype) -> bool {
    !matches!(dtype, Dtype::U64 | Dtype::I64)
}

fn refuse(dtype: Dtype) -> Error {
    Error::InvalidArgument(format!(
        "regional maxima cannot be taken over {} here. A plateau's value travels between \
         blocks as an f64, which cannot hold every {} exactly, and two values that differ \
         inside a block but round together across a seam would make the answer depend on \
         where the seam fell. Narrow the volume first, or take the maxima with the kernels \
         directly, which are generic over the element type.",
        dtype.numpy_name(),
        dtype.numpy_name()
    ))
}

/// Phase 0: label each block's plateaus and say what crosses its faces.
///
/// **Reach zero.** A block-local labelling reads nothing outside its own core;
/// everything that would need a neighbour is in the fragment instead. That is
/// the point of the split — the expensive, per-voxel half is fully parallel and
/// halo-free, and only the cheap, global half is a reduction.
pub struct LabelPlateauxOp {
    name: &'static str,
    stream: String,
    lifecycle: Lifecycle,
    /// The one relation this op is defined over: which equal-valued voxels are
    /// one plateau, and which neighbours are consulted for an ascent. One field
    /// because it is one relation — the module header is where that is argued.
    /// [`Connectivity::Faces`] unless a caller said otherwise, so every existing
    /// caller gets the answer it always got.
    connectivity: Connectivity,
}

impl LabelPlateauxOp {
    pub fn new(name: &'static str, stream: impl Into<String>, lifecycle: Lifecycle) -> Self {
        Self {
            name,
            stream: stream.into(),
            lifecycle,
            connectivity: Connectivity::Faces,
        }
    }

    /// The same op, over a stated [`Connectivity`] — **both** the plateau's and
    /// the ascent's, which are one relation.
    ///
    /// A consuming builder rather than a fourth argument to [`Self::new`], for
    /// `detect::RegionPointsOp::emitting`'s reason: every call site that does not
    /// say this word keeps its signature and its answer.
    ///
    /// The merge has to be told the same thing; [`regional_phases`] refuses a
    /// pair that disagrees.
    pub fn connecting(mut self, connectivity: Connectivity) -> Self {
        self.connectivity = connectivity;
        self
    }

    /// The relation this op labels and measures ascents over.
    pub fn connectivity(&self) -> Connectivity {
        self.connectivity
    }

    pub fn stream(&self) -> &str {
        &self.stream
    }
}

impl FragmentOp for LabelPlateauxOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reads_pixels(&self) -> bool {
        true
    }

    fn writes_pixels(&self) -> bool {
        true
    }

    /// **`u32` labels, whatever width the volume arrived in.** A component index
    /// is not a sample, and saying so here is what lets the plan allocate the
    /// label level at the width it is actually written at rather than at the
    /// input's.
    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::U32
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            self.stream.clone(),
            self.lifecycle,
            // Every block, always. A block that found no plateau at all — every
            // voxel NaN — still has six faces of zeros and a label count of
            // nought, and the merge needs to see that rather than infer it from
            // an absence.
            Coverage::EveryBlock,
        )]
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let BlockBuf::Array(pixels) = at.pixels()? else {
            // An accounting run has no data. It still writes a fragment and a
            // block, because what it is measuring is the IO, and a phase that
            // silently produced nothing would measure a different program.
            return Ok(
                BlockOutput::fragment(self.stream.clone(), PlateauFaces::empty().encode())
                    .with_pixels(at.output_buffer(0.0)?),
            );
        };
        // One widening copy, and `Voxels::widened` is blunt that it is not free.
        // It is paid because the *fragment* is stated in `f64`, so comparing in
        // anything else inside the block would be comparing in two relations —
        // exactly the seam-dependent answer this op refuses `u64` to avoid.
        // Through `as_values` rather than `widened` directly, so that the shell's
        // idea of what it accepts is written down once.
        let values = as_values(pixels)?;

        let mut buffer = at.output_buffer(0.0)?;
        let BlockBuf::Array(out) = &mut buffer else {
            unreachable!("the environment gave data for the input and none for the output");
        };
        let (count, plateau_values) = {
            let labels = out.view_mut::<u32>()?;
            label_plateaux_into_with(values.view(), self.connectivity, labels)?
        };

        let labels = out.view::<u32>()?;
        let ascends = ascending_neighbours_with(values.view(), labels, count, self.connectivity)?;
        let faces = PlateauFaces::of(labels, count, plateau_values, ascends)?;
        let bytes = faces.encode();
        Ok(BlockOutput::fragment(self.stream.clone(), bytes).with_pixels(buffer))
    }
}

/// Phase 1: close the plateaus and write the mask.
///
/// Declares a **whole-lattice** fragment reach, which is what makes it a
/// planning barrier: nothing can be fused across it, because its answer depends
/// on every block.
pub struct RegionalMaximaOp {
    name: &'static str,
    stream: String,
    faces_phase: usize,
    mask: Dtype,
    lattice: [usize; 3],
    /// The relation the seam is closed under, which has to be the one the
    /// labelling used. [`Connectivity::Faces`] unless a caller said otherwise.
    connectivity: Connectivity,
}

impl RegionalMaximaOp {
    /// `faces_phase` is the phase whose blocks wrote the faces — part of the
    /// address rather than a default, for `FragmentInput`'s reason: a stream
    /// written by two phases holds two generations and "the fragments of stream
    /// s" is not a well-formed request.
    ///
    /// `mask` is the element type the answer is written in. Stated rather than
    /// inherited: this op reads a `u32` label level and hands back a mask, so
    /// "unchanged" is exactly the wrong default and there is no width it could
    /// infer. `Dtype::Bool` is the natural one — the output is a mask and
    /// nothing else — and a wider type is offered only so that a mask can be
    /// carried down a chain whose level is already `f64`.
    pub fn new(
        name: &'static str,
        stream: impl Into<String>,
        faces_phase: usize,
        mask: Dtype,
        grid: &BlockGrid,
    ) -> Self {
        Self {
            name,
            stream: stream.into(),
            faces_phase,
            mask,
            lattice: grid.blocks_per_axis(),
            connectivity: Connectivity::Faces,
        }
    }

    /// The same op, closing the seams under a stated [`Connectivity`].
    ///
    /// **It must be the labelling's**, for `ops::fill::agree_on_connectivity`'s
    /// reason: the flood inside a block and the walk across a seam are two halves
    /// of one relation. [`regional_phases`] and [`append_connected`] check the
    /// pair; this builder alone cannot, because it sees one of the two ops.
    pub fn connecting(mut self, connectivity: Connectivity) -> Self {
        self.connectivity = connectivity;
        self
    }

    /// The relation this op closes the seams under.
    pub fn connectivity(&self) -> Connectivity {
        self.connectivity
    }

    /// The same op, addressed by a [`Phase`] handle instead of a number.
    ///
    /// **This is the constructor that cannot be got wrong.** `new` takes the
    /// producing phase's index, which is a `usize` a caller can write as a
    /// literal — and a literal that is off by one is not refused, it reads a
    /// different generation of the same stream and answers differently. A
    /// [`Phase`] can only come from the [`PlanBuilder`](crate::PlanBuilder) call
    /// that created the phase, so there is no number to write. `new` stays for
    /// hand-assembled plans, and `check_phase_work` refuses a phase that does
    /// not write the stream by name.
    pub fn reading(
        name: &'static str,
        stream: impl Into<String>,
        faces: Phase,
        mask: Dtype,
        grid: &BlockGrid,
    ) -> Self {
        Self::new(name, stream, faces.index(), mask, grid)
    }

    /// The lattice this op was built for, which is also the reach it declares.
    pub fn lattice(&self) -> [usize; 3] {
        self.lattice
    }
}

impl FragmentOp for RegionalMaximaOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reads_pixels(&self) -> bool {
        true
    }

    fn writes_pixels(&self) -> bool {
        true
    }

    /// The mask width the caller asked for; see [`RegionalMaximaOp::new`].
    fn produces(&self, _input: Dtype) -> Dtype {
        self.mask
    }

    /// The whole lattice, stated as the lattice rather than as a large number.
    ///
    /// This is why the constructor takes a grid. "Everything" is a different
    /// integer on every lattice, and a saturating sentinel is not a way out: the
    /// reach is multiplied by the block edge to get a halo, and a sentinel
    /// overflows the geometry rather than clamping.
    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own(self.stream.clone(), self.faces_phase).with_reach(self.lattice)]
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        Vec::new()
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let BlockBuf::Array(pixels) = at.pixels()? else {
            return Ok(BlockOutput::nothing().with_pixels(at.output_buffer(0.0)?));
        };
        let labels = pixels.view::<u32>()?;

        let mut reports = BTreeMap::new();
        for (key, bytes) in at.fragments(&self.stream) {
            reports.insert(key.block, PlateauFaces::decode(bytes)?);
        }
        let ascends = merge_plateaux_with(&reports, at.grid.blocks_per_axis(), self.connectivity)?;
        let mine = ascends.get(&at.index).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "the merge produced no answer for block {:?}, which is the block asking",
                at.index
            ))
        })?;

        // Only the core: the read extent is the whole volume, so `labels` holds
        // every block's labels under every block's own numbering, and `mine`
        // decodes one of them. See `components::core_within_read`.
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
                maxima_from_labels_into(core_labels, mine, view.slice_mut(window))?;
            }
            _ => {
                let mut flags = Array3::from_elem((extent[0], extent[1], extent[2]), false);
                maxima_from_labels_into(core_labels, mine, flags.view_mut())?;
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
/// declarations rather than from this function: zero for the labelling, the
/// whole lattice for the maxima. `ops::fill`'s header explains why the second
/// one has to be that and what it costs — the halo is the dependency edge
/// between pipelined phases, not merely a fetch extent.
///
/// `input_dtype` is the element type of the level the volume arrives in, and is
/// checked here rather than at the first block: a plan that cannot run is worth
/// refusing before anything is scheduled.
pub fn regional_phases(
    grid: BlockGrid,
    input_dtype: Dtype,
    label: &LabelPlateauxOp,
    maxima: &RegionalMaximaOp,
) -> Result<Decomposition> {
    if !accepts(input_dtype) {
        return Err(refuse(input_dtype));
    }
    agree_on_connectivity(label.connectivity(), maxima.connectivity())?;
    let volume = grid.volume();
    let mut labelling = fragment_phase(label, grid.clone())?;
    labelling.dtype = Some(label.produces(input_dtype));
    let mut finding = fragment_phase(maxima, grid)?;
    finding.dtype = Some(maxima.produces(Dtype::U32));
    let plan = Decomposition {
        volume,
        dtype: input_dtype,
        phases: vec![labelling, finding],
        chain_reach: [0, 0, 0],
    };
    plan.check()?;
    Ok(plan)
}

/// The same two phases, **appended to a plan that already has some**.
///
/// [`regional_phases`] builds a whole `Decomposition` and is therefore unusable
/// the moment these two phases are part of something larger: its body has to be
/// re-derived by hand, including the `labelling.dtype` line, whose omission is
/// refused by a message about a level's width rather than about the missing
/// line. This is that body, against a builder instead of a fresh plan — and the
/// element types are not restated here at all, because [`PlanBuilder::fragments`]
/// asks the ops.
///
/// `stream` is where phase one's faces go and where phase two reads them; it is
/// one name because it is one stream, and the phase half of the address is wired
/// here rather than being a number the caller writes. `mask` is the element type
/// the answer is written in — stated rather than inherited, for
/// [`RegionalMaximaOp::new`]'s reason.
///
/// Returns the merge's phase, which is the one that writes the mask.
pub fn append_to(
    plan: &mut PlanBuilder,
    stream: impl Into<String>,
    lifecycle: Lifecycle,
    mask: Dtype,
) -> Result<Phase> {
    append_connected(plan, stream, lifecycle, mask, Connectivity::Faces)
}

/// [`append_to`], with the [`Connectivity`] said out loud.
///
/// The general one; `append_to` is this at [`Connectivity::Faces`], which is what
/// the two phases have always been. Kept as two functions rather than one with a
/// fifth argument so that no existing call site has to be edited to say what it
/// was already doing.
///
/// **One argument, because it is one relation** — the plateau's and the ascent's
/// are the same graph and the module header says why they cannot differ. The
/// choice goes to both phases from here, which is what makes this the safe way to
/// ask for a wider one.
pub fn append_connected(
    plan: &mut PlanBuilder,
    stream: impl Into<String>,
    lifecycle: Lifecycle,
    mask: Dtype,
    connectivity: Connectivity,
) -> Result<Phase> {
    let input_dtype = plan.reads();
    if !accepts(input_dtype) {
        return Err(refuse(input_dtype));
    }
    let stream = stream.into();
    let grid = plan.grid().clone();
    let plateaux = plan.fragments(
        LabelPlateauxOp::new("plateau labelling", stream.clone(), lifecycle)
            .connecting(connectivity),
    )?;
    plan.fragments(
        RegionalMaximaOp::reading("regional maxima", stream, plateaux, mask, &grid)
            .connecting(connectivity),
    )
}

/// The whole-volume answer: the same kernels, called once, over everything.
///
/// **Not a second implementation.** It is what the blocked path is measured
/// against, so if it were written a second way a disagreement would be a
/// modelling difference rather than a decomposition bug. It is `pub` because the
/// acceptance suite in `tests/` is a separate crate and needs exactly this.
pub fn regional_maxima(values: ArrayView3<'_, f64>) -> Result<Array3<bool>> {
    regional_maxima_with(values, Connectivity::Faces)
}

/// [`regional_maxima`] under a stated [`Connectivity`].
///
/// The same three kernels at the same choice, which is what makes this a
/// reference for a blocked run at that choice rather than a second
/// implementation of one.
pub fn regional_maxima_with(
    values: ArrayView3<'_, f64>,
    connectivity: Connectivity,
) -> Result<Array3<bool>> {
    let mut labels = Array3::<u32>::zeros(values.raw_dim());
    let (count, _) = label_plateaux_into_with(values, connectivity, labels.view_mut())?;
    let ascends = ascending_neighbours_with(values, labels.view(), count, connectivity)?;
    let mut out = Array3::from_elem(values.raw_dim(), false);
    maxima_from_labels_into(labels.view(), &ascends, out.view_mut())?;
    Ok(out)
}

/// A block's pixels as `f64`, or the refusal that names why not.
///
/// Offered because a caller composing the kernels by hand needs the same bridge
/// the op's shell uses, and two spellings of "which types this accepts" is one
/// too many.
pub fn as_values(pixels: &Voxels) -> Result<Array3<f64>> {
    if !accepts(pixels.dtype()) {
        return Err(refuse(pixels.dtype()));
    }
    Ok(pixels.widened())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn maxima(values: &Array3<f64>) -> Array3<bool> {
        regional_maxima(values.view()).unwrap()
    }

    /// The degenerate case, and the one an off-by-one in the adjacency test gets
    /// wrong: nothing is greater than anything, so the whole volume is one
    /// plateau and the whole volume is a maximum.
    #[test]
    fn a_constant_volume_is_one_regional_maximum_covering_everything() {
        let flat = Array3::from_elem((5, 4, 3), 7.0);
        let mut labels = Array3::<u32>::zeros(flat.raw_dim());
        let (count, values) = label_plateaux_into(flat.view(), labels.view_mut()).unwrap();
        assert_eq!(count, 1, "a constant volume is one plateau");
        assert_eq!(values, vec![7.0]);
        assert!(labels.iter().all(|&label| label == 1));
        assert!(
            maxima(&flat).iter().all(|&set| set),
            "every voxel of a constant volume is in the one regional maximum"
        );
    }

    /// The four shapes the definition distinguishes, in one array so that they
    /// are asserted against each other rather than one at a time.
    #[test]
    fn a_spike_is_a_maximum_a_plateau_is_a_maximum_and_a_plateau_under_a_spike_is_not() {
        let mut values = Array3::from_elem((9, 9, 3), 1.0);
        // a single voxel, higher than everything around it
        values[[1, 1, 1]] = 5.0;
        // a flat 3 x 3 plateau with nothing higher anywhere near it
        for j in 3..=5 {
            for k in 0..=2 {
                values[[4, j, k]] = 4.0;
            }
        }
        // a flat plateau of the same value with exactly one higher neighbour
        for j in 3..=5 {
            for k in 0..=2 {
                values[[7, j, k]] = 4.0;
            }
        }
        values[[8, 4, 1]] = 6.0;

        let found = maxima(&values);
        assert!(found[[1, 1, 1]], "a single higher voxel is a maximum");
        for j in 3..=5 {
            for k in 0..=2 {
                assert!(
                    found[[4, j, k]],
                    "a plateau with nothing higher is a maximum"
                );
                assert!(
                    !found[[7, j, k]],
                    "one higher neighbour disqualifies the whole plateau, at {j},{k}"
                );
            }
        }
        assert!(
            found[[8, 4, 1]],
            "and the voxel that disqualified it is one"
        );
        assert!(
            !found[[0, 0, 0]],
            "the flat surround has higher voxels on it"
        );
    }

    /// A monotone ramp: every voxel can be ascended from except the top slice,
    /// which is where the ramp stops.
    #[test]
    fn a_monotone_ramp_has_no_maximum_except_its_top() {
        let values = Array3::from_shape_fn((6, 3, 3), |(i, _, _)| i as f64);
        let found = maxima(&values);
        for i in 0..6 {
            for j in 0..3 {
                for k in 0..3 {
                    assert_eq!(
                        found[[i, j, k]],
                        i == 5,
                        "voxel {i},{j},{k} of a ramp along the first axis"
                    );
                }
            }
        }
        // and the plateaus of a ramp are its slices, one per step
        let mut labels = Array3::<u32>::zeros(values.raw_dim());
        let (count, held) = label_plateaux_into(values.view(), labels.view_mut()).unwrap();
        assert_eq!(count, 6);
        assert_eq!(held, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    /// The two connectivities are both face connectivity, asserted rather than
    /// assumed. A test that did not distinguish them would pass under either.
    #[test]
    fn both_the_plateau_and_the_ascent_are_six_connected() {
        // two equal voxels that touch only at a corner are two plateaus
        let mut values = Array3::from_elem((4, 4, 4), 0.0);
        values[[1, 1, 1]] = 3.0;
        values[[2, 2, 2]] = 3.0;
        let mut labels = Array3::<u32>::zeros(values.raw_dim());
        let (count, _) = label_plateaux_into(values.view(), labels.view_mut()).unwrap();
        assert_eq!(count, 3, "the surround and two corner-touching plateaus");
        assert_ne!(labels[[1, 1, 1]], labels[[2, 2, 2]]);
        let found = maxima(&values);
        assert!(found[[1, 1, 1]] && found[[2, 2, 2]]);

        // a higher voxel that touches a plateau only at a corner does not
        // disqualify it, which is the ascent half of the same decision
        let mut corner = Array3::from_elem((4, 4, 4), 0.0);
        corner[[1, 1, 1]] = 3.0;
        corner[[2, 2, 2]] = 9.0;
        assert!(
            maxima(&corner)[[1, 1, 1]],
            "under six-connectivity a corner neighbour is not an ascent"
        );
        // and a face neighbour does
        let mut face = corner.clone();
        face[[2, 2, 2]] = 0.0;
        face[[2, 1, 1]] = 9.0;
        assert!(!maxima(&face)[[1, 1, 1]]);
    }

    /// **The header's own counter-example, run.** The one fixture that separates
    /// "one connectivity" from "two that happen to be equal".
    ///
    /// `v(0,0,0) = 5` and `v(1,1,1) = 5` are one plateau under twenty-six and two
    /// under six. `v(2,2,2) = 6` is a corner step from `(1,1,1)` and 6-adjacent
    /// to nothing. So:
    ///
    /// * at six — two plateaus of 5, neither with anything greater beside it, so
    ///   both are maxima;
    /// * at twenty-six — one plateau, and the 6 is adjacent to it, so it is not.
    ///
    /// And the mistake the header forbids is asserted to *be* a mistake: labelling
    /// the plateau at twenty-six while taking the ascent at six reports the
    /// plateau as a maximum, even though one can walk out of it upward using
    /// exactly the steps that built it. That is the answer the single parameter
    /// makes unreachable.
    #[test]
    fn the_plateau_and_the_ascent_widen_together_and_a_split_pair_would_be_wrong() {
        let mut values = Array3::from_elem((4, 4, 4), 0.0);
        values[[0, 0, 0]] = 5.0;
        values[[1, 1, 1]] = 5.0;
        values[[2, 2, 2]] = 6.0;

        let six = regional_maxima_with(values.view(), Connectivity::Faces).unwrap();
        assert!(
            six[[0, 0, 0]] && six[[1, 1, 1]],
            "two plateaus, both maxima"
        );

        let full = regional_maxima_with(values.view(), Connectivity::FacesEdgesAndCorners).unwrap();
        assert!(
            !full[[0, 0, 0]] && !full[[1, 1, 1]],
            "one plateau, with a greater voxel a corner step away"
        );
        assert!(full[[2, 2, 2]], "and the voxel that disqualified it is one");

        // the forbidden pairing, driven by hand: plateau at 26, ascent at 6
        let mut labels = Array3::<u32>::zeros(values.raw_dim());
        let (count, _) = label_plateaux_into_with(
            values.view(),
            Connectivity::FacesEdgesAndCorners,
            labels.view_mut(),
        )
        .unwrap();
        assert_eq!(labels[[0, 0, 0]], labels[[1, 1, 1]], "one plateau");
        let split =
            ascending_neighbours_with(values.view(), labels.view(), count, Connectivity::Faces)
                .unwrap();
        let mut wrong = Array3::from_elem(values.raw_dim(), false);
        maxima_from_labels_into(labels.view(), &split, wrong.view_mut()).unwrap();
        assert!(
            wrong[[0, 0, 0]],
            "the split pair calls it a maximum, which is the answer the definition \
             forbids and the reason this op takes one connectivity rather than two"
        );
        assert_ne!(wrong, full, "so the split pair is a different answer");
    }

    /// Exact equality, with no tolerance: two values a hair apart are two
    /// plateaus, and the lower one is not a maximum.
    #[test]
    fn plateaux_are_components_of_exactly_equal_value() {
        let mut values = Array3::from_elem((4, 1, 1), 0.0);
        values[[1, 0, 0]] = 1.0;
        values[[2, 0, 0]] = 1.0 + f64::EPSILON;
        let mut labels = Array3::<u32>::zeros(values.raw_dim());
        let (count, _) = label_plateaux_into(values.view(), labels.view_mut()).unwrap();
        // the 0.0 at each end, and the two near-equal values between them,
        // which are two plateaus rather than one
        assert_eq!(count, 4, "one ulp apart is not equal");
        let found = maxima(&values);
        assert!(!found[[1, 0, 0]], "the lower of the two is not a maximum");
        assert!(found[[2, 0, 0]]);
    }

    /// `-0.0` and `+0.0` are one plateau, because equality and the strict order
    /// have to be the two halves of one relation. Comparing bit patterns would
    /// split them into two plateaus neither of which is above the other, and
    /// report two maxima where the order says there is one.
    #[test]
    fn the_two_zeroes_are_one_plateau_because_the_order_says_they_are_equal() {
        let mut values = Array3::from_elem((3, 1, 1), -1.0);
        values[[0, 0, 0]] = -0.0;
        values[[1, 0, 0]] = 0.0;
        assert_ne!((-0.0f64).to_bits(), 0.0f64.to_bits());
        let mut labels = Array3::<u32>::zeros(values.raw_dim());
        let (count, _) = label_plateaux_into(values.view(), labels.view_mut()).unwrap();
        assert_eq!(count, 2, "the two zeroes, and the -1.0 beside them");
        assert_eq!(labels[[0, 0, 0]], labels[[1, 0, 0]]);
    }

    /// NaN, as the module header defines it: in no plateau, in no maximum, and
    /// disqualifying nothing.
    #[test]
    fn a_nan_is_in_no_plateau_and_disqualifies_nothing() {
        let mut values = Array3::from_elem((5, 1, 1), 0.0);
        values[[1, 0, 0]] = f64::NAN;
        values[[2, 0, 0]] = 3.0;
        values[[3, 0, 0]] = f64::NAN;

        let mut labels = Array3::<u32>::zeros(values.raw_dim());
        let (count, held) = label_plateaux_into(values.view(), labels.view_mut()).unwrap();
        assert_eq!(labels[[1, 0, 0]], UNLABELLED);
        assert_eq!(labels[[3, 0, 0]], UNLABELLED);
        // 0.0 at index 0, 3.0 at index 2, 0.0 at index 4 — and the two 0.0s are
        // *not* one plateau, because the NaN between them is in neither
        assert_eq!(count, 3);
        assert_eq!(held, vec![0.0, 3.0, 0.0]);

        let found = maxima(&values);
        assert!(!found[[1, 0, 0]], "a NaN voxel is never in a maximum");
        assert!(!found[[3, 0, 0]]);
        assert!(found[[2, 0, 0]], "the 3.0 stands above both its neighbours");
        assert!(
            found[[0, 0, 0]] && found[[4, 0, 0]],
            "a plateau whose only other neighbour is NaN is a maximum: NaN is not \
             strictly greater than anything"
        );

        // two adjacent NaNs do not become a plateau however their bits compare
        let mut pair = Array3::from_elem((2, 1, 1), f64::NAN);
        pair[[1, 0, 0]] = f64::NAN;
        let mut two = Array3::<u32>::zeros(pair.raw_dim());
        let (none, _) = label_plateaux_into(pair.view(), two.view_mut()).unwrap();
        assert_eq!(none, 0);
        assert!(maxima(&pair).iter().all(|&set| !set));
    }

    /// A plateau that spans the whole block, which is what an implementation
    /// written as a depth-first recursion overflows the stack on.
    #[test]
    fn a_plateau_spanning_the_whole_block_is_labelled_without_recursing() {
        let values = Array3::from_elem((64, 64, 64), 2.0);
        let mut labels = Array3::<u32>::zeros(values.raw_dim());
        let (count, _) = label_plateaux_into(values.view(), labels.view_mut()).unwrap();
        assert_eq!(count, 1);
        assert!(labels.iter().all(|&label| label == 1));
    }

    /// Labels are numbered in the order their lowest voxel is met, which is what
    /// makes the intermediate level reproducible between runs of one
    /// decomposition.
    #[test]
    fn labels_are_numbered_in_scan_order_and_are_reproducible() {
        let mut values = Array3::from_elem((5, 5, 5), 0.0);
        values[[1, 1, 1]] = 1.0;
        values[[3, 3, 3]] = 2.0;
        let mut labels = Array3::<u32>::zeros(values.raw_dim());
        let (count, held) = label_plateaux_into(values.view(), labels.view_mut()).unwrap();
        assert_eq!(count, 3);
        assert_eq!(labels[[0, 0, 0]], 1);
        assert_eq!(labels[[1, 1, 1]], 2);
        assert_eq!(labels[[3, 3, 3]], 3);
        assert_eq!(held, vec![0.0, 1.0, 2.0]);

        let mut again = Array3::<u32>::zeros(values.raw_dim());
        label_plateaux_into(values.view(), again.view_mut()).unwrap();
        assert_eq!(labels, again);
    }

    #[test]
    fn a_plateau_faces_fragment_survives_a_round_trip_and_a_wrong_one_is_refused() {
        let mut values = Array3::from_elem((5, 4, 3), 1.0);
        values[[2, 2, 1]] = 9.0;
        values[[0, 0, 0]] = f64::NAN;
        let mut labels = Array3::<u32>::zeros(values.raw_dim());
        let (count, held) = label_plateaux_into(values.view(), labels.view_mut()).unwrap();
        let ascends = ascending_neighbours(values.view(), labels.view(), count).unwrap();
        let faces = PlateauFaces::of(labels.view(), count, held, ascends).unwrap();

        let bytes = faces.encode();
        assert_eq!(PlateauFaces::decode(&bytes).unwrap(), faces);

        // a stream written by something else — including, specifically, by the
        // other six-plane op in this crate
        let mut foreign = bytes.clone();
        foreign[0] ^= 0xff;
        assert!(PlateauFaces::decode(&foreign).is_err());
        assert!(super::super::fill::BlockFaces::decode(&bytes).is_err());
        assert!(PlateauFaces::decode(&super::super::fill::BlockFaces::empty().encode()).is_err());
        // a truncated one
        assert!(PlateauFaces::decode(&bytes[..bytes.len() - 4]).is_err());
        // one with something appended
        let mut extra = bytes;
        extra.extend_from_slice(&[0, 0, 0, 0]);
        assert!(PlateauFaces::decode(&extra).is_err());
        // and something that is not words at all
        assert!(PlateauFaces::decode(&[1, 2, 3]).is_err());
    }

    /// A NaN value survives the fragment exactly, which is the reason the values
    /// travel as bits rather than as anything else.
    #[test]
    fn a_fragment_carries_a_nan_value_through_unchanged() {
        let faces = PlateauFaces {
            labels: 2,
            values: vec![f64::NAN, -0.0],
            ascends: vec![false, true],
            faces: empty_planes(),
        };
        let back = PlateauFaces::decode(&faces.encode()).unwrap();
        assert!(back.values[0].is_nan());
        assert_eq!(back.values[1].to_bits(), (-0.0f64).to_bits());
        assert_eq!(back.ascends, vec![false, true]);
    }

    /// The merge refuses a lattice it was not given every block of. "Absent" and
    /// "present and empty" are different facts, and only one of them is a block
    /// that ran.
    #[test]
    fn the_merge_refuses_a_lattice_with_a_block_missing() {
        let mut reports = BTreeMap::new();
        reports.insert([0, 0, 0], PlateauFaces::empty());
        assert!(merge_plateaux(&reports, [2, 1, 1]).is_err());
        reports.insert([1, 0, 0], PlateauFaces::empty());
        assert!(merge_plateaux(&reports, [2, 1, 1]).is_ok());
    }

    /// Two blocks whose faces are different shapes came from two different
    /// lattices, and the merge says so rather than silently zipping the shorter.
    #[test]
    fn the_merge_refuses_faces_from_two_different_lattices() {
        use super::super::components::face_index;
        let mut small = PlateauFaces::empty();
        small.faces[face_index(0, 1)] = ([2, 2], vec![0; 4]);
        let mut large = PlateauFaces::empty();
        large.faces[face_index(0, 0)] = ([3, 3], vec![0; 9]);
        let mut reports = BTreeMap::new();
        reports.insert([0, 0, 0], small);
        reports.insert([1, 0, 0], large);
        let message = merge_plateaux(&reports, [2, 1, 1]).unwrap_err().to_string();
        assert!(message.contains("two different lattices"), "{message}");
    }

    /// The three branches of a seam meeting, driven directly rather than through
    /// a run, so that each one is asserted on its own.
    #[test]
    fn a_seam_meeting_unions_on_equal_and_flags_on_unequal() {
        use super::super::components::face_index;

        // Two blocks, one label each, meeting on axis 0.
        let build = |here: f64, there: f64| {
            let mut left = PlateauFaces {
                labels: 1,
                values: vec![here],
                ascends: vec![false],
                faces: empty_planes(),
            };
            left.faces[face_index(0, 1)] = ([1, 1], vec![1]);
            let mut right = PlateauFaces {
                labels: 1,
                values: vec![there],
                ascends: vec![false],
                faces: empty_planes(),
            };
            right.faces[face_index(0, 0)] = ([1, 1], vec![1]);
            let mut reports = BTreeMap::new();
            reports.insert([0, 0, 0], left);
            reports.insert([1, 0, 0], right);
            merge_plateaux(&reports, [2, 1, 1]).unwrap()
        };

        // equal: one component, and nothing above it
        let joined = build(3.0, 3.0);
        assert_eq!(joined[&[0, 0, 0]], vec![false]);
        assert_eq!(joined[&[1, 0, 0]], vec![false]);

        // the right one higher: the left has an ascent, the right does not
        let right_higher = build(3.0, 4.0);
        assert_eq!(right_higher[&[0, 0, 0]], vec![true]);
        assert_eq!(right_higher[&[1, 0, 0]], vec![false]);

        // and symmetrically
        let left_higher = build(4.0, 3.0);
        assert_eq!(left_higher[&[0, 0, 0]], vec![false]);
        assert_eq!(left_higher[&[1, 0, 0]], vec![true]);

        // NaN on one side: no join and no flag, in either direction
        let unordered = build(3.0, f64::NAN);
        assert_eq!(unordered[&[0, 0, 0]], vec![false]);
        assert_eq!(unordered[&[1, 0, 0]], vec![false]);
    }

    /// The two phases are two halves of one relation, and a plan whose halves
    /// disagree is refused before it is scheduled.
    #[test]
    fn a_plan_whose_two_phases_disagree_about_connectivity_is_refused() {
        use crate::geometry::BlockGrid;

        let grid = BlockGrid::new([8, 8, 8], [4, 4, 4]).unwrap();
        let label = LabelPlateauxOp::new("label", "s", Lifecycle::DeleteOnExit)
            .connecting(Connectivity::FacesAndEdges);
        let maxima = RegionalMaximaOp::new("maxima", "s", 0, Dtype::Bool, &grid);
        let message = regional_phases(grid.clone(), Dtype::F64, &label, &maxima)
            .unwrap_err()
            .to_string();
        assert!(message.contains("same connectivity"), "{message}");

        for connectivity in [
            Connectivity::Faces,
            Connectivity::FacesAndEdges,
            Connectivity::FacesEdgesAndCorners,
        ] {
            let label = LabelPlateauxOp::new("label", "s", Lifecycle::DeleteOnExit)
                .connecting(connectivity);
            let maxima = RegionalMaximaOp::new("maxima", "s", 0, Dtype::Bool, &grid)
                .connecting(connectivity);
            assert_eq!(label.connectivity(), connectivity);
            assert_eq!(maxima.connectivity(), connectivity);
            assert!(regional_phases(grid.clone(), Dtype::F64, &label, &maxima).is_ok());
        }
    }

    /// The element types the shell bridges, and the two it will not.
    #[test]
    fn the_shell_accepts_every_width_but_the_sixty_four_bit_integers() {
        for dtype in [
            Dtype::Bool,
            Dtype::U8,
            Dtype::U16,
            Dtype::U32,
            Dtype::I8,
            Dtype::I16,
            Dtype::I32,
            Dtype::F32,
            Dtype::F64,
        ] {
            assert!(accepts(dtype), "{dtype:?}");
        }
        assert!(!accepts(Dtype::U64));
        assert!(!accepts(Dtype::I64));
        let message = as_values(&Voxels::zeros(Dtype::U64, [1, 1, 1]).unwrap())
            .unwrap_err()
            .to_string();
        assert!(message.contains("where the seam fell"), "{message}");
    }

    /// The kernels are generic over the element type, and an integer volume goes
    /// through them with no widening at all.
    #[test]
    fn the_kernels_take_the_element_type_they_are_handed() {
        let mut values = Array3::<u16>::zeros((4, 1, 1));
        values[[1, 0, 0]] = 7;
        values[[2, 0, 0]] = 7;
        let mut labels = Array3::<u32>::zeros(values.raw_dim());
        let (count, held) = label_plateaux_into(values.view(), labels.view_mut()).unwrap();
        assert_eq!(count, 3);
        assert_eq!(held, vec![0u16, 7, 0]);
        let ascends = ascending_neighbours(values.view(), labels.view(), count).unwrap();
        let mut out = Array3::from_elem(values.raw_dim(), false);
        maxima_from_labels_into(labels.view(), &ascends, out.view_mut()).unwrap();
        assert_eq!(
            out.iter().copied().collect::<Vec<_>>(),
            vec![false, true, true, false]
        );
    }
}
