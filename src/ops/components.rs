// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The machinery two **fragment-and-join** ops share, extracted rather than
// copied.
//
// Why there is a second op with this shape at all
// -----------------------------------------------
// `ops::fill` and `ops::regional` answer completely different questions — does
// this background component reach the outside of the volume, and does anything
// adjacent to this plateau stand higher than it — but they are the same
// *program*, and the header of `fill.rs` is where that program is argued for.
// Both are a global connected-component question that no halo can express, so
// both are:
//
// | phase | shape | what it does |
// |---|---|---|
// | 0 | `volume -> fragments`, and writes pixels | label the block's components locally at reach 0, write the labels as a `u32` image, emit the block's six face planes plus one fact per label |
// | 1 | `fragments -> volume` | read every block's faces, close the local labels into global components with a union-find, fold the per-label fact over each component, and rewrite this block's labels into the answer |
//
// What is genuinely common is everything except the two things the ops differ
// in: **what a component is** (a background run, or a run of equal value) and
// **what one seam pair does** (always union, or compare-then-union-or-flag).
// Those two stay in the ops. Everything below is here.
//
// * the disjoint sets, with path halving and union by size;
// * the six-face geometry — the neighbour offsets, the face ordering, which two
//   axes a face spans;
// * reading the six label planes off a block and encoding them as words;
// * the flat `(block, label)` numbering, the seam walk over the lattice, and the
//   fold of a per-label boolean onto its component's root;
// * the seeded flood fill itself, over a **per-voxel** membership test.
//
// The last one is here because two of the three ops built on this differ only in
// that test — `fill` labels the voxels a mask leaves clear, `detect` labels the
// ones it sets — and a second copy of a flood fill is a second place for a
// traversal to be subtly different. `regional`'s labelling is deliberately *not*
// expressed through it: what makes two voxels one plateau there is a comparison
// between them rather than a fact about each, and a pairwise relation does not
// fit a per-voxel predicate without carrying the seed's value into it.
//
// The extraction made `fill` shorter and changed none of its behaviour: its
// fragment type, its magic, its public functions and its error messages are
// where they were, and its tests are unchanged.
//
// Connectivity is a parameter, and six is the default
// ---------------------------------------------------
// [`Connectivity`] names which of the twenty-six voxels around one count as
// adjacent: the six that share a face, those plus the twelve that share an edge,
// or all of them. It is a *choice* rather than a correction — a caller that
// wants the face-connected answer and a caller that wants the full one are both
// right — so every entry point here comes in two forms, the bare one that means
// [`Connectivity::Faces`] and a `_with` one that takes the choice. Nothing that
// predates the parameter moved: the bare forms are the parameterised ones at
// their default, and the labels, the seam pairs and the bytes of a
// face-connected run are what they were.
//
// **The fragment did not have to change, and the reason is worth stating**,
// because the obvious guess is that it did. The guess goes: face connectivity
// crosses a seam only through the shared plane, so six planes are a block's
// whole contribution; wider connectivity also crosses edges and corners, so a
// fragment would have to carry twelve edge lines and eight corner voxels as
// well. The second half is wrong. A voxel with any neighbour outside its block
// lies on a face of that block, so the block's six faces **are** its whole
// boundary shell — and the twelve edge lines and the eight corner voxels are
// slices of those faces rather than anything new. `planes_of` already sends
// them; what was missing was a merge that reads them.
//
// So what changed is the *walk*, and only its inputs:
//
// | | face-connected | wider |
// |---|---|---|
// | fragment | six planes | six planes, unchanged |
// | lattice neighbours | 3 forward, the axis steps | up to 13 forward, the axis steps plus the lattice's own edges and corners |
// | pairs at one seam | voxel against the voxel opposite | voxel against a 3x3 window of them |
// | the relation | transitive closure of those pairs | transitive closure of those pairs |
//
// The last row is the point. The union-find, [`LabelIndex`], [`Union::fold_or`]
// and every op's per-label fact are untouched, because the equivalence relation
// is the same relation — the transitive closure of an adjacency — and only the
// set of pairs that generates it grew. That is why this is an addition to the
// merge rather than a second merge.
//
// Eighteen is offered, and here is why. It is one entry in the offsets table
// rather than a third code path: the walk is driven by the offset set, so the
// middle case costs a line. It also earns its place as a *test*, which the
// standing lesson about fixtures argues for directly — with only six and
// twenty-six, the predicate that decides which offsets cross a seam is only ever
// asked "exactly one step?" or "anything at all?", and is blind at its own
// boundary. Eighteen is the case that can see it.
//
// Which ops take the choice, and how many each has
// -------------------------------------------------
// An op built on this states its own connectivity in its own header, because
// what the relation is *about* is the op's. All three that exist take it from a
// caller and default to [`Connectivity::Faces`], so nothing that predates the
// parameter moved — and each has exactly one, for a different reason:
//
// | op | what its one connectivity names | why only one |
// |---|---|---|
// | `ops::fill` | the **background**'s | the background is the only thing it labels; the foreground's is `detect`'s and the caller pairs them |
// | `ops::detect` | the **foreground**'s | likewise, from the other side |
// | `ops::regional` | the plateau's **and** the ascent's | they are provably one relation; two would let a caller state something the definition does not admit |
//
// The pairing between the first two is worth naming because it is the one thing
// a caller has to do by hand: the *complementary pair* convention analyses a
// 6-connected foreground against a 26-connected background and vice versa, and
// neither op can honour it on the other's behalf, because each sees one of the
// two sets. `ops::fill`'s header is where that is spelt out.
//
// Every op that takes the choice takes it **twice** — once in the phase that
// floods and once in the phase that walks the seams — and refuses a pair that
// disagrees, at planning time. `ops::fill::agree_on_connectivity` is the one
// check and the one message; the reason is that the flood and the walk generate
// one equivalence relation between them, so a mismatched pair joins inside a
// block what it keeps apart across a seam and answers differently depending on
// where the volume was cut.

use std::collections::BTreeMap;

use ndarray::{ArrayView3, ArrayViewMut3};

use crate::error::{Error, Result};
use crate::fragment::BlockView;

// ------------------------------------------------------------- geometry --

/// The label reserved for "this voxel belongs to no component".
///
/// Zero rather than a sentinel at the top of the range, so that an image
/// allocated as zeros starts out entirely unlabelled and a lookup table is
/// indexed by `label - 1` with no offset arithmetic anywhere else.
///
/// What "no component" *means* is the op's: for `fill` it is a foreground voxel,
/// which is not part of the background being labelled; for `regional` it is a
/// voxel whose value is unordered with everything, itself included.
pub const UNLABELLED: u32 = 0;

/// The six face neighbours, as `(axis, step)`.
pub const FACE_NEIGHBOURS: [(usize, isize); 6] =
    [(0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)];

/// Which of the twenty-six voxels around one count as adjacent to it.
///
/// Named for what they touch rather than numbered, because the numbers are a
/// fact about three dimensions and the names are not: the conventional 6, 18 and
/// 26 are `Faces`, `FacesAndEdges` and `FacesEdgesAndCorners` here, and the
/// conventional numbers are given in each variant's own documentation for
/// whoever arrives looking for them.
///
/// The ordering is by strength — a `Faces` component is contained in the
/// `FacesAndEdges` component it is part of, which is contained in the
/// `FacesEdgesAndCorners` one — which is why the derive is here rather than
/// omitted, and it is the same order the offsets table is grouped in.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Connectivity {
    /// The six voxels sharing a face. **The default**, and what every op in this
    /// crate asks for; conventionally "6-connected".
    #[default]
    Faces,
    /// The six faces and the twelve edges: conventionally "18-connected".
    FacesAndEdges,
    /// Every voxel of the 3x3x3 neighbourhood: the six faces, the twelve edges
    /// and the eight corners, conventionally "26-connected".
    FacesEdgesAndCorners,
}

/// The twenty-six neighbour offsets, **grouped by how many axes they step
/// along**: six that step along one, twelve along two, eight along three.
///
/// One table rather than three, so that each connectivity's offsets are a prefix
/// of it and there is a single place a typo could live. The first six are in
/// [`FACE_NEIGHBOURS`]'s order, which is what keeps a face-connected traversal
/// visiting neighbours in exactly the order it always did.
const NEIGHBOUR_OFFSETS: [[isize; 3]; 26] = [
    // one step: the faces
    [-1, 0, 0],
    [1, 0, 0],
    [0, -1, 0],
    [0, 1, 0],
    [0, 0, -1],
    [0, 0, 1],
    // two steps: the edges
    [-1, -1, 0],
    [-1, 0, -1],
    [-1, 0, 1],
    [-1, 1, 0],
    [0, -1, -1],
    [0, -1, 1],
    [0, 1, -1],
    [0, 1, 1],
    [1, -1, 0],
    [1, 0, -1],
    [1, 0, 1],
    [1, 1, 0],
    // three steps: the corners
    [-1, -1, -1],
    [-1, -1, 1],
    [-1, 1, -1],
    [-1, 1, 1],
    [1, -1, -1],
    [1, -1, 1],
    [1, 1, -1],
    [1, 1, 1],
];

/// The directions from a block to a lattice neighbour it can meet, each listed
/// **once for the pair** — the first non-zero component is `+1`, so a seam is
/// walked from the lower block and every meeting is seen once.
///
/// Grouped by steps like [`NEIGHBOUR_OFFSETS`] and for the same reason: the
/// three axis steps, then the six lattice edges, then the four lattice corners,
/// so each connectivity's directions are a prefix. A direction that steps along
/// *n* axes can only be crossed by an offset that steps along at least those
/// *n*, which is exactly why the two tables share a grouping.
const FORWARD_DIRECTIONS: [[isize; 3]; 13] = [
    // one step
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    // two steps
    [1, -1, 0],
    [1, 0, -1],
    [1, 0, 1],
    [1, 1, 0],
    [0, 1, -1],
    [0, 1, 1],
    // three steps
    [1, -1, -1],
    [1, -1, 1],
    [1, 1, -1],
    [1, 1, 1],
];

/// How many axes an offset steps along, which is what a connectivity is a bound
/// on.
pub fn steps_of(by: [isize; 3]) -> usize {
    by.iter().filter(|&&step| step != 0).count()
}

impl Connectivity {
    /// The most axes one step of this connectivity may move along.
    pub fn steps(self) -> usize {
        match self {
            Self::Faces => 1,
            Self::FacesAndEdges => 2,
            Self::FacesEdgesAndCorners => 3,
        }
    }

    /// Does this connectivity make two voxels `by` apart adjacent?
    ///
    /// A zero offset is **not** adjacency: a voxel is trivially in its own
    /// component and the question here is about a pair.
    pub fn joins(self, by: [isize; 3]) -> bool {
        let steps = steps_of(by);
        steps > 0 && steps <= self.steps()
    }

    /// The neighbour offsets, faces first, in a fixed order.
    pub fn offsets(self) -> &'static [[isize; 3]] {
        &NEIGHBOUR_OFFSETS[..match self {
            Self::Faces => 6,
            Self::FacesAndEdges => 18,
            Self::FacesEdgesAndCorners => 26,
        }]
    }

    /// The directions from one block to a lattice neighbour it can meet under
    /// this connectivity, axis steps first and each listed **once for the
    /// pair** — the first non-zero component is `+1`, so a seam is walked from
    /// the lower block and every meeting is seen once.
    ///
    /// Three of them for faces, and thirteen for the widest: the three axis
    /// steps, then the six ways two blocks share only a lattice edge, then the
    /// four ways they share only a lattice corner.
    pub fn directions(self) -> &'static [[isize; 3]] {
        &FORWARD_DIRECTIONS[..match self {
            Self::Faces => 3,
            Self::FacesAndEdges => 9,
            Self::FacesEdgesAndCorners => 13,
        }]
    }
}

/// `at` moved by `by`, or `None` if that leaves an array of `shape`.
///
/// Clamped by refusing rather than by saturating, for [`offset`]'s reason. Also
/// the step from a block index to a neighbouring block's: a lattice is an array
/// of blocks, and "outside the lattice" and "outside the block" are the same
/// arithmetic.
pub fn offset_by(at: [usize; 3], by: [isize; 3], shape: [usize; 3]) -> Option<[usize; 3]> {
    let mut to = [0usize; 3];
    for axis in 0..3 {
        let moved = at[axis] as isize + by[axis];
        if moved < 0 || moved >= shape[axis] as isize {
            return None;
        }
        to[axis] = moved as usize;
    }
    Some(to)
}

/// `at` moved by `step` along `axis`, or `None` if that leaves the array.
///
/// Clamped by refusing rather than by saturating: a neighbour outside the block
/// is *absent*, not a copy of the edge voxel, and the difference matters — a
/// saturating step would make every edge voxel its own neighbour and a plateau
/// its own higher neighbour.
pub fn offset(at: [usize; 3], axis: usize, step: isize, shape: [usize; 3]) -> Option<[usize; 3]> {
    let mut by = [0isize; 3];
    by[axis] = step;
    offset_by(at, by, shape)
}

/// The face index for `axis` on the low (`side == 0`) or high side.
pub fn face_index(axis: usize, side: usize) -> usize {
    axis * 2 + side
}

/// The two axes a face on `axis` spans, in increasing order.
pub fn face_axes(axis: usize) -> [usize; 2] {
    match axis {
        0 => [1, 2],
        1 => [0, 2],
        _ => [0, 1],
    }
}

// ------------------------------------------------------------ labelling --

/// Label the face-connected components of the voxels `member` accepts, into
/// `out`, and return how many were found.
///
/// The membership test is per voxel and takes a position rather than a value, so
/// that the caller keeps its own array and this borrows nothing: `fill` passes
/// `|at| !mask[at]`, `detect` passes `|at| mask[at]`, and the *program* — which
/// is everything else — is written once.
///
/// Deterministic, and deterministic in a way that matters: components are
/// numbered in the order their lowest voxel is met in row-major order, so the
/// same block always produces the same labels. Two runs of one decomposition are
/// then byte-identical in the label volume as well as in whatever is derived from
/// it, which is what makes the labels worth looking at when something is wrong.
///
/// Iterative rather than recursive. A component can span the whole block — a
/// mask that is set everywhere is one — and a depth-first recursion over a
/// 256-cube is a stack overflow rather than a slow answer.
pub fn label_members_into(
    shape: [usize; 3],
    member: impl Fn([usize; 3]) -> bool,
    out: ArrayViewMut3<'_, u32>,
) -> Result<u32> {
    crate::ops::shapes_agree(&shape, out.shape(), "label_members_into")?;
    label_into(shape, Connectivity::Faces, member, out)
}

/// [`label_members_into`] under a stated [`Connectivity`].
///
/// Everything that function promises holds here: the numbering, the
/// determinism, the iterative traversal. What the connectivity changes is which
/// voxels a flood reaches, and nothing else — the seeds are still met in
/// row-major order and the *rule* is still that a component is numbered by
/// where its lowest voxel sits. Widening does change the numbers, because
/// merging two components leaves a shorter list, and it changes them by that
/// rule rather than by traversal order.
pub fn label_members_into_with(
    shape: [usize; 3],
    connectivity: Connectivity,
    member: impl Fn([usize; 3]) -> bool,
    out: ArrayViewMut3<'_, u32>,
) -> Result<u32> {
    crate::ops::shapes_agree(&shape, out.shape(), "label_members_into_with")?;
    label_into(shape, connectivity, member, out)
}

/// The one traversal both forms above are, with the shape already checked.
fn label_into(
    shape: [usize; 3],
    connectivity: Connectivity,
    member: impl Fn([usize; 3]) -> bool,
    mut out: ArrayViewMut3<'_, u32>,
) -> Result<u32> {
    out.fill(UNLABELLED);

    let mut next = UNLABELLED;
    let mut stack: Vec<[usize; 3]> = Vec::new();
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let seed = [i, j, k];
                if out[seed] != UNLABELLED || !member(seed) {
                    continue;
                }
                next += 1;
                out[seed] = next;
                stack.push(seed);
                while let Some(at) = stack.pop() {
                    for &by in connectivity.offsets() {
                        let Some(to) = offset_by(at, by, shape) else {
                            continue;
                        };
                        if out[to] != UNLABELLED || !member(to) {
                            continue;
                        }
                        out[to] = next;
                        stack.push(to);
                    }
                }
            }
        }
    }
    Ok(next)
}

// --------------------------------------------------------- face planes --

/// A block's six faces of labels, ordered `axis * 2 + side` with side 0 low and
/// 1 high, each as `(shape, labels)` in row-major order over the two axes that
/// are not this face's.
///
/// A type alias rather than a struct on purpose: both ops hold this inside their
/// own fragment type as a field named `faces`, and a wrapper would have renamed
/// that field in `fill`'s public API for no gain.
pub type FacePlanes = [([usize; 2], Vec<u32>); 6];

/// Read a block's six face planes off its label volume.
pub fn planes_of(labels: ArrayView3<'_, u32>) -> FacePlanes {
    let shape = [labels.shape()[0], labels.shape()[1], labels.shape()[2]];
    let mut faces: Vec<([usize; 2], Vec<u32>)> = Vec::with_capacity(6);
    for axis in 0..3 {
        for side in 0..2 {
            let [u, v] = face_axes(axis);
            let plane = [shape[u], shape[v]];
            let fixed = if side == 0 { 0 } else { shape[axis] - 1 };
            let mut values = Vec::with_capacity(plane[0] * plane[1]);
            for a in 0..plane[0] {
                for b in 0..plane[1] {
                    let mut at = [0usize; 3];
                    at[axis] = fixed;
                    at[u] = a;
                    at[v] = b;
                    values.push(labels[at]);
                }
            }
            faces.push((plane, values));
        }
    }
    faces.try_into().expect("six faces were pushed")
}

/// Six empty planes: what a block with nothing to say reports, which is a
/// different fact from no fragment at all.
pub fn empty_planes() -> FacePlanes {
    std::array::from_fn(|_| ([0, 0], Vec::new()))
}

/// Append the planes to a word stream, each as `[rows, columns, labels...]`.
pub fn push_planes(planes: &FacePlanes, words: &mut Vec<u32>) {
    for (plane, values) in planes {
        words.push(plane[0] as u32);
        words.push(plane[1] as u32);
        words.extend_from_slice(values);
    }
}

/// The inverse of [`push_planes`], advancing `at` past what it read.
///
/// `noun` names the fragment kind in the diagnostics, because a truncated
/// fragment's only useful message is which stream it should have come from.
pub fn take_planes(words: &[u32], at: &mut usize, noun: &str) -> Result<FacePlanes> {
    let mut faces: Vec<([usize; 2], Vec<u32>)> = Vec::with_capacity(6);
    for _ in 0..6 {
        if words.len() < *at + 2 {
            return Err(truncated(noun, "ends inside a face header"));
        }
        let plane = [words[*at] as usize, words[*at + 1] as usize];
        *at += 2;
        let count = plane[0]
            .checked_mul(plane[1])
            .ok_or_else(|| truncated(noun, "declares a face larger than the address space"))?;
        if words.len() < *at + count {
            return Err(truncated(noun, "ends inside a face"));
        }
        faces.push((plane, words[*at..*at + count].to_vec()));
        *at += count;
    }
    Ok(faces.try_into().expect("six faces were pushed"))
}

fn truncated(noun: &str, what: &str) -> Error {
    Error::InvalidArgument(format!("{noun} {what}"))
}

// ------------------------------------------------------------- the wire --

/// Little-endian `u32` words as bytes.
pub fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

/// The inverse. A length that is not a whole number of words is a truncated
/// fragment and says so rather than dropping the tail.
pub fn bytes_to_words(bytes: &[u8], noun: &str) -> Result<Vec<u32>> {
    if bytes.len() % 4 != 0 {
        return Err(truncated(noun, "is not a whole number of 32-bit words"));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

/// Check a `[magic, version, labels]` header and return the label count.
///
/// The magic is not decoration: a fragment is addressed by `(stream, phase,
/// block)` and nothing in that address says what the bytes mean, so a stream
/// name reused by two ops would otherwise be decoded as whatever the reader
/// expected. Two ops in this crate now write six-plane fragments that differ
/// only in their per-label payload, which is exactly the confusion this refuses.
/// It is cheaper to refuse.
pub fn read_header(words: &[u32], magic: u32, version: u32, noun: &str) -> Result<u32> {
    if words.len() < 3 {
        return Err(truncated(noun, "is shorter than its own header"));
    }
    if words[0] != magic {
        return Err(truncated(
            noun,
            "does not begin with its own magic; the stream it was read from was \
             written by something else",
        ));
    }
    if words[1] != version {
        return Err(truncated(
            noun,
            &format!(
                "is version {} and this build reads version {version}",
                words[1]
            ),
        ));
    }
    Ok(words[2])
}

/// Refuse a fragment with bytes after its last field.
pub fn expect_end(words: &[u32], at: usize, noun: &str) -> Result<()> {
    if at != words.len() {
        return Err(truncated(noun, "has bytes after its last face"));
    }
    Ok(())
}

// ------------------------------------------------------------ the merge --

/// Disjoint sets over a flat node numbering, with path halving and union by
/// size.
///
/// Not generic and not a crate-wide utility: it exists to close block-local
/// labels into global components, its nodes are `(block, label)` pairs flattened
/// by [`LabelIndex`], and it is `pub` only so that both ops can drive it.
pub struct Union {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl Union {
    pub fn new(count: usize) -> Self {
        Self {
            parent: (0..count).collect(),
            size: vec![1; count],
        }
    }

    pub fn find(&mut self, mut node: usize) -> usize {
        while self.parent[node] != node {
            self.parent[node] = self.parent[self.parent[node]];
            node = self.parent[node];
        }
        node
    }

    pub fn union(&mut self, a: usize, b: usize) {
        let (mut a, mut b) = (self.find(a), self.find(b));
        if a == b {
            return;
        }
        if self.size[a] < self.size[b] {
            std::mem::swap(&mut a, &mut b);
        }
        self.parent[b] = a;
        self.size[a] += self.size[b];
    }

    /// OR one flag per node onto the root of its component, and return the
    /// per-root answer.
    ///
    /// Folded onto the roots first and read back afterwards, rather than
    /// propagated as the unions happen, so the answer does not depend on the
    /// order the unions arrived in. That is what makes the result a function of
    /// the volume rather than of the block iteration order.
    pub fn fold_or(&mut self, flags: &[bool]) -> Vec<bool> {
        let mut roots = vec![false; flags.len()];
        for (node, &flag) in flags.iter().enumerate() {
            let root = self.find(node);
            roots[root] |= flag;
        }
        roots
    }

    /// Reduce one `u64` per node onto the root of its component by `min`, and
    /// return the per-root answer. A node's own value is included, so a
    /// singleton component folds to itself.
    ///
    /// **The same argument [`Self::fold_or`] rests on, and it is the whole
    /// reason this is a fold rather than a running minimum kept as the unions
    /// happen.** `min` is associative, commutative and idempotent, so the
    /// per-root answer is a function of the *set* of nodes in a component and
    /// not of the order they were joined in — which is what makes a quantity
    /// derived from it the same under every decomposition.
    ///
    /// A root that is not the least node of its component still gets the least
    /// *value*: the fold visits every node and asks `find` where it lives, so
    /// nothing depends on which member the union-by-size left holding the
    /// component.
    pub fn fold_min(&mut self, values: &[u64]) -> Vec<u64> {
        let mut roots = vec![u64::MAX; values.len()];
        for (node, &value) in values.iter().enumerate() {
            let root = self.find(node);
            if value < roots[root] {
                roots[root] = value;
            }
        }
        roots
    }
}

/// A flat numbering of every `(block, label)` in a lattice.
///
/// The merge is a union-find, and a union-find wants integers. This is the map
/// from "block `[1,0,2]`'s label 7" to one of them, plus the inverse walk.
pub struct LabelIndex {
    /// Every block of the lattice in row-major order.
    order: Vec<[usize; 3]>,
    /// Block -> (first flat node, how many labels).
    span: BTreeMap<[usize; 3], (usize, usize)>,
    total: usize,
}

impl LabelIndex {
    /// Number every label of every block of `counts`.
    ///
    /// A block of the lattice that is missing from `reports` is **refused**
    /// rather than assumed empty, because "absent" and "present with nothing to
    /// say" are different facts and only one of them is a block that ran. That
    /// is the same distinction `Coverage::EveryBlock` exists to check, checked
    /// again here where the answer would silently be wrong.
    pub fn build<R>(
        reports: &BTreeMap<[usize; 3], R>,
        counts: [usize; 3],
        count_of: impl Fn(&R) -> u32,
    ) -> Result<Self> {
        let mut order: Vec<[usize; 3]> = Vec::with_capacity(counts[0] * counts[1] * counts[2]);
        for i in 0..counts[0] {
            for j in 0..counts[1] {
                for k in 0..counts[2] {
                    order.push([i, j, k]);
                }
            }
        }
        let mut span = BTreeMap::new();
        let mut total = 0usize;
        for &block in &order {
            let report = reports.get(&block).ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "block {block:?} wrote no faces fragment. The stream is declared \
                     every-block, so a missing one is a block that did not run rather than a \
                     block with nothing to say."
                ))
            })?;
            let labels = count_of(report) as usize;
            span.insert(block, (total, labels));
            total += labels;
        }
        Ok(Self { order, span, total })
    }

    /// How many `(block, label)` pairs there are.
    pub fn total(&self) -> usize {
        self.total
    }

    /// Every block of the lattice, row-major.
    pub fn order(&self) -> &[[usize; 3]] {
        &self.order
    }

    /// The flat node for a block's `label`, which is **one-based** exactly as it
    /// is in the label volume.
    pub fn node(&self, block: [usize; 3], label: u32) -> usize {
        self.span[&block].0 + label as usize - 1
    }

    /// Lay one per-label value from every block out in flat order.
    ///
    /// `missing` fills a slot the report did not supply. It should never be
    /// reached — a fragment carries exactly as many payload entries as it
    /// declares labels — and it is here so that a short report is a wrong
    /// *answer* rather than a panic in the merge of somebody's overnight run.
    pub fn gather<R, T: Copy>(
        &self,
        reports: &BTreeMap<[usize; 3], R>,
        per_label: impl Fn(&R) -> &[T],
        missing: T,
    ) -> Vec<T> {
        let mut out = Vec::with_capacity(self.total);
        for block in &self.order {
            let (_, labels) = self.span[block];
            let values = per_label(&reports[block]);
            for label in 0..labels {
                out.push(values.get(label).copied().unwrap_or(missing));
            }
        }
        out
    }

    /// Read a per-*component* answer back out as one flag per label of each
    /// block, in label order, so `flags[label - 1]` is the lookup a rewrite
    /// wants.
    pub fn per_block(
        &self,
        sets: &mut Union,
        per_root: &[bool],
    ) -> BTreeMap<[usize; 3], Vec<bool>> {
        self.per_block_of(sets, per_root)
    }

    /// [`Self::per_block`] for a per-component answer of any `Copy` type.
    ///
    /// The `bool` form above is this at `T = bool` and did not move; the reason
    /// for the generalisation is that a per-component **label** is a `u32` and
    /// the walk that reads it back out is the same walk, character for
    /// character. Two copies of it would be two chances for the flat numbering
    /// and the inverse of the flat numbering to disagree, and the flat numbering
    /// is the one thing every op built on this module shares.
    pub fn per_block_of<T: Copy>(
        &self,
        sets: &mut Union,
        per_root: &[T],
    ) -> BTreeMap<[usize; 3], Vec<T>> {
        let mut out = BTreeMap::new();
        for &block in &self.order {
            let (base, labels) = self.span[&block];
            let mut values = Vec::with_capacity(labels);
            for label in 0..labels {
                let root = sets.find(base + label);
                values.push(per_root[root]);
            }
            out.insert(block, values);
        }
        out
    }
}

/// Walk every seam of the lattice and hand `meet` the two flat nodes that touch
/// at each pair of face voxels.
///
/// Each seam is visited from exactly one side — this block's high face against
/// the next block's low face — which is why only the `+1` neighbour is looked
/// at and why `meet` sees each meeting once. A pair where either side is
/// [`UNLABELLED`] is skipped and `meet` never sees it: a voxel in no component
/// takes part in no join.
///
/// What `meet` does is the whole of the difference between the two ops built on
/// this. `fill` unions unconditionally, because two background labels that touch
/// across a seam are one background component. `regional` compares first,
/// because two plateaux that touch are the same plateau only if their values are
/// equal, and if they are not then one of them has just been shown to have a
/// higher neighbour.
pub fn walk_seams<R>(
    reports: &BTreeMap<[usize; 3], R>,
    counts: [usize; 3],
    index: &LabelIndex,
    planes_of: impl Fn(&R) -> &FacePlanes,
    meet: impl FnMut(usize, usize),
) -> Result<()> {
    walk_seams_with(reports, counts, index, Connectivity::Faces, planes_of, meet)
}

/// [`walk_seams`] under a stated [`Connectivity`], which is the whole of what
/// the wider connectivities needed.
///
/// **The fragment is the same six planes.** A voxel with a neighbour outside its
/// block lies on one of that block's faces, so the six planes are its entire
/// boundary shell; the edge lines and corner voxels a wider connectivity meets
/// across are rows and single entries *of those planes*, not extra data. See the
/// module header.
///
/// Three things generalise together, and each is one line of the table there:
///
/// * **which lattice neighbours are visited** — `connectivity.directions()`,
///   three for faces and up to thirteen, the extra ones being the blocks that
///   share only a lattice edge or only a lattice corner with this one;
/// * **which face is read** — the direction's first stepped axis picks the
///   plane, and the other stepped axes pin a row or a column of it, which is how
///   an edge line and a corner voxel are addressed without being stored;
/// * **which voxels of it pair** — an axis the direction does not step along is
///   free, and a free axis pairs each index with the three around it rather than
///   with itself. A pair costs one step per axis it shifts on, and the total,
///   the direction's own steps included, must be within the connectivity.
///
/// `meet` sees each meeting once and never sees [`UNLABELLED`], exactly as in
/// [`walk_seams`], and under [`Connectivity::Faces`] it sees precisely the pairs
/// it saw before this parameter existed, in the same order.
pub fn walk_seams_with<R>(
    reports: &BTreeMap<[usize; 3], R>,
    counts: [usize; 3],
    index: &LabelIndex,
    connectivity: Connectivity,
    planes_of: impl Fn(&R) -> &FacePlanes,
    mut meet: impl FnMut(usize, usize),
) -> Result<()> {
    let budget = connectivity.steps();
    for &block in index.order() {
        for &towards in connectivity.directions() {
            let Some(ahead) = offset_by(block, towards, counts) else {
                continue;
            };
            // The first stepped axis names the pair of faces that meet. It is
            // always a `+1` step — that is what makes the direction table
            // one-per-pair — so it is this block's high face against the next
            // block's low one, the same two planes `walk_seams` always read.
            let pivot = towards
                .iter()
                .position(|&step| step != 0)
                .expect("a direction steps along at least one axis");
            let here = &planes_of(&reports[&block])[face_index(pivot, 1)];
            let there = &planes_of(&reports[&ahead])[face_index(pivot, 0)];
            let span = face_axes(pivot);

            // Checked before anything is skipped, so that a lattice mismatch is
            // still refused when one of the two blocks reported nothing.
            for slot in 0..2 {
                if towards[span[slot]] == 0 && here.0[slot] != there.0[slot] {
                    return Err(unequal_faces(block, ahead, towards, here.0, there.0));
                }
            }
            if here.0.contains(&0) || there.0.contains(&0) {
                continue;
            }

            // Per spanned axis, the index pairs that meet along it and what
            // each costs. A stepped axis contributes one pinned pair and no
            // extra cost — the step is already counted in the direction.
            //
            // A shift the budget cannot afford on its own is dropped here
            // rather than in the loop below, which is what keeps the
            // face-connected case building one pair per index instead of three
            // and discarding two. Dropping strictly fewer than the loop would
            // discard, so the survivors and their order are the same either way.
            let base = steps_of(towards);
            let mut along: [Vec<(usize, usize, usize)>; 2] = [Vec::new(), Vec::new()];
            for slot in 0..2 {
                let (mine, theirs) = (here.0[slot], there.0[slot]);
                match towards[span[slot]] {
                    0 => {
                        for a in 0..mine {
                            for shift in [-1isize, 0, 1] {
                                let cost = usize::from(shift != 0);
                                let b = a as isize + shift;
                                if base + cost > budget || b < 0 || b >= theirs as isize {
                                    continue;
                                }
                                along[slot].push((a, b as usize, cost));
                            }
                        }
                    }
                    step if step > 0 => along[slot].push((mine - 1, 0, 0)),
                    _ => along[slot].push((0, theirs - 1, 0)),
                }
            }

            for &(a_here, a_there, a_cost) in &along[0] {
                for &(b_here, b_there, b_cost) in &along[1] {
                    if base + a_cost + b_cost > budget {
                        continue;
                    }
                    let a = here.1[a_here * here.0[1] + b_here];
                    let b = there.1[a_there * there.0[1] + b_there];
                    if a == UNLABELLED || b == UNLABELLED {
                        continue;
                    }
                    meet(index.node(block, a), index.node(ahead, b));
                }
            }
        }
    }
    Ok(())
}

/// Two blocks whose shared extent disagrees, which means their fragments came
/// from two different lattices.
///
/// Two messages because the single-axis case has one and it is the one a
/// face-connected run has always produced; a diagonal meeting has no "the other
/// two" to talk about.
fn unequal_faces(
    block: [usize; 3],
    ahead: [usize; 3],
    towards: [isize; 3],
    here: [usize; 2],
    there: [usize; 2],
) -> Error {
    if steps_of(towards) == 1 {
        let axis = towards
            .iter()
            .position(|&step| step != 0)
            .expect("one step is along some axis");
        return Error::InvalidArgument(format!(
            "blocks {block:?} and {ahead:?} share a seam on axis {axis} but their \
             faces are {here:?} and {there:?}. Two blocks adjacent on one axis have the same \
             extent on the other two, so unequal faces mean the fragments came from \
             two different lattices."
        ));
    }
    Error::InvalidArgument(format!(
        "blocks {block:?} and {ahead:?} meet along {towards:?} but their faces are {here:?} \
         and {there:?}. Two blocks differing only in the axes they step along have the same \
         extent on the rest, so unequal faces mean the fragments came from two different \
         lattices."
    ))
}

// ----------------------------------------------------------- the rewrite --

/// Where the block's core sits inside its read extent, as an offset and an
/// extent, both in the read buffer's own indices.
///
/// **Both phase-1 ops need this and neither of them is optimising.** A
/// whole-lattice fragment reach is also a halo, so the read extent of such a
/// phase is the whole volume and the buffer it is handed holds every block's
/// labels — numbered per block. A per-block answer decodes that block's
/// numbering and no other, so rewriting the whole read extent with it would read
/// block 3's label 2 as this block's label 2. The executor slices the valid
/// sub-box out of what is returned, and valid is the core, so writing only the
/// core is both correct and the whole of what is used.
pub fn core_within_read(at: &BlockView<'_>) -> Result<([usize; 3], [usize; 3])> {
    let mut offset = [0usize; 3];
    let mut extent = [0usize; 3];
    let core_end = at.core.end();
    for axis in 0..3 {
        let core_start = at.core.start[axis];
        let read_start = at.read.start[axis];
        offset[axis] = core_start.checked_sub(read_start).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "block {:?} has a core starting at {core_start} on axis {axis} and a read \
                 extent starting at {read_start}. A core outside its own read extent is a \
                 geometry that cannot be sliced.",
                at.index
            ))
        })?;
        extent[axis] = core_end[axis] - core_start;
    }
    Ok((offset, extent))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    /// **`fold_min` is order-independent, and that is the whole of why it is a
    /// fold.** The same component built by two different sequences of unions
    /// must reduce to the same least value — because the value a component is
    /// *named* by downstream is derived from it, and a name that depended on the
    /// union order would be a name that depended on the block iteration order.
    #[test]
    fn fold_min_reduces_to_the_least_value_whatever_order_the_unions_arrived_in() {
        let values = [70u64, 30, 50, 90];
        let mut forwards = Union::new(4);
        forwards.union(0, 1);
        forwards.union(1, 2);
        let mut backwards = Union::new(4);
        backwards.union(2, 1);
        backwards.union(1, 0);

        let a = forwards.fold_min(&values);
        let b = backwards.fold_min(&values);
        for node in 0..4 {
            assert_eq!(
                a[forwards.find(node)],
                b[backwards.find(node)],
                "node {node} disagreed between two union orders"
            );
        }
        assert_eq!(
            a[forwards.find(0)],
            30,
            "the least reaches the whole component"
        );
        assert_eq!(
            a[forwards.find(3)],
            90,
            "and no further: a singleton is itself"
        );
    }

    /// The generic `per_block_of` and the `bool` `per_block` are one walk. If
    /// they ever stop agreeing, the flat numbering has two inverses.
    #[test]
    fn per_block_and_per_block_of_are_the_same_walk() {
        let mut reports: BTreeMap<[usize; 3], u32> = BTreeMap::new();
        reports.insert([0, 0, 0], 2);
        reports.insert([0, 0, 1], 3);
        let index = LabelIndex::build(&reports, [1, 1, 2], |&count| count).expect("an index");
        let mut sets = Union::new(index.total());
        sets.union(index.node([0, 0, 0], 1), index.node([0, 0, 1], 2));

        let flags: Vec<bool> = (0..index.total()).map(|node| node % 2 == 0).collect();
        let by_root = sets.fold_or(&flags);
        let bools = index.per_block(&mut sets, &by_root);
        let generic = index.per_block_of(&mut sets, &by_root);
        assert_eq!(bools, generic);

        // and the generic form carries a `u32`, which is what the label volume
        // wants and what the `bool` form cannot say
        let named: Vec<u32> = (0..index.total()).map(|node| node as u32 + 100).collect();
        let labels = index.per_block_of(&mut sets, &named);
        assert_eq!(labels[&[0, 0, 0]].len(), 2);
        assert_eq!(labels[&[0, 0, 1]].len(), 3);
    }

    #[test]
    fn a_union_is_by_size_and_folds_without_regard_to_order() {
        // The same three-element component built by two different sequences of
        // unions must fold to the same answer, which is the property `fold_or`
        // exists for.
        let mut forwards = Union::new(4);
        forwards.union(0, 1);
        forwards.union(1, 2);
        let mut backwards = Union::new(4);
        backwards.union(2, 1);
        backwards.union(1, 0);

        let flags = [false, false, true, false];
        let a = forwards.fold_or(&flags);
        let b = backwards.fold_or(&flags);
        for node in 0..4 {
            assert_eq!(
                a[forwards.find(node)],
                b[backwards.find(node)],
                "node {node} disagreed between two union orders"
            );
        }
        assert!(
            a[forwards.find(0)],
            "the flag must reach the whole component"
        );
        assert!(!a[forwards.find(3)], "and no further");
    }

    /// The one traversal, under the two predicates the ops built on it pass.
    ///
    /// A shell with a cavity: labelling the set voxels finds one component, and
    /// labelling the clear ones finds two — the outside and the cavity — which
    /// is the pair of answers `detect` and `fill` are respectively asking for.
    /// Also the two things the numbering promises: scan order, and reproducible.
    #[test]
    fn the_labelling_takes_its_membership_from_the_caller_and_numbers_in_scan_order() {
        let mut mask = Array3::from_elem((5, 5, 5), false);
        for i in 1..=3 {
            for j in 1..=3 {
                for k in 1..=3 {
                    mask[[i, j, k]] = !(i == 2 && j == 2 && k == 2);
                }
            }
        }
        let shape = [5usize, 5, 5];

        let mut set = Array3::<u32>::zeros(mask.raw_dim());
        assert_eq!(
            label_members_into(shape, |at| mask[at], set.view_mut()).unwrap(),
            1,
            "the shell is one component"
        );
        assert_eq!(set[[0, 0, 0]], UNLABELLED);
        assert_eq!(set[[1, 1, 1]], 1);

        let mut clear = Array3::<u32>::zeros(mask.raw_dim());
        assert_eq!(
            label_members_into(shape, |at| !mask[at], clear.view_mut()).unwrap(),
            2,
            "the outside and the cavity"
        );
        // Numbered in the order their lowest voxel is met: the outside first.
        assert_eq!(clear[[0, 0, 0]], 1);
        assert_eq!(clear[[2, 2, 2]], 2);
        assert_ne!(clear[[0, 0, 0]], clear[[2, 2, 2]]);

        // and the same input twice gives the same labels
        let mut again = Array3::<u32>::zeros(mask.raw_dim());
        label_members_into(shape, |at| !mask[at], again.view_mut()).unwrap();
        assert_eq!(clear, again);

        // a mismatched output shape is refused rather than half-filled
        let mut wrong = Array3::<u32>::zeros((4, 5, 5));
        assert!(label_members_into(shape, |_| true, wrong.view_mut()).is_err());
    }

    /// The tables are data, and data is where a typo hides silently. So they are
    /// checked against the definition rather than against themselves: the
    /// twenty-six offsets must be exactly `{-1, 0, 1}^3` without the origin,
    /// grouped by how many axes they step along, and each connectivity's prefix
    /// must be exactly the offsets it says it joins.
    #[test]
    fn the_offsets_are_the_whole_neighbourhood_grouped_by_steps() {
        let mut expected: Vec<[isize; 3]> = Vec::new();
        for i in [-1isize, 0, 1] {
            for j in [-1isize, 0, 1] {
                for k in [-1isize, 0, 1] {
                    if [i, j, k] != [0, 0, 0] {
                        expected.push([i, j, k]);
                    }
                }
            }
        }
        let mut sorted = NEIGHBOUR_OFFSETS.to_vec();
        sorted.sort();
        expected.sort();
        assert_eq!(sorted, expected, "the table is the 3x3x3 neighbourhood");

        // grouped, so that a prefix is a connectivity
        let steps: Vec<usize> = NEIGHBOUR_OFFSETS.iter().map(|&by| steps_of(by)).collect();
        assert!(steps.windows(2).all(|pair| pair[0] <= pair[1]));
        assert_eq!(steps.iter().filter(|&&n| n == 1).count(), 6);
        assert_eq!(steps.iter().filter(|&&n| n == 2).count(), 12);
        assert_eq!(steps.iter().filter(|&&n| n == 3).count(), 8);

        for connectivity in [
            Connectivity::Faces,
            Connectivity::FacesAndEdges,
            Connectivity::FacesEdgesAndCorners,
        ] {
            let joined: Vec<[isize; 3]> = NEIGHBOUR_OFFSETS
                .into_iter()
                .filter(|&by| connectivity.joins(by))
                .collect();
            assert_eq!(
                connectivity.offsets(),
                joined,
                "{connectivity:?}'s offsets are the ones it joins"
            );
            assert!(
                !connectivity.joins([0, 0, 0]),
                "a voxel is not its own pair"
            );
        }
        assert_eq!(Connectivity::Faces.offsets().len(), 6);
        assert_eq!(Connectivity::FacesAndEdges.offsets().len(), 18);
        assert_eq!(Connectivity::FacesEdgesAndCorners.offsets().len(), 26);

        // and the first six are `FACE_NEIGHBOURS`, which is what keeps a
        // face-connected traversal visiting neighbours in the order it did.
        for (slot, (axis, step)) in FACE_NEIGHBOURS.into_iter().enumerate() {
            let mut by = [0isize; 3];
            by[axis] = step;
            assert_eq!(NEIGHBOUR_OFFSETS[slot], by);
        }
    }

    /// Every lattice direction, checked to be one per pair and to cover every
    /// way one block can meet another.
    ///
    /// The cover is the load-bearing half: if a direction were missing, a
    /// component joined only across it would silently be two, and no fixture
    /// that happens not to touch there would notice.
    #[test]
    fn the_directions_are_every_block_meeting_taken_from_one_side() {
        for &towards in &FORWARD_DIRECTIONS {
            let first = towards.iter().find(|&&step| step != 0);
            assert_eq!(first, Some(&1), "{towards:?} is not taken from one side");
        }
        let mut seen: Vec<[isize; 3]> = Vec::new();
        for &towards in &FORWARD_DIRECTIONS {
            seen.push(towards);
            seen.push([-towards[0], -towards[1], -towards[2]]);
        }
        seen.sort();
        seen.dedup();
        let mut expected = NEIGHBOUR_OFFSETS.to_vec();
        expected.sort();
        assert_eq!(
            seen, expected,
            "the directions and their opposites are every neighbouring block"
        );

        // and a direction stepping along n axes needs a connectivity of at
        // least n, which is why the prefixes line up
        for connectivity in [
            Connectivity::Faces,
            Connectivity::FacesAndEdges,
            Connectivity::FacesEdgesAndCorners,
        ] {
            let reachable: Vec<[isize; 3]> = FORWARD_DIRECTIONS
                .into_iter()
                .filter(|&towards| steps_of(towards) <= connectivity.steps())
                .collect();
            assert_eq!(connectivity.directions(), reachable);
        }
        assert_eq!(Connectivity::Faces.directions().len(), 3);
        assert_eq!(Connectivity::FacesAndEdges.directions().len(), 9);
        assert_eq!(Connectivity::FacesEdgesAndCorners.directions().len(), 13);
    }

    /// The discriminating case, inside one block: two voxels that touch only at
    /// a corner, and two that touch only along an edge.
    ///
    /// A fixture whose parts touch face to face answers the same under all
    /// three and would pass with the connectivity ignored entirely, which is
    /// exactly why this one does not touch that way.
    #[test]
    fn a_corner_join_needs_twenty_six_and_an_edge_join_needs_eighteen() {
        let shape = [4usize, 4, 4];
        let corner = [[1usize, 1, 1], [2, 2, 2]];
        let edge = [[1usize, 1, 0], [2, 2, 0]];
        let face = [[1usize, 1, 1], [1, 1, 2]];

        let counts = |pair: [[usize; 3]; 2], connectivity| {
            let mut out = Array3::<u32>::zeros((4, 4, 4));
            label_members_into_with(
                shape,
                connectivity,
                |at| at == pair[0] || at == pair[1],
                out.view_mut(),
            )
            .unwrap()
        };

        assert_eq!(counts(corner, Connectivity::Faces), 2);
        assert_eq!(counts(corner, Connectivity::FacesAndEdges), 2);
        assert_eq!(counts(corner, Connectivity::FacesEdgesAndCorners), 1);

        assert_eq!(counts(edge, Connectivity::Faces), 2);
        assert_eq!(counts(edge, Connectivity::FacesAndEdges), 1);
        assert_eq!(counts(edge, Connectivity::FacesEdgesAndCorners), 1);

        // and the case that discriminates nothing, present to show it does not
        assert_eq!(counts(face, Connectivity::Faces), 1);
        assert_eq!(counts(face, Connectivity::FacesEdgesAndCorners), 1);

        // the default form is the face-connected one
        let mut bare = Array3::<u32>::zeros((4, 4, 4));
        assert_eq!(
            label_members_into(
                shape,
                |at| at == corner[0] || at == corner[1],
                bare.view_mut()
            )
            .unwrap(),
            2
        );
    }

    #[test]
    fn the_face_planes_of_a_block_are_its_six_boundary_slices() {
        let mut labels = Array3::<u32>::zeros((2, 3, 4));
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    labels[[i, j, k]] = (i * 12 + j * 4 + k) as u32 + 1;
                }
            }
        }
        let planes = planes_of(labels.view());
        assert_eq!(planes[face_index(0, 0)].0, [3, 4]);
        assert_eq!(planes[face_index(0, 0)].1[0], labels[[0, 0, 0]]);
        assert_eq!(planes[face_index(0, 1)].1[0], labels[[1, 0, 0]]);
        assert_eq!(planes[face_index(2, 1)].0, [2, 3]);
        assert_eq!(planes[face_index(2, 1)].1[0], labels[[0, 0, 3]]);
    }

    #[test]
    fn planes_round_trip_through_words_and_a_truncated_one_is_refused() {
        let labels = Array3::<u32>::from_shape_fn((3, 3, 3), |(i, j, k)| (i + j + k) as u32 + 1);
        let planes = planes_of(labels.view());
        let mut words = Vec::new();
        push_planes(&planes, &mut words);
        let mut at = 0usize;
        assert_eq!(
            take_planes(&words, &mut at, "a test fragment").unwrap(),
            planes
        );
        assert_eq!(at, words.len());

        let mut short = 0usize;
        assert!(take_planes(&words[..words.len() - 1], &mut short, "a test fragment").is_err());
    }

    #[test]
    fn a_header_from_another_op_is_refused() {
        let words = [0x1111_1111u32, 1, 5];
        assert_eq!(read_header(&words, 0x1111_1111, 1, "x").unwrap(), 5);
        assert!(read_header(&words, 0x2222_2222, 1, "x").is_err());
        assert!(read_header(&words, 0x1111_1111, 2, "x").is_err());
        assert!(read_header(&words[..2], 0x1111_1111, 1, "x").is_err());
    }

    #[test]
    fn a_lattice_with_a_block_missing_is_refused_and_the_numbering_is_dense() {
        let mut reports: BTreeMap<[usize; 3], u32> = BTreeMap::new();
        reports.insert([0, 0, 0], 2);
        assert!(LabelIndex::build(&reports, [2, 1, 1], |&count| count).is_err());
        reports.insert([1, 0, 0], 3);
        let index = LabelIndex::build(&reports, [2, 1, 1], |&count| count).unwrap();
        assert_eq!(index.total(), 5);
        assert_eq!(index.node([0, 0, 0], 1), 0);
        assert_eq!(index.node([0, 0, 0], 2), 1);
        assert_eq!(index.node([1, 0, 0], 1), 2);
        assert_eq!(index.node([1, 0, 0], 3), 4);
    }
}
