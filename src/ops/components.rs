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
// | 0 | `volume -> fragments`, and writes pixels | label the block's components locally at reach 0, write the labels as a `u32` level, emit the block's six face planes plus one fact per label |
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
//   fold of a per-label boolean onto its component's root.
//
// The extraction made `fill` shorter and changed none of its behaviour: its
// fragment type, its magic, its public functions and its error messages are
// where they were, and its tests are unchanged.
//
// Connectivity is six, here as well as there
// ------------------------------------------
// The face planes *are* the six-connectivity decision made structural. Under
// face connectivity a component crosses a block seam only through the shared
// plane, so a block's whole contribution to the merge is six planes. Under
// 26-connectivity it can also cross an edge or a corner, so a fragment would
// have to carry twelve edge lines and eight corner voxels as well, and the merge
// would walk three kinds of adjacency instead of one. That is a different
// fragment shape and a different merge, and it is not what this module is.
//
// Any op built on this is therefore six-connected, and says so in its own
// header rather than leaving a reader to infer it from here.

use std::collections::BTreeMap;

use ndarray::ArrayView3;

use crate::error::{Error, Result};
use crate::fragment::BlockView;

// ------------------------------------------------------------- geometry --

/// The label reserved for "this voxel belongs to no component".
///
/// Zero rather than a sentinel at the top of the range, so that a level
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

/// `at` moved by `step` along `axis`, or `None` if that leaves the array.
///
/// Clamped by refusing rather than by saturating: a neighbour outside the block
/// is *absent*, not a copy of the edge voxel, and the difference matters — a
/// saturating step would make every edge voxel its own neighbour and a plateau
/// its own higher neighbour.
pub fn offset(at: [usize; 3], axis: usize, step: isize, shape: [usize; 3]) -> Option<[usize; 3]> {
    let moved = at[axis] as isize + step;
    if moved < 0 || moved >= shape[axis] as isize {
        return None;
    }
    let mut to = at;
    to[axis] = moved as usize;
    Some(to)
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
        let mut out = BTreeMap::new();
        for &block in &self.order {
            let (base, labels) = self.span[&block];
            let mut flags = Vec::with_capacity(labels);
            for label in 0..labels {
                let root = sets.find(base + label);
                flags.push(per_root[root]);
            }
            out.insert(block, flags);
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
    mut meet: impl FnMut(usize, usize),
) -> Result<()> {
    for &block in index.order() {
        for axis in 0..3 {
            let mut ahead = block;
            ahead[axis] += 1;
            if ahead[axis] >= counts[axis] {
                continue;
            }
            let here = &planes_of(&reports[&block])[face_index(axis, 1)];
            let there = &planes_of(&reports[&ahead])[face_index(axis, 0)];
            if here.0 != there.0 {
                return Err(Error::InvalidArgument(format!(
                    "blocks {block:?} and {ahead:?} share a seam on axis {axis} but their \
                     faces are {:?} and {:?}. Two blocks adjacent on one axis have the same \
                     extent on the other two, so unequal faces mean the fragments came from \
                     two different lattices.",
                    here.0, there.0
                )));
            }
            for (&a, &b) in here.1.iter().zip(there.1.iter()) {
                if a == UNLABELLED || b == UNLABELLED {
                    continue;
                }
                meet(index.node(block, a), index.node(ahead, b));
            }
        }
    }
    Ok(())
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
