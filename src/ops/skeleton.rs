// SPDX-License-Identifier: MIT
//
// Original work for this crate. Written from the definitions the operation is
// stated in — border point, simple point, end point, subfield — and not adapted
// from any implementation of them.
//
// Binary skeletonization by **topological thinning**: delete voxels from the
// border of an object, over and over, while every deletion is one the topology
// of neither the object nor its background can notice, until what is left is
// thin.
//
// One sub-iteration is one op, and the sequence is exact
// ------------------------------------------------------
// The unit here is **one sub-iteration** ([`ThinningOp`]), not "thin until
// nothing changes". A `BlockOp` states what it reads, and the planner sizes
// halos from that; an op that iterated internally to a data-dependent fixed
// point would have to declare a reach it cannot know, which is the one
// declaration this crate treats as unforgivable. So the loop is lifted out of
// the op and into the chain: [`thinning_pass`] is the eight sub-iterations that
// make one full pass, [`thin`] is `n` of those, and the reach of the whole is
// the sum the `Chain` fold computes — `8n` — rather than a number written down
// beside it.
//
// **Why writing it as a plain sequence is exact rather than convenient.** Each
// sub-iteration reads *only the current volume*: the deletion decision at a
// voxel is a function of that voxel's 3x3x3 neighbourhood in the volume the
// sub-iteration was handed, and of nothing else. A `Chain::Sequence` hands each
// child what the previous child produced, which is precisely that. Contrast
// `super::deconvolve`: its step needs the current estimate **and the original
// observation**, so the same shape of loop is not a sequence at all — it is a
// fan-in per step, and writing it as a sequence would silently drop an operand.
// The difference is a property of the two algorithms, not a style choice, and it
// is why this one gets to be the cheap form.
//
// The price is stated rather than hidden: `n` passes reach `8n`, so a chain that
// thins a large object is a chain with a large halo, and the guard will refuse a
// decomposition that cannot afford it. That is the honest answer. A caller who
// holds the whole volume and wants the fixed point calls
// [`thin_to_fixed_point`], which is not a `BlockOp` and does not pretend to be.
//
// Why the deletion rule is subfield-restricted, and not directional
// -----------------------------------------------------------------
// The textbook 6-subiteration directional schemes delete, in sub-iteration `d`,
// every voxel that is a `d`-border point, is simple, and is not an end point.
// Taken literally as a **parallel** rule — every decision computed from the same
// input snapshot, which is what decomposition invariance requires — that rule
// **does not preserve topology**, and the counterexample is small enough to hold
// in the head:
//
// ```text
//   two columns joined by a flat two-voxel-wide bridge at the top.
//   every bridge voxel has background directly above it, so every one of them
//   is an "up"-border point; every one of them is simple (its neighbours in the
//   bridge stay connected without it); none is an end point. The sub-iteration
//   deletes the whole bridge at once and the object falls into two pieces.
// ```
//
// `the_naive_directional_rule_falls_apart_where_this_one_does_not` builds that
// object and measures it. The published directional algorithms avoid this by
// giving up one of the two things this crate cannot give up: Palagyi-Kuba
// replace "simple and not an end point" with a fixed list of deletion templates
// whose parallel safety is proved case by case, and Lee-Kashyap-Chu keep the
// predicate but **re-test each candidate against the partially updated volume**,
// which makes the answer a function of the scan order — the exact defect this
// crate exists to prevent, since two block decompositions scan in two orders.
//
// What is implemented instead is the **subfield** member of the same family
// (Bertrand-Aktouf; Saha and co-workers; Nemeth-Kardos-Palagyi), and it is
// chosen because its safety proof is three lines and is checkable rather than
// tabulated:
//
// 1. The lattice is partitioned into **eight classes by coordinate parity**
//    ([`Subfield`]). Sub-iteration `k` may delete only voxels of class `k`.
// 2. Two voxels of one class differ by at least two on some axis, so **no two
//    deleted voxels are ever 26-adjacent**.
// 3. A voxel's simplicity is a function of its 3x3x3 neighbourhood alone. If `p`
//    and `q` are both deleted they are not 26-adjacent, so `p` is not in `q`'s
//    neighbourhood: deleting `p` cannot change whether `q` is simple.
//
// So the parallel deletion of the whole class is identical to deleting its
// members one at a time **in any order whatever**, each step being the deletion
// of a simple point, and the deletion of a simple point preserves topology by
// definition. Parallel and sequential do not merely agree here; they are the
// same function, which is what makes this the member of the family a block
// framework can use. `a_class_can_be_deleted_one_at_a_time_in_any_order` asserts
// exactly that, by shuffling.
//
// The one thing this costs: the sub-iteration is **anchored to the global
// grid**, because a voxel's parity class is a fact about where it is in the
// volume rather than where it is in the block. That is what [`Anchor`] is for,
// and `super::local`'s sample lattice already sets the precedent — an op whose
// arithmetic re-anchored to the block would produce a complete, well-formed,
// wrong volume. `the_block_runs_agree_with_the_whole_volume_reference` is the
// test that would fail if the parity were taken from the buffer instead.
//
// The simple point test
// ---------------------
// `p` is **simple** in `X` when deleting it changes the topology of neither `X`
// nor its complement. That is a global statement with a purely local
// characterisation over the 26-neighbourhood, and the local form is what is
// implemented ([`is_simple_point`]):
//
// * `C*(p)` — the object voxels of the 26-neighbourhood, minus `p` itself, form
//   exactly **one** 26-connected component (connectivity taken inside the 3x3x3
//   box);
// * `Cbar(p)` — the background voxels of the 18-neighbourhood form exactly
//   **one** 6-connected component that is 6-adjacent to `p`.
//
// The convention is `(26, 6)`: 26-connectivity for the object, 6 for the
// background. It is the usual one for skeletons, and it is the one under which
// the object's continuous analogue is the union of its closed voxel cubes —
// which is what makes the Euler characteristic below computable exactly.
//
// **The characterisation is a theorem, so it is checked rather than trusted.**
// `the_simple_point_test_agrees_with_the_topology_it_stands_for` measures the
// three Betti numbers of thousands of neighbourhoods before and after deleting
// the centre, by a completely different route ([`betti_numbers`], which counts
// components and cells and knows nothing about `C*` or `Cbar`), and asserts the
// predicate agrees with the measurement in every case. A wrong predicate is the
// failure that would make every other property in this file vacuous, so it is
// the one test that does not take the implementation's word for anything.
//
// Note what is **not** a separate condition: "`p` is a border point" — some face
// neighbour of `p` is background — is *implied* by `Cbar(p) = 1`, since a voxel
// whose six face neighbours are all object has no background component adjacent
// to it at all and `Cbar` is zero. [`is_border_point`] exists as a cheap early
// exit and as something a reader can test against, not as an extra clause.
//
// The end point condition, and what it makes the answer
// -----------------------------------------------------
// Without a second condition, thinning by simple points alone shrinks a
// simply-connected object to a single voxel: topology is preserved all the way
// down and nothing about a point's shape is preserved at all. The condition that
// stops it is the **end point** rule, and which rule is chosen is what decides
// what kind of skeleton comes out. Here: a voxel with **at most one**
// 26-neighbour in the object is never deleted ([`is_end_point`]), which makes
// the result a **curve skeleton** — a set of one-voxel-wide arcs.
//
// A surface-preserving variant is not offered, and the reason is that it does
// *not* fall out of the same test. A surface end point is a genuinely different
// predicate over the same neighbourhood — one that recognises a voxel as lying
// in a one-voxel-thick sheet rather than at the end of an arc — and adding it
// would be adding a second algorithm with its own correctness argument, not a
// parameter. What does fall out for free is the observation that the *interior*
// of a one-voxel sheet is never simple anyway (deleting it opens a tunnel), so
// sheets survive until they are eroded from their rims; the end point rule is
// what decides where that erosion stops.
//
// Edge behaviour
// --------------
// **Outside the array handed in is background.** `super::morphology` clamps its
// element to the array and lets the clamp take the identity of the operation;
// the same question here has a sharper answer, because the object is a set and
// "there is nothing there" is what background means. Treating the outside as
// object instead would make every voxel on the volume's own face a non-border
// point, and an object that reached the volume face would never thin there.
//
// So at a real volume boundary this is right, and the whole-volume reference
// does the same thing. At a **block seam** it is deliberately wrong — a block's
// halo edge is not a volume edge, and a voxel there sees background that is not
// background — which is what turns a short halo into a loud failure instead of a
// silent one. It is not fixed at the seam, on purpose: that is the guard's job.
//
// The reach
// ---------
// **One, on every axis, and there is no field that sets it.** Every clause of
// the deletion rule — border, end point, simple — is a function of the 3x3x3
// neighbourhood and of nothing wider, so a sub-iteration reads one voxel beyond
// the one it writes. It is *tight*: `the_reach_of_one_is_tight` builds an object
// whose answer at a voxel changes when a voxel exactly one away changes, so the
// declaration is the dependency rather than a safe over-statement of it.
//
// A pass is eight sub-iterations, so `thinning_pass` reaches 8 and `thin(n)`
// reaches `8n`, and both of those come out of the `Chain` fold rather than out
// of arithmetic written here. [`thinning_reach`] states the same number a second
// time for a caller sizing a halo before it has a chain to ask, and the test
// asserts the two agree — the arrangement `super::background::background_reach`
// already uses.
//
// Costs
// -----
// Measured; see [`cost_report`], which is runnable and prints the table the
// constant at the bottom of this file was read off. The measurement has a
// caveat this op cannot make go away, and it is stated with the constant: the
// cost of a sub-iteration is **data-dependent** to about an order of magnitude,
// because the expensive test runs only at border voxels.

use ndarray::{Array3, ArrayView3, ArrayViewMut3};

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::op::{Anchor, BlockOp, Chain};
use crate::voxels::Voxels;

use super::shapes_agree;
use super::voxelwise::{from_set, is_set};

// ------------------------------------------------------ the neighbourhood --

/// Voxels in a 3x3x3 neighbourhood, the centre included.
pub const NEIGHBOURHOOD: usize = 27;

/// Where the centre sits in a gathered neighbourhood.
pub const CENTRE: usize = 13;

/// The index an offset in `-1..=1` on each axis occupies.
///
/// Row-major over `(i, j, k)`, so `CENTRE` is `(0, 0, 0)` and the arithmetic is
/// the same one `ndarray` does; nothing here depends on the order beyond its
/// being fixed, and everything that indexes a neighbourhood goes through this.
pub const fn neighbour_index(di: isize, dj: isize, dk: isize) -> usize {
    ((di + 1) * 9 + (dj + 1) * 3 + (dk + 1)) as usize
}

/// The offset an index stands for. The inverse of [`neighbour_index`].
pub const fn neighbour_offset(index: usize) -> [isize; 3] {
    [
        (index / 9) as isize - 1,
        ((index / 3) % 3) as isize - 1,
        (index % 3) as isize - 1,
    ]
}

/// The six face neighbours: the ones sharing a face with the centre.
pub const FACE_NEIGHBOURS: [usize; 6] = [
    neighbour_index(-1, 0, 0),
    neighbour_index(1, 0, 0),
    neighbour_index(0, -1, 0),
    neighbour_index(0, 1, 0),
    neighbour_index(0, 0, -1),
    neighbour_index(0, 0, 1),
];

/// Chebyshev distance between two neighbourhood positions: `1` exactly when they
/// are 26-adjacent.
fn chebyshev(a: usize, b: usize) -> isize {
    let (a, b) = (neighbour_offset(a), neighbour_offset(b));
    (0..3)
        .map(|axis| (a[axis] - b[axis]).abs())
        .max()
        .unwrap_or(0)
}

/// Manhattan distance between two neighbourhood positions: `1` exactly when they
/// are 6-adjacent.
fn manhattan(a: usize, b: usize) -> isize {
    let (a, b) = (neighbour_offset(a), neighbour_offset(b));
    (0..3).map(|axis| (a[axis] - b[axis]).abs()).sum()
}

/// A stack of neighbourhood indices that never allocates.
///
/// A flood fill inside a 3x3x3 box can never hold more than the box, and this
/// predicate runs once per border voxel of every block of every sub-iteration —
/// a `Vec` here is a heap allocation in the innermost loop of the op.
struct Frontier {
    items: [usize; NEIGHBOURHOOD],
    len: usize,
}

impl Frontier {
    fn new() -> Self {
        Self {
            items: [0; NEIGHBOURHOOD],
            len: 0,
        }
    }

    fn push(&mut self, index: usize) {
        self.items[self.len] = index;
        self.len += 1;
    }

    fn pop(&mut self) -> Option<usize> {
        (self.len > 0).then(|| {
            self.len -= 1;
            self.items[self.len]
        })
    }
}

// -------------------------------------------------------- the predicates --

/// The number of 26-connected components the **object** voxels of the
/// neighbourhood form once the centre is taken out, connectivity being taken
/// inside the 3x3x3 box.
///
/// This is `C*(p)` of the simple point characterisation. `n[CENTRE]` is ignored:
/// the count is about what surrounds the centre.
pub fn object_components(n: &[bool; NEIGHBOURHOOD]) -> usize {
    let mut seen = [false; NEIGHBOURHOOD];
    seen[CENTRE] = true;
    let mut count = 0;
    for start in 0..NEIGHBOURHOOD {
        if seen[start] || !n[start] {
            continue;
        }
        count += 1;
        seen[start] = true;
        let mut frontier = Frontier::new();
        frontier.push(start);
        while let Some(at) = frontier.pop() {
            for other in 0..NEIGHBOURHOOD {
                if seen[other] || !n[other] || chebyshev(at, other) != 1 {
                    continue;
                }
                seen[other] = true;
                frontier.push(other);
            }
        }
    }
    count
}

/// The number of 6-connected components the **background** voxels of the
/// 18-neighbourhood form that are 6-adjacent to the centre.
///
/// This is `Cbar(p)`. Two restrictions carry the whole meaning and neither is
/// decoration:
///
/// * the **18**-neighbourhood, not the 26: a corner voxel of the box is not
///   6-connected to anything else in it, so including corners would count
///   components that cannot reach the centre;
/// * only components **6-adjacent to the centre** — that is, containing one of
///   the six face neighbours — are counted. A background voxel walled off from
///   the centre by object voxels is not a piece of background the centre's
///   deletion would join anything to.
pub fn background_components(n: &[bool; NEIGHBOURHOOD]) -> usize {
    let mut member = [false; NEIGHBOURHOOD];
    for index in 0..NEIGHBOURHOOD {
        if index == CENTRE || n[index] {
            continue;
        }
        let offset = neighbour_offset(index);
        // the 18-neighbourhood: everything but the eight corners
        if offset[0].abs() + offset[1].abs() + offset[2].abs() <= 2 {
            member[index] = true;
        }
    }

    let mut seen = [false; NEIGHBOURHOOD];
    let mut count = 0;
    for &face in &FACE_NEIGHBOURS {
        if seen[face] || !member[face] {
            continue;
        }
        count += 1;
        seen[face] = true;
        let mut frontier = Frontier::new();
        frontier.push(face);
        while let Some(at) = frontier.pop() {
            for other in 0..NEIGHBOURHOOD {
                if seen[other] || !member[other] || manhattan(at, other) != 1 {
                    continue;
                }
                seen[other] = true;
                frontier.push(other);
            }
        }
    }
    count
}

/// Is the centre a **simple point**: would deleting it leave the topology of the
/// object and of the background as they were?
///
/// `C*(p) = 1 and Cbar(p) = 1`, which is the local characterisation of a global
/// property. The centre's own value is not consulted — the question only means
/// anything for an object voxel, and the callers here have already established
/// that.
pub fn is_simple_point(n: &[bool; NEIGHBOURHOOD]) -> bool {
    object_components(n) == 1 && background_components(n) == 1
}

/// Is the centre a **border point**: is at least one of its six face neighbours
/// background?
///
/// Implied by [`is_simple_point`] — a voxel with six object face neighbours has
/// `Cbar = 0` — so this is an early exit and a thing a reader can test against,
/// not a clause of the rule. `the_border_condition_is_implied_by_simplicity`
/// asserts the implication over every neighbourhood it is checked on.
pub fn is_border_point(n: &[bool; NEIGHBOURHOOD]) -> bool {
    FACE_NEIGHBOURS.iter().any(|&face| !n[face])
}

/// Is the centre an **end point**: does it have at most one 26-neighbour in the
/// object?
///
/// The condition that decides what kind of skeleton this is. An arc's last voxel
/// has one neighbour and an isolated voxel has none; both are kept, so arcs do
/// not retreat from their ends and specks do not vanish.
pub fn is_end_point(n: &[bool; NEIGHBOURHOOD]) -> bool {
    let mut neighbours = 0;
    for index in 0..NEIGHBOURHOOD {
        if index != CENTRE && n[index] {
            neighbours += 1;
            if neighbours > 1 {
                return false;
            }
        }
    }
    true
}

/// The whole deletion rule for one voxel, given its neighbourhood: an object
/// voxel, on the border, not an end point, and simple.
///
/// The subfield restriction is **not** here, because it is a fact about where
/// the voxel is rather than about what surrounds it; [`thin_subfield_into`]
/// applies it, and this stays a pure function of the neighbourhood so that it
/// can be checked against topology directly.
pub fn is_deletable(n: &[bool; NEIGHBOURHOOD]) -> bool {
    n[CENTRE] && is_border_point(n) && !is_end_point(n) && is_simple_point(n)
}

// ----------------------------------------------------------- the classes --

/// One of the eight parity classes of the lattice.
///
/// The class of a voxel is the parity of its **global** coordinates, which is
/// why every function that computes one takes a position in the volume rather
/// than in a buffer. Two voxels of one class differ by at least two on some
/// axis, so they are never 26-adjacent — the property the whole correctness
/// argument rests on, and the reason it is parity rather than any other
/// eight-way partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Subfield(u8);

impl Subfield {
    /// How many classes there are, which is also how many sub-iterations make a
    /// full pass: every voxel belongs to exactly one, so eight sub-iterations
    /// offer every voxel exactly one chance to be deleted.
    pub const COUNT: usize = 8;

    /// The class with this index, or an error naming the range.
    pub fn new(index: usize) -> Result<Self> {
        if index >= Self::COUNT {
            return Err(Error::InvalidArgument(format!(
                "subfield {index}: the lattice has {} parity classes, indexed 0..{}",
                Self::COUNT,
                Self::COUNT
            )));
        }
        Ok(Self(index as u8))
    }

    /// The class a **global** position belongs to.
    pub fn of(position: [usize; 3]) -> Self {
        Self((((position[0] & 1) << 2) | ((position[1] & 1) << 1) | (position[2] & 1)) as u8)
    }

    pub fn index(self) -> usize {
        self.0 as usize
    }

    pub fn contains(self, position: [usize; 3]) -> bool {
        Self::of(position) == self
    }
}

/// The order the eight classes are visited in, one full pass.
///
/// Every order is correct — each sub-iteration is topology-preserving on its
/// own, so any sequence of them is — and the order affects only *where* inside a
/// thick object the surviving arc ends up. This one alternates between a class
/// and its antipode (`k`, then `7 - k`), so consecutive sub-iterations erode
/// from opposite corners of the parity cube rather than repeatedly from the same
/// side, which keeps the result closer to the middle of the object than the
/// natural order `0..8` does.
pub const SWEEP: [Subfield; Subfield::COUNT] = [
    Subfield(0),
    Subfield(7),
    Subfield(1),
    Subfield(6),
    Subfield(2),
    Subfield(5),
    Subfield(3),
    Subfield(4),
];

/// The name each sub-iteration carries into a plan, a log and a progress
/// display. Fixed literals because [`BlockOp::name`] is `&'static str`, and
/// distinct so that a phase table listing eight of them says which is which.
const SUBFIELD_NAMES: [&str; Subfield::COUNT] = [
    "skeleton.thin.0",
    "skeleton.thin.1",
    "skeleton.thin.2",
    "skeleton.thin.3",
    "skeleton.thin.4",
    "skeleton.thin.5",
    "skeleton.thin.6",
    "skeleton.thin.7",
];

// ------------------------------------------------------------ the kernel --

/// One thinning sub-iteration: delete every voxel of `subfield` that the
/// deletion rule allows, computing **every** decision from `input`.
///
/// `origin` is where `input`'s lower corner sits in the volume, and it is what
/// makes the parity global; a caller holding the whole volume passes
/// `[0, 0, 0]`.
///
/// The purity is structural rather than disciplined: `out` is a different array
/// from `input`, nothing reads `out` back, and the loop never writes a voxel
/// twice. There is no version of this kernel that could accidentally consult a
/// half-updated volume, which is what the in-place textbook form does and what
/// makes its answer depend on the scan order.
pub fn thin_subfield_into(
    input: ArrayView3<'_, bool>,
    subfield: Subfield,
    origin: [usize; 3],
    mut out: ArrayViewMut3<'_, bool>,
) -> Result<()> {
    shapes_agree(input.shape(), out.shape(), "thin_subfield_into")?;
    let shape = [input.shape()[0], input.shape()[1], input.shape()[2]];
    // **Every voxel this pass does not delete comes out as it went in**, so the
    // output starts as a copy and the loop's only job is to find the deletions.
    // Written as a bulk assign rather than a voxel a time: it is the same bytes
    // by the same rule, moved by a memcpy instead of by `ndarray`'s indexing.
    out.assign(&input);
    // **The subfield is reached, not tested for.** A class is the parity of the
    // *global* position on each axis, so the buffer indices in it are an
    // arithmetic progression of stride two — one iteration in eight. The loop
    // used to visit all eight eighths and discard seven, and that discarding was
    // where the time went: with only a subfield's object voxels reaching
    // `is_deletable`, about **one iteration in a hundred** did any of the work
    // this function is for.
    //
    // `the_cost_of_the_border_early_exit` is what made that visible, by failing
    // to find anything: four different neighbourhood kernels measured the same,
    // because none of them was what the time was going to. Worth `1.12-1.19x`
    // against an A/A control of `0.95-0.98`, on both a filament and a blob.
    let first = |axis: usize| -> usize {
        let wanted = (subfield.index() >> (2 - axis)) & 1;
        wanted ^ (origin[axis] & 1)
    };
    let mut i = first(0);
    while i < shape[0] {
        let mut j = first(1);
        while j < shape[1] {
            let mut k = first(2);
            while k < shape[2] {
                if input[[i, j, k]] && is_deletable(&gather(input, [i, j, k])) {
                    out[[i, j, k]] = false;
                }
                k += 2;
            }
            j += 2;
        }
        i += 2;
    }
    Ok(())
}

/// The 3x3x3 neighbourhood around a position, with **outside the array read as
/// background**.
///
/// That convention is the operation's, not a fallback: see the header. It is
/// right at a real volume boundary and deliberately wrong at a block seam.
fn gather(input: ArrayView3<'_, bool>, at: [usize; 3]) -> [bool; NEIGHBOURHOOD] {
    let shape = [
        input.shape()[0] as isize,
        input.shape()[1] as isize,
        input.shape()[2] as isize,
    ];
    let mut n = [false; NEIGHBOURHOOD];
    for di in -1..=1isize {
        for dj in -1..=1isize {
            for dk in -1..=1isize {
                let position = [
                    at[0] as isize + di,
                    at[1] as isize + dj,
                    at[2] as isize + dk,
                ];
                if (0..3).all(|axis| position[axis] >= 0 && position[axis] < shape[axis]) {
                    n[neighbour_index(di, dj, dk)] = input[[
                        position[0] as usize,
                        position[1] as usize,
                        position[2] as usize,
                    ]];
                }
            }
        }
    }
    n
}

/// How many passes an iteration is allowed before it is declared broken.
///
/// **A guard, not a parameter.** This is the distinction the design record
/// settles: `thin(n)` states `n` as part of the *answer* — the caller is asking
/// for exactly `n` passes and gets them — whereas running to convergence has no
/// `n` in its definition at all, and any number attached to it is there only to
/// stop a non-terminating op from running forever with no diagnostic.
///
/// So exceeding it is an **error naming the op and the count**, never a silent
/// truncation. A truncated skeleton is a plausible, well-formed, wrong answer,
/// which is the failure mode this crate is arranged against; a loud one says
/// either the op does not converge or the limit was set below what this data
/// needs, and both are things a caller can act on.
///
/// Thinning itself provably terminates — every pass deletes at least one voxel
/// or is the last, and the object is finite — so for this op the limit is a
/// backstop against a future change rather than an expected outcome. It is
/// stated anyway, because "this one happens to terminate" is not a property the
/// caller can see.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PassLimit(usize);

impl PassLimit {
    /// Generous rather than tuned. A thinning pass peels one layer, so the
    /// passes a volume needs is about half the thickness of the thickest object
    /// in it — small for anything filamentous, and bounded above by half the
    /// volume's own shortest axis, since nothing thicker fits.
    pub fn of(passes: usize) -> Result<Self> {
        if passes == 0 {
            return Err(Error::InvalidArgument(
                "a pass limit of zero would refuse before doing anything; the limit is a \
                 backstop, and one that fires immediately is a limit that means \"do not run\""
                    .to_string(),
            ));
        }
        Ok(Self(passes))
    }

    /// The limit implied by a volume: half its shortest axis, rounded up, plus
    /// one. Nothing thicker than that fits in the volume, so no correct run can
    /// need more — which makes it a bound rather than a guess.
    pub fn for_volume(volume: [usize; 3]) -> Self {
        let shortest = volume.iter().copied().min().unwrap_or(1).max(1);
        Self(shortest / 2 + 2)
    }

    pub fn passes(self) -> usize {
        self.0
    }
}

/// Thin `input` until a full pass changes nothing, over the **whole volume**.
///
/// Returns the fixed point and how many passes it took. Not a [`BlockOp`] and
/// deliberately not dressed as one: the number of passes is a function of the
/// data, and an op that iterated until the data said stop could not state its
/// reach. A caller who has the whole volume in hand is the caller this is for;
/// a caller who does not builds [`thin`] with a stated number of passes and
/// lets the planner price it.
///
/// **It terminates.** Every pass either deletes at least one voxel or is the
/// last one, and the object is finite. `limit` is nonetheless required and is
/// nonetheless an error to exceed — see [`PassLimit`] for why a proof in this
/// file is not a substitute for a guard a caller can see.
pub fn thin_to_fixed_point(
    input: ArrayView3<'_, bool>,
    limit: PassLimit,
) -> Result<(Array3<bool>, usize)> {
    let mut current = input.to_owned();
    let mut passes = 0;
    loop {
        let mut next = current.clone();
        for &subfield in &SWEEP {
            let source = next.clone();
            thin_subfield_into(source.view(), subfield, [0, 0, 0], next.view_mut())?;
        }
        passes += 1;
        if next == current {
            return Ok((current, passes));
        }
        current = next;
        if passes >= limit.passes() {
            return Err(Error::InvalidArgument(format!(
                "thinning did not reach a fixed point in {} pass(es) over a {:?} volume. \
                 Either the limit is below what this data needs — raise it, or take it from \
                 `PassLimit::for_volume` — or a thinning pass has stopped being monotone, \
                 which would be a defect in the deletion rule rather than in the data. The \
                 partially thinned volume is deliberately not returned: it is a plausible, \
                 well-formed, wrong skeleton.",
                limit.passes(),
                [input.shape()[0], input.shape()[1], input.shape()[2]]
            )));
        }
    }
}

// ------------------------------------------------------------- the shell --

/// One thinning sub-iteration as a [`BlockOp`].
pub struct ThinningOp {
    name: &'static str,
    subfield: Subfield,
    cost: f64,
}

impl ThinningOp {
    pub fn new(name: &'static str, subfield: Subfield) -> Self {
        Self {
            name,
            subfield,
            cost: THINNING_COST,
        }
    }

    /// The sub-iteration for a class, under this module's own name for it.
    pub fn for_subfield(subfield: Subfield) -> Self {
        Self::new(SUBFIELD_NAMES[subfield.index()], subfield)
    }

    pub fn subfield(&self) -> Subfield {
        self.subfield
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }
}

impl BlockOp for ThinningOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// **One**, on every axis. The deletion rule reads the 3x3x3 neighbourhood
    /// and nothing wider, so this is the neighbourhood's own radius; there is no
    /// parameter it could be derived from and no field that sets it.
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        1
    }

    /// A mask, held as a mask or held as `f64`.
    ///
    /// `Bool` is what a binary volume is and what this op is for; `F64` is kept
    /// because a chain may carry a mask as `f64` under this module's
    /// `is_set`/`from_set` convention, and refusing it would break such a chain
    /// for no gain. The kernel is a `bool` kernel either way — the `f64` arm
    /// narrows on the way in and widens on the way out, which is the shell's
    /// work and not the kernel's.
    fn accepts(&self, dtype: Dtype) -> bool {
        matches!(dtype, Dtype::Bool | Dtype::F64)
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, at: &Anchor) -> Result<()> {
        match input.dtype() {
            Dtype::Bool => thin_subfield_into(
                input.view::<bool>()?,
                self.subfield,
                at.offset,
                out.view_mut::<bool>()?,
            ),
            _ => {
                let mask = input.view::<f64>()?.mapv(is_set);
                let mut result = Array3::from_elem(mask.raw_dim(), false);
                thin_subfield_into(mask.view(), self.subfield, at.offset, result.view_mut())?;
                let mut out = out.view_mut::<f64>()?;
                ndarray::Zip::from(&mut out)
                    .and(&result)
                    .for_each(|slot, &value| *slot = from_set(value));
                Ok(())
            }
        }
    }

    /// **Clear maps to clear, and set maps to nothing at all.**
    ///
    /// A volume with no object voxels has nothing to delete, so the answer is
    /// the input and the declaration is exact.
    ///
    /// An all-**set** block is the interesting half, and the answer is that
    /// nothing may be declared for it. Interior voxels of a solid block have no
    /// background face neighbour, so they are not border points and are never
    /// deleted; but the voxels on the *buffer's own faces* see the outside as
    /// background — this op's stated convention — and are deleted where their
    /// class comes up. The output of an all-set block is therefore not constant,
    /// and a short circuit that filled it with `1.0` would disagree with
    /// computing it at exactly the places the halo exists to get right. So the
    /// mapping is `None` there, the block is computed, and
    /// `an_all_set_block_is_not_constant_and_nothing_is_declared_for_it` asserts
    /// both halves: that the declaration is withheld, and that it had to be.
    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        (!is_set(value)).then_some(0.0)
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

// ---------------------------------------------------------- the assembly --

/// One full pass: the eight sub-iterations, in [`SWEEP`] order.
///
/// **A [`Chain::Sequence`] rather than one op, deliberately**, on the same
/// argument `super::background::background_estimate` makes: `Chain::slots`
/// flattens sequences, so a planner may cut between sub-iterations and give them
/// separate phases if the cost model prefers that. An op that ran all eight
/// internally would be one indivisible slot and would have to state its own
/// reach of 8; this way the 8 is the sum the fold computes.
pub fn thinning_pass() -> Chain {
    Chain::sequence(
        SWEEP
            .iter()
            .map(|&subfield| Chain::op(ThinningOp::for_subfield(subfield)))
            .collect(),
    )
}

/// `passes` full passes: `8 * passes` sub-iterations, reaching `8 * passes`.
///
/// Zero passes is the empty sequence, which `Chain::apply` defines as the
/// identity — the honest answer for that parameter rather than an error, in the
/// same way an element of one voxel opens nothing.
pub fn thin(passes: usize) -> Chain {
    Chain::sequence((0..passes).map(|_| thinning_pass()).collect())
}

/// What [`thin`] reads beyond the voxel it writes, per axis.
///
/// **The second statement of one quantity, and that is what it is for.** The
/// authority is `Chain::reach`, which folds the sequence and is what every plan
/// is built from; this derives the same number from the two facts it follows
/// from — one voxel per sub-iteration, [`Subfield::COUNT`] sub-iterations per
/// pass — so that a caller sizing a halo before it has a chain can ask, and the
/// test asserts the two agree.
pub fn thinning_reach(passes: usize) -> usize {
    passes * Subfield::COUNT
}

// ----------------------------------------------- measuring what it claims --

/// Which voxels are adjacent to which, for [`connected_components`].
///
/// The two that matter are the two the `(26, 6)` convention names: the object is
/// counted with 26-adjacency and the background with 6, and using one for both
/// is the classic way to count a connectivity paradox instead of a component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Adjacency {
    /// Sharing a face.
    Six,
    /// Sharing a face, an edge or a corner.
    TwentySix,
}

impl Adjacency {
    fn joins(self, step: [isize; 3]) -> bool {
        match self {
            Adjacency::Six => step.iter().map(|value| value.abs()).sum::<isize>() == 1,
            Adjacency::TwentySix => step != [0, 0, 0] && step.iter().all(|value| value.abs() <= 1),
        }
    }
}

/// How many connected components the voxels equal to `value` form.
pub fn connected_components(input: ArrayView3<'_, bool>, value: bool, how: Adjacency) -> usize {
    let shape = [input.shape()[0], input.shape()[1], input.shape()[2]];
    let mut seen = Array3::from_elem((shape[0], shape[1], shape[2]), false);
    let mut count = 0;
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                if seen[[i, j, k]] || input[[i, j, k]] != value {
                    continue;
                }
                count += 1;
                seen[[i, j, k]] = true;
                let mut frontier = vec![[i, j, k]];
                while let Some(at) = frontier.pop() {
                    for di in -1..=1isize {
                        for dj in -1..=1isize {
                            for dk in -1..=1isize {
                                if !how.joins([di, dj, dk]) {
                                    continue;
                                }
                                let next = [
                                    at[0] as isize + di,
                                    at[1] as isize + dj,
                                    at[2] as isize + dk,
                                ];
                                if (0..3).any(|axis| {
                                    next[axis] < 0 || next[axis] >= shape[axis] as isize
                                }) {
                                    continue;
                                }
                                let next = [next[0] as usize, next[1] as usize, next[2] as usize];
                                if seen[next] || input[next] != value {
                                    continue;
                                }
                                seen[next] = true;
                                frontier.push(next);
                            }
                        }
                    }
                }
            }
        }
    }
    count
}

/// The Euler characteristic of a binary volume under the `(26, 6)` convention.
///
/// **Computed from the definition, not from a table.** Under `(26, 6)` the
/// object's continuous analogue is the union of its **closed** unit voxel cubes:
/// two cubes that share only an edge or a corner still touch, which is
/// 26-connectivity, and the open complement is then separated at that edge,
/// which is 6-connectivity. So the object is a cubical complex and its Euler
/// characteristic is the alternating count of its cells,
///
/// ```text
///     chi = vertices - edges + faces - cubes
/// ```
///
/// where a cell is present when any voxel incident to it is object. Outside the
/// array is background, matching the operation's own convention, so the answer
/// is the one for the object as handed in.
pub fn euler_characteristic(input: ArrayView3<'_, bool>) -> i64 {
    let shape = [input.shape()[0], input.shape()[1], input.shape()[2]];
    let object = |i: isize, j: isize, k: isize| -> bool {
        i >= 0
            && j >= 0
            && k >= 0
            && i < shape[0] as isize
            && j < shape[1] as isize
            && k < shape[2] as isize
            && input[[i as usize, j as usize, k as usize]]
    };
    // Any voxel incident to the cell at a corner-space position.
    let any = |lo: [isize; 3], hi: [isize; 3]| -> bool {
        for i in lo[0]..=hi[0] {
            for j in lo[1]..=hi[1] {
                for k in lo[2]..=hi[2] {
                    if object(i, j, k) {
                        return true;
                    }
                }
            }
        }
        false
    };

    let (mut vertices, mut edges, mut faces, mut cubes) = (0i64, 0i64, 0i64, 0i64);
    for i in 0..=shape[0] as isize {
        for j in 0..=shape[1] as isize {
            for k in 0..=shape[2] as isize {
                // the lattice vertex at (i, j, k): eight incident voxels
                if any([i - 1, j - 1, k - 1], [i, j, k]) {
                    vertices += 1;
                }
                // the three edges leaving it in the positive direction: four
                // incident voxels each
                for axis in 0..3 {
                    let mut lo = [i - 1, j - 1, k - 1];
                    let mut hi = [i, j, k];
                    lo[axis] = [i, j, k][axis];
                    hi[axis] = [i, j, k][axis];
                    if hi[axis] < shape[axis] as isize && any(lo, hi) {
                        edges += 1;
                    }
                }
                // the three faces spanning from it in the positive directions:
                // two incident voxels each, the pair sharing that face
                for axis in 0..3 {
                    let mut lo = [i, j, k];
                    let hi = [i, j, k];
                    lo[axis] -= 1;
                    let spans = (0..3)
                        .all(|other| other == axis || [i, j, k][other] < shape[other] as isize);
                    if spans && any(lo, hi) {
                        faces += 1;
                    }
                }
                // the cube whose lower corner this is
                if object(i, j, k) {
                    cubes += 1;
                }
            }
        }
    }
    vertices - edges + faces - cubes
}

/// The three Betti numbers of a binary volume under the `(26, 6)` convention:
/// components, tunnels, cavities.
///
/// **Why this exists in this file.** The central claim of a thinning operation
/// is "topology is preserved", and a claim nothing can measure is not a claim.
/// This is the measurement, by a route that shares no code and no idea with
/// [`is_simple_point`]: component counts and a cell count. The tests use it on
/// the predicate itself, and on whole shapes before and after a pass.
///
/// The volume is padded with one voxel of background on every side first, which
/// is this operation's convention for the outside anyway, and which is what
/// makes the second line below true.
///
/// * `b0` is the number of 26-connected object components;
/// * `b2` is the number of 6-connected background components **less one**, the
///   one being the unbounded outside; the rest are cavities;
/// * `b1` follows from `chi = b0 - b1 + b2`, which is the only way to see a
///   tunnel: a tunnel changes neither component count, and a test that watched
///   only those two would pass a pass that cut a ring open.
pub fn betti_numbers(input: ArrayView3<'_, bool>) -> [usize; 3] {
    let shape = [input.shape()[0], input.shape()[1], input.shape()[2]];
    let mut padded = Array3::from_elem((shape[0] + 2, shape[1] + 2, shape[2] + 2), false);
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                padded[[i + 1, j + 1, k + 1]] = input[[i, j, k]];
            }
        }
    }
    let components = connected_components(padded.view(), true, Adjacency::TwentySix);
    let cavities = connected_components(padded.view(), false, Adjacency::Six) - 1;
    let chi = euler_characteristic(padded.view());
    let tunnels = components as i64 + cavities as i64 - chi;
    [
        components,
        usize::try_from(tunnels).expect(
            "the component counts and the Euler characteristic are one arithmetic; a negative \
             tunnel count means one of the three is wrong",
        ),
        cavities,
    ]
}

// ---------------------------------------------------------------- costs --

/// Measured; see [`cost_report`], and `super::COST_MEASUREMENT` for the method.
/// Relative to the voxelwise map, which is this module's unit of work.
///
/// **The caveat is part of the number, and it is a limit of the trait rather
/// than of this op.** A sub-iteration runs the expensive test — two flood fills
/// in a 3x3x3 box — only where a voxel is an object voxel, of the current class,
/// **and on the border**. The border's size is a property of the *data*, so the
/// per-voxel cost is too: over the two inputs in [`cost_report`], one built of
/// solid blocks and one a speckle, it differs by a factor of two (25.5 against
/// 51.0 nanoseconds per voxel, both stable to a few percent across runs).
///
/// `cost_per_voxel` takes no argument the data could arrive through, and
/// `Decomposition` is data-blind by design — two datasets must not produce two
/// plans — so there is nowhere for the honest answer to go. The stored figure is
/// therefore the **expensive** input's, on the argument that a schedule chosen
/// with an op over-priced costs some redundancy while one chosen with an op
/// under-priced fuses something it should have cut.
///
/// **The spread is stated because the ratio is noisier than the op is.** Four
/// runs gave 12.2, 15.2, 16.5 and 11.5; `14.0` is stored. The thinning column
/// itself varied by under three percent across those runs — the movement is all
/// in the *denominator*, since the voxelwise map costs three to four nanoseconds
/// a voxel and at that size a ratio inherits the noise of memory bandwidth. What
/// the planner needs from this number is that a sub-iteration is an expensive
/// neighbourhood pass rather than a cheap one, and every run says that by an
/// order of magnitude.
pub const THINNING_COST: f64 = 14.0;

/// The measurement the constant above came from, kept as text so that a re-run
/// somewhere else can be **compared** against it rather than merely replacing
/// it. `--release`, one thread, 96 x 64 x 64, best of 5, on the machine this was
/// written on:
///
/// ```text
/// case                                                       ns/voxel   relative
/// voxelwise map (the unit)                                      4.215       1.00
/// thinning sub-iteration, solid blocks                          26.11       6.20
/// thinning sub-iteration, speckle                               51.42      12.20
/// ```
///
/// The two thinning rows moved by under three percent over four runs and the
/// unit moved between 3.1 and 4.4, which is why the doc comment on
/// [`THINNING_COST`] gives the range of the ratio rather than one figure.
pub const COST_MEASUREMENT: &str = "ops::skeleton::cost_report";

/// Retake the measurement, through the same `BlockOp::apply` the executor calls.
/// Runnable; `print_the_cost_table` below is the one command.
///
/// Two inputs, because one number would hide the thing worth knowing: a mask
/// built of solid blocks, most of whose voxels are interior and cost almost
/// nothing, and a speckle, nearly all of whose voxels are on a border and pay
/// the full test. The unit is the voxelwise map, as in `super::cost` and
/// `super::ridge::cost_report`.
pub fn cost_report(shape: [usize; 3], repetitions: usize) -> String {
    use std::time::Instant;

    let voxels = (shape[0] * shape[1] * shape[2]) as f64;
    let anchor = Anchor::whole(shape);
    let repetitions = repetitions.max(1);

    let best_of = |mut run: Box<dyn FnMut()>| -> f64 {
        // One untimed pass first: a freshly allocated output pays a page fault
        // per page on first touch, and that fault is the measurement for the
        // cheapest op here.
        run();
        let mut best = f64::INFINITY;
        for _ in 0..repetitions {
            let started = Instant::now();
            run();
            best = best.min(started.elapsed().as_secs_f64() * 1e9 / voxels);
        }
        best
    };

    let mut rows: Vec<(String, f64)> = Vec::new();

    {
        let mut ramp = Array3::<f64>::zeros((shape[0], shape[1], shape[2]));
        for (flat, value) in ramp.iter_mut().enumerate() {
            *value = ((flat * 7919) % 1013) as f64;
        }
        let input: Voxels = ramp.into();
        let op = super::voxelwise::VoxelwiseMapOp::threshold("map", 500.0, 1.0, 0.0);
        let mut out = Voxels::zeros(Dtype::F64, shape).unwrap();
        rows.push((
            "voxelwise map (the unit)".to_string(),
            best_of(Box::new(move || {
                op.apply(&input, &mut out, &anchor).unwrap();
            })),
        ));
    }

    for (what, mask) in [
        (
            "thinning sub-iteration, solid blocks (little of it on a border)",
            Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
                (i / 8 + j / 8 + k / 8) % 4 != 0
            }),
        ),
        (
            "thinning sub-iteration, speckle (nearly all of it on a border)",
            Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
                (i * 31 + j * 17 + k * 7) % 5 < 2
            }),
        ),
    ] {
        let input: Voxels = mask.into();
        let op = ThinningOp::for_subfield(SWEEP[0]);
        let mut out = Voxels::zeros(Dtype::Bool, shape).unwrap();
        let anchor = Anchor::whole(shape);
        rows.push((
            what.to_string(),
            best_of(Box::new(move || {
                op.apply(&input, &mut out, &anchor).unwrap();
            })),
        ));
    }

    let unit = rows.first().map(|(_, nanos)| *nanos).unwrap_or(1.0);
    let mut out = format!(
        "skeleton cost, {}x{}x{}, best of {repetitions}\n{:<56} {:>10} {:>10} {:>10}\n",
        shape[0], shape[1], shape[2], "case", "ns/voxel", "relative", "stored"
    );
    for (name, nanos) in &rows {
        out.push_str(&format!(
            "{name:<56} {nanos:>10.3} {:>10.2} {THINNING_COST:>10.2}\n",
            nanos / unit
        ));
    }
    out
}

#[cfg(test)]
mod tests {

    /// Is the centre a border point, asked of the **array** rather than of a
    /// gathered neighbourhood?
    ///
    /// [`is_border_point`] is the same question and is already the first clause
    /// [`is_deletable`] evaluates — but it is asked of a `[bool; 27]` that
    /// [`gather`] has to fill first, so the twenty-seven reads are paid before the
    /// six that decide the answer. Asking the array directly is what turns a
    /// documented "early exit" into one.
    ///
    /// **The rewrite is an identity, not an approximation.** `is_deletable` is
    /// `n[CENTRE] && is_border_point(n) && ...`, so a voxel that is not a border
    /// point is not deletable, and the caller writes `!is_deletable(..) == true`
    /// for it — which is exactly what the arm that skips the test writes, because
    /// the voxel is in the object. No neighbourhood answers differently; there is
    /// no case to check for.
    ///
    /// Outside the array reads as background, which is [`gather`]'s convention and
    /// the operation's: a voxel against a face of the buffer is on a border by
    /// that alone, and gets the answer without touching memory.
    fn on_a_border(input: ArrayView3<'_, bool>, at: [usize; 3], shape: [usize; 3]) -> bool {
        for axis in 0..3 {
            if at[axis] == 0 || at[axis] + 1 == shape[axis] {
                return true;
            }
        }
        let mut position = at;
        for axis in 0..3 {
            for step in [usize::wrapping_sub(0, 1), 1] {
                position[axis] = at[axis].wrapping_add(step);
                if !input[position] {
                    return true;
                }
            }
            position[axis] = at[axis];
        }
        false
    }

    // --------------------------------------------- the cost of the early exit --

    /// A deterministic binary volume with a stated **thickness**.
    ///
    /// Thickness is what decides this measurement, because the border test's
    /// whole value is skipping voxels that are not on a border — and a thin tube
    /// has almost none of those. `radius` 1 is a filament a voxel or two across,
    /// where nearly nine in ten object voxels are on a border; `radius` 6 packs
    /// most of them inside, where under four in ten are. Both are run, because a benchmark on one of them would
    /// recommend the opposite of what a benchmark on the other does.
    fn blobs(shape: [usize; 3], seeds: usize, radius: isize) -> Array3<bool> {
        let mut out = Array3::from_elem((shape[0], shape[1], shape[2]), false);
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..seeds {
            let centre = [
                (next() as usize) % shape[0],
                (next() as usize) % shape[1],
                (next() as usize) % shape[2],
            ];
            for di in -radius..=radius {
                for dj in -radius..=radius {
                    for dk in -radius..=radius {
                        if di * di + dj * dj + dk * dk > radius * radius {
                            continue;
                        }
                        let at = [
                            centre[0] as isize + di,
                            centre[1] as isize + dj,
                            centre[2] as isize + dk,
                        ];
                        if (0..3).all(|axis| at[axis] >= 0 && (at[axis] as usize) < shape[axis]) {
                            out[[at[0] as usize, at[1] as usize, at[2] as usize]] = true;
                        }
                    }
                }
            }
        }
        out
    }

    /// [`gather`] without the per-neighbour bounds test, for a centre whose whole
    /// neighbourhood is inside the buffer.
    ///
    /// The shipped gather asks, for each of twenty-seven neighbours, whether each
    /// of three coordinates is in range, and then indexes through `ndarray`. For
    /// an interior voxel every one of those answers is yes, and the buffer is one
    /// contiguous run, so the whole neighbourhood is twenty-seven loads at fixed
    /// offsets from one base.
    fn gather_flat(values: &[bool], strides: [isize; 3], base: isize) -> [bool; NEIGHBOURHOOD] {
        let mut n = [false; NEIGHBOURHOOD];
        for di in -1..=1isize {
            for dj in -1..=1isize {
                for dk in -1..=1isize {
                    let at = base + di * strides[0] + dj * strides[1] + dk * strides[2];
                    n[neighbour_index(di, dj, dk)] = values[at as usize];
                }
            }
        }
        n
    }

    /// The arms, over one volume, **interleaved and repeated**.
    ///
    /// `gathering` is the kernel as it stood before the border test was hoisted:
    /// gather twenty-seven, then let `is_deletable` decide. The others are the
    /// two changes and their combination, and the last is `gather always` a
    /// second time — the **A/A control**, which is the whole reason this is
    /// shaped the way it is. Run once each in order, the first arm pays for
    /// bringing the volume into cache and every later arm looks faster than it
    /// is; the first version of this measurement reported a `1.84x` that was
    /// entirely that. Rounds are interleaved and the best of each arm is kept,
    /// so a difference has to survive being measured in every position.
    fn arms(input: &Array3<bool>, subfield: Subfield) -> Vec<(&'static str, f64, Array3<bool>)> {
        let shape = [input.shape()[0], input.shape()[1], input.shape()[2]];
        let view = input.view();
        let values = input.as_slice().expect("a standard-layout volume");
        let strides = [(shape[1] * shape[2]) as isize, shape[2] as isize, 1];
        let interior =
            |at: [usize; 3]| (0..3).all(|axis| at[axis] > 0 && at[axis] + 1 < shape[axis]);
        let kernels: Vec<(&'static str, Box<dyn Fn([usize; 3], isize) -> bool>)> = vec![
            (
                "gather always",
                Box::new(move |at, _| !is_deletable(&gather(view, at))),
            ),
            (
                "border, then gather",
                Box::new(move |at, _| {
                    !(on_a_border(view, at, shape) && is_deletable(&gather(view, at)))
                }),
            ),
            (
                "flat gather",
                Box::new(move |at, base| {
                    let n = if interior(at) {
                        gather_flat(values, strides, base)
                    } else {
                        gather(view, at)
                    };
                    !is_deletable(&n)
                }),
            ),
            (
                "border, then flat gather",
                Box::new(move |at, base| {
                    if !on_a_border(view, at, shape) {
                        return true;
                    }
                    let n = if interior(at) {
                        gather_flat(values, strides, base)
                    } else {
                        gather(view, at)
                    };
                    !is_deletable(&n)
                }),
            ),
            (
                "gather always (A/A)",
                Box::new(move |at, _| !is_deletable(&gather(view, at))),
            ),
        ];
        let once = |kernel: &dyn Fn([usize; 3], isize) -> bool| {
            let mut out = Array3::from_elem((shape[0], shape[1], shape[2]), false);
            let started = std::time::Instant::now();
            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    for k in 0..shape[2] {
                        let base = i as isize * strides[0] + j as isize * strides[1] + k as isize;
                        out[[i, j, k]] = if input[[i, j, k]] && subfield.contains([i, j, k]) {
                            kernel([i, j, k], base)
                        } else {
                            input[[i, j, k]]
                        };
                    }
                }
            }
            (started.elapsed().as_secs_f64(), out)
        };
        // A warm-up round, discarded: the first pass over the volume is paying
        // for the volume, not for the kernel.
        for (_, kernel) in &kernels {
            once(kernel.as_ref());
        }
        let mut best: Vec<(&'static str, f64, Array3<bool>)> = Vec::new();
        for _ in 0..7 {
            for (slot, (name, kernel)) in kernels.iter().enumerate() {
                let (seconds, out) = once(kernel.as_ref());
                match best.get_mut(slot) {
                    None => best.push((name, seconds, out)),
                    Some(held) => {
                        if seconds < held.1 {
                            held.1 = seconds;
                        }
                    }
                }
            }
        }
        best
    }

    /// The sub-iteration as it stood: **every** voxel visited, the subfield
    /// tested for, and the output written a voxel at a time.
    fn thin_subfield_visiting_all(
        input: ArrayView3<'_, bool>,
        subfield: Subfield,
        origin: [usize; 3],
        mut out: ArrayViewMut3<'_, bool>,
    ) {
        let shape = [input.shape()[0], input.shape()[1], input.shape()[2]];
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    out[[i, j, k]] = if input[[i, j, k]]
                        && subfield.contains([origin[0] + i, origin[1] + j, origin[2] + k])
                    {
                        !is_deletable(&gather(input, [i, j, k]))
                    } else {
                        input[[i, j, k]]
                    };
                }
            }
        }
    }

    /// The strided loop **without** the border early-exit, to price the exit
    /// itself now that it is no longer hidden under a loop a hundred times its
    /// size.
    fn thin_subfield_strided_with_border_exit(
        input: ArrayView3<'_, bool>,
        subfield: Subfield,
        origin: [usize; 3],
        mut out: ArrayViewMut3<'_, bool>,
    ) {
        let shape = [input.shape()[0], input.shape()[1], input.shape()[2]];
        out.assign(&input);
        let first = |axis: usize| -> usize {
            let wanted = (subfield.index() >> (2 - axis)) & 1;
            wanted ^ (origin[axis] & 1)
        };
        let mut i = first(0);
        while i < shape[0] {
            let mut j = first(1);
            while j < shape[1] {
                let mut k = first(2);
                while k < shape[2] {
                    if input[[i, j, k]]
                        && on_a_border(input, [i, j, k], shape)
                        && is_deletable(&gather(input, [i, j, k]))
                    {
                        out[[i, j, k]] = false;
                    }
                    k += 2;
                }
                j += 2;
            }
            i += 2;
        }
    }

    /// **What reaching the subfield is worth against testing for it.**
    ///
    /// Same shape of measurement as `the_cost_of_the_border_early_exit`, and for
    /// the same reason: interleaved rounds, a discarded warm-up, and the old
    /// kernel run twice so an A/A control says what a difference has to beat.
    /// The answers are asserted identical before any time is reported.
    #[test]
    #[ignore = "a measurement, not an assertion"]
    fn the_cost_of_visiting_every_voxel() {
        println!();
        for (what, radius, seeds) in [("filament", 1isize, 9000usize), ("blob", 6, 120)] {
            let input = blobs([96, 96, 96], seeds, radius);
            let shape = (96, 96, 96);
            let subfield = Subfield::of([0, 0, 0]);
            let mut old = Array3::from_elem(shape, false);
            let mut new = Array3::from_elem(shape, false);
            thin_subfield_visiting_all(input.view(), subfield, [0, 0, 0], old.view_mut());
            thin_subfield_into(input.view(), subfield, [0, 0, 0], new.view_mut()).unwrap();
            assert_eq!(old, new, "{what}: the two loops thin differently");

            let mut times = [f64::MAX; 4];
            for round in 0..8 {
                let mut scratch = Array3::from_elem(shape, false);
                let started = std::time::Instant::now();
                thin_subfield_visiting_all(input.view(), subfield, [0, 0, 0], scratch.view_mut());
                let all = started.elapsed().as_secs_f64();
                let started = std::time::Instant::now();
                thin_subfield_into(input.view(), subfield, [0, 0, 0], scratch.view_mut()).unwrap();
                let strided = started.elapsed().as_secs_f64();
                let started = std::time::Instant::now();
                thin_subfield_strided_with_border_exit(
                    input.view(),
                    subfield,
                    [0, 0, 0],
                    scratch.view_mut(),
                );
                let bare = started.elapsed().as_secs_f64();
                let started = std::time::Instant::now();
                thin_subfield_visiting_all(input.view(), subfield, [0, 0, 0], scratch.view_mut());
                let control = started.elapsed().as_secs_f64();
                if round > 0 {
                    times[0] = times[0].min(all);
                    times[1] = times[1].min(strided);
                    times[2] = times[2].min(control);
                    times[3] = times[3].min(bare);
                }
            }
            println!("  {what}:");
            for (name, seconds) in [
                ("every voxel, parity tested", times[0]),
                ("the subfield, reached", times[1]),
                ("the subfield, border exit", times[3]),
                ("every voxel (A/A)", times[2]),
            ] {
                println!(
                    "    {name:>28}  {seconds:>8.4} s  {:>6.2}x",
                    times[0] / seconds
                );
            }
        }
    }

    /// **What the border early-exit costs and buys, against a straight gather.**
    ///
    /// An early exit is a branch, and a branch whose outcome follows the data is
    /// worse than a longer straight line when it mispredicts. Whether it does
    /// depends on the object: on a filament nearly every object voxel is on a
    /// border, so the test is taken almost always and buys nothing; in a blob
    /// most are interior, so it is taken almost never and skips twenty-seven
    /// reads. Both are measured, and the answers of all four arms are asserted
    /// equal first — a faster kernel that thins differently is not a faster
    /// kernel.
    #[test]
    #[ignore = "a measurement, not an assertion"]
    fn the_cost_of_the_border_early_exit() {
        println!();
        for (what, radius, seeds) in [("filament", 1isize, 9000usize), ("blob", 6, 120)] {
            let input = blobs([96, 96, 96], seeds, radius);
            let object = input.iter().filter(|value| **value).count();
            let border = {
                let shape = [96, 96, 96];
                input
                    .indexed_iter()
                    .filter(|((i, j, k), value)| {
                        **value && on_a_border(input.view(), [*i, *j, *k], shape)
                    })
                    .count()
            };
            let measured = arms(&input, Subfield::of([0, 0, 0]));
            for arm in &measured[1..] {
                assert_eq!(arm.2, measured[0].2, "{} thins differently", arm.0);
            }
            println!(
                "  {what}: {object} object voxels, {border} on a border ({:.0}%)",
                100.0 * border as f64 / object.max(1) as f64
            );
            for (name, seconds, _) in &measured {
                println!(
                    "    {name:>24}  {seconds:>8.4} s  {:>6.2}x",
                    measured[0].1 / seconds
                );
            }
        }
    }
    /// A limit no test shape can reach, so a test that hits it has found a real
    /// failure to converge rather than a limit set too low.
    fn generous() -> PassLimit {
        PassLimit::of(1000).unwrap()
    }

    use super::*;

    use crate::decomposition::{Decomposition, PhaseDecomposition};
    use crate::env::ArrayEnvironment;
    use crate::geometry::BlockGrid;
    use crate::strategy::{execute, Hints, Workflow};

    // ------------------------------------------------------- fixtures --

    /// A deterministic bit source. Not a good generator and does not need to be:
    /// what the tests want is a reproducible walk through the configuration
    /// space, and a seed that names the walk.
    struct Bits(u64);

    impl Bits {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }

        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            self.0 >> 11
        }

        fn chance(&mut self, in_percent: u64) -> bool {
            self.next() % 100 < in_percent
        }

        fn below(&mut self, limit: usize) -> usize {
            (self.next() % limit as u64) as usize
        }
    }

    fn empty(shape: [usize; 3]) -> Array3<bool> {
        Array3::from_elem((shape[0], shape[1], shape[2]), false)
    }

    fn solid_box(shape: [usize; 3], lo: [usize; 3], hi: [usize; 3]) -> Array3<bool> {
        let mut out = empty(shape);
        for i in lo[0]..hi[0] {
            for j in lo[1]..hi[1] {
                for k in lo[2]..hi[2] {
                    out[[i, j, k]] = true;
                }
            }
        }
        out
    }

    /// Apply one sub-iteration over the whole volume.
    fn sub_iteration(input: &Array3<bool>, subfield: Subfield) -> Array3<bool> {
        let mut out = Array3::from_elem(input.raw_dim(), false);
        thin_subfield_into(input.view(), subfield, [0, 0, 0], out.view_mut()).unwrap();
        out
    }

    /// Apply one full pass over the whole volume.
    fn pass(input: &Array3<bool>) -> Array3<bool> {
        let mut current = input.clone();
        for &subfield in &SWEEP {
            current = sub_iteration(&current, subfield);
        }
        current
    }

    /// Every shape the topology tests are held to, and why each is here.
    fn shapes() -> Vec<(&'static str, Array3<bool>)> {
        let volume = [14, 14, 14];

        // a solid box: the base case, one component and nothing else
        let solid = solid_box(volume, [3, 3, 3], [11, 11, 11]);

        // a box with a tunnel bored through it: one component, one tunnel, no
        // cavity. The only shape here whose invariant lives in `b1`, and the
        // reason `betti_numbers` computes an Euler characteristic at all.
        let mut tunnel = solid.clone();
        for i in 3..11 {
            tunnel[[i, 7, 7]] = false;
            tunnel[[i, 7, 6]] = false;
            tunnel[[i, 6, 7]] = false;
            tunnel[[i, 6, 6]] = false;
        }

        // a box with a sealed cavity: one component, one cavity, no tunnel
        let mut cavity = solid.clone();
        for i in 6..8 {
            for j in 6..8 {
                for k in 6..8 {
                    cavity[[i, j, k]] = false;
                }
            }
        }

        // two blobs meeting at a corner: one 26-connected component, and the
        // shape a wrong simple point test is most likely to break in two
        let mut touching = solid_box(volume, [2, 2, 2], [7, 7, 7]);
        for i in 7..12 {
            for j in 7..12 {
                for k in 7..12 {
                    touching[[i, j, k]] = true;
                }
            }
        }

        // a hollow shell: one component, one cavity, no tunnel, and every voxel
        // of it is a border voxel
        let mut shell = solid_box(volume, [2, 2, 2], [12, 12, 12]);
        for i in 4..10 {
            for j in 4..10 {
                for k in 4..10 {
                    shell[[i, j, k]] = false;
                }
            }
        }

        // a one-voxel arc, which must not move at all
        let mut arc = empty(volume);
        for k in 3..11 {
            arc[[7, 7, k]] = true;
        }

        // one voxel
        let mut speck = empty(volume);
        speck[[7, 7, 7]] = true;

        // a speckle, which is not a shape anybody draws but is the input most
        // likely to contain a neighbourhood nobody thought of
        let mut speckle = empty(volume);
        let mut bits = Bits::new(20260806);
        for value in speckle.iter_mut() {
            *value = bits.chance(45);
        }

        vec![
            ("solid box", solid),
            ("box with a tunnel", tunnel),
            ("box with a cavity", cavity),
            ("two touching blobs", touching),
            ("hollow shell", shell),
            ("one-voxel arc", arc),
            ("single voxel", speck),
            ("speckle", speckle),
        ]
    }

    // ------------------------------- the predicate against the topology --

    /// **The test the rest of the file rests on.**
    ///
    /// `is_simple_point` is a local characterisation of a global property, and
    /// the characterisation is a theorem rather than a definition. So it is
    /// checked against the property: for each of several thousand
    /// neighbourhoods, the three Betti numbers of the configuration are measured
    /// before and after deleting the centre — by [`betti_numbers`], which counts
    /// components and cells and shares nothing with the predicate — and the
    /// predicate must agree with whether they changed.
    ///
    /// If the predicate were wrong, every "topology is preserved" assertion in
    /// this file would be measuring the same mistake twice. This one cannot be:
    /// the two sides are computed by different means.
    #[test]
    fn the_simple_point_test_agrees_with_the_topology_it_stands_for() {
        let mut bits = Bits::new(4242);
        let mut simple = 0;
        let mut not_simple = 0;
        // a spread of densities, so the sample contains sparse configurations
        // (where the object falls apart) and dense ones (where the background
        // does)
        for density in [10u64, 25, 40, 50, 60, 75, 90] {
            for _ in 0..400 {
                let mut n = [false; NEIGHBOURHOOD];
                for slot in n.iter_mut() {
                    *slot = bits.chance(density);
                }
                n[CENTRE] = true;

                let mut before = empty([3, 3, 3]);
                for index in 0..NEIGHBOURHOOD {
                    let offset = neighbour_offset(index);
                    before[[
                        (offset[0] + 1) as usize,
                        (offset[1] + 1) as usize,
                        (offset[2] + 1) as usize,
                    ]] = n[index];
                }
                let mut after = before.clone();
                after[[1, 1, 1]] = false;

                let unchanged = betti_numbers(before.view()) == betti_numbers(after.view());
                assert_eq!(
                    is_simple_point(&n),
                    unchanged,
                    "the predicate and the measurement disagree on {:?} (before {:?}, after {:?})",
                    n,
                    betti_numbers(before.view()),
                    betti_numbers(after.view())
                );
                if unchanged {
                    simple += 1;
                } else {
                    not_simple += 1;
                }
            }
        }
        assert!(
            simple > 100 && not_simple > 100,
            "the sample was one-sided: {simple} simple, {not_simple} not — a predicate that \
             answered a constant would have passed"
        );
    }

    /// The cases a reader can check by hand, stated separately from the sample
    /// so that a failure says *which* configuration rather than "one of 2800".
    #[test]
    fn the_simple_point_test_answers_the_cases_that_can_be_checked_by_hand() {
        let index = |di, dj, dk| neighbour_index(di, dj, dk);

        // an isolated voxel: deleting it destroys a component
        let mut isolated = [false; NEIGHBOURHOOD];
        isolated[CENTRE] = true;
        assert!(!is_simple_point(&isolated), "an isolated voxel");
        assert!(is_end_point(&isolated), "an isolated voxel is an end point");

        // the interior of a solid: deleting it opens a cavity
        let solid = [true; NEIGHBOURHOOD];
        assert!(!is_simple_point(&solid), "the interior of a solid");
        assert!(!is_border_point(&solid), "and it is not on the border");

        // a voxel on the flat face of a solid: the textbook simple point
        let mut face = [true; NEIGHBOURHOOD];
        for dj in -1..=1 {
            for dk in -1..=1 {
                face[index(1, dj, dk)] = false;
            }
        }
        assert!(is_simple_point(&face), "a flat face of a solid");

        // the interior of a one-voxel sheet: deleting it opens a tunnel, which
        // neither component count would notice
        let mut sheet = [false; NEIGHBOURHOOD];
        for dj in -1..=1 {
            for dk in -1..=1 {
                sheet[index(0, dj, dk)] = true;
            }
        }
        assert!(
            !is_simple_point(&sheet),
            "the interior of a sheet is not simple: deleting it opens a tunnel"
        );

        // the interior of an arc: deleting it cuts the arc in two
        let mut arc = [false; NEIGHBOURHOOD];
        arc[CENTRE] = true;
        arc[index(0, 0, -1)] = true;
        arc[index(0, 0, 1)] = true;
        assert!(!is_simple_point(&arc), "the interior of an arc");

        // the end of an arc: simple, and saved only by the end point rule
        let mut end = [false; NEIGHBOURHOOD];
        end[CENTRE] = true;
        end[index(0, 0, -1)] = true;
        assert!(is_simple_point(&end), "the end of an arc is simple");
        assert!(is_end_point(&end), "and is an end point");
        assert!(!is_deletable(&end), "so it is not deleted");
    }

    /// The border condition is a consequence of simplicity, not a clause of the
    /// rule; that is stated in the documentation, so it is asserted.
    #[test]
    fn the_border_condition_is_implied_by_simplicity() {
        let mut bits = Bits::new(99);
        let mut checked = 0;
        for _ in 0..4000 {
            let mut n = [false; NEIGHBOURHOOD];
            for slot in n.iter_mut() {
                *slot = bits.chance(70);
            }
            n[CENTRE] = true;
            if is_simple_point(&n) {
                assert!(is_border_point(&n), "simple but not on the border: {n:?}");
                checked += 1;
            }
        }
        assert!(checked > 50, "only {checked} simple points in the sample");
    }

    // ------------------------------------------------- purity and order --

    /// A sub-iteration is a pure function of the buffer it is handed.
    ///
    /// Two halves, and the second is the one that matters: the output is exactly
    /// the input with the voxels the *input's own* predicate selects removed. An
    /// implementation that consulted a partially updated volume would delete a
    /// different set, and the difference would be a set of voxels the input
    /// predicate says are deletable and which survived.
    #[test]
    fn a_sub_iteration_is_a_pure_function_of_its_input() {
        for (name, input) in shapes() {
            for &subfield in &SWEEP {
                let once = sub_iteration(&input, subfield);
                let twice = sub_iteration(&input, subfield);
                assert_eq!(once, twice, "{name}: not deterministic");

                let shape = [input.shape()[0], input.shape()[1], input.shape()[2]];
                let mut expected = input.clone();
                for i in 0..shape[0] {
                    for j in 0..shape[1] {
                        for k in 0..shape[2] {
                            if subfield.contains([i, j, k])
                                && is_deletable(&gather(input.view(), [i, j, k]))
                            {
                                expected[[i, j, k]] = false;
                            }
                        }
                    }
                }
                assert_eq!(
                    once,
                    expected,
                    "{name}, subfield {}: the applied output is not the input's own \
                     deletion set removed",
                    subfield.index()
                );
            }
        }
    }

    /// **The correctness argument, asserted rather than argued.**
    ///
    /// The claim is that the voxels one sub-iteration deletes can be deleted one
    /// at a time, in any order, each still being a simple non-end point at the
    /// moment it goes. That is what makes the parallel form and the sequential
    /// form the same function, and it is what a directional scheme cannot say.
    ///
    /// Here it is checked directly: the deleted set is walked in several shuffled
    /// orders and each member is re-tested against the *partially deleted*
    /// volume.
    #[test]
    fn a_class_can_be_deleted_one_at_a_time_in_any_order() {
        let mut bits = Bits::new(77);
        let mut checked = 0;
        for (name, input) in shapes() {
            for &subfield in &SWEEP {
                let shape = [input.shape()[0], input.shape()[1], input.shape()[2]];
                let mut deleted = Vec::new();
                for i in 0..shape[0] {
                    for j in 0..shape[1] {
                        for k in 0..shape[2] {
                            if subfield.contains([i, j, k])
                                && is_deletable(&gather(input.view(), [i, j, k]))
                            {
                                deleted.push([i, j, k]);
                            }
                        }
                    }
                }
                if deleted.is_empty() {
                    continue;
                }
                checked += deleted.len();
                for _ in 0..3 {
                    let mut order = deleted.clone();
                    for at in (1..order.len()).rev() {
                        order.swap(at, bits.below(at + 1));
                    }
                    let mut current = input.clone();
                    for position in &order {
                        let n = gather(current.view(), *position);
                        assert!(
                            is_deletable(&n),
                            "{name}, subfield {}: {position:?} stopped being deletable once its \
                             class-mates went, so parallel and sequential deletion differ",
                            subfield.index()
                        );
                        current[*position] = false;
                    }
                    assert_eq!(
                        current,
                        sub_iteration(&input, subfield),
                        "{name}, subfield {}: a shuffled sequential deletion gave a different \
                         volume from the parallel one",
                        subfield.index()
                    );
                }
            }
        }
        assert!(checked > 200, "only {checked} deletions were exercised");
    }

    // --------------------------------------------------------- topology --

    /// **The bar.** Every sub-iteration, on every shape, leaves all three Betti
    /// numbers where it found them — and so does a whole pass, and so does
    /// thinning to a fixed point.
    ///
    /// This is the test a wrong simple point predicate fails. It is also the one
    /// the naive directional rule fails, which is why this file does not
    /// implement it.
    #[test]
    fn thinning_preserves_the_topology_of_every_shape() {
        for (name, input) in shapes() {
            let want = betti_numbers(input.view());
            let mut current = input.clone();
            for &subfield in &SWEEP {
                current = sub_iteration(&current, subfield);
                assert_eq!(
                    betti_numbers(current.view()),
                    want,
                    "{name}: subfield {} changed the topology",
                    subfield.index()
                );
            }
            let (converged, passes) = thin_to_fixed_point(input.view(), generous()).unwrap();
            assert_eq!(
                betti_numbers(converged.view()),
                want,
                "{name}: thinning to a fixed point in {passes} passes changed the topology"
            );
        }
    }

    /// The shapes are the shapes they are meant to be. Without this the test
    /// above could be preserving the topology of something that has none.
    #[test]
    fn the_test_shapes_have_the_topology_they_are_named_for() {
        let measured: Vec<(&str, [usize; 3])> = shapes()
            .iter()
            .map(|(name, shape)| (*name, betti_numbers(shape.view())))
            .collect();
        for (name, want) in [
            ("solid box", [1, 0, 0]),
            ("box with a tunnel", [1, 1, 0]),
            ("box with a cavity", [1, 0, 1]),
            ("two touching blobs", [1, 0, 0]),
            ("hollow shell", [1, 0, 1]),
            ("one-voxel arc", [1, 0, 0]),
            ("single voxel", [1, 0, 0]),
        ] {
            let got = measured
                .iter()
                .find(|(other, _)| *other == name)
                .unwrap_or_else(|| panic!("{name} is missing from the shape list"))
                .1;
            assert_eq!(got, want, "{name}");
        }
    }

    /// **Why this file does not implement the directional rule its family is
    /// usually written in.**
    ///
    /// The rule "delete every `d`-border point that is simple and is not an end
    /// point", applied in parallel from one snapshot, is stated in the header as
    /// unsound. Here is the object that shows it: two columns joined by a flat
    /// two-voxel-wide bridge whose every voxel has background above it. The
    /// naive rule deletes the whole bridge at once and the object falls in two.
    /// The implemented rule, on the same object, does not.
    ///
    /// This is the test that makes the design decision evidence rather than
    /// assertion.
    #[test]
    fn the_naive_directional_rule_falls_apart_where_this_one_does_not() {
        let volume = [8, 10, 8];
        let mut object = empty(volume);
        for j in 2..8 {
            for i in 3..5 {
                object[[i, j, 5]] = true;
            }
        }
        for k in 2..5 {
            object[[3, 2, k]] = true;
            object[[3, 7, k]] = true;
        }
        assert_eq!(
            betti_numbers(object.view()),
            [1, 0, 0],
            "the fixture must start as one component"
        );

        // the naive rule: an "up"-border point (nothing at k + 1) that is simple
        // and is not an end point, every decision from the same snapshot
        let mut naive = object.clone();
        let mut deleted = 0;
        for i in 0..volume[0] {
            for j in 0..volume[1] {
                for k in 0..volume[2] {
                    let n = gather(object.view(), [i, j, k]);
                    if n[CENTRE]
                        && !n[neighbour_index(0, 0, 1)]
                        && !is_end_point(&n)
                        && is_simple_point(&n)
                    {
                        naive[[i, j, k]] = false;
                        deleted += 1;
                    }
                }
            }
        }
        assert!(
            deleted > 0,
            "the naive rule deleted nothing to be wrong about"
        );
        assert_ne!(
            betti_numbers(naive.view()),
            [1, 0, 0],
            "the naive directional rule was supposed to break this object; if it no longer \
             does, the reason this file gives for not implementing it needs rewriting"
        );

        // and the implemented rule, on the same object, does not
        let (converged, _) = thin_to_fixed_point(object.view(), generous()).unwrap();
        assert_eq!(
            betti_numbers(converged.view()),
            [1, 0, 0],
            "the subfield rule broke the object the naive one breaks"
        );
    }

    // -------------------------------------------- what the answer looks like --

    /// Thinning stops, and once it has stopped another pass changes nothing.
    #[test]
    fn thinning_terminates_and_is_idempotent_at_its_fixed_point() {
        for (name, input) in shapes() {
            let (converged, passes) = thin_to_fixed_point(input.view(), generous()).unwrap();
            assert!(passes >= 1, "{name}");
            assert_eq!(
                pass(&converged),
                converged,
                "{name}: a pass after convergence changed the volume"
            );
            for &subfield in &SWEEP {
                assert_eq!(
                    sub_iteration(&converged, subfield),
                    converged,
                    "{name}: subfield {} changed the fixed point",
                    subfield.index()
                );
            }
        }
    }

    /// A structure that is already thin is already the answer.
    ///
    /// The two conditions together are what does it: the interior of the arc is
    /// not simple, and its two ends are end points. Drop either and this fails.
    #[test]
    fn a_one_voxel_arc_is_returned_unchanged() {
        let volume = [12, 12, 12];
        let mut straight = empty(volume);
        for k in 2..10 {
            straight[[6, 6, k]] = true;
        }
        // and a diagonal one, which is 26-connected and not 6-connected
        let mut diagonal = empty(volume);
        for step in 2..9 {
            diagonal[[step, step, step]] = true;
        }
        for (name, arc) in [("straight", straight), ("diagonal", diagonal)] {
            assert_eq!(pass(&arc), arc, "{name} arc moved");
            assert_eq!(
                thin_to_fixed_point(arc.view(), generous()).unwrap().0,
                arc,
                "{name} arc"
            );
        }
    }

    /// **The one arc that does move, and why that is right.**
    ///
    /// A right-angled arc has a redundant corner: the two voxels flanking the
    /// corner are themselves 26-adjacent, so the corner joins nothing and is a
    /// simple non-end point. Thinning deletes it, the arc stays connected
    /// through the diagonal step, and the result is one voxel shorter and
    /// topologically identical.
    ///
    /// This is here because it is the case a reader would file as a bug. It is
    /// not one: an implementation that *kept* the corner would be keeping a
    /// voxel its own deletion rule says nothing depends on, and the only way to
    /// keep it would be an extra condition nothing in the definition asks for.
    #[test]
    fn a_right_angled_arc_loses_its_redundant_corner_and_nothing_else() {
        let volume = [12, 12, 12];
        let mut bent = empty(volume);
        for k in 2..7 {
            bent[[6, 6, k]] = true;
        }
        for j in 7..10 {
            bent[[6, j, 6]] = true;
        }
        let corner = [6usize, 6, 6];
        assert!(bent[corner]);

        let (thinned, _) = thin_to_fixed_point(bent.view(), generous()).unwrap();
        assert_eq!(
            betti_numbers(thinned.view()),
            betti_numbers(bent.view()),
            "the arc came apart"
        );
        assert!(!thinned[corner], "the redundant corner survived");
        let mut expected = bent.clone();
        expected[corner] = false;
        assert_eq!(
            thinned, expected,
            "exactly one voxel — the corner — should have gone"
        );
    }

    /// A single voxel survives everything.
    #[test]
    fn a_single_voxel_survives() {
        let mut speck = empty([5, 5, 5]);
        speck[[2, 2, 2]] = true;
        assert_eq!(
            thin_to_fixed_point(speck.view(), generous()).unwrap().0,
            speck
        );

        // including at the very corner of the volume, where the outside-is-
        // background convention gives it the most background it can have
        let mut corner = empty([5, 5, 5]);
        corner[[0, 0, 0]] = true;
        assert_eq!(
            thin_to_fixed_point(corner.view(), generous()).unwrap().0,
            corner
        );
    }

    /// A solid rod thins to something one voxel thick along its length, and
    /// stays in one piece.
    ///
    /// Both halves are needed. "It got smaller" is satisfied by deleting
    /// everything; "it stayed connected" is satisfied by doing nothing.
    ///
    /// **What it does *not* claim, because it is not true and hiding it would
    /// be worse.** The rod's two flat end faces have corners, and a corner of a
    /// receding face becomes an end point before it becomes simple — so the
    /// result carries a short spur into each corner at each end. That is what
    /// thinning a box does, in every scheme with an end point condition; it is
    /// the price of the condition that keeps the arc from retreating. The
    /// assertion below is therefore about the rod's **middle**, and the ends are
    /// covered by the thinness and connectedness properties, which hold
    /// everywhere.
    #[test]
    fn a_solid_rod_thins_to_one_voxel_along_its_length_and_stays_connected() {
        let volume = [11, 11, 24];
        let rod = solid_box(volume, [3, 3, 2], [8, 8, 22]);
        let (thinned, _) = thin_to_fixed_point(rod.view(), generous()).unwrap();

        assert_eq!(
            connected_components(thinned.view(), true, Adjacency::TwentySix),
            1,
            "the rod came apart"
        );
        assert_eq!(betti_numbers(thinned.view()), [1, 0, 0]);

        let solid_voxels = rod.iter().filter(|value| **value).count();
        let thin_voxels = thinned.iter().filter(|value| **value).count();
        assert!(
            thin_voxels * 10 < solid_voxels,
            "{thin_voxels} of {solid_voxels} voxels left, which is not thinning"
        );
        assert!(
            thin_voxels >= 8,
            "the rod collapsed to {thin_voxels} voxels; an arc's ends are end points and must \
             stop retreating"
        );

        // **Nowhere two voxels thick.** No axis-aligned 2x2 square of the
        // result is entirely object — the shape-independent statement of "one
        // voxel thick" that does not assume which way the surviving arc runs.
        for i in 0..volume[0] - 1 {
            for j in 0..volume[1] - 1 {
                for k in 0..volume[2] - 1 {
                    for (a, b) in [(0usize, 1usize), (0, 2), (1, 2)] {
                        let mut corners = [[i, j, k]; 4];
                        corners[1][a] += 1;
                        corners[2][b] += 1;
                        corners[3][a] += 1;
                        corners[3][b] += 1;
                        assert!(
                            !corners.iter().all(|corner| thinned[*corner]),
                            "the thinned rod is two voxels thick at {:?} in the {a}-{b} plane",
                            [i, j, k]
                        );
                    }
                }
            }
        }

        // and through the middle of the rod, well away from either end face,
        // every slice across the long axis holds exactly one voxel
        for k in 8..17 {
            let in_slice = (0..volume[0])
                .flat_map(|i| (0..volume[1]).map(move |j| (i, j)))
                .filter(|(i, j)| thinned[[*i, *j, k]])
                .count();
            assert_eq!(
                in_slice, 1,
                "slice {k} through the middle of the thinned rod holds {in_slice} voxels"
            );
        }
    }

    /// A ring thins to a ring: the tunnel is what a curve skeleton of a loop is
    /// *for*, and cutting it would leave one component and no tunnel — which
    /// only `b1` can see.
    #[test]
    fn a_thick_ring_thins_to_a_loop() {
        let volume = [16, 16, 8];
        let mut ring = empty(volume);
        for i in 0..volume[0] {
            for j in 0..volume[1] {
                let (di, dj) = (i as f64 - 7.5, j as f64 - 7.5);
                let radius = (di * di + dj * dj).sqrt();
                if (3.0..6.0).contains(&radius) {
                    for k in 3..6 {
                        ring[[i, j, k]] = true;
                    }
                }
            }
        }
        assert_eq!(
            betti_numbers(ring.view()),
            [1, 1, 0],
            "the fixture is a ring"
        );

        let (thinned, _) = thin_to_fixed_point(ring.view(), generous()).unwrap();
        assert_eq!(
            betti_numbers(thinned.view()),
            [1, 1, 0],
            "the loop was cut or doubled"
        );
        assert!(
            thinned.iter().filter(|value| **value).count() * 4
                < ring.iter().filter(|value| **value).count(),
            "the ring did not thin"
        );
    }

    // ------------------------------------------------------------ reach --

    /// The reach is one, and it is **tight**: an object exists whose answer at a
    /// voxel changes when a voxel exactly one away changes.
    ///
    /// A declared reach that is larger than the dependency costs halo for
    /// nothing; one that is smaller is the silent wrongness this crate exists to
    /// prevent. This asserts the first is not happening.
    #[test]
    fn the_reach_of_one_is_tight() {
        let op = ThinningOp::for_subfield(SWEEP[0]);
        for axis in 0..3 {
            assert_eq!(op.reach(axis, 1000), 1);
        }

        // `p` at [2, 2, 2] — a class-0 voxel — with two object neighbours on
        // opposite sides along `axis`. They are not 26-adjacent to each other,
        // so `p` has two object components around it and is not simple: it
        // survives. One more voxel, adjacent to both, joins them into one
        // component and `p` goes. That voxel sits **exactly one** away, on
        // `bridged`; place it two away instead and nothing changes.
        for axis in 0..3 {
            let bridged = (axis + 1) % 3;
            let mut base = empty([7, 7, 7]);
            base[[2, 2, 2]] = true;
            let mut lower = [2usize, 2, 2];
            let mut upper = [2usize, 2, 2];
            lower[axis] = 1;
            upper[axis] = 3;
            base[lower] = true;
            base[upper] = true;
            assert!(
                sub_iteration(&base, SWEEP[0])[[2, 2, 2]],
                "axis {axis}: the fixture must start out undeletable"
            );

            let mut one_away = [2usize, 2, 2];
            let mut two_away = [2usize, 2, 2];
            one_away[bridged] = 3;
            two_away[bridged] = 4;

            let mut with_far = base.clone();
            with_far[two_away] = true;
            assert!(
                sub_iteration(&with_far, SWEEP[0])[[2, 2, 2]],
                "axis {bridged}: a voxel two away changed the answer, so the reach is larger \
                 than one and the declaration is short"
            );

            let mut with_near = base.clone();
            with_near[one_away] = true;
            assert!(
                !sub_iteration(&with_near, SWEEP[0])[[2, 2, 2]],
                "axis {bridged}: a voxel one away must reach the answer, or the declared reach \
                 of one is larger than the dependency and the halo is paid for nothing"
            );
        }
    }

    /// A pass is eight sub-iterations and the chain's reach is the sum, and the
    /// second statement of that number agrees with the fold.
    #[test]
    fn the_chain_states_the_reach_the_sub_iterations_add_up_to() {
        assert_eq!(thinning_pass().slots().len(), Subfield::COUNT);
        for passes in [0usize, 1, 3, 7] {
            let chain = thin(passes);
            assert_eq!(chain.slots().len(), passes * Subfield::COUNT);
            for axis in 0..3 {
                assert_eq!(
                    chain.reach(axis, 1000),
                    thinning_reach(passes),
                    "{passes} passes, axis {axis}"
                );
            }
        }
    }

    /// Every voxel gets exactly one chance per pass, which is what makes eight
    /// sub-iterations *a* pass rather than an arbitrary number of them.
    #[test]
    fn the_eight_classes_partition_the_lattice() {
        let mut seen = [0usize; Subfield::COUNT];
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    let mut classes = SWEEP
                        .iter()
                        .filter(|subfield| subfield.contains([i, j, k]))
                        .count();
                    assert_eq!(classes, 1, "[{i}, {j}, {k}] is in {classes} classes");
                    classes = Subfield::of([i, j, k]).index();
                    seen[classes] += 1;
                }
            }
        }
        assert_eq!(seen, [8; Subfield::COUNT]);
        let mut sorted: Vec<usize> = SWEEP.iter().map(|subfield| subfield.index()).collect();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..Subfield::COUNT).collect::<Vec<_>>());
        assert!(Subfield::new(Subfield::COUNT).is_err());
    }

    // ------------------------------------------------------ the shell --

    /// The `bool` arm and the `f64` arm are one operation, and the `bool` arm
    /// holds an eighth of the bytes.
    #[test]
    fn a_mask_held_as_bool_and_as_f64_gives_the_same_answer() {
        let volume = [10, 9, 8];
        let mut bits = Bits::new(5);
        let mut mask = empty(volume);
        for value in mask.iter_mut() {
            *value = bits.chance(55);
        }
        let op = ThinningOp::for_subfield(SWEEP[2]);
        let at = Anchor::whole(volume);

        let narrow_in: Voxels = mask.clone().into();
        let mut narrow = Voxels::zeros(Dtype::Bool, volume).unwrap();
        op.apply(&narrow_in, &mut narrow, &at).unwrap();

        let wide_in: Voxels = mask.mapv(from_set).into();
        let mut wide = Voxels::zeros(Dtype::F64, volume).unwrap();
        op.apply(&wide_in, &mut wide, &at).unwrap();

        let narrow_view = narrow.view::<bool>().unwrap();
        let wide_view = wide.view::<f64>().unwrap();
        for (flag, value) in narrow_view.iter().zip(wide_view.iter()) {
            assert_eq!(from_set(*flag), *value);
        }
        assert_eq!(wide.bytes(), narrow.bytes() * 8);
    }

    /// The one constant that may be declared is declared, and it is computed and
    /// checked; the one that may not be is withheld, and the reason it may not be
    /// is measured rather than argued.
    #[test]
    fn an_all_set_block_is_not_constant_and_nothing_is_declared_for_it() {
        let volume = [6, 6, 6];
        let op = ThinningOp::for_subfield(SWEEP[0]);
        let at = Anchor::whole(volume);

        assert_eq!(op.constant_maps_to(0.0), Some(0.0));
        assert_eq!(op.constant_maps_to(1.0), None);
        assert_eq!(op.constant_maps_to(9.0), None);

        // clear stays clear, computed
        let clear: Voxels = empty(volume).into();
        let mut out = Voxels::zeros(Dtype::Bool, volume).unwrap();
        op.apply(&clear, &mut out, &at).unwrap();
        assert!(out.view::<bool>().unwrap().iter().all(|value| !value));

        // and set does not stay set, which is why nothing is declared for it
        let set: Voxels = Array3::from_elem((6, 6, 6), true).into();
        let mut out = Voxels::zeros(Dtype::Bool, volume).unwrap();
        op.apply(&set, &mut out, &at).unwrap();
        let view = out.view::<bool>().unwrap();
        assert!(
            view.iter().any(|value| !value),
            "an all-set block was expected to lose its face voxels; if it no longer does, the \
             reason `constant_maps_to` withholds the set case has changed"
        );
        assert!(
            view[[3, 3, 3]],
            "and the interior must be untouched: it has no background face neighbour"
        );
    }

    // -------------------------------------------- through the framework --

    const BLOCK_VOLUME: [usize; 3] = [20, 16, 14];

    fn framework_input() -> Array3<bool> {
        let mut bits = Bits::new(31337);
        let mut mask = empty(BLOCK_VOLUME);
        for value in mask.iter_mut() {
            *value = bits.chance(48);
        }
        // and some solid structure, so the volume is not only speckle
        for i in 4..14 {
            for j in 4..12 {
                for k in 4..10 {
                    mask[[i, j, k]] = true;
                }
            }
        }
        mask
    }

    /// One phase holding the whole chain, at a given block edge and split axes,
    /// with the halo derived from the chain's own reach and from nothing else.
    fn plan(workflow: &Workflow, block: usize, split_axes: &[usize]) -> Decomposition {
        let slots = workflow.chain.slots();
        let names: Vec<String> = slots.iter().map(|slot| slot.display_name()).collect();
        let reach = workflow.chain.reach3(&BLOCK_VOLUME);
        let grid = BlockGrid::along(BLOCK_VOLUME, split_axes, block).unwrap();
        let mut plan = Decomposition {
            volume: BLOCK_VOLUME,
            dtype: workflow.dtype,
            phases: vec![PhaseDecomposition::derive(
                (0..slots.len()).collect(),
                names,
                reach,
                reach,
                grid,
            )],
            chain_reach: reach,
        };
        plan.declare_dtypes(&workflow.chain).unwrap();
        plan
    }

    fn reference(chain: &Chain, input: &Array3<bool>) -> Array3<bool> {
        let source: Voxels = input.clone().into();
        let mut out = Voxels::zeros(Dtype::Bool, BLOCK_VOLUME).unwrap();
        chain
            .apply(&source, &mut out, &Anchor::whole(BLOCK_VOLUME))
            .expect("the whole-volume reference must run");
        out.view::<bool>().unwrap().to_owned()
    }

    fn run(
        workflow: &Workflow,
        decomposition: &Decomposition,
        input: &Array3<bool>,
    ) -> Array3<bool> {
        let env = ArrayEnvironment::new(input.clone().into(), decomposition.n_phases(), [4, 4, 4])
            .unwrap();
        execute("skeleton", workflow, decomposition, &Hints::default(), &env).unwrap();
        env.output().view::<bool>().unwrap().to_owned()
    }

    /// **Decomposition invariance.** Byte-identical output against the
    /// whole-volume reference, under every decomposition, for a single
    /// sub-iteration and for a whole pass.
    ///
    /// This is also the test that would fail if the parity class were taken from
    /// the block's own indices rather than from the anchor: every block but the
    /// first would then thin a different eighth of the lattice, and the seams
    /// would disagree with the reference without any halo being short.
    #[test]
    fn the_block_runs_agree_with_the_whole_volume_reference() {
        let input = framework_input();
        for (name, chain) in [
            (
                "one sub-iteration",
                Chain::op(ThinningOp::for_subfield(SWEEP[3])),
            ),
            ("a full pass", thinning_pass()),
        ] {
            let want = reference(&chain, &input);
            let workflow = Workflow::new(chain, BLOCK_VOLUME, Dtype::Bool);
            let mut ran = 0;
            for block in [4usize, 7, 20] {
                for split_axes in [vec![0], vec![2], vec![0, 1], vec![0, 1, 2]] {
                    let decomposition = plan(&workflow, block, &split_axes);
                    decomposition
                        .check()
                        .unwrap_or_else(|err| panic!("{name}: an honest plan must tile: {err}"));
                    assert_eq!(
                        run(&workflow, &decomposition, &input),
                        want,
                        "{name}: block {block}, axes {split_axes:?} disagreed with the \
                         whole-volume reference"
                    );
                    ran += 1;
                }
            }
            assert!(ran >= 12, "{name}: the sweep did not run");
        }
    }

    /// **The guard, seen firing.** A halo one voxel short of the derived reach
    /// must make the valid regions stop tiling, and the executor must refuse the
    /// plan for the same reason.
    #[test]
    fn a_halo_short_of_the_derived_reach_is_caught() {
        let input = framework_input();
        for (name, chain, reach) in [
            (
                "one sub-iteration",
                Chain::op(ThinningOp::for_subfield(SWEEP[3])),
                1usize,
            ),
            ("a full pass", thinning_pass(), Subfield::COUNT),
        ] {
            assert_eq!(chain.reach(0, BLOCK_VOLUME[0]), reach, "{name}");
            let workflow = Workflow::new(chain, BLOCK_VOLUME, Dtype::Bool);
            let honest = plan(&workflow, 8, &[0]);
            honest.check().unwrap();

            let forced = honest.with_forced_halo([reach - 1, 0, 0]);
            let err = forced
                .check()
                .expect_err(&format!("{name}: a short halo must not check out"))
                .to_string();
            assert!(
                err.contains("do not tile the volume exactly"),
                "{name}: expected the tiling guard, got: {err}"
            );

            let env = ArrayEnvironment::new(input.clone().into(), 1, [4, 4, 4]).unwrap();
            let err = execute("short", &workflow, &forced, &Hints::default(), &env)
                .expect_err(&format!("{name}: the executor must refuse a short halo"))
                .to_string();
            assert!(
                err.contains("do not tile the volume exactly"),
                "{name}: got {err}"
            );
        }
    }

    /// The silent version of the same failure: a phase that **understates** its
    /// reach tiles perfectly and produces wrong values. A guard nobody has
    /// watched fail is not known to work, and a reach nobody has watched matter
    /// is not known to be needed.
    #[test]
    fn an_understated_reach_tiles_perfectly_and_produces_wrong_values() {
        let input = framework_input();
        let chain = Chain::op(ThinningOp::for_subfield(SWEEP[3]));
        let want = reference(&chain, &input);
        let workflow = Workflow::new(chain, BLOCK_VOLUME, Dtype::Bool);

        let slots = workflow.chain.slots();
        let names: Vec<String> = slots.iter().map(|slot| slot.display_name()).collect();
        let mut wrong = false;
        for block in [4usize, 5, 7] {
            for axis in 0..3 {
                let grid = BlockGrid::along(BLOCK_VOLUME, &[axis], block).unwrap();
                let mut plan = Decomposition {
                    volume: BLOCK_VOLUME,
                    dtype: workflow.dtype,
                    phases: vec![PhaseDecomposition::derive(
                        (0..slots.len()).collect(),
                        names.clone(),
                        [0, 0, 0],
                        [0, 0, 0],
                        grid,
                    )],
                    chain_reach: [0, 0, 0],
                };
                plan.declare_dtypes(&workflow.chain).unwrap();
                plan.check()
                    .expect("an understated reach is self-consistent and tiles");
                if run(&workflow, &plan, &input) != want {
                    wrong = true;
                }
            }
        }
        assert!(
            wrong,
            "a phase declaring no reach at all reproduced the reference everywhere, which \
             would mean the declared reach of one is not needed"
        );
    }

    // ------------------------------------------------------------ costs --

    /// Retaking the measurement. Ignored because timing in a test suite measures
    /// the machine's mood, not the code — but it is here, it runs, and it is one
    /// command:
    ///
    /// ```text
    /// cargo test --release -- --ignored --nocapture ops::skeleton
    /// ```
    #[test]
    #[ignore = "a measurement, not an assertion"]
    fn print_the_cost_table() {
        println!("{}", cost_report([96, 64, 64], 5));
    }

    /// What can be asserted about a measured cost without measuring: that the
    /// order the constant encodes is the order the ops actually have. A
    /// neighbourhood op with two flood fills in it must cost more per voxel than
    /// a voxelwise map.
    #[test]
    fn the_stored_cost_keeps_the_order_the_measurement_found() {
        let map = super::super::voxelwise::VoxelwiseMapOp::new("map", |value| value);
        let thin = ThinningOp::for_subfield(SWEEP[0]);
        assert!(thin.cost_per_voxel() > map.cost_per_voxel());
        assert_eq!(
            thin.cost_per_voxel(),
            THINNING_COST,
            "the op must report the measured constant, not a default"
        );
        assert_eq!(
            ThinningOp::for_subfield(SWEEP[0])
                .with_cost(3.5)
                .cost_per_voxel(),
            3.5
        );
    }
}

#[cfg(test)]
mod limit_tests {
    use super::*;

    /// A limit is a **guard**: exceeding it is an error naming the op, never a
    /// truncated answer. A partially thinned volume is plausible, well-formed
    /// and wrong, which is the one outcome this must not produce.
    #[test]
    fn a_run_that_does_not_converge_in_time_fails_rather_than_truncating() {
        // a solid rod needs several passes; one is not enough
        let mut rod = Array3::from_elem((21, 9, 9), false);
        for i in 2..19 {
            for j in 3..6 {
                for k in 3..6 {
                    rod[[i, j, k]] = true;
                }
            }
        }
        let error = thin_to_fixed_point(rod.view(), PassLimit::of(1).unwrap()).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("did not reach a fixed point"), "{message}");
        assert!(message.contains("wrong skeleton"), "{message}");

        // and with room it converges and reports how many it took
        let (thinned, passes) =
            thin_to_fixed_point(rod.view(), PassLimit::for_volume([21, 9, 9])).unwrap();
        assert!(
            passes > 1,
            "the shape must actually need more than one pass"
        );
        assert!(thinned.iter().any(|&set| set));
    }

    /// The bound from the volume is a bound rather than a guess: nothing thicker
    /// than half the shortest axis fits, so no correct run can need more passes.
    #[test]
    fn the_volume_bound_is_enough_for_the_thickest_thing_that_fits() {
        let volume = [24usize, 16, 12];
        let limit = PassLimit::for_volume(volume);
        assert_eq!(limit.passes(), 12 / 2 + 2);

        // a solid block filling the volume is the thickest object it can hold
        let solid = Array3::from_elem((volume[0], volume[1], volume[2]), true);
        let (_, passes) = thin_to_fixed_point(solid.view(), limit).unwrap();
        assert!(
            passes <= limit.passes(),
            "the thickest object that fits took {passes} passes against a bound of {}",
            limit.passes()
        );
    }

    /// A limit that fires before anything runs is not a backstop.
    #[test]
    fn a_limit_of_zero_is_refused_where_it_is_stated() {
        assert!(PassLimit::of(0).is_err());
        assert!(PassLimit::of(1).is_ok());
    }
}
