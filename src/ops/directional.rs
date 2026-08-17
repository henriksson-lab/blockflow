// SPDX-License-Identifier: MIT
//
// Original work for this crate. The rule implemented here is the **published
// 12-subiteration parallel thinning algorithm of Palagyi and Kuba** (Graphical
// Models and Image Processing 61, 199-221, 1999); its fourteen deletion
// templates are encoded below from that published definition, as a table of
// positions and required states, and nothing in this file is adapted from any
// implementation of it. See "Where the templates come from" for the full note,
// because the provenance of a template set is exactly the kind of thing that is
// impossible to reconstruct later.
//
// What this is, and why it exists beside `super::skeleton`
// --------------------------------------------------------
// Both thin a binary volume to a curve skeleton and both are *parallel*
// algorithms in the sense this crate needs — every deletion decision in a
// sub-iteration is computed from one snapshot and written back afterwards, so
// the answer does not depend on the order voxels are visited in. They are not
// variants of one thing, and neither is a tuning of the other:
//
// | | `skeleton` | here |
// |---|---|---|
// | family | subfield | **directional, template-driven** |
// | sub-iterations per pass | 8, one per coordinate-parity class | **12**, one per rotation of the neighbourhood |
// | what makes a deletion safe | the deleted set is pairwise non-adjacent, so parallel *is* sequential, and each deletion is of a simple point | the templates are a **fixed list whose parallel safety is proved case by case** in the paper they come from |
// | anchoring | **global**: a voxel's parity class is a fact about where it is | **none**: translation-invariant, so a block needs no `Anchor` at all |
// | which voxels are candidates | recomputed every sub-iteration | the border set is computed **once per pass** and reused, deliberately stale |
//
// The last two rows are the ones that matter to a block framework, and they
// point in opposite directions. Being translation-invariant is a gift: nothing
// here reads `Anchor::offset`, so a block's answer is a function of its buffer
// alone. The stale border set is the price, and it is what forces the *pass*
// rather than the sub-iteration to be the unit of work — see below.
//
// What the answer looks like, measured
// ------------------------------------
// A solid circular column 40 voxels long thins to **one voxel per slice** — the
// axis and nothing else — at every radius tried, and
// `the_column_thins_to_its_own_axis` pins the whole answer rather than its
// count. That is worth stating because the near miss is so plausible: a rule
// that let alternate slabs freeze would leave the axis intact with a transverse
// rung every second slice, and such a ladder has the right number of
// components, the right Betti numbers, no 2x2 square anywhere, and looks like a
// skeleton. The per-slice profile is what tells the two apart, so it is what the
// test asserts.
//
// The pass is the op, and the sub-iteration is not
// -----------------------------------------------
// `super::skeleton` makes each sub-iteration a `BlockOp` and a pass a
// `Chain::Sequence` of eight of them, and it can, because its sub-iteration is a
// function of the volume it is handed and of nothing else. **That is not true
// here.** A sub-iteration reads two things: the current volume, and the border
// set computed at the *start of the pass* — before any of the twelve
// sub-iterations ran. A voxel that loses a face neighbour during sub-iteration 3
// becomes a border voxel, and this algorithm does not consider it until the next
// pass.
//
// That staleness is not an accident to be tidied away; it is part of the
// algorithm's proof. Recomputing the border set per sub-iteration is a one-line
// "fix" that produces a different, unproved algorithm which still returns a
// plausible-looking skeleton, so
// `the_border_set_is_stale_within_a_pass` measures it rather than trusting a
// comment.
//
// A `BlockOp` carries one input buffer and one output buffer, so there is
// nowhere to thread a second array through a `Chain::Sequence`. The honest
// consequence is that **one pass is one op** ([`DirectionalPassOp`]), running
// its twelve sub-iterations internally, and declaring the reach that follows
// from doing so. It is not a data-dependent loop — twelve is twelve — so the
// declaration is exact rather than hopeful, which is the line this crate draws.
//
// The reach, derived
// ------------------
// **Twelve per pass**, on every axis. Write `S(k)` for the volume after
// sub-iteration `k` and `B` for the border set taken from `S(0)`:
//
// * `B(v)` reads `S(0)` within radius 1 (a face neighbour count);
// * `S(k)(v)` reads `S(k-1)` within radius 1 (a 3x3x3 template) and `B(v)`.
//
// Unrolling, `S(12)(v)` reads `S(0)` within radius 12, and the `B` it consults
// along the way is needed only within radius 11, which reads `S(0)` within
// radius 12 as well. So 12 is the bound and [`directional_reach`] states
// `12 * passes`, which is what `Chain::reach` folds independently.
//
// **It is a bound and not a measurement, and the two differ.** Deriving 12
// charges every sub-iteration a full voxel of dependency, and in practice the
// twelve orientations do not chain that way — measurement on structured and
// random volumes finds blocked and whole-volume runs already agreeing at a halo
// of four or five, and `an_understated_reach_tiles_perfectly_and_is_wrong`
// searches for and prints the number on its own fixture. Declaring the measured
// number would be declaring a coincidence: there is no argument
// that some volume cannot chain further, and a reach that is right for the data
// tried so far is precisely the failure this crate is arranged against. So the
// declaration is the derivation. `tests/directional_thinning.rs` pins both
// sides — that the declared halo is exact across block sizes, and that a halo
// *below* what the algorithm needs breaks, which is what makes the declaration a
// claim rather than a decoration.
//
// The fixed point is not a `BlockOp`, and per-block convergence is not offered
// ------------------------------------------------------------------------------
// [`directional_to_fixed_point`] runs passes until one changes nothing, over a
// whole volume, and is deliberately not dressed as an op: the number of passes
// is a function of the data. A caller who cannot hold the volume runs
// [`directional_thin`] with a stated pass count per round and asks a **global**
// reduction whether anything changed, then runs another round.
//
// There is no per-block convergence entry point here and that omission is
// load-bearing. "This block removed nothing" is **not** a safe stopping test: a
// region buried inside structure thicker than itself has no border voxels at all
// until the erosion front arrives, so it removes nothing in one round and
// something several rounds later. Only a reduction over the whole volume can
// say the work is done, and an API shaped like the unsafe test would invite the
// unsafe test.
//
// Where the templates come from
// -----------------------------
// The fourteen templates in [`TEMPLATES`] are the deleting templates of the
// published algorithm named at the top of this file, transcribed **from the
// algorithm's definition** — a template is a list of neighbourhood positions
// that must be object, a list that must be background, and (for three of them) a
// disjunction or a parity clause over named pairs — into the position tables
// below. They are stated as *data* here rather than as a chain of boolean
// operators, which is both how the source defines them and the form a reader can
// check against a figure.
//
// Nothing was copied from any implementation, and in particular **no lookup
// table was transcribed**. A `2^26`-entry table over this predicate is a pure
// speed choice, and where one is wanted it is *generated* from the predicate in
// this file: [`deletion_table_word`] produces any 64 entries of it and
// [`is_deletable_index`] is the predicate the generator calls, so the table
// cannot drift from the rule and there is no third-party artefact in the
// dependency chain. This module itself does not build one — the predicate
// compiles to a handful of masked integer comparisons ([`CompiledTemplate`]),
// which measures at about the same cost per sub-iteration as the flood fill
// `super::skeleton` runs — so the table is offered rather than used, and a
// caller for whom this is the bottleneck can build it in eight mebibytes as one
// bit per index without waiting on anything in here.
//
// The twelve orientations, and the order they come in
// --------------------------------------------------
// The templates above are stated once, for a single orientation, and the twelve
// sub-iterations apply them through twelve **rotations of the neighbourhood**
// ([`sub_iteration_sources`]). Each rotation is a permutation of the 26
// non-centre positions that fixes the centre, so it is a relabelling of the
// neighbourhood and not a second predicate.
//
// **The order of the twelve is part of the algorithm, and getting it wrong is
// the failure that looks like success.** A quarter turn and a three-quarter turn
// about one axis are each other's inverse, so swapping them yields *the same
// twelve rotations in a different order* — every rotation is still present,
// every one is still a proper rotation, the pass still terminates, topology is
// still preserved, and the result is still a one-voxel-wide curve skeleton with
// the same number of components. It is simply a different skeleton, displaced by
// about a voxel. No count, no invariant and no eyeball catches that, so
// [`directional_pass_with_sources_into`] takes the orientation table as an
// argument and `the_order_of_the_twelve_orientations_is_part_of_the_answer`
// feeds it the transposed one and asserts the two disagree.
//
// Edge behaviour
// --------------
// **Outside the array handed in is background**, the same convention
// `super::skeleton` states and for the same reason: the object is a set, and
// "there is nothing there" is what background means. It is right at a real
// volume boundary and deliberately wrong at a block seam, where it is what turns
// a short halo into a loud failure.
//
// The algorithm's own precondition is stronger and it is a **volume-level** one:
// the object must not touch the faces of the volume. [`clear_faces`] imposes it
// and [`faces_are_clear`] checks it, and both are free functions over a whole
// array rather than anything a block runs. Applying either per block would zero
// a face of every interior block — which is to say, would carve a hole along
// every seam — so neither is wired into [`DirectionalPassOp`] and the header of
// each says so.

use std::sync::OnceLock;

use ndarray::{Array3, ArrayView3, ArrayViewMut3};

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::op::{Anchor, BlockOp, Chain};
use crate::voxels::Voxels;

use super::shapes_agree;
use super::skeleton::{neighbour_index, neighbour_offset, PassLimit, CENTRE, NEIGHBOURHOOD};
use super::voxelwise::{from_set, is_set};

// ------------------------------------------------------------ the templates --

/// Sub-iterations in one pass: one per orientation of the neighbourhood.
pub const SUB_ITERATIONS: usize = 12;

/// A position in the 3x3x3 template frame, in `0..3` on each axis, with
/// `[1, 1, 1]` the voxel under test.
type Position = [usize; 3];

/// One deleting template.
///
/// A template is satisfied by a neighbourhood when every clause is: the
/// positions in `set` are object, those in `clear` are background, at least one
/// of `any_set` is object (an empty list means the clause is absent), neither
/// member of any `not_both` pair is object at the same time as the other, and
/// exactly one member of each `exactly_one` pair is object.
///
/// The centre is listed in `set` wherever the definition lists it, even though
/// the predicate is only ever asked about a voxel that is object. Leaving it
/// implicit would make the table one step further from the figure it was read
/// off, and the cost is one bit test that is always true.
struct Template {
    set: &'static [Position],
    clear: &'static [Position],
    any_set: &'static [Position],
    not_both: &'static [(Position, Position)],
    exactly_one: &'static [(Position, Position)],
}

/// The whole `z = 2` plane, which the first and third templates require to be
/// background.
const PLANE_Z2: &[Position] = &[
    [0, 0, 2],
    [1, 0, 2],
    [2, 0, 2],
    [0, 1, 2],
    [1, 1, 2],
    [2, 1, 2],
    [0, 2, 2],
    [1, 2, 2],
    [2, 2, 2],
];

/// The whole `y = 0` plane.
const PLANE_Y0: &[Position] = &[
    [0, 0, 0],
    [1, 0, 0],
    [2, 0, 0],
    [0, 0, 1],
    [1, 0, 1],
    [2, 0, 1],
    [0, 0, 2],
    [1, 0, 2],
    [2, 0, 2],
];

/// The fourteen deleting templates, in the order the algorithm states them.
///
/// The order is immaterial to the answer — a neighbourhood is deletable when it
/// satisfies *any* of them — and is kept only so that the table can be read
/// beside the definition.
const TEMPLATES: &[Template] = &[
    // 1. The face above is background across the whole plane, the face below is
    //    object, and the neighbourhood is not otherwise empty.
    Template {
        set: &[[1, 1, 1], [1, 1, 0]],
        clear: PLANE_Z2,
        any_set: &[
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [0, 1, 0],
            [2, 1, 0],
            [0, 2, 0],
            [1, 2, 0],
            [2, 2, 0],
            [0, 0, 1],
            [1, 0, 1],
            [2, 0, 1],
            [0, 1, 1],
            [2, 1, 1],
            [0, 2, 1],
            [1, 2, 1],
            [2, 2, 1],
        ],
        not_both: &[],
        exactly_one: &[],
    },
    // 2. The same statement a quarter turn away: the `y = 0` plane empty, the
    //    `+y` neighbour object.
    Template {
        set: &[[1, 1, 1], [1, 2, 1]],
        clear: PLANE_Y0,
        any_set: &[
            [0, 1, 0],
            [1, 1, 0],
            [2, 1, 0],
            [0, 2, 0],
            [1, 2, 0],
            [2, 2, 0],
            [0, 1, 1],
            [2, 1, 1],
            [0, 2, 1],
            [2, 2, 1],
            [0, 1, 2],
            [1, 1, 2],
            [2, 1, 2],
            [0, 2, 2],
            [1, 2, 2],
            [2, 2, 2],
        ],
        not_both: &[],
        exactly_one: &[],
    },
    // 3. Both planes empty at once, with the object continuing diagonally.
    Template {
        set: &[[1, 1, 1], [1, 2, 0]],
        clear: &[
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [0, 0, 1],
            [1, 0, 1],
            [2, 0, 1],
            [0, 0, 2],
            [1, 0, 2],
            [2, 0, 2],
            [0, 1, 2],
            [1, 1, 2],
            [2, 1, 2],
            [0, 2, 2],
            [1, 2, 2],
            [2, 2, 2],
        ],
        any_set: &[
            [0, 1, 0],
            [2, 1, 0],
            [0, 2, 0],
            [2, 2, 0],
            [0, 1, 1],
            [2, 1, 1],
            [0, 2, 1],
            [2, 2, 1],
        ],
        not_both: &[],
        exactly_one: &[],
    },
    // 4-10. The seven templates of the second group. They share a spine —
    //    `[1,1,0]`, the centre and `[1,2,1]` object, `[1,0,1]` and `[1,1,2]`
    //    background — and differ in what they demand of the two far columns.
    Template {
        set: &[[1, 1, 0], [1, 1, 1], [1, 2, 1]],
        clear: &[[1, 0, 1], [0, 0, 2], [1, 0, 2], [2, 0, 2], [1, 1, 2]],
        any_set: &[],
        not_both: &[([0, 0, 1], [0, 1, 2]), ([2, 0, 1], [2, 1, 2])],
        exactly_one: &[],
    },
    Template {
        set: &[[1, 1, 0], [1, 1, 1], [1, 2, 1], [2, 0, 2]],
        clear: &[[1, 0, 1], [0, 0, 2], [1, 0, 2], [1, 1, 2]],
        any_set: &[],
        not_both: &[([0, 0, 1], [0, 1, 2])],
        exactly_one: &[([2, 0, 1], [2, 1, 2])],
    },
    Template {
        set: &[[1, 1, 0], [1, 1, 1], [1, 2, 1], [0, 0, 2]],
        clear: &[[1, 0, 1], [1, 0, 2], [2, 0, 2], [1, 1, 2]],
        any_set: &[],
        not_both: &[([2, 0, 1], [2, 1, 2])],
        exactly_one: &[([0, 0, 1], [0, 1, 2])],
    },
    Template {
        set: &[[1, 1, 0], [1, 1, 1], [2, 1, 1], [1, 2, 1]],
        clear: &[[1, 0, 1], [0, 0, 2], [1, 0, 2], [1, 1, 2]],
        any_set: &[],
        not_both: &[([0, 0, 1], [0, 1, 2])],
        exactly_one: &[],
    },
    Template {
        set: &[[1, 1, 0], [0, 1, 1], [1, 1, 1], [1, 2, 1]],
        clear: &[[1, 0, 1], [1, 0, 2], [2, 0, 2], [1, 1, 2]],
        any_set: &[],
        not_both: &[([2, 0, 1], [2, 1, 2])],
        exactly_one: &[],
    },
    Template {
        set: &[[1, 1, 0], [1, 1, 1], [2, 1, 1], [0, 0, 2], [1, 2, 1]],
        clear: &[[1, 0, 1], [1, 0, 2], [1, 1, 2]],
        any_set: &[],
        not_both: &[],
        exactly_one: &[([0, 0, 1], [0, 1, 2])],
    },
    Template {
        set: &[[1, 1, 0], [0, 1, 1], [1, 1, 1], [2, 0, 2], [1, 2, 1]],
        clear: &[[1, 0, 1], [1, 0, 2], [1, 1, 2]],
        any_set: &[],
        not_both: &[],
        exactly_one: &[([2, 0, 1], [2, 1, 2])],
    },
    // 11-14. The four templates of the third group: a diagonal continuation with
    //    a large background region and no disjunctive clause at all.
    Template {
        set: &[[2, 1, 0], [1, 1, 1], [1, 2, 0]],
        clear: &[
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 0, 2],
            [1, 0, 2],
            [2, 0, 2],
            [0, 1, 2],
            [1, 1, 2],
            [2, 1, 2],
            [0, 2, 2],
            [1, 2, 2],
            [2, 2, 2],
        ],
        any_set: &[],
        not_both: &[],
        exactly_one: &[],
    },
    Template {
        set: &[[0, 1, 0], [1, 2, 0], [1, 1, 1]],
        clear: &[
            [1, 0, 0],
            [2, 0, 0],
            [1, 0, 1],
            [2, 0, 1],
            [0, 0, 2],
            [1, 0, 2],
            [2, 0, 2],
            [0, 1, 2],
            [1, 1, 2],
            [2, 1, 2],
            [0, 2, 2],
            [1, 2, 2],
            [2, 2, 2],
        ],
        any_set: &[],
        not_both: &[],
        exactly_one: &[],
    },
    Template {
        set: &[[1, 2, 0], [1, 1, 1], [2, 2, 1]],
        clear: &[
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [0, 0, 1],
            [1, 0, 1],
            [2, 0, 1],
            [0, 0, 2],
            [1, 0, 2],
            [2, 0, 2],
            [0, 1, 2],
            [1, 1, 2],
            [0, 2, 2],
            [1, 2, 2],
        ],
        any_set: &[],
        not_both: &[],
        exactly_one: &[],
    },
    Template {
        set: &[[1, 2, 0], [1, 1, 1], [0, 2, 1]],
        clear: &[
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [0, 0, 1],
            [1, 0, 1],
            [2, 0, 1],
            [0, 0, 2],
            [1, 0, 2],
            [2, 0, 2],
            [1, 1, 2],
            [2, 1, 2],
            [1, 2, 2],
            [2, 2, 2],
        ],
        any_set: &[],
        not_both: &[],
        exactly_one: &[],
    },
];

/// A [`Template`] with every position resolved to a bit of a 27-bit
/// neighbourhood word, so matching is a handful of masked comparisons.
///
/// Purely a change of representation: [`compile`] is the only thing that builds
/// one, from [`TEMPLATES`], and `the_compiled_templates_agree_with_the_table`
/// checks the two agree position by position.
struct CompiledTemplate {
    set: u32,
    clear: u32,
    any_set: u32,
    not_both: Vec<u32>,
    exactly_one: Vec<u32>,
}

impl CompiledTemplate {
    #[inline]
    fn matches(&self, word: u32) -> bool {
        word & self.set == self.set
            && word & self.clear == 0
            && (self.any_set == 0 || word & self.any_set != 0)
            && self
                .not_both
                .iter()
                .all(|&pair| (word & pair).count_ones() < 2)
            && self
                .exactly_one
                .iter()
                .all(|&pair| (word & pair).count_ones() == 1)
    }
}

/// The bit a template position occupies in a 27-bit neighbourhood word.
///
/// The same numbering `super::skeleton::neighbour_index` uses, so a
/// neighbourhood gathered by either module means the same thing.
fn position_bit(position: Position) -> u32 {
    1 << neighbour_index(
        position[0] as isize - 1,
        position[1] as isize - 1,
        position[2] as isize - 1,
    )
}

fn mask_of(positions: &[Position]) -> u32 {
    positions
        .iter()
        .copied()
        .map(position_bit)
        .fold(0, |a, b| a | b)
}

fn compile() -> Vec<CompiledTemplate> {
    TEMPLATES
        .iter()
        .map(|template| CompiledTemplate {
            set: mask_of(template.set),
            clear: mask_of(template.clear),
            any_set: mask_of(template.any_set),
            not_both: template
                .not_both
                .iter()
                .map(|&(a, b)| position_bit(a) | position_bit(b))
                .collect(),
            exactly_one: template
                .exactly_one
                .iter()
                .map(|&(a, b)| position_bit(a) | position_bit(b))
                .collect(),
        })
        .collect()
}

fn compiled() -> &'static [CompiledTemplate] {
    static COMPILED: OnceLock<Vec<CompiledTemplate>> = OnceLock::new();
    COMPILED.get_or_init(compile)
}

/// Whether any deleting template is satisfied by the neighbourhood `word`, which
/// carries one bit per position under [`super::skeleton::neighbour_index`].
///
/// This is the whole deletion rule. There is no separate simplicity or end-point
/// test: the templates *are* the proof obligation, discharged once in the
/// algorithm's own source rather than recomputed per voxel.
#[inline]
pub fn matches_a_template(word: u32) -> bool {
    compiled().iter().any(|template| template.matches(word))
}

// ---------------------------------------------------------- the rotations --

/// A rotation of the neighbourhood, as the integer matrix it acts by.
type Matrix = [[i8; 3]; 3];

const IDENTITY: Matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

/// A quarter turn about each axis, all three in the same handedness.
const QUARTER_TURN: [Matrix; 3] = [
    [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
    [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
    [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
];

fn multiply(left: Matrix, right: Matrix) -> Matrix {
    let mut out = [[0i8; 3]; 3];
    for (row, slot) in out.iter_mut().enumerate() {
        for (column, entry) in slot.iter_mut().enumerate() {
            *entry = (0..3).map(|k| left[row][k] * right[k][column]).sum();
        }
    }
    out
}

fn turn(axis: usize, steps: usize) -> Matrix {
    (0..steps % 4).fold(IDENTITY, |acc, _| multiply(acc, QUARTER_TURN[axis]))
}

/// One orientation, named by the turns that build it.
///
/// `[(a, i), (b, j)]` is a turn of `i` quarters about axis `a` followed by one
/// of `j` quarters about axis `b`, composed in that order. An empty list is the
/// identity.
type Orientation = &'static [(usize, usize)];

/// The twelve orientations of a pass, **in order**.
///
/// The order is the algorithm's and the header explains why transposing two
/// entries is the error that survives every cheap check. Nothing here may be
/// sorted, deduplicated or "tidied".
const ORIENTATIONS: [Orientation; SUB_ITERATIONS] = [
    &[],
    &[(2, 2), (1, 3)],
    &[(1, 2), (2, 1)],
    &[(1, 3)],
    &[(2, 1)],
    &[(1, 2), (2, 2)],
    &[(1, 1)],
    &[(2, 2)],
    &[(1, 2), (2, 3)],
    &[(2, 2), (1, 1)],
    &[(2, 3)],
    &[(1, 2)],
];

/// Where each template position reads from, for each of the twelve
/// sub-iterations.
///
/// `sources[k][t]` is the neighbourhood position (a
/// [`super::skeleton::neighbour_index`]) that template position `t` takes its
/// value from during sub-iteration `k`. Every entry is a permutation of the 27
/// positions fixing the centre, which is what makes the twelve sub-iterations
/// one predicate seen from twelve sides rather than twelve predicates.
pub fn sub_iteration_sources() -> &'static [[usize; NEIGHBOURHOOD]; SUB_ITERATIONS] {
    static SOURCES: OnceLock<[[usize; NEIGHBOURHOOD]; SUB_ITERATIONS]> = OnceLock::new();
    SOURCES.get_or_init(|| {
        let mut sources = [[0usize; NEIGHBOURHOOD]; SUB_ITERATIONS];
        for (orientation, slot) in ORIENTATIONS.iter().zip(sources.iter_mut()) {
            let matrix = orientation.iter().fold(IDENTITY, |acc, &(axis, steps)| {
                multiply(acc, turn(axis, steps))
            });
            // The template frame is the rotated one, so a template position is
            // read from the neighbourhood position the *inverse* rotation sends
            // it to; a rotation matrix's inverse is its transpose.
            for (index, source) in slot.iter_mut().enumerate() {
                let offset = neighbour_offset(index);
                let mut turned = [0isize; 3];
                for (axis, entry) in turned.iter_mut().enumerate() {
                    *entry = (0..3).map(|k| matrix[k][axis] as isize * offset[k]).sum();
                }
                *source = neighbour_index(turned[0], turned[1], turned[2]);
            }
        }
        sources
    })
}

// -------------------------------------------------------- the neighbourhood --

/// The 27-bit neighbourhood word around `at`, read through `sources`, with
/// **outside the array read as background**.
///
/// The permutation is applied while gathering rather than afterwards, which is
/// why there is one function here and not a gather plus a shuffle.
///
/// This is the form that copes with the outside; [`interior_word`] is the same
/// value where the whole neighbourhood is in the array, and
/// `the_two_gathers_agree_wherever_both_apply` asserts they never disagree.
fn oriented_word(
    data: &[bool],
    shape: [usize; 3],
    at: [usize; 3],
    sources: &[usize; NEIGHBOURHOOD],
) -> u32 {
    let mut word = 0u32;
    for (target, &source) in sources.iter().enumerate() {
        let offset = neighbour_offset(source);
        let mut flat = 0usize;
        let mut inside = true;
        for axis in 0..3 {
            let position = at[axis] as isize + offset[axis];
            if position < 0 || position >= shape[axis] as isize {
                inside = false;
                break;
            }
            flat = flat * shape[axis] + position as usize;
        }
        if inside && data[flat] {
            word |= 1 << target;
        }
    }
    word
}

/// [`oriented_word`] for a voxel whose whole neighbourhood is inside the array,
/// which is every voxel but the one-voxel shell around it.
///
/// `offsets` is `sources` resolved to flat displacements once per
/// sub-iteration, which is the only reason this exists: the general form spends
/// most of its time deciding, twenty-seven times per voxel, that the answer is
/// inside the array.
#[inline]
fn interior_word(data: &[bool], at: usize, offsets: &[isize; NEIGHBOURHOOD]) -> u32 {
    let mut word = 0u32;
    for (target, &offset) in offsets.iter().enumerate() {
        if data[at.wrapping_add_signed(offset)] {
            word |= 1 << target;
        }
    }
    word
}

/// `sources` as flat displacements into an array of this shape.
fn flat_offsets(shape: [usize; 3], sources: &[usize; NEIGHBOURHOOD]) -> [isize; NEIGHBOURHOOD] {
    let strides = [(shape[1] * shape[2]) as isize, shape[2] as isize, 1];
    let mut offsets = [0isize; NEIGHBOURHOOD];
    for (slot, &source) in offsets.iter_mut().zip(sources.iter()) {
        let offset = neighbour_offset(source);
        *slot = (0..3).map(|axis| offset[axis] * strides[axis]).sum();
    }
    offsets
}

/// The bit a non-centre neighbourhood position occupies in a **26-bit** index.
///
/// The 27-bit word above is the natural form for the predicate; a lookup table
/// wants a dense domain, and dropping the centre — which is object by
/// construction wherever the predicate is asked — gives `2^26` rather than
/// `2^27`. The two forms differ only by that one bit and
/// [`is_deletable_index`] is the bridge.
#[inline]
pub fn index_bit(neighbourhood_index: usize) -> u32 {
    debug_assert!(neighbourhood_index != CENTRE);
    (neighbourhood_index - usize::from(neighbourhood_index > CENTRE)) as u32
}

/// Bits in the dense index [`is_deletable_index`] takes.
pub const INDEX_BITS: u32 = 26;

/// Entries a full table over that index would have.
pub const TABLE_LEN: usize = 1 << INDEX_BITS;

/// The deletion rule as a function of a dense 26-bit neighbourhood index, the
/// centre taken to be object.
///
/// This is what a generated lookup table would be filled from, and it is the
/// only definition of the rule in dense-index form. See the header for why the
/// table itself is not built here.
pub fn is_deletable_index(index: u32) -> bool {
    debug_assert!((index as usize) < TABLE_LEN);
    let mut word = 1u32 << CENTRE;
    for position in 0..NEIGHBOURHOOD {
        if position != CENTRE && (index >> index_bit(position)) & 1 != 0 {
            word |= 1 << position;
        }
    }
    matches_a_template(word)
}

/// Sixty-four entries of a generated table, packed one bit per index.
///
/// Offered so that a caller for whom the predicate is the bottleneck can build
/// `TABLE_LEN / 64` of these — in parallel, in any order — and get a table that
/// is by construction the predicate in this file. `word * 64 + bit` is the index
/// bit `bit` stands for.
pub fn deletion_table_word(word: u32) -> u64 {
    let base = word << 6;
    (0..64)
        .filter(|offset| is_deletable_index(base | offset))
        .fold(0u64, |bits, offset| bits | 1u64 << offset)
}

// --------------------------------------------------------------- the pass --

/// The voxels a pass may consider: object voxels with fewer than six object
/// face neighbours.
///
/// **Computed once, from the volume as it stands at the start of the pass.** The
/// header explains why it is then not recomputed; this function exists as a
/// separate, testable thing precisely so that the staleness is visible in the
/// signature of [`directional_pass_with_sources_into`] rather than buried in it.
pub fn border_mask(input: ArrayView3<'_, bool>) -> Array3<bool> {
    let shape = [input.shape()[0], input.shape()[1], input.shape()[2]];
    let mut mask = Array3::from_elem(shape, false);
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                if !input[[i, j, k]] {
                    continue;
                }
                let mut faces = 0;
                for axis in 0..3 {
                    for step in [-1isize, 1] {
                        let mut position = [i as isize, j as isize, k as isize];
                        position[axis] += step;
                        if (0..3).all(|a| position[a] >= 0 && position[a] < shape[a] as isize)
                            && input[[
                                position[0] as usize,
                                position[1] as usize,
                                position[2] as usize,
                            ]]
                        {
                            faces += 1;
                        }
                    }
                }
                mask[[i, j, k]] = faces < 6;
            }
        }
    }
    mask
}

/// The flat index of a position in a C-ordered array of this shape.
#[inline]
fn flat_index(shape: [usize; 3], at: [usize; 3]) -> usize {
    (at[0] * shape[1] + at[1]) * shape[2] + at[2]
}

/// One sub-iteration: delete every voxel of `border` that is still object and
/// whose neighbourhood, read through `sources`, satisfies a deleting template.
///
/// **Batched, not in place.** Every decision is taken against `current` as it
/// stands on entry and applied afterwards, so the answer is a function of the
/// input volume and not of the order voxels are visited in. That is what lets a
/// pass be split across blocks at all, and it is why this takes a separate
/// scratch buffer rather than writing as it goes.
///
/// Returns how many voxels it deleted.
fn sub_iteration(
    current: &mut [bool],
    shape: [usize; 3],
    border: &[bool],
    sources: &[usize; NEIGHBOURHOOD],
    doomed: &mut Vec<usize>,
) -> usize {
    doomed.clear();
    let offsets = flat_offsets(shape, sources);
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            let interior_row = i > 0 && i + 1 < shape[0] && j > 0 && j + 1 < shape[1];
            for k in 0..shape[2] {
                let at = flat_index(shape, [i, j, k]);
                if !border[at] || !current[at] {
                    continue;
                }
                let word = if interior_row && k > 0 && k + 1 < shape[2] {
                    interior_word(current, at, &offsets)
                } else {
                    oriented_word(current, shape, [i, j, k], sources)
                };
                if matches_a_template(word) {
                    doomed.push(at);
                }
            }
        }
    }
    for &at in doomed.iter() {
        current[at] = false;
    }
    doomed.len()
}

/// One sub-iteration, over a border set the caller supplies.
///
/// The border set is an *argument* rather than something this computes, and
/// that is the whole point of the signature: a pass takes it once and hands the
/// same one to all twelve sub-iterations, and a caller who recomputed it between
/// them would be running a different algorithm. Exposed so that
/// `the_border_set_is_stale_within_a_pass` can build that different algorithm
/// and measure the difference rather than assert a comment.
pub fn directional_sub_iteration_into(
    input: ArrayView3<'_, bool>,
    border: ArrayView3<'_, bool>,
    sources: &[usize; NEIGHBOURHOOD],
    mut out: ArrayViewMut3<'_, bool>,
) -> Result<()> {
    shapes_agree(input.shape(), out.shape(), "directional_sub_iteration_into")?;
    shapes_agree(
        input.shape(),
        border.shape(),
        "directional_sub_iteration_into: border set",
    )?;
    let shape = [input.shape()[0], input.shape()[1], input.shape()[2]];
    let mut current: Vec<bool> = input.iter().copied().collect();
    let border: Vec<bool> = border.iter().copied().collect();
    let mut doomed = Vec::new();
    sub_iteration(&mut current, shape, &border, sources, &mut doomed);
    for (slot, value) in out.iter_mut().zip(current) {
        *slot = value;
    }
    Ok(())
}

/// One full pass: the border set taken once, then the twelve sub-iterations
/// through `sources`, in the order `sources` gives them.
///
/// The orientation table is an argument rather than a constant so that the order
/// can be asserted on — see the header. Callers who want the algorithm pass
/// [`sub_iteration_sources`], which is what [`directional_pass_into`] does.
pub fn directional_pass_with_sources_into(
    input: ArrayView3<'_, bool>,
    sources: &[[usize; NEIGHBOURHOOD]; SUB_ITERATIONS],
    mut out: ArrayViewMut3<'_, bool>,
) -> Result<()> {
    shapes_agree(
        input.shape(),
        out.shape(),
        "directional_pass_with_sources_into",
    )?;
    let shape = [input.shape()[0], input.shape()[1], input.shape()[2]];
    let mut current: Vec<bool> = input.iter().copied().collect();
    let border: Vec<bool> = border_mask(input).into_iter().collect();
    let mut doomed = Vec::new();
    for source in sources.iter() {
        sub_iteration(&mut current, shape, &border, source, &mut doomed);
    }
    for (slot, value) in out.iter_mut().zip(current) {
        *slot = value;
    }
    Ok(())
}

/// One full pass of the algorithm.
pub fn directional_pass_into(
    input: ArrayView3<'_, bool>,
    out: ArrayViewMut3<'_, bool>,
) -> Result<()> {
    directional_pass_with_sources_into(input, sub_iteration_sources(), out)
}

/// `passes` full passes.
///
/// Zero passes is the identity, which is the honest answer for that parameter
/// rather than an error — the same reading `super::skeleton::thin` gives it.
pub fn directional_passes_into(
    input: ArrayView3<'_, bool>,
    passes: usize,
    mut out: ArrayViewMut3<'_, bool>,
) -> Result<()> {
    shapes_agree(input.shape(), out.shape(), "directional_passes_into")?;
    let mut current = input.to_owned();
    for _ in 0..passes {
        let source = current.clone();
        directional_pass_into(source.view(), current.view_mut())?;
    }
    out.assign(&current);
    Ok(())
}

/// Thin until a pass changes nothing, over the **whole volume**, returning the
/// fixed point and the number of passes it took.
///
/// Not a [`BlockOp`] and deliberately not dressed as one: the pass count is a
/// function of the data. The header says what a caller who cannot hold the
/// volume does instead, and why the alternative that suggests itself — letting
/// each block decide it has converged — is wrong.
///
/// **It terminates**: every pass either deletes a voxel or is the last, and the
/// object is finite. `limit` is required anyway, and exceeding it is an error
/// rather than a truncation, for the reason
/// [`super::skeleton::PassLimit`] states.
pub fn directional_to_fixed_point(
    input: ArrayView3<'_, bool>,
    limit: PassLimit,
) -> Result<(Array3<bool>, usize)> {
    let mut current = input.to_owned();
    let mut passes = 0;
    loop {
        let mut next = current.clone();
        directional_pass_into(current.view(), next.view_mut())?;
        passes += 1;
        if next == current {
            return Ok((current, passes));
        }
        current = next;
        if passes >= limit.passes() {
            return Err(Error::InvalidArgument(format!(
                "directional thinning did not reach a fixed point in {} pass(es) over a {:?} \
                 volume. Either the limit is below what this data needs — raise it, or take it \
                 from `PassLimit::for_volume` — or a pass has stopped being monotone, which \
                 would be a defect in the deletion rule rather than in the data. The partially \
                 thinned volume is deliberately not returned: it is a plausible, well-formed, \
                 wrong skeleton.",
                limit.passes(),
                [input.shape()[0], input.shape()[1], input.shape()[2]]
            )));
        }
    }
}

// ------------------------------------------------- the volume-level margin --

/// Clear the six faces of `volume`.
///
/// **A whole-volume operation, and never a per-block one.** The algorithm's
/// precondition is that the object does not touch the boundary of the array it
/// is thinned in; imposing it on a block's buffer instead would zero a face of
/// every interior block, which is a hole along every seam. Nothing in
/// [`DirectionalPassOp`] calls this, and a pipeline applies it once to the
/// volume before the first round.
pub fn clear_faces(volume: &mut Array3<bool>) {
    let shape = [volume.shape()[0], volume.shape()[1], volume.shape()[2]];
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let on_face = (0..3).any(|axis| {
                    let position = [i, j, k][axis];
                    position == 0 || position + 1 == shape[axis]
                });
                if on_face {
                    volume[[i, j, k]] = false;
                }
            }
        }
    }
}

/// Whether the six faces of `volume` are free of object voxels: the
/// precondition [`clear_faces`] imposes, checkable without imposing it.
pub fn faces_are_clear(volume: ArrayView3<'_, bool>) -> bool {
    let shape = [volume.shape()[0], volume.shape()[1], volume.shape()[2]];
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let on_face = (0..3).any(|axis| {
                    let position = [i, j, k][axis];
                    position == 0 || position + 1 == shape[axis]
                });
                if on_face && volume[[i, j, k]] {
                    return false;
                }
            }
        }
    }
    true
}

// -------------------------------------------------------------- the shell --

/// One full pass as a [`BlockOp`].
pub struct DirectionalPassOp {
    name: &'static str,
    cost: f64,
}

impl DirectionalPassOp {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            cost: DIRECTIONAL_PASS_COST,
        }
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }
}

impl Default for DirectionalPassOp {
    fn default() -> Self {
        Self::new("directional thinning pass")
    }
}

impl BlockOp for DirectionalPassOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// **Twelve**, on every axis, and it is derived rather than configured: one
    /// voxel per sub-iteration, [`SUB_ITERATIONS`] of them, plus a border set
    /// that is taken before the first of them and therefore costs nothing
    /// further. There is no field that sets it. The header derives it in full
    /// and says why the number that measurement would support is smaller and
    /// would still be wrong to declare.
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        SUB_ITERATIONS
    }

    /// A mask, held as a mask or held as `f64` — the arrangement
    /// `super::skeleton` uses, for the same reason: the kernel is a `bool`
    /// kernel, and the `f64` arm is the shell narrowing on the way in and
    /// widening on the way out.
    fn accepts(&self, dtype: Dtype) -> bool {
        matches!(dtype, Dtype::Bool | Dtype::F64)
    }

    /// **`Anchor` is unused, and that is a property of the algorithm.** Every
    /// clause of the rule is stated in offsets from the voxel under test, so a
    /// block's answer is a function of its buffer and nothing else.
    /// `super::skeleton` cannot say this — a parity class is a fact about
    /// absolute position — and it is the one respect in which this op is the
    /// easier of the two to decompose.
    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        match input.dtype() {
            Dtype::Bool => directional_pass_into(input.view::<bool>()?, out.view_mut::<bool>()?),
            _ => {
                let mask = input.view::<f64>()?.mapv(is_set);
                let mut result = Array3::from_elem(mask.raw_dim(), false);
                directional_pass_into(mask.view(), result.view_mut())?;
                let mut out = out.view_mut::<f64>()?;
                ndarray::Zip::from(&mut out)
                    .and(&result)
                    .for_each(|slot, &value| *slot = from_set(value));
                Ok(())
            }
        }
    }

    /// **Clear maps to clear, and set maps to nothing at all**, exactly as in
    /// `super::skeleton` and for the same reason. An empty block has nothing to
    /// delete. A full block does: its interior voxels have six object face
    /// neighbours and are not border voxels, but the voxels on the *buffer's own
    /// faces* see the outside as background, so the output of an all-set block
    /// is not constant and no declaration may be made for it.
    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        (!is_set(value)).then_some(0.0)
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

// ---------------------------------------------------------- the assembly --

/// One pass, as a chain of one op.
///
/// A single op rather than a `Chain::Sequence` of twelve, and the header says
/// why it has to be: a sub-iteration reads the pass's border set as well as the
/// volume, and a `Chain::Sequence` can thread only the volume.
pub fn directional_pass() -> Chain {
    Chain::op(DirectionalPassOp::default())
}

/// `passes` full passes, reaching `12 * passes`.
///
/// Zero passes is the empty sequence, which `Chain::apply` defines as the
/// identity.
pub fn directional_thin(passes: usize) -> Chain {
    Chain::sequence((0..passes).map(|_| directional_pass()).collect())
}

/// What [`directional_thin`] reads beyond the voxel it writes, per axis.
///
/// The second statement of one quantity, and that is what it is for: the
/// authority is `Chain::reach`, and this derives the same number from the two
/// facts it follows from so a caller sizing a halo before it has a chain can
/// ask. The test asserts the two agree.
pub fn directional_reach(passes: usize) -> usize {
    passes * SUB_ITERATIONS
}

/// Cost of one pass per voxel, relative to a voxelwise map.
///
/// Measured by [`cost_report`], which is runnable. Twelve sub-iterations, each
/// of which gathers a permuted 3x3x3 neighbourhood into a word and runs the
/// fourteen masked template comparisons over it, at the voxels the pass's border
/// set admits.
///
/// **Data-dependent by about a factor of three**, the same caveat
/// `super::skeleton::THINNING_COST` carries and for the same reason: the
/// template comparisons run only where the border mask admits a voxel, so solid
/// structure is cheap and speckle is not. Measured at **99** relative units over
/// solid blocks and **315** over speckle on a 64-cubed volume; the constant is
/// the geometric middle of that range rather than either end, because a planner
/// wants the ordering between op families right and cannot be given a number
/// that is right for both.
///
/// For scale, `super::skeleton::THINNING_COST` is 14 for one *sub-iteration*,
/// and a pass there is eight of them: the two families cost about the same per
/// sub-iteration, twelve against eight.
pub const DIRECTIONAL_PASS_COST: f64 = 175.0;

/// How [`DIRECTIONAL_PASS_COST`] was obtained.
pub const COST_MEASUREMENT: &str = "ops::directional::cost_report";

/// Time one pass over a synthetic volume and print the per-voxel figure the
/// constant above was read off.
///
/// Run with
/// `cargo test --release -- --ignored --nocapture ops::directional::cost`.
pub fn cost_report(shape: [usize; 3], repetitions: usize) -> String {
    use std::time::Instant;

    use crate::op::Anchor;

    let voxels = (shape[0] * shape[1] * shape[2]) as f64;
    let repetitions = repetitions.max(1);
    let anchor = Anchor::whole(shape);
    let best_of = |mut run: Box<dyn FnMut()>| -> f64 {
        // One untimed run first: a freshly allocated output pays a page fault
        // per page on first touch, which is most of the measurement for the
        // cheapest case here.
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
        let anchor = anchor.clone();
        rows.push((
            "voxelwise map (the unit)".to_string(),
            best_of(Box::new(move || {
                op.apply(&input, &mut out, &anchor).unwrap();
            })),
        ));
    }

    for (what, mask) in [
        (
            "pass, solid blocks (little of it on a border)",
            Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
                (i / 8 + j / 8 + k / 8) % 4 != 0
            }),
        ),
        (
            "pass, speckle (nearly all of it on a border)",
            Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
                (i * 31 + j * 17 + k * 7) % 5 < 2
            }),
        ),
    ] {
        let input: Voxels = mask.into();
        let op = DirectionalPassOp::default();
        let mut out = Voxels::zeros(Dtype::Bool, shape).unwrap();
        let anchor = anchor.clone();
        rows.push((
            what.to_string(),
            best_of(Box::new(move || {
                op.apply(&input, &mut out, &anchor).unwrap();
            })),
        ));
    }

    let unit = rows.first().map(|(_, nanos)| *nanos).unwrap_or(1.0);
    let mut out = format!(
        "directional cost, {}x{}x{}, best of {repetitions}\n{:<48} {:>10} {:>10} {:>10}\n",
        shape[0], shape[1], shape[2], "case", "ns/voxel", "relative", "stored"
    );
    for (name, nanos) in &rows {
        out.push_str(&format!(
            "{name:<48} {nanos:>10.3} {:>10.2} {DIRECTIONAL_PASS_COST:>10.2}\n",
            nanos / unit
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_twelve_orientations_are_distinct_permutations_that_fix_the_centre() {
        let sources = sub_iteration_sources();
        for source in sources.iter() {
            assert_eq!(source[CENTRE], CENTRE, "a rotation moved the centre");
            let mut seen = [false; NEIGHBOURHOOD];
            for &position in source.iter() {
                assert!(!seen[position], "a rotation was not a permutation");
                seen[position] = true;
            }
        }
        for (a, first) in sources.iter().enumerate() {
            for second in sources.iter().skip(a + 1) {
                assert_ne!(first, second, "two of the twelve orientations coincide");
            }
        }
    }

    #[test]
    fn the_compiled_templates_agree_with_the_table() {
        for (template, compiled) in TEMPLATES.iter().zip(compiled()) {
            assert_eq!(mask_of(template.set), compiled.set);
            assert_eq!(mask_of(template.clear), compiled.clear);
            assert_eq!(mask_of(template.any_set), compiled.any_set);
            assert_eq!(template.not_both.len(), compiled.not_both.len());
            assert_eq!(template.exactly_one.len(), compiled.exactly_one.len());
            for (&(a, b), &pair) in template.not_both.iter().zip(&compiled.not_both) {
                assert_eq!(position_bit(a) | position_bit(b), pair);
            }
            for (&(a, b), &pair) in template.exactly_one.iter().zip(&compiled.exactly_one) {
                assert_eq!(position_bit(a) | position_bit(b), pair);
            }
        }
    }

    #[test]
    fn a_template_never_asks_for_a_position_in_two_states_at_once() {
        for template in TEMPLATES {
            let set = mask_of(template.set);
            let clear = mask_of(template.clear);
            assert_eq!(set & clear, 0, "a template requires a position both ways");
            assert_eq!(
                set & mask_of(template.any_set),
                0,
                "a disjunction repeats a position the template already requires"
            );
        }
    }

    #[test]
    fn the_dense_index_and_the_neighbourhood_word_are_the_same_rule() {
        // The bridge between the two forms, checked on the low indices rather
        // than on all `2 ** 26` of them: a full sweep is a table build, which is
        // exactly what this file declines to do.
        for index in 0..4096u32 {
            let mut word = 1u32 << CENTRE;
            for position in 0..NEIGHBOURHOOD {
                if position != CENTRE && (index >> index_bit(position)) & 1 != 0 {
                    word |= 1 << position;
                }
            }
            assert_eq!(is_deletable_index(index), matches_a_template(word));
        }
    }

    #[test]
    fn a_generated_table_word_is_the_predicate() {
        for word in 0..64u32 {
            let bits = deletion_table_word(word);
            for offset in 0..64u32 {
                assert_eq!(
                    (bits >> offset) & 1 != 0,
                    is_deletable_index((word << 6) | offset)
                );
            }
        }
    }

    #[test]
    fn the_two_gathers_agree_wherever_both_apply() {
        // The fast path drops the bounds test, so it is right only away from
        // the array's own shell. Everywhere it applies it must be the same
        // value as the general form, or the interior and the seam of a block
        // would be thinned by two different rules.
        let shape = [7usize, 6, 5];
        let mut bits = 0x9e3779b97f4a7c15u64;
        let data: Vec<bool> = (0..shape[0] * shape[1] * shape[2])
            .map(|_| {
                bits ^= bits << 13;
                bits ^= bits >> 7;
                bits ^= bits << 17;
                bits & 3 != 0
            })
            .collect();
        for sources in sub_iteration_sources().iter() {
            let offsets = flat_offsets(shape, sources);
            for i in 1..shape[0] - 1 {
                for j in 1..shape[1] - 1 {
                    for k in 1..shape[2] - 1 {
                        let at = flat_index(shape, [i, j, k]);
                        assert_eq!(
                            interior_word(&data, at, &offsets),
                            oriented_word(&data, shape, [i, j, k], sources)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn an_isolated_voxel_is_never_deletable() {
        assert!(!matches_a_template(1 << CENTRE));
        assert!(!is_deletable_index(0));
    }

    #[test]
    #[ignore = "a measurement, not an assertion"]
    fn cost() {
        println!("{}", cost_report([64, 64, 64], 3));
    }
}
