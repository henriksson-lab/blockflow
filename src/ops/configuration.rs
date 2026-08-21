// SPDX-License-Identifier: MIT
//
// Original work for this crate. Written from the definition of the operation,
// not adapted from any implementation of it.
//
// A binary volume rewritten by a **table indexed on the 3x3x3 neighbourhood**.
//
// What it computes
// ----------------
// Every voxel's 3x3x3 neighbourhood is 27 bits. Read them in a fixed order and
// they are an integer in `0..2^27` — the voxel's *configuration* — and a table
// with one entry per configuration says what the voxel becomes. One pass writes
// a fresh volume from a volume it never mutates, so within a pass every voxel is
// independent; a run is some number of passes.
//
// That is the whole of it, and the generality is the point: the table is the
// caller's, so the same op is a majority vote, a boundary-preserving smoothing,
// a hit-or-miss transform or a cellular automaton depending on what is in it.
// Nothing here knows which, and nothing here ships a table beyond
// [`ConfigurationTable::identity`], which is the one whose meaning is fixed by
// the index convention rather than chosen.
//
// Why a table and not a predicate
// -------------------------------
// A predicate over 27 booleans and a table over 2^27 entries are the same
// function, and the table is the better representation for three reasons that
// are all about *this* crate rather than about speed:
//
// * a table is **data**, so a caller that derives one from a rule can check the
//   derivation against the rule once, at 2^27 configurations, instead of hoping
//   a hot predicate stayed faithful under optimisation;
// * a table has **no branch structure to disagree with itself** between the
//   whole-volume reference and the block-decomposed run, which is the comparison
//   every op here has to pass;
// * a table is 16 MiB bit-packed — an `Arc` per op, resident, shared by every
//   block and every thread — while an equivalent predicate is re-evaluated per
//   voxel per pass.
//
// The cost is that building one is the caller's problem, and a naive build
// evaluates a rule 134 million times. [`ConfigurationTable::assign_matching`]
// exists so that it need not: a rule stated as *templates* — these neighbours
// set, these clear, the rest free — is written into the table in one word
// operation per free high bit combination, not one per configuration.
//
// Two shells, because there are two questions
// -------------------------------------------
// | shell | what it runs | reach | stops when |
// |---|---|---|---|
// | [`ConfigurationPassOp`] | a **stated** number of passes | `passes`, because one pass reaches one voxel and the next pass reads what that one wrote | the count is reached |
// | [`ConfigurationFixedPointOp`] | one pass per substage | **1**, whatever the count | nothing anywhere changed |
//
// They are not two ways of saying one thing and the difference is not
// stylistic. A stated count is a *definition*: "three passes" has a three in the
// answer, the answer exists whether or not the sequence has settled, and the
// halo is three because the third pass reads what the second wrote two voxels
// away. Running to a fixed point has no count in its definition at all, so no
// halo can be derived from one — and it does not need to be, because
// [`crate::iterate`] pays the depth in private round trips instead: substage
// `k+1` reads the *cores its neighbours wrote* at `k`, so the phase's external
// reach is one substage's however many substages there are.
//
// **Convergence is a global predicate, and only a global one.** A voxel that
// changes can change its neighbour on the next pass, and that neighbour may sit
// in another block; so "this block did not change" is not "this block will never
// change again", and a per-block fixed point is a different and wrong thing.
// `crate::strategy`'s executor is already arranged for this — it runs *every*
// block at *every* substage and only stops when no block anywhere changed — so
// the fixed-point shell inherits the right predicate rather than implementing
// one. What it costs is that a run is as deep as its slowest region: a volume
// that settles everywhere but one corner keeps paying for the whole volume until
// that corner settles. A dirty set would fix that and is the executor's to build.
//
// **Neither shell is a planning barrier, and that was checked rather than
// assumed.** An op that reaches a whole axis declares `AxisReach::All` and gets
// its own phase over one block — `crate::decomposition::is_planning_barrier` —
// which is the honest shape for something whose answer is not expressible over
// blocks at all. It is not the shape here, for either shell, and the difference
// is worth stating because "iterative" reads like "global":
//
// * a stated pass count is exactly expressible at a halo of `passes`. At six
//   passes over a 256^3 core that is 19.25 Mvoxel read for 16.8 written — **15%**
//   more IO, and nothing resident but one block;
// * a fixed point is exactly expressible at a halo of **one**, because the
//   executor's convergence test is already the global disjunction over blocks.
//   What it costs is not halo but *residency*: `crate::iterate` allocates two
//   private whole-volume buffers, so the phase's working set is twice the volume
//   however it is blocked.
//
// So the arithmetic decides which shell a large run wants, and it is a real
// decision rather than a preference. A mask of 1.8 Gvoxel is 1.8 GB as `bool`,
// and two buffers of it are 3.5 GB — comfortable on any node that could hold the
// intensities it came from. A mask of 94 Gvoxel is 94 GB, and two buffers are
// **188 GB**, which no such node has. The stated-count shell streams at either
// size; the fixed-point shell does not, and a caller reaching for it on a volume
// that does not fit should know that from here rather than from an allocator.
//
// **The runaway limit is required and is not a decoration.** For a general table
// there is no bound derivable from the volume, because there is no argument that
// the iteration terminates at all: the table that maps every configuration to
// the complement of its centre has period two on any non-empty volume and
// converges nowhere. So [`ConfigurationFixedPointOp::new`] takes a
// [`SubstageLimit`] from the caller and there is no `bound_for` here to imply
// otherwise. Whether a *particular* table converges is a property of that table
// and belongs with whoever built it.
//
// The index convention, which is load-bearing
// -------------------------------------------
// Bit `(di + 1) + 3 * (dj + 1) + 9 * (dk + 1)` of the configuration is the voxel
// at offset `(di, dj, dk)` from the centre, each offset in `-1..=1`. The centre
// is therefore bit **13**, axis 0 is the fastest-varying term and axis 2 the
// slowest. Any consistent convention would do — but a table is data, and a table
// built under one convention and probed under another is silently a different
// operation, so the convention is stated here, is the only one used, and
// [`configuration_bit`] is the only place it is written down.
//
// Edge behaviour
// --------------
// **Neighbours outside the array read as clear.** The neighbourhood is resolved
// against the array the op is handed, as everything in this module is, and the
// bits for positions that do not exist are simply left at zero. At a real volume
// boundary that is the definition of the operation and the whole-volume
// reference resolves it identically; at a block seam it is deliberately *wrong*,
// which is what turns a short halo into a visible disagreement rather than a
// plausible one.

use std::sync::Arc;

use ndarray::{Array3, ArrayView3, ArrayViewMut3};

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::iterate::{IterativeOp, Substage, SubstageLimit, SubstageOperand};
use crate::op::{Anchor, BlockOp};
use crate::voxels::Voxels;

use super::shapes_agree;
use super::voxelwise::{from_set, is_set};

/// Voxels in a 3x3x3 neighbourhood, and therefore bits in a configuration.
pub const CONFIGURATION_BITS: u32 = 27;

/// How many entries a table has: one per configuration.
pub const CONFIGURATION_COUNT: usize = 1 << CONFIGURATION_BITS;

/// The bit the voxel itself occupies. `configuration_bit([0, 0, 0])`, and the
/// one entry point that hard-codes it is [`ConfigurationTable::identity`].
pub const CENTRE_BIT: u32 = 13;

/// Which bit of a configuration a neighbour occupies.
///
/// The convention this module is defined by; see the header. Refuses an offset
/// outside `-1..=1` rather than wrapping it, because a wrapped offset would
/// address a real bit belonging to a different neighbour and would be wrong
/// without being detectable.
pub fn configuration_bit(offset: [isize; 3]) -> Result<u32> {
    let mut bit = 0u32;
    for axis in (0..3).rev() {
        let step = offset[axis];
        if !(-1..=1).contains(&step) {
            return Err(Error::InvalidArgument(format!(
                "a 3x3x3 neighbourhood offset is -1, 0 or 1 on every axis, and this one is \
                 {offset:?}. A configuration has {CONFIGURATION_BITS} bits and there is no bit \
                 for a voxel two away."
            )));
        }
        bit = bit * 3 + (step + 1) as u32;
    }
    Ok(bit)
}

/// A set of configurations, stated as what must be set, what must be clear, and
/// nothing at all about the rest.
///
/// This is how a rule gets into a table without 2^27 evaluations of the rule.
/// A template naming `s` set positions and `c` clear positions matches
/// `2^(27 - s - c)` configurations, and [`ConfigurationTable::assign_matching`]
/// writes all of them without visiting them one at a time.
///
/// Built by accumulation rather than from two integers, so that a position named
/// twice with two different demands is an **error** at the place it is named. A
/// template requiring a voxel both set and clear matches nothing; it is
/// well-formed, it silently contributes no configurations, and a rule assembled
/// out of such templates would be quietly missing whichever cases the mistake
/// covered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ConfigurationTemplate {
    set: u32,
    clear: u32,
}

impl ConfigurationTemplate {
    /// Every configuration: nothing is required either way.
    pub fn any() -> Self {
        Self::default()
    }

    /// The same, from two masks over [`configuration_bit`]'s bit positions.
    ///
    /// The accumulating form is the one to reach for while *writing* a rule, and
    /// this one is for a caller **generating** templates rather than writing
    /// them: a rule of the form "any six of these twenty-six clear" is hundreds
    /// of thousands of templates, and going through one `with` call per position
    /// per template makes building the table dominated by the building rather
    /// than by the table. Refuses the same overlap for the same reason, and
    /// refuses a bit outside the neighbourhood, which a hand-built mask can have
    /// and an offset cannot.
    pub fn from_masks(set: u32, clear: u32) -> Result<Self> {
        let outside = (set | clear) & !(CONFIGURATION_COUNT as u32 - 1);
        if outside != 0 {
            return Err(Error::InvalidArgument(format!(
                "a template's masks address bits {outside:#x} outside the \
                 {CONFIGURATION_BITS} of a configuration; there is no voxel there."
            )));
        }
        if set & clear != 0 {
            return Err(Error::InvalidArgument(format!(
                "a template requires the voxels at bits {:#x} to be both set and clear, so it \
                 matches no configuration at all — a rule that silently covers nothing rather \
                 than a rule that covers something.",
                set & clear
            )));
        }
        Ok(Self { set, clear })
    }

    /// Also require the voxel at `offset` to be set.
    pub fn with_set(self, offset: [isize; 3]) -> Result<Self> {
        self.with(offset, true)
    }

    /// Also require the voxel at `offset` to be clear.
    pub fn with_clear(self, offset: [isize; 3]) -> Result<Self> {
        self.with(offset, false)
    }

    /// Also require the voxel at `offset` to hold `value`.
    pub fn with(mut self, offset: [isize; 3], value: bool) -> Result<Self> {
        let bit = 1u32 << configuration_bit(offset)?;
        let (wanted, opposite) = if value {
            (&mut self.set, self.clear)
        } else {
            (&mut self.clear, self.set)
        };
        if opposite & bit != 0 {
            return Err(Error::InvalidArgument(format!(
                "the voxel at {offset:?} is already required to be {} by this template, and is \
                 now required to be {}. A template demanding both matches no configuration at \
                 all, which is a rule that silently covers nothing rather than a rule that \
                 covers something.",
                if value { "clear" } else { "set" },
                if value { "set" } else { "clear" }
            )));
        }
        *wanted |= bit;
        Ok(self)
    }

    /// The mask of positions required set.
    pub fn set_mask(&self) -> u32 {
        self.set
    }

    /// The mask of positions required clear.
    pub fn clear_mask(&self) -> u32 {
        self.clear
    }

    /// Positions this template says nothing about.
    pub fn free_mask(&self) -> u32 {
        !(self.set | self.clear) & (CONFIGURATION_COUNT as u32 - 1)
    }

    /// How many configurations match. `2` to the number of free positions.
    pub fn matches(&self) -> usize {
        1usize << self.free_mask().count_ones()
    }

    /// Does `configuration` match?
    pub fn admits(&self, configuration: u32) -> bool {
        configuration & self.set == self.set && configuration & self.clear == 0
    }

    /// The template with every requirement inverted: set for clear, clear for
    /// set, and the free positions still free.
    ///
    /// A rule expressed over the *complement* of a neighbourhood — "this pattern,
    /// with object and background exchanged" — is this, and stating it as one
    /// method keeps a caller from writing the two lists out twice and getting one
    /// of them wrong.
    pub fn inverted(self) -> Self {
        Self {
            set: self.clear,
            clear: self.set,
        }
    }
}

/// One entry per configuration, one bit per entry.
///
/// 2^27 bits is **16 MiB**, which fits in cache on the machines this crate runs
/// on and is shared read-only by every block and every thread of a phase. The
/// same table as one byte per entry is 128 MiB and would be read from memory
/// rather than from cache at every voxel.
#[derive(Clone, PartialEq, Eq)]
pub struct ConfigurationTable {
    words: Vec<u64>,
}

impl std::fmt::Debug for ConfigurationTable {
    /// The contents are 16 MiB of bits and printing them helps nobody; what a
    /// reader of a failure message wants is which table this is, and the count is
    /// the cheapest thing that distinguishes two of them.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ConfigurationTable {{ entries: {CONFIGURATION_COUNT}, set: {} }}",
            self.count_set()
        )
    }
}

impl ConfigurationTable {
    const LOW_BITS: u32 = 6;
    const LOW_MASK: u32 = (1 << Self::LOW_BITS) - 1;
    const HIGH_MASK: u32 = (CONFIGURATION_COUNT as u32 - 1) >> Self::LOW_BITS;

    /// Every configuration maps to clear.
    pub fn zeros() -> Self {
        Self {
            words: vec![0u64; CONFIGURATION_COUNT / 64],
        }
    }

    /// Every configuration maps to its own centre voxel: the table that changes
    /// nothing, at any number of passes.
    ///
    /// The one table this module ships, because it is the only one whose entries
    /// are fixed by the index convention rather than chosen by a caller — and
    /// because a rule is almost always stated as departures from it.
    pub fn identity() -> Self {
        let mut table = Self::zeros();
        table.assign_matching(
            &ConfigurationTemplate::any()
                .with_set([0, 0, 0])
                .expect("the centre is a neighbourhood offset"),
            true,
        );
        table
    }

    /// What `configuration` maps to.
    ///
    /// Infallible on a `u32` that fits in [`CONFIGURATION_BITS`]; a wider value
    /// is masked rather than refused, because this is the per-voxel probe and a
    /// `Result` here would be a branch on a condition the kernel establishes
    /// once. [`configuration_index_at`] can only produce a 27-bit value.
    #[inline]
    pub fn get(&self, configuration: u32) -> bool {
        let index = (configuration & (CONFIGURATION_COUNT as u32 - 1)) as usize;
        self.words[index >> 6] >> (index & 63) & 1 != 0
    }

    /// Set one entry.
    pub fn assign(&mut self, configuration: u32, value: bool) {
        let index = (configuration & (CONFIGURATION_COUNT as u32 - 1)) as usize;
        let bit = 1u64 << (index & 63);
        if value {
            self.words[index >> 6] |= bit;
        } else {
            self.words[index >> 6] &= !bit;
        }
    }

    /// Set **every** entry the template matches, without visiting them one at a
    /// time.
    ///
    /// This is what makes a rule-derived table affordable, and the derivation is
    /// worth stating because it is the whole reason [`ConfigurationTemplate`]
    /// exists. Split a configuration into its low six bits — the offset within a
    /// 64-bit word — and its high twenty-one — the word. The template's demands
    /// on the low bits are the same for every word, so they are one 64-bit mask
    /// computed once; its demands on the high bits pick out which words, and the
    /// words it leaves free are enumerated as submasks. So the work is one
    /// `|=` or `&=` per matching *word*, and a template with twenty free
    /// positions costs about 32 thousand word operations rather than a million
    /// bit operations.
    ///
    /// Submask enumeration — `sub = (sub - 1) & free` — visits each submask once
    /// in constant time per submask, which is what keeps the *whole* build linear
    /// in the number of matching words rather than in the number of free bits
    /// times it.
    pub fn assign_matching(&mut self, template: &ConfigurationTemplate, value: bool) {
        let low_set = template.set & Self::LOW_MASK;
        let low_clear = template.clear & Self::LOW_MASK;
        let mut within_word = 0u64;
        for offset in 0..64u32 {
            if offset & low_set == low_set && offset & low_clear == 0 {
                within_word |= 1u64 << offset;
            }
        }

        let high_set = (template.set >> Self::LOW_BITS) & Self::HIGH_MASK;
        let high_clear = (template.clear >> Self::LOW_BITS) & Self::HIGH_MASK;
        let free = Self::HIGH_MASK & !(high_set | high_clear);
        let mut sub = free;
        loop {
            let word = (high_set | sub) as usize;
            if value {
                self.words[word] |= within_word;
            } else {
                self.words[word] &= !within_word;
            }
            if sub == 0 {
                break;
            }
            sub = (sub - 1) & free;
        }
    }

    /// How many configurations map to set. A whole-table fact, and the cheapest
    /// thing that distinguishes two tables.
    pub fn count_set(&self) -> usize {
        self.words
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    /// How many configurations this table maps to something other than their own
    /// centre voxel: the size of its departure from [`Self::identity`].
    ///
    /// **A table that moves nothing passes every comparison.** A pass under such
    /// a table is the identity, and a decomposition-invariance test over it is
    /// invariant for the wrong reason, so a test that asserts a table does
    /// something is not a nicety. This is the whole-table half of that; the
    /// per-volume half is how many voxels a pass actually moved, which only the
    /// data can say.
    pub fn count_moving(&self) -> usize {
        let identity = Self::identity();
        self.words
            .iter()
            .zip(identity.words.iter())
            .map(|(ours, theirs)| (ours ^ theirs).count_ones() as usize)
            .sum()
    }
}

/// The configuration of the voxel at `(i, j, k)`, with neighbours outside the
/// array reading clear.
///
/// The definition of the index, in code, and the only place the traversal has to
/// agree with. [`configuration_pass_into`] computes the same integers by a route
/// that reads each voxel once instead of 27 times, and
/// `a_pass_agrees_with_the_index_definition_voxel_by_voxel` is what holds the two
/// together.
pub fn configuration_index_at(input: ArrayView3<'_, bool>, at: [usize; 3]) -> u32 {
    let extent = input.shape();
    let mut index = 0u32;
    for dk in -1..=1isize {
        for dj in -1..=1isize {
            for di in -1..=1isize {
                let offset = [di, dj, dk];
                let mut neighbour = [0usize; 3];
                let mut inside = true;
                for axis in 0..3 {
                    let coordinate = at[axis] as isize + offset[axis];
                    if coordinate < 0 || coordinate >= extent[axis] as isize {
                        inside = false;
                        break;
                    }
                    neighbour[axis] = coordinate as usize;
                }
                if inside && input[neighbour] {
                    index |= 1 << configuration_bit(offset).expect("a unit offset");
                }
            }
        }
    }
    index
}

/// One pass: every voxel's configuration, looked up, written to `out`.
///
/// `out` is written in full from an `input` that is never mutated, so no voxel's
/// answer depends on another's having been computed first and there is nothing
/// here whose order could be reassociated.
///
/// The traversal reads nine voxels per output rather than twenty-seven. The
/// nine `(i, j)` lines the neighbourhood touches are found once per output line
/// and each contributes one bit position; a step along axis 2 then rolls the
/// three planes — what was above becomes here, and one new plane is gathered —
/// so the 27 bits cost nine loads and two shifts. Every bit still lands at
/// exactly the offset [`configuration_bit`] gives it.
pub fn configuration_pass_into(
    input: ArrayView3<'_, bool>,
    table: &ConfigurationTable,
    mut out: ArrayViewMut3<'_, bool>,
) -> Result<()> {
    shapes_agree(input.shape(), out.shape(), "configuration_pass_into")?;
    let extent = [input.shape()[0], input.shape()[1], input.shape()[2]];

    // The nine lines, as (i, j, shift). Rebuilt per output line, which is once
    // per `extent[2]` outputs.
    let mut lines: [(usize, usize, u32); 9] = [(0, 0, 0); 9];
    for i in 0..extent[0] {
        for j in 0..extent[1] {
            let mut count = 0usize;
            for dj in -1..=1isize {
                let jj = j as isize + dj;
                if jj < 0 || jj >= extent[1] as isize {
                    continue;
                }
                for di in -1..=1isize {
                    let ii = i as isize + di;
                    if ii < 0 || ii >= extent[0] as isize {
                        continue;
                    }
                    // The axis-2 term is added by which plane the bits go into,
                    // so only the axis-0 and axis-1 terms are here.
                    let shift = (di + 1) as u32 + 3 * (dj + 1) as u32;
                    lines[count] = (ii as usize, jj as usize, shift);
                    count += 1;
                }
            }
            let lines = &lines[..count];
            let plane = |k: isize| -> u32 {
                if k < 0 || k >= extent[2] as isize {
                    return 0;
                }
                let k = k as usize;
                let mut bits = 0u32;
                for &(ii, jj, shift) in lines {
                    bits |= u32::from(input[[ii, jj, k]]) << shift;
                }
                bits
            };

            let mut below = 0u32;
            let mut here = plane(0);
            let mut above = plane(1);
            for k in 0..extent[2] {
                out[[i, j, k]] = table.get(below | (here << 9) | (above << 18));
                below = here;
                here = above;
                above = plane(k as isize + 2);
            }
        }
    }
    Ok(())
}

/// `passes` passes, each reading what the last one wrote.
///
/// **Reaches `passes` voxels**, which is the price of a stated count: the last
/// pass reads one voxel of the pass before it, which read one voxel of the pass
/// before that. `passes == 0` is the identity and is allowed — a caller sweeping
/// a parameter should not have to special-case the bottom of its range.
pub fn configuration_passes_into(
    input: ArrayView3<'_, bool>,
    table: &ConfigurationTable,
    passes: usize,
    mut out: ArrayViewMut3<'_, bool>,
) -> Result<()> {
    shapes_agree(input.shape(), out.shape(), "configuration_passes_into")?;
    if passes == 0 {
        out.assign(&input);
        return Ok(());
    }
    // One pass writes straight into `out`; the rest ping-pong through a single
    // scratch volume, so live storage is one extra volume whatever the count.
    configuration_pass_into(input, table, out.view_mut())?;
    if passes == 1 {
        return Ok(());
    }
    let mut scratch = Array3::from_elem(input.raw_dim(), false);
    for pass in 1..passes {
        if pass % 2 == 1 {
            configuration_pass_into(out.view(), table, scratch.view_mut())?;
        } else {
            configuration_pass_into(scratch.view(), table, out.view_mut())?;
        }
    }
    if (passes - 1) % 2 == 1 {
        out.assign(&scratch);
    }
    Ok(())
}

/// Passes until one changes nothing, and how many were run.
///
/// The whole-volume statement of what [`ConfigurationFixedPointOp`] computes, and
/// the reference a block-decomposed run is checked against: no blocks, no halo,
/// and a convergence test over the entire volume at once, which is the predicate
/// the executor's per-block test adds up to.
///
/// The count includes the pass that changed nothing, because that pass is how the
/// fixed point was established. Returns an error at `limit` rather than the
/// volume it had reached: a partially converged volume is plausible, well-formed
/// and wrong, and it is the same refusal `crate::iterate` makes for the same
/// reason.
pub fn configuration_to_fixed_point(
    input: ArrayView3<'_, bool>,
    table: &ConfigurationTable,
    limit: SubstageLimit,
) -> Result<(Array3<bool>, usize)> {
    let mut current = input.to_owned();
    let mut next = Array3::from_elem(input.raw_dim(), false);
    for pass in 1..=limit.substages() {
        configuration_pass_into(current.view(), table, next.view_mut())?;
        let settled = next == current;
        std::mem::swap(&mut current, &mut next);
        if settled {
            return Ok((current, pass));
        }
    }
    Err(Error::InvalidArgument(format!(
        "a configuration table iteration did not settle in {} pass(es). Either the limit is \
         below what this data needs, or this table has no fixed point on this volume — a table \
         mapping every configuration to the complement of its centre has period two everywhere, \
         and no bound derived from the volume would make that converge.",
        limit.substages()
    )))
}

/// A stated number of passes of a configuration table over a mask.
pub struct ConfigurationPassOp {
    name: &'static str,
    table: Arc<ConfigurationTable>,
    passes: usize,
    cost: f64,
}

impl ConfigurationPassOp {
    pub fn new(name: &'static str, table: Arc<ConfigurationTable>, passes: usize) -> Self {
        Self {
            name,
            table,
            passes,
            cost: PASS_COST * passes as f64,
        }
    }

    pub fn table(&self) -> &Arc<ConfigurationTable> {
        &self.table
    }

    pub fn passes(&self) -> usize {
        self.passes
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }
}

impl BlockOp for ConfigurationPassOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// **The pass count**, on every axis. One pass reads the 3x3x3
    /// neighbourhood, so it reaches one; `n` of them reach `n`, because pass `n`
    /// consumes values pass `n-1` derived from a voxel one further out. Derived
    /// from the parameter and there is no field that sets it.
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        self.passes
    }

    /// A mask, held as a mask or held as `f64`.
    ///
    /// `Bool` is what a binary volume is and is what this op is for. `F64` is
    /// kept because a chain may carry a mask as `f64` under this module's
    /// `is_set`/`from_set` convention; the kernel is a `bool` kernel either way.
    fn accepts(&self, dtype: Dtype) -> bool {
        matches!(dtype, Dtype::Bool | Dtype::F64)
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        match input.dtype() {
            Dtype::Bool => configuration_passes_into(
                input.view::<bool>()?,
                &self.table,
                self.passes,
                out.view_mut::<bool>()?,
            ),
            _ => {
                let mask = input.view::<f64>()?.mapv(is_set);
                let mut result = Array3::from_elem(mask.raw_dim(), false);
                configuration_passes_into(
                    mask.view(),
                    &self.table,
                    self.passes,
                    result.view_mut(),
                )?;
                let mut out = out.view_mut::<f64>()?;
                ndarray::Zip::from(&mut out)
                    .and(&result)
                    .for_each(|slot, &value| *slot = from_set(value));
                Ok(())
            }
        }
    }

    /// **Only an all-clear block, and only under a table that leaves the empty
    /// configuration clear.**
    ///
    /// Every voxel of an all-clear block has configuration zero — including the
    /// ones on the buffer's own faces, since this op's convention is that what
    /// lies outside is clear — so one pass makes it uniformly `table.get(0)`. If
    /// that is clear the block is all-clear again and every further pass agrees,
    /// which makes the declaration exact for any count.
    ///
    /// The all-**set** block is the interesting half and nothing may be declared
    /// for it: its interior voxels see a full neighbourhood, its face voxels see
    /// the outside as clear, so the output is not uniform and a short circuit
    /// would disagree with computing it at exactly the voxels the halo exists to
    /// get right.
    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        if !is_set(value) && !self.table.get(0) {
            Some(from_set(false))
        } else {
            None
        }
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

/// Passes of a configuration table until nothing anywhere changes.
///
/// One pass per substage, so the phase's external reach is **1** whatever the
/// substage count — see the header for why that is the point of the shape, and
/// for why the convergence predicate has to be global.
pub struct ConfigurationFixedPointOp {
    name: &'static str,
    table: Arc<ConfigurationTable>,
    limit: SubstageLimit,
    cost: f64,
}

impl ConfigurationFixedPointOp {
    /// The limit is the caller's and is required; there is deliberately no
    /// `bound_for` here, because a bound would be a claim that this table
    /// terminates and that is a property of the table rather than of the volume.
    pub fn new(name: &'static str, table: Arc<ConfigurationTable>, limit: SubstageLimit) -> Self {
        Self {
            name,
            table,
            limit,
            cost: PASS_COST,
        }
    }

    pub fn table(&self) -> &Arc<ConfigurationTable> {
        &self.table
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }
}

impl IterativeOp for ConfigurationFixedPointOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// One operand and one voxel of it.
    ///
    /// There is no `Fixed` operand: a pass consults the table and the running
    /// estimate and nothing else, so declaring the input image as a second
    /// operand would fetch a halo per substage that no voxel reads.
    fn operands(&self) -> Vec<SubstageOperand> {
        vec![SubstageOperand::running([1, 1, 1])]
    }

    fn limit(&self) -> SubstageLimit {
        self.limit
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        matches!(dtype, Dtype::Bool | Dtype::F64)
    }

    fn substage(&self, at: &Substage<'_>, out: &mut Voxels) -> Result<()> {
        let input = at.operand(0)?;
        match input.dtype() {
            Dtype::Bool => {
                configuration_pass_into(input.view::<bool>()?, &self.table, out.view_mut::<bool>()?)
            }
            _ => {
                let mask = input.view::<f64>()?.mapv(is_set);
                let mut result = Array3::from_elem(mask.raw_dim(), false);
                configuration_pass_into(mask.view(), &self.table, result.view_mut())?;
                let mut out = out.view_mut::<f64>()?;
                ndarray::Zip::from(&mut out)
                    .and(&result)
                    .for_each(|slot, &value| *slot = from_set(value));
                Ok(())
            }
        }
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

// ---------------------------------------------------------------- costs --

/// Measured; see [`cost_report`], and `super::COST_MEASUREMENT` for the method.
/// Relative to the voxelwise map, which is this module's unit of work. **One
/// pass**; [`ConfigurationPassOp`] multiplies by its own count, as
/// `MorphologyOp` multiplies by its composition factor.
///
/// **The number is stable across inputs in a way the thinning constant is not**,
/// and that was the prediction and is now the measurement: 19.71 and 19.65
/// ns/voxel over the two inputs in [`cost_report`], one built of solid blocks and
/// one a speckle, agreeing to three parts in a thousand where a thinning
/// sub-iteration differs by a factor of two. The reason is structural — a pass
/// does the same nine loads, two shifts and one table probe at every voxel
/// whatever the data holds, so there is no data-dependent branch for the two
/// inputs to disagree about — and it is worth having a measurement of, because it
/// is the property that makes one `cost_per_voxel` honest for this op where the
/// trait cannot express a data-dependent one.
///
/// **The spread is stated because the ratio is noisier than the op is.** Four
/// runs put the op at 19.6-19.7 ns/voxel every time and the *unit* between 0.71
/// and 1.12, so the ratio came out 19.7, 23.8, 25.5 and 27.7; `22.0` is stored.
/// All of the movement is in the denominator — a voxelwise map costs under a
/// nanosecond a voxel and at that size a ratio inherits the noise of memory
/// bandwidth. What the planner needs from this number is that a configuration
/// pass is an expensive neighbourhood pass rather than a cheap one, and every run
/// says so by more than an order of magnitude.
pub const PASS_COST: f64 = 22.0;

/// The measurement [`PASS_COST`] came from, kept as text so that a re-run
/// somewhere else can be **compared** against it rather than merely replacing it.
/// `--release`, one thread, 96 x 64 x 64, best of 5, on the machine this was
/// written on:
///
/// ```text
/// case                                                       ns/voxel   relative
/// voxelwise map (the unit)                                      0.827       1.00
/// configuration pass, solid blocks                             19.675      23.80
/// configuration pass, speckle                                  19.638      23.76
/// ```
pub const COST_MEASUREMENT: &str = "ops::configuration::cost_report";

/// A table that is not the identity and is cheap to build: a voxel survives
/// only if it and its six face neighbours are set.
///
/// Used by [`cost_report`] and by the tests, so that neither prices or checks a
/// probe the compiler could see through.
fn face_erosion_table() -> ConfigurationTable {
    let mut template = ConfigurationTemplate::any();
    for offset in [
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ] {
        template = template.with_set(offset).expect("a unit offset");
    }
    let mut table = ConfigurationTable::zeros();
    table.assign_matching(&template, true);
    table
}

/// Retake the measurement, through the same `BlockOp::apply` the executor calls.
///
/// Two inputs, as `super::skeleton::cost_report` uses two, so that the claim
/// above — that this op does not care what the data holds — is a measurement
/// rather than an assertion.
pub fn cost_report(shape: [usize; 3], repetitions: usize) -> String {
    use std::time::Instant;

    let voxels = (shape[0] * shape[1] * shape[2]) as f64;
    let anchor = Anchor::whole(shape);
    let repetitions = repetitions.max(1);

    let best_of = |mut run: Box<dyn FnMut()>| -> f64 {
        // One untimed pass first, for the page faults a fresh output pays.
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
        let mut out = Voxels::zeros(Dtype::F64, shape).expect("a buffer");
        rows.push((
            "voxelwise map (the unit)".to_string(),
            best_of(Box::new(move || {
                op.apply(&input, &mut out, &anchor).unwrap();
            })),
        ));
    }

    let table = Arc::new(face_erosion_table());
    for (what, mask) in [
        (
            "configuration pass, solid blocks",
            Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
                (i / 8 + j / 8 + k / 8) % 4 != 0
            }),
        ),
        (
            "configuration pass, speckle",
            Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
                (i * 31 + j * 17 + k * 7) % 5 < 2
            }),
        ),
    ] {
        let input: Voxels = mask.into();
        let op = ConfigurationPassOp::new("pass", table.clone(), 1);
        let mut out = Voxels::zeros(Dtype::Bool, shape).expect("a buffer");
        let anchor = Anchor::whole(shape);
        rows.push((
            what.to_string(),
            best_of(Box::new(move || {
                op.apply(&input, &mut out, &anchor).unwrap();
            })),
        ));
    }

    let unit = rows.first().map(|(_, nanos)| *nanos).unwrap_or(1.0);
    let mut report = format!(
        "configuration cost, {}x{}x{}, best of {repetitions}\n{:<56} {:>10} {:>10} {:>10}\n",
        shape[0], shape[1], shape[2], "case", "ns/voxel", "relative", "stored"
    );
    for (name, nanos) in &rows {
        report.push_str(&format!(
            "{name:<56} {nanos:>10.3} {:>10.2} {PASS_COST:>10.2}\n",
            nanos / unit
        ));
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The convention, spelled out at three positions rather than trusted.
    #[test]
    fn the_centre_is_bit_thirteen_and_axis_zero_is_the_fastest_term() {
        assert_eq!(configuration_bit([0, 0, 0]).unwrap(), CENTRE_BIT);
        assert_eq!(configuration_bit([-1, -1, -1]).unwrap(), 0);
        assert_eq!(configuration_bit([1, 1, 1]).unwrap(), 26);
        assert_eq!(configuration_bit([1, 0, 0]).unwrap(), 14);
        assert_eq!(configuration_bit([0, 1, 0]).unwrap(), 16);
        assert_eq!(configuration_bit([0, 0, 1]).unwrap(), 22);
        assert!(configuration_bit([2, 0, 0]).is_err());
    }

    #[test]
    fn a_template_that_demands_both_is_refused_where_it_is_written() {
        let template = ConfigurationTemplate::any().with_set([1, 0, 0]).unwrap();
        let message = template.with_clear([1, 0, 0]).unwrap_err().to_string();
        assert!(message.contains("matches no configuration"), "{message}");
    }

    #[test]
    fn the_two_ways_of_stating_a_template_agree_and_refuse_the_same_things() {
        let accumulated = ConfigurationTemplate::any()
            .with_set([0, 0, 0])
            .unwrap()
            .with_clear([1, -1, 0])
            .unwrap();
        let from_masks = ConfigurationTemplate::from_masks(
            1 << configuration_bit([0, 0, 0]).unwrap(),
            1 << configuration_bit([1, -1, 0]).unwrap(),
        )
        .unwrap();
        assert_eq!(accumulated, from_masks);

        let message = ConfigurationTemplate::from_masks(0b11, 0b10)
            .unwrap_err()
            .to_string();
        assert!(message.contains("matches no configuration"), "{message}");
        let message = ConfigurationTemplate::from_masks(1 << 27, 0)
            .unwrap_err()
            .to_string();
        assert!(message.contains("outside"), "{message}");
    }

    /// `assign_matching` is a word-level trick, and this is the naive statement
    /// of what it must do.
    #[test]
    fn assigning_a_template_touches_exactly_the_configurations_it_admits() {
        let template = ConfigurationTemplate::any()
            .with_set([0, 0, 0])
            .unwrap()
            .with_set([0, 0, 1])
            .unwrap()
            .with_clear([-1, -1, -1])
            .unwrap()
            .with_clear([1, 1, 1])
            .unwrap();
        let mut table = ConfigurationTable::zeros();
        table.assign_matching(&template, true);
        assert_eq!(table.count_set(), template.matches());
        assert_eq!(template.matches(), 1 << 23);

        // Over a sample rather than all 2^27, and including the boundaries of
        // the word split the implementation turns on.
        for configuration in (0..CONFIGURATION_COUNT as u32).step_by(9973) {
            assert_eq!(
                table.get(configuration),
                template.admits(configuration),
                "configuration {configuration:#029b}"
            );
        }
        for configuration in 0..256u32 {
            assert_eq!(table.get(configuration), template.admits(configuration));
        }
    }

    #[test]
    fn assigning_can_clear_as_well_as_set() {
        let template = ConfigurationTemplate::any().with_clear([0, 0, 0]).unwrap();
        let mut table = ConfigurationTable::identity();
        assert_eq!(table.count_set(), CONFIGURATION_COUNT / 2);
        table.assign_matching(&template, true);
        assert_eq!(table.count_set(), CONFIGURATION_COUNT);
        table.assign_matching(&template.inverted(), false);
        assert_eq!(table.count_set(), CONFIGURATION_COUNT / 2);
        // And what survives is the *other* half.
        assert!(table.get(0));
        assert!(!table.get(1 << CENTRE_BIT));
    }

    #[test]
    fn the_identity_table_moves_nothing_and_says_so() {
        let table = ConfigurationTable::identity();
        assert_eq!(table.count_moving(), 0);
        for configuration in (0..CONFIGURATION_COUNT as u32).step_by(7919) {
            assert_eq!(
                table.get(configuration),
                configuration >> CENTRE_BIT & 1 == 1
            );
        }
    }

    fn checkerboard(shape: [usize; 3]) -> Array3<bool> {
        Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
            (i * 5 + j * 3 + k * 11) % 7 < 3
        })
    }

    /// The fast traversal against the definition, **one bit at a time**, at
    /// every voxel of a volume small enough to check exhaustively and irregular
    /// enough to be worth checking.
    ///
    /// A table that reports a single bit of the configuration turns the pass
    /// into a projection, so comparing all 27 of them pins the whole index
    /// rather than whatever part of it one table happens to be sensitive to. A
    /// transposed axis or a plane rolled the wrong way fails on some bit.
    #[test]
    fn a_pass_agrees_with_the_index_definition_bit_by_bit() {
        let shape = [7, 5, 6];
        let input = checkerboard(shape);
        for dk in -1..=1isize {
            for dj in -1..=1isize {
                for di in -1..=1isize {
                    let offset = [di, dj, dk];
                    let bit = configuration_bit(offset).unwrap();
                    let mut table = ConfigurationTable::zeros();
                    table.assign_matching(
                        &ConfigurationTemplate::any().with_set(offset).unwrap(),
                        true,
                    );
                    let mut out = Array3::from_elem(input.raw_dim(), false);
                    configuration_pass_into(input.view(), &table, out.view_mut()).unwrap();
                    for i in 0..shape[0] {
                        for j in 0..shape[1] {
                            for k in 0..shape[2] {
                                let index = configuration_index_at(input.view(), [i, j, k]);
                                assert_eq!(
                                    out[[i, j, k]],
                                    index >> bit & 1 == 1,
                                    "bit {bit} (offset {offset:?}) at {:?}",
                                    [i, j, k]
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn the_identity_table_leaves_a_volume_alone_at_any_pass_count() {
        let input = checkerboard([6, 6, 6]);
        let table = ConfigurationTable::identity();
        for passes in 0..4 {
            let mut out = Array3::from_elem(input.raw_dim(), false);
            configuration_passes_into(input.view(), &table, passes, out.view_mut()).unwrap();
            assert_eq!(out, input, "at {passes} pass(es)");
        }
    }

    /// The ping-pong in `configuration_passes_into` is the kind of thing that is
    /// off by one pass and still plausible, so the composition is checked
    /// against passes applied one at a time.
    #[test]
    fn n_passes_are_n_applications_of_one() {
        let input = checkerboard([6, 7, 5]);
        let table = face_erosion_table();
        assert!(table.count_moving() > 0, "the table must do something");

        let mut stepwise = input.clone();
        for passes in 1..5usize {
            let mut once = Array3::from_elem(input.raw_dim(), false);
            configuration_pass_into(stepwise.view(), &table, once.view_mut()).unwrap();
            stepwise = once;

            let mut together = Array3::from_elem(input.raw_dim(), false);
            configuration_passes_into(input.view(), &table, passes, together.view_mut()).unwrap();
            assert_eq!(together, stepwise, "at {passes} pass(es)");
        }
    }

    /// Retaking the measurement. Ignored because timing in a test suite measures
    /// the suite, and because the constant is `--release`'s number.
    #[test]
    #[ignore = "a measurement, not an assertion"]
    fn print_the_cost_table() {
        println!("{}", cost_report([96, 64, 64], 5));
    }

    /// What can be asserted about a measured cost without measuring it: that the
    /// op reports the constant rather than the trait's default, and that the
    /// stated pass count multiplies it.
    #[test]
    fn the_stored_cost_is_reported_and_scales_with_the_pass_count() {
        let table = Arc::new(ConfigurationTable::identity());
        let one = ConfigurationPassOp::new("one", table.clone(), 1);
        let six = ConfigurationPassOp::new("six", table.clone(), 6);
        assert_eq!(one.cost_per_voxel(), PASS_COST);
        assert_eq!(six.cost_per_voxel(), PASS_COST * 6.0);
        assert!(
            PASS_COST > 1.0,
            "a neighbourhood pass is not a voxelwise map"
        );
        assert_eq!(
            ConfigurationFixedPointOp::new("fixed", table, SubstageLimit::of(4).unwrap())
                .cost_per_voxel(),
            PASS_COST,
            "one substage is one pass"
        );
    }

    /// A table with no fixed point, which is why the limit is required.
    #[test]
    fn a_table_that_never_settles_ends_at_the_limit_rather_than_at_an_answer() {
        let mut table = ConfigurationTable::identity();
        // Every configuration maps to the complement of its centre.
        let centre = ConfigurationTemplate::any().with_set([0, 0, 0]).unwrap();
        table.assign_matching(&centre, false);
        table.assign_matching(&centre.inverted(), true);

        let input = checkerboard([4, 4, 4]);
        let limit = SubstageLimit::of(20).unwrap();
        let error = configuration_to_fixed_point(input.view(), &table, limit)
            .unwrap_err()
            .to_string();
        assert!(error.contains("did not settle in 20"), "{error}");
    }
}
