// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// Voxelwise ops: reach 0, no neighbourhood, no anchor arithmetic. Small, and
// worth having first for a reason that is structural rather than about the
// arithmetic — `docs/design/BLOCK_OPS.md` §"`Chain` has no fan-in" records that
// the shape this crate could not express was a **diamond**: one input feeding
// two arms which are then combined. Every part of that except the combine
// existed. This is the combine.
//
// The second operand, and why it is held rather than fetched
// ----------------------------------------------------------
// `BlockOp::apply` takes one input buffer, so a two-input op has to get its
// second operand from somewhere else. [`CombineOp`] holds it as a whole-volume
// array and slices it at the [`Anchor`] the executor supplies. That is honest
// about what today's signature can do and it makes the anchor load-bearing:
// slice at the wrong offset and the op combines the wrong voxels, everywhere,
// which the decomposition-invariance tests would see immediately.
//
// It is *not* a claim that this is how fan-in should eventually work. When
// `Chain` grows a fan-in node the second arm becomes a sibling subtree and the
// operand arrives as a buffer like the first; the kernel below — a voxelwise map
// of two views into a third — is what that node would call either way, which is
// the reason the kernel is a free function and the op is a shell around it.
//
// That node now exists, and the sentence above was taken literally
// ---------------------------------------------------------------
// [`Chain::Parallel`] holds a `Box<dyn Combine>`; [`LogicCombine`] is the
// implementation of it that this module owes. It calls exactly the two kernels
// [`CombineOp`] calls — `logic_into` for two `bool` buffers, `combine_into`
// through this module's mask convention for `f64` — and the **arrangement** is
// what changed: the second operand is a sibling branch's result, produced by
// the same block from the same input, instead of a whole-volume array held by
// the op and sliced at the anchor.
//
// `CombineOp` stays. It is not a worse `LogicCombine`; it is a different shape
// — one arm computed, one arm supplied from outside the chain — and it is the
// only way to combine against an array a chain did not produce. What it can no
// longer claim is to be the diamond.
//
// The map is a type parameter, and what that buys
// ------------------------------------------------
// [`VoxelwiseMapOp`] used to hold `Box<dyn Fn(f64) -> f64>` and call it once per
// voxel. That is an optimisation barrier in the innermost loop of the cheapest
// op in the crate: nothing inlines through an indirect call, nothing vectorises
// across it, nothing reorders past it. The kernel [`map_into`] was already
// generic over `impl Fn` and vectorised fine when handed a real closure, so the
// entire loss was the shell. It now holds an `M: MapFn` type parameter instead,
// and the `dyn` boundary moves out to `Box<dyn BlockOp>` in
// [`Chain`](crate::op::Chain) — one virtual call per *block*, which is free at
// that rate.
//
// [`MapFn`] is a trait rather than an enum of the maps this crate happens to
// need, because the set is not this crate's to close: a third party writes
// `impl MapFn for MyMap` outside it and gets a fully monomorphised op, with no
// variant added here and no coordination. [`Compose`] is the same idea one level
// up — a `MapFn` built from two others, generic in both, which is the worked
// example somebody writing their *own* fusion op copies. Everything such an op
// needs is public: the kernels in this module and its siblings are generic free
// functions over `ArrayView3`, callable from inside a user's own loop.
//
// **A user-injected fusion cannot be discovered by the planner**, and that is a
// limitation rather than an oversight. The planner would have to *name* the
// fused type to decide to build one, and it cannot name a type it has never
// heard of. Fusion is therefore a decision made when the chain is **constructed**
// — by whoever writes `Chain::op(VoxelwiseMapOp::from_map(name, Compose::new(a, b)))`
// — and not a plan-time rewrite of two adjacent ops. There is no planner support
// for it to go looking for.

use std::sync::Arc;

use ndarray::{ArrayView3, ArrayViewMut3};

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::op::{Anchor, BlockOp, Combine};
use crate::voxels::Voxels;

use super::shapes_agree;

/// Apply `map` to every voxel, writing into `out`.
///
/// Generic over both element types and bounded by nothing: a voxelwise map
/// imposes no requirement on what it maps between. The adapters below are the
/// only part that has to know about `f64`, which is why the element-type change
/// rewrote them and left this alone.
pub fn map_into<A, B>(
    input: ArrayView3<'_, A>,
    mut out: ArrayViewMut3<'_, B>,
    map: impl Fn(&A) -> B,
) -> Result<()> {
    shapes_agree(input.shape(), out.shape(), "map_into")?;
    ndarray::Zip::from(&mut out)
        .and(input)
        .for_each(|slot, value| *slot = map(value));
    Ok(())
}

/// Apply `combine` to every pair of co-located voxels, writing into `out`.
///
/// The three element types are independent because the useful cases have them
/// different: a mask and an image producing a masked image, two images producing
/// a difference, two masks producing a mask.
pub fn combine_into<A, B, C>(
    left: ArrayView3<'_, A>,
    right: ArrayView3<'_, B>,
    mut out: ArrayViewMut3<'_, C>,
    combine: impl Fn(&A, &B) -> C,
) -> Result<()> {
    shapes_agree(left.shape(), right.shape(), "combine_into operands")?;
    shapes_agree(left.shape(), out.shape(), "combine_into")?;
    ndarray::Zip::from(&mut out)
        .and(left)
        .and(right)
        .for_each(|slot, left, right| *slot = combine(left, right));
    Ok(())
}

/// The binary connectives, over `bool`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Logic {
    And,
    Or,
    Xor,
}

impl Logic {
    pub fn apply(self, left: bool, right: bool) -> bool {
        match self {
            Logic::And => left && right,
            Logic::Or => left || right,
            Logic::Xor => left != right,
        }
    }

    /// What this connective gives for **every** value of the other operand, if
    /// there is such a value, when one operand is `known`.
    ///
    /// This is the whole of the constant algebra for a two-input op: the op is
    /// told what one side is and knows nothing about the other, so it may only
    /// answer where the connective has an absorbing element. `false AND x` is
    /// `false` and `true OR x` is `true` whatever `x` is; nothing else here is
    /// determined, and `XOR` never is.
    pub fn absorbs(self, known: bool) -> Option<bool> {
        match (self, known) {
            (Logic::And, false) => Some(false),
            (Logic::Or, true) => Some(true),
            _ => None,
        }
    }
}

/// Combine two `bool` volumes with a connective.
pub fn logic_into(
    left: ArrayView3<'_, bool>,
    right: ArrayView3<'_, bool>,
    out: ArrayViewMut3<'_, bool>,
    logic: Logic,
) -> Result<()> {
    combine_into(left, right, out, |&left, &right| logic.apply(left, right))
}

/// The complement. The unary member of the set above, and a voxelwise map.
pub fn not_into(input: ArrayView3<'_, bool>, out: ArrayViewMut3<'_, bool>) -> Result<()> {
    map_into(input, out, |&value| !value)
}

// ------------------------------------------------------------- adapters --

/// How this module reads a `f64` buffer as a mask: **non-zero is true**.
///
/// Stated once and used everywhere, because the alternative — each op choosing
/// its own predicate — is how two ops in one chain come to disagree about what a
/// mask is. `true` is written back as `1.0`, so a mask this crate produces reads
/// back as the same mask.
pub fn is_set(value: f64) -> bool {
    value != 0.0
}

/// The `f64` a mask voxel is written as.
pub fn from_set(value: bool) -> f64 {
    if value {
        1.0
    } else {
        0.0
    }
}

// --------------------------------------------------------- the map plug-in --

/// A pure, position-independent `f64 -> f64` map, as a **type** rather than a
/// boxed closure.
///
/// Purity is a precondition of implementing this, not something checked, and it
/// is what licenses [`constant_maps_to`](Self::constant_maps_to): if a map
/// consulted anything but its argument, the answer for a short-circuited block
/// and for a computed one would differ.
///
/// Implement it outside this crate freely — that is the point of it being a
/// trait. The one method with no default is [`map`](Self::map); the other three
/// have defaults that are correct for every pure map, and are overridden only
/// where the type knows something the general case cannot.
pub trait MapFn: Send + Sync + 'static {
    /// The map, on one value.
    fn map(&self, value: f64) -> f64;

    /// The map, over a run of contiguous voxels.
    ///
    /// **The default is genuinely free**, which it would not have been behind a
    /// trait object: `Self` is concrete at every call site, so `map` inlines
    /// into this loop and the loop vectorises. An implementer writes one method
    /// and gets the fast path; an implementer who can do better than one `map`
    /// per voxel — [`Identity`] turns it into a `memcpy`, [`Threshold`] hoists
    /// its comparison out of the loop — overrides this and gets that instead.
    ///
    /// `src` and `dst` are the same length; [`VoxelwiseMapOp::apply`] checks the
    /// two buffers agree before it gets here. The default zips, which would
    /// silently stop at the shorter of the two, so the check is the caller's job
    /// rather than this method's.
    fn map_slice(&self, src: &[f64], dst: &mut [f64]) {
        for (slot, &value) in dst.iter_mut().zip(src.iter()) {
            *slot = self.map(value);
        }
    }

    /// What this map gives for a block that is entirely `value`, if that is
    /// known. `None` means "not known", which disables the short circuit.
    ///
    /// The default is the map applied to the constant — exactly right for a pure
    /// map, and exactly the semantics the boxed closure had. A type overrides it
    /// not to change the answer but to *derive* it rather than evaluate it; see
    /// [`Compose`], where deriving is conservative in the one way that matters.
    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        Some(self.map(value))
    }

    /// What one voxel of this map costs, in the unit
    /// [`COST_MEASUREMENT`](super::COST_MEASUREMENT) defines.
    ///
    /// Derived from the expression, which is why it is on the map and not on the
    /// op: a ten-deep [`Compose`] is ten maps' worth of arithmetic per voxel and
    /// an [`Identity`] is a copy, and one flat constant for both is a planner
    /// input that is wrong in both directions. The default is the flat
    /// [`MAP_COST`], which is what a map that has not been asked can honestly
    /// say.
    fn cost(&self) -> f64 {
        MAP_COST
    }
}

/// Any pure `f64 -> f64` closure is a [`MapFn`].
///
/// This is the escape hatch, and it is not speculative: a closure is how most of
/// this workspace's voxelwise maps are written today, and they will not all be
/// one of the named types below however many of those are added. Removing it
/// would mean either a variant per caller — which is the closed set this design
/// exists to avoid — or rewriting every call site to say the same thing longer.
///
/// **It costs nothing the named types do not also cost.** `F` is a type
/// parameter here too, so a closure passed to `new` is monomorphised into the
/// loop exactly as [`Threshold`] is; there is no indirect call on this path
/// either. What a closure gives up is only what a closure cannot *say*:
/// `constant_maps_to` falls back to calling it, and `cost` falls back to the
/// flat [`MAP_COST`]. A map that wants either derived becomes a named type.
impl<F> MapFn for F
where
    F: Fn(f64) -> f64 + Send + Sync + 'static,
{
    fn map(&self, value: f64) -> f64 {
        self(value)
    }
}

/// The map that is not a map.
///
/// Call sites all over this workspace ask for `|value| value`, and at least one
/// of them cannot simply be deleted: a downstream pipeline uses an identity
/// phase to hold a level index still when an optional stage is switched off, so
/// that the plan's wiring does not change with a parameter. The identity
/// therefore has to exist, and has to be as cheap as the shape allows.
///
/// **What it costs, and why it is not elided further.** `map_slice` is a
/// `copy_from_slice`, so an identity phase is one `memcpy` from the input block
/// into the output block — not a `map` call per voxel and not a vectorised loop,
/// but the widest copy the platform has. It cannot be less than that *here*:
/// [`BlockOp::apply`] is handed an input buffer and a separate output buffer the
/// executor has already allocated, and there is no way through this signature to
/// say "the output *is* the input". Removing the copy altogether is a planner
/// decision — drop the phase, or alias its level to its input — and neither is
/// expressible from inside an op. What is expressible is that the copy is a
/// copy, and that is what this does.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Identity;

impl MapFn for Identity {
    fn map(&self, value: f64) -> f64 {
        value
    }

    /// A `memcpy`. `copy_from_slice` panics on unequal lengths rather than
    /// truncating, which is the stronger of the two behaviours available and
    /// costs nothing given the shell has already checked the shapes.
    fn map_slice(&self, src: &[f64], dst: &mut [f64]) {
        dst.copy_from_slice(src);
    }

    fn cost(&self) -> f64 {
        IDENTITY_COST
    }
}

/// The complement, under this module's mask convention.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Not;

impl MapFn for Not {
    fn map(&self, value: f64) -> f64 {
        from_set(!is_set(value))
    }
}

/// Which side of the level a [`Threshold`] counts as above.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ThresholdTest {
    /// `value > level`. What [`VoxelwiseMapOp::threshold`] has always meant.
    Above,
    /// `value >= level`.
    AtOrAbove,
}

/// Compare against a fixed level: `above` where the test holds, `below` where it
/// does not.
///
/// A *global* threshold, with no position dependence at all —
/// [`super::AdaptiveThresholdOp`] is the one that varies with position.
///
/// **Both tests, because both are asked for and they are not interchangeable.**
/// The difference is one voxel wide and lands exactly where data piles up: at a
/// level of `0.0` over a volume whose background is exactly `0.0`, `>` and `>=`
/// disagree on every background voxel. A caller that needs the non-strict test
/// and only has the strict one writes it out as a closure — which is what
/// several did — and gives up the derived cost and the derived constant for a
/// difference of one character.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Threshold {
    pub level: f64,
    pub test: ThresholdTest,
    pub above: f64,
    pub below: f64,
}

impl Threshold {
    /// `above` where `value > level`.
    pub fn above(level: f64, above: f64, below: f64) -> Self {
        Self {
            level,
            test: ThresholdTest::Above,
            above,
            below,
        }
    }

    /// `above` where `value >= level`.
    pub fn at_or_above(level: f64, above: f64, below: f64) -> Self {
        Self {
            level,
            test: ThresholdTest::AtOrAbove,
            above,
            below,
        }
    }
}

impl MapFn for Threshold {
    /// Branchless, and measured to be worth it.
    ///
    /// The obvious form is `match self.test { Above => value > level, AtOrAbove
    /// => value >= level }`. As the whole map that is fine, because
    /// [`map_slice`](Self::map_slice) below hoists the match out of the loop
    /// entirely. Inside a [`Compose`] it is not: there the loop body is
    /// `map`, the match cannot be hoisted through the composition, and four
    /// composed thresholds measured **5.05 ns/voxel** against **0.60** for the
    /// same arithmetic written as closures. Computing both comparisons and
    /// combining them with non-short-circuiting `|` and `&` takes that to 3.15.
    ///
    /// `at_or_above` is loop-invariant and the two comparisons are two
    /// instructions, so this is never slower than the branch and is much faster
    /// where the branch would not predict. The remaining gap to a closure is the
    /// three field loads a closure bakes in as immediates, which is the price of
    /// the parameters being data.
    ///
    /// `NaN` compares false both ways and lands on `below`, which is what the
    /// `match` form did too.
    fn map(&self, value: f64) -> f64 {
        let at_or_above = matches!(self.test, ThresholdTest::AtOrAbove);
        let set = (value > self.level) | (at_or_above & (value >= self.level));
        if set {
            self.above
        } else {
            self.below
        }
    }

    /// The test is chosen **once**, outside the loop, and each arm is then a
    /// loop over one comparison against three values held in registers. Written
    /// out rather than left to the default because the choice is a field, and a
    /// per-voxel load and branch on it is what the optimiser is not obliged to
    /// hoist; hoisting it by hand costs six lines. This is the path a threshold
    /// takes when it is the whole map, and it is why `map` above only has to be
    /// good enough for the composed case.
    fn map_slice(&self, src: &[f64], dst: &mut [f64]) {
        let (level, above, below) = (self.level, self.above, self.below);
        match self.test {
            ThresholdTest::Above => {
                for (slot, &value) in dst.iter_mut().zip(src.iter()) {
                    *slot = if value > level { above } else { below };
                }
            }
            ThresholdTest::AtOrAbove => {
                for (slot, &value) in dst.iter_mut().zip(src.iter()) {
                    *slot = if value >= level { above } else { below };
                }
            }
        }
    }
}

/// `then` after `first`: two maps as one map.
///
/// This is what makes voxelwise fusion **data** rather than a rewrite. Two ops
/// that would each read a block and write a block become one op that reads once
/// and writes once, with no intermediate buffer and no second pass over memory —
/// and because both operands are type parameters, the fused body is one
/// monomorphised loop with both maps inlined into it, not two calls.
///
/// It is also the worked example for a *user's* fusion op. Nothing about it is
/// privileged: it is a public generic struct implementing a public trait, and a
/// third party writes the equivalent for their own combination — three maps, a
/// map and a reduction, whatever the arrangement is — with no access to anything
/// private here. What such an op cannot get is planner support; see the module
/// header for why fusion is a construction-time decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Compose<A: MapFn, B: MapFn> {
    pub first: A,
    pub then: B,
}

impl<A: MapFn, B: MapFn> Compose<A, B> {
    pub fn new(first: A, then: B) -> Self {
        Self { first, then }
    }
}

impl<A: MapFn, B: MapFn> MapFn for Compose<A, B> {
    fn map(&self, value: f64) -> f64 {
        self.then.map(self.first.map(value))
    }

    /// One pass, and deliberately **not** the operands' own `map_slice`.
    ///
    /// Running `first.map_slice` and then `then.map_slice` would need somewhere
    /// to put the intermediate — a buffer this op has not been given and would
    /// have to allocate per block — and would touch the data twice. Fusing at
    /// the element instead keeps the value in a register between the two maps.
    /// The cost is that an operand's specialised `map_slice` is not used inside
    /// a composition: composing with [`Identity`] gets the elementwise identity
    /// rather than the `memcpy`, which is the right trade when the other operand
    /// has real work to do and is why nobody should compose with `Identity`.
    fn map_slice(&self, src: &[f64], dst: &mut [f64]) {
        for (slot, &value) in dst.iter_mut().zip(src.iter()) {
            *slot = self.then.map(self.first.map(value));
        }
    }

    /// Derived, by threading the constant through both — not by evaluating the
    /// composed map.
    ///
    /// The two agree wherever both operands use the default. They differ when an
    /// operand *declines* to answer: if `first` returns `None` this returns
    /// `None`, even though calling the composition would have produced a number.
    /// That is the conservative direction, and the right one — an operand that
    /// says it does not know what a constant block maps to is not to be
    /// second-guessed by its wrapper.
    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        self.then
            .constant_maps_to(self.first.constant_maps_to(value)?)
    }

    /// The sum, which is what a fused pass actually does per voxel. This is the
    /// case the single flat constant priced wrong: a ten-deep composition cost
    /// the planner the same as an identity.
    fn cost(&self) -> f64 {
        self.first.cost() + self.then.cost()
    }
}

// ------------------------------------------------------------- the shell --

/// `out = map(in)`, voxelwise.
///
/// Generic in the map, so there is no indirect call in the loop: see the module
/// header. The map must be **pure and position-independent**, which is
/// [`MapFn`]'s precondition and not this type's to re-check.
pub struct VoxelwiseMapOp<M: MapFn> {
    name: &'static str,
    map: M,
    cost: f64,
}

impl<M: MapFn> VoxelwiseMapOp<M> {
    /// From any [`MapFn`] — a named one from this module, or a caller's own.
    ///
    /// The cost is taken from the map at construction rather than read on every
    /// call, because [`MapFn::cost`] on a deep [`Compose`] walks the whole tree
    /// and `cost_per_voxel` is asked repeatedly by the planner.
    pub fn from_map(name: &'static str, map: M) -> Self {
        let cost = map.cost();
        Self { name, map, cost }
    }

    /// The map this op holds, for a caller that built it generically.
    pub fn map(&self) -> &M {
        &self.map
    }

    /// Override the derived cost. For a `map` that is much more expensive than
    /// the arithmetic the default was measured on and cannot say so itself —
    /// a named [`MapFn`] says so in [`MapFn::cost`] instead.
    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }
}

/// The closure constructor, kept separate from [`VoxelwiseMapOp::from_map`] for
/// one concrete reason: **closure signature inference**.
///
/// A generic `M: MapFn` gives rustc nothing to infer a closure's parameter type
/// from, so `VoxelwiseMapOp::from_map("floor", |value| value.max(0.0))` fails to
/// compile with "type annotations needed" — the blanket `impl MapFn for F where
/// F: Fn(f64) -> f64` is found only *after* the signature is known, not used to
/// derive it. Stating the `Fn` bound on the impl instead puts the signature back
/// where the closure can see it, and every call site that passed a closure to
/// the old boxed constructor keeps compiling with no annotation.
///
/// So this is one constructor per kind of argument, not one constructor and a
/// convenience. Both produce the same monomorphised op.
impl<F> VoxelwiseMapOp<F>
where
    F: Fn(f64) -> f64 + Send + Sync + 'static,
{
    /// `map` must be pure: same argument, same answer, wherever it is called.
    pub fn new(name: &'static str, map: F) -> Self {
        Self::from_map(name, map)
    }
}

impl VoxelwiseMapOp<Identity> {
    /// The identity. See [`Identity`] for what it costs and why it is not
    /// nothing.
    pub fn identity(name: &'static str) -> Self {
        Self::from_map(name, Identity)
    }
}

impl VoxelwiseMapOp<Not> {
    /// The complement of a mask, under this module's mask convention.
    pub fn not(name: &'static str) -> Self {
        Self::from_map(name, Not)
    }
}

impl VoxelwiseMapOp<Threshold> {
    /// Compare against a fixed level: `above` where `value > level`, `below`
    /// elsewhere.
    pub fn threshold(name: &'static str, level: f64, above: f64, below: f64) -> Self {
        Self::from_map(name, Threshold::above(level, above, below))
    }

    /// The same, with the non-strict test: `above` where `value >= level`.
    pub fn at_or_above(name: &'static str, level: f64, above: f64, below: f64) -> Self {
        Self::from_map(name, Threshold::at_or_above(level, above, below))
    }
}

impl<M: MapFn> BlockOp for VoxelwiseMapOp<M> {
    fn name(&self) -> &'static str {
        self.name
    }

    /// Zero, on every axis and at every volume size. A voxelwise op reads the
    /// voxel it writes and nothing else.
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    /// `f64` only, because the map it holds is an `f64 -> f64` map the
    /// **caller** supplied. Accepting a narrower type would mean choosing a
    /// conversion on the caller's behalf, and there is no honest choice: a map
    /// written for intensities is not a map on flags. A caller wanting a
    /// narrower map supplies a narrower op.
    ///
    /// Two paths, and the shapes are checked once for both. The buffers are
    /// whole `Array3`s in practice, so the contiguous path is the one taken and
    /// it is the one [`MapFn::map_slice`] exists for; the `map_into` fallback is
    /// there because an `ArrayView` need not be in standard layout and a wrong
    /// answer for a strided view would be a silent one. The failure label stays
    /// `map_into` in both so the error text is the one this op has always
    /// produced.
    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        let input = input.view::<f64>()?;
        let mut out = out.view_mut::<f64>()?;
        shapes_agree(input.shape(), out.shape(), "map_into")?;
        if input.is_standard_layout() && out.is_standard_layout() {
            let src = input.as_slice().expect("standard layout is contiguous");
            let dst = out.as_slice_mut().expect("standard layout is contiguous");
            self.map.map_slice(src, dst);
            return Ok(());
        }
        map_into(input, out, |&value| self.map.map(value))
    }

    /// Whatever the map says, which for a map that says nothing is the map
    /// applied to the constant. True because the map is pure and the op reads
    /// nothing else.
    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        self.map.constant_maps_to(value)
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

/// `out = logic(in, operand)`, voxelwise, against a second whole-volume operand.
///
/// The operand is in **volume** coordinates and is sliced at the anchor, so this
/// op is position-dependent in the one way that matters: it must know where its
/// buffer sits to know which part of the operand to combine it with.
///
/// **Bounded to volumes that fit in memory, and there is now a form that is
/// not.** The operand is an `Arc<Voxels>` of the *whole* volume — asserted equal
/// to `at.volume` on every block — so this op costs one full copy of the second
/// array, resident for the length of the run, at the sizes an out-of-core
/// framework exists for. The out-of-core form is
/// [`Chain::Source`](crate::op::Chain::Source): a leaf that reads a stored level
/// at the block's own read extent, so that
///
/// ```text
/// Chain::parallel(
///     vec![computed_arm, Chain::source(level, dtype)],
///     Box::new(LogicCombine::new("and", Logic::And)),
/// )
/// ```
///
/// is the same answer with the second operand read a block at a time. Prefer it
/// wherever the second array is a level of the run. This one stays for the case
/// it is honestly good at — a small operand a caller already holds, combined
/// against a chain that has no plan around it yet — and it is not a worse
/// `LogicCombine`; it is a different arrangement of the same kernels.
pub struct CombineOp {
    name: &'static str,
    logic: Logic,
    operand: Arc<Voxels>,
    cost: f64,
}

impl CombineOp {
    /// `operand` must have the shape of the whole volume; `apply` checks it
    /// against the anchor rather than trusting it, because an operand of the
    /// wrong shape would otherwise slice successfully for some blocks and not
    /// others.
    pub fn new(name: &'static str, logic: Logic, operand: Arc<Voxels>) -> Self {
        Self {
            name,
            logic,
            operand,
            cost: COMBINE_COST,
        }
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }
}

impl BlockOp for CombineOp {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    /// The element type of the operand it was built with, and nothing else.
    ///
    /// A two-input op has two element types to agree about, and the second one
    /// is fixed when the op is constructed. So this is not a policy — it is what
    /// the operand *is*, and a block of any other type has nothing to be
    /// combined with.
    fn accepts(&self, dtype: Dtype) -> bool {
        dtype == self.operand.dtype() && matches!(dtype, Dtype::Bool | Dtype::F64)
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, at: &Anchor) -> Result<()> {
        let volume = self.operand.shape();
        if volume != at.volume {
            return Err(Error::InvalidArgument(format!(
                "{}: the operand is {:?} but the volume is {:?}",
                self.name, volume, at.volume
            )));
        }
        let shape = input.shape();
        for axis in 0..3 {
            if at.offset[axis] + shape[axis] > at.volume[axis] {
                return Err(Error::InvalidArgument(format!(
                    "{}: a buffer of {:?} at {:?} does not fit a volume of {:?}",
                    self.name, shape, at.offset, at.volume
                )));
            }
        }
        let window = crate::region::Region::new(&at.offset, &shape);
        let logic = self.logic;
        match input.dtype() {
            // Two `bool` buffers into the `bool` connective: the diamond's sink
            // with no conversion on either arm.
            Dtype::Bool => {
                let operand = self.operand.slice_region(&window)?;
                logic_into(
                    input.view::<bool>()?,
                    operand.view::<bool>()?,
                    out.view_mut::<bool>()?,
                    logic,
                )
            }
            _ => {
                let operand = self.operand.slice_region(&window)?;
                combine_into(
                    input.view::<f64>()?,
                    operand.view::<f64>()?,
                    out.view_mut::<f64>()?,
                    |&left, &right| from_set(logic.apply(is_set(left), is_set(right))),
                )
            }
        }
    }

    /// Only where the connective absorbs, because the op is told about one
    /// operand and knows nothing about the other.
    ///
    /// `AND` with an all-zero input is zero whatever the operand holds, and `OR`
    /// with an all-set input is set. Those two are exactly true. Everything else
    /// — including every `XOR` — depends on data this op has not been shown, and
    /// the default `None` is the right answer for it.
    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        self.logic.absorbs(is_set(value)).map(from_set)
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

/// The sink of a diamond: join several branch results with one connective.
///
/// This is [`CombineOp`]'s arithmetic with the operands arranged the way
/// [`Chain::Parallel`](crate::op::Chain::Parallel) supplies them — every branch
/// of the fan-in produced a buffer from the same input at the same anchor, and
/// they arrive as a slice in branch order. Nothing is held and nothing is
/// sliced, so the anchor is not consulted at all: a voxelwise join of buffers
/// that are already co-located has no position to be wrong about.
///
/// **Why it takes any arity above one rather than exactly two.** `Logic` is a
/// binary connective, but `And` and `Or` are associative and `Xor` is too, so
/// folding left over `n` branches is well defined and is what a three-arm
/// diamond wants. The arity is checked rather than assumed: `accepts` refuses a
/// list of one, and `cost_per_voxel` is told `n` so a three-arm join is charged
/// for the two pairs it actually does.
///
/// **The element types must agree across branches**, which is a statement about
/// *this* combine and not about fan-in. A connective is a function of two
/// values of the same kind; a combine joining an image with a mask is a
/// different combine, and `Chain::produces` deliberately leaves that decision
/// here rather than imposing agreement on every fan-in there could be.
pub struct LogicCombine {
    name: &'static str,
    logic: Logic,
    cost: f64,
}

impl LogicCombine {
    pub fn new(name: &'static str, logic: Logic) -> Self {
        Self {
            name,
            logic,
            cost: COMBINE_COST,
        }
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }

    /// One pair, through this module's kernels and nothing else.
    fn pair(&self, left: &Voxels, right: &Voxels, out: &mut Voxels) -> Result<()> {
        match left.dtype() {
            Dtype::Bool => logic_into(
                left.view::<bool>()?,
                right.view::<bool>()?,
                out.view_mut::<bool>()?,
                self.logic,
            ),
            _ => combine_into(
                left.view::<f64>()?,
                right.view::<f64>()?,
                out.view_mut::<f64>()?,
                |&left, &right| from_set(self.logic.apply(is_set(left), is_set(right))),
            ),
        }
    }
}

impl Combine for LogicCombine {
    fn name(&self) -> &'static str {
        self.name
    }

    /// Zero, on every axis and at every volume size — this reads the voxel it
    /// writes, in each of its operands, and nothing else. Stated explicitly
    /// because [`Combine::reach`] has no default: a fan-in's halo is the widest
    /// branch's plus *this*, so a silent zero here would be a silent zero in
    /// the halo.
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn accepts(&self, inputs: &[Dtype]) -> bool {
        inputs.len() >= 2
            && matches!(inputs[0], Dtype::Bool | Dtype::F64)
            && inputs.iter().all(|&dtype| dtype == inputs[0])
    }

    fn produces(&self, inputs: &[Dtype]) -> Dtype {
        inputs[0]
    }

    /// Every branch must have produced the same extent, and the two that did
    /// not are named.
    ///
    /// This is the fallible half of [`Combine::output_shape`] being used for
    /// what it is for. A voxelwise join has no meaning for buffers of different
    /// extents — there is no correspondence between their voxels — and the
    /// alternative to refusing is to pick one and silently combine the wrong
    /// pairs, which is the class of failure this crate is arranged against.
    fn output_shape(&self, inputs: &[[usize; 3]]) -> Result<[usize; 3]> {
        let first = *inputs.first().ok_or_else(|| {
            Error::InvalidArgument(format!("{}: no branch results to join", self.name))
        })?;
        for (branch, shape) in inputs.iter().enumerate() {
            if shape != &first {
                return Err(Error::InvalidArgument(format!(
                    "{}: branch 0 produced {first:?} and branch {branch} produced {shape:?}. A \
                     voxelwise join pairs co-located voxels, and buffers of different extents \
                     have no such pairing.",
                    self.name
                )));
            }
        }
        Ok(first)
    }

    fn apply(&self, inputs: &[Voxels], out: &mut Voxels, _at: &Anchor) -> Result<()> {
        if inputs.len() < 2 {
            return Err(Error::InvalidArgument(format!(
                "{}: a connective joins at least two results and was handed {}",
                self.name,
                inputs.len()
            )));
        }
        // Fold left, writing the **last** pair straight into `out` so the
        // two-branch case — which is the diamond — allocates nothing at all.
        let mut folded: Option<Voxels> = None;
        for index in 1..inputs.len() {
            let last = index + 1 == inputs.len();
            if last {
                self.pair(folded.as_ref().unwrap_or(&inputs[0]), &inputs[index], out)?;
            } else {
                let mut next = Voxels::zeros(inputs[0].dtype(), inputs[0].shape())?;
                self.pair(
                    folded.as_ref().unwrap_or(&inputs[0]),
                    &inputs[index],
                    &mut next,
                )?;
                folded = Some(next);
            }
        }
        Ok(())
    }

    /// The exact fold, because *every* operand is known.
    ///
    /// [`CombineOp::constant_maps_to`] may only answer where the connective
    /// absorbs, since it is told about one operand and knows nothing about the
    /// array it holds. This one is different in kind: `Chain::constant_maps_to`
    /// only reaches here when every branch has folded to a constant, so there
    /// is no unknown operand and the connective can simply be applied.
    fn constant_maps_to(&self, values: &[f64]) -> Option<f64> {
        let mut folded = is_set(*values.first()?);
        for &value in &values[1..] {
            folded = self.logic.apply(folded, is_set(value));
        }
        Some(from_set(folded))
    }

    /// `branches - 1` pairs' worth of work, which is what folding a binary
    /// connective over a list costs.
    fn cost_per_voxel(&self, branches: usize) -> f64 {
        self.cost * branches.saturating_sub(1) as f64
    }
}

/// The unit of the whole cost model: one voxelwise map's worth of work per
/// voxel.
///
/// `1.0` by construction rather than by measurement — `ops::cost` divides every
/// other op by this one — but **the absolute figure behind it moved when the
/// boxed closure went away**, and that is a fact about every other constant in
/// `super`. On the machine this crate was developed on, through the same
/// `BlockOp::apply` and the same ramp, a `threshold` block cost **1.91
/// ns/voxel** through `Box<dyn Fn>` and costs **0.71 ns/voxel** through an
/// `M: MapFn`: the unit got 2.7 times cheaper. Every stored
/// constant in `rank`, `morphology`, `local` and `smooth` is a multiple of this
/// unit and was read off the table before that, so each of them now *understates*
/// its op by roughly the same factor. The ordering the planner acts on is
/// unaffected — the unit moved, not the ops — which is why nothing was rescaled
/// here; a rescale is its own change, and one that should not be made on a table
/// whose neighbourhood rows this crate has since found to swing 35% with
/// codegen-unit partitioning alone (see [`cost_report`]).
///
/// Public because it is what a third party's [`MapFn::cost`] is denominated in.
/// A map that does about as much arithmetic as a threshold returns this.
pub const MAP_COST: f64 = 1.0;

/// What an [`Identity`] costs: a `memcpy`, not a map.
///
/// Measured; see [`cost_report`], which prices the identity as its own row
/// precisely so this is not a guess — and the measurement is the interesting
/// part. It is **0.68 against the threshold's 0.71 ns/voxel**: removing the
/// arithmetic entirely buys 4%. At one `f64` read and one `f64` write per voxel
/// a voxelwise op is bound by memory and not by what it computes, so the floor
/// this constant describes is nearly the whole pass.
///
/// It is stored below [`MAP_COST`] rather than equal to it because the ordering
/// is real and the planner should have it — but a planner that treated an
/// identity phase as free would be wrong by a factor of twenty-five, and that,
/// not the 4%, is what this number is for.
pub const IDENTITY_COST: f64 = 0.95;

/// Measured; see `super::COST_MEASUREMENT`.
pub(super) const COMBINE_COST: f64 = 0.49;

/// The measurement [`IDENTITY_COST`] and [`MAP_COST`]'s absolute figure came
/// from, kept as text so a re-run somewhere else can be compared against it
/// rather than merely replacing it.
///
/// `--release`, 96 x 64 x 64, best of 40, on the machine this crate was
/// developed on. Best of **40** rather than the 5 `ops::cost` uses, because at a
/// nanosecond a voxel five samples is not enough to separate 0.60 from 0.71:
///
/// ```text
/// case                                     ns/voxel   relative     stored
/// threshold, Box<dyn Fn> (the old shell)       1.91       2.69          -
/// threshold                                    0.71       1.00       1.00
/// identity                                     0.68       0.96       0.95
/// four thresholds composed                     3.15       4.44       4.00
/// four closures composed                       0.60       0.85       4.00
/// threshold, as a closure                      0.60       0.85       1.00
/// ```
///
/// Four things it says, in order of how much they matter:
///
/// * **The indirect call was the cost.** 1.91 to 0.71 for the same arithmetic,
///   which is what this whole change was for.
/// * **A closure is not a slow path.** The last row is the first row's
///   arithmetic written as `|value| ...` and handed to
///   [`VoxelwiseMapOp::new`]. It is not merely as fast as the named
///   [`Threshold`], it is slightly *faster* — a closure bakes its level and its
///   two outputs in as immediates where `Threshold` loads them from fields.
///   Monomorphisation is doing the work here, not the named types.
/// * **A voxelwise op is bound by memory, not by arithmetic.** An identity
///   `memcpy` and a threshold differ by 4%, and four fused closures cost no more
///   than one map. That is the real argument for [`Compose`]: the second map is
///   nearly free, and the buffer it avoids is not.
/// * **The derived cost of a [`Compose`] over-charges the good case and
///   under-charges the bad one.** Four thresholds fused measure 4.44 units where
///   the sum of the parts says 4.00; four closures measure 0.85. The sum is kept
///   for both, because it is the model the *arithmetic* justifies and because
///   the spread between those two rows is an artefact of how much of a map lives
///   in fields — see [`Threshold::map`], which is written branchless to close
///   most of it (5.05 before, 3.15 after). A planner constant should not encode
///   that.
///
/// **This is its own measurement rather than rows in `ops::cost::measure`, and
/// that was not a preference.** Adding these cases to that function's list moved
/// the `gaussian smooth` row from 55 to 73 ns/voxel with `smooth.rs` untouched.
/// Rebuilding both with `-C codegen-units=1` made the two agree at 78, which
/// says the 55 was a lucky codegen-unit partition rather than a property of the
/// convolution — but it also says that any edit to `ops::cost` reshuffles the
/// numbers the constants in four other modules were read off. `super`'s header
/// already records three ops that measure themselves in their own file for
/// reasons of their own; this is a fourth, for that one.
pub fn cost_report(shape: [usize; 3], repetitions: usize) -> String {
    use std::time::Instant;

    let voxels = (shape[0] * shape[1] * shape[2]) as f64;
    let anchor = Anchor::whole(shape);
    let mut input = ndarray::Array3::<f64>::zeros((shape[0], shape[1], shape[2]));
    for (flat, value) in input.iter_mut().enumerate() {
        *value = ((flat * 7919) % 1013) as f64;
    }
    let input: Voxels = input.into();

    // Four thresholds, so the composed row is far enough from one map to say
    // something: whether the derived cost — the sum of the parts — is the model.
    let composed = Compose::new(
        Compose::new(
            Threshold::above(500.0, 1.0, 0.0),
            Threshold::above(0.5, 700.0, 300.0),
        ),
        Compose::new(
            Threshold::above(500.0, 1.0, 0.0),
            Threshold::above(0.5, 700.0, 300.0),
        ),
    );
    let cases: Vec<(String, Box<dyn BlockOp>)> = vec![
        (
            "threshold".to_string(),
            Box::new(VoxelwiseMapOp::threshold("map", 500.0, 1.0, 0.0)),
        ),
        (
            "identity".to_string(),
            Box::new(VoxelwiseMapOp::identity("identity")),
        ),
        (
            "four thresholds composed".to_string(),
            Box::new(VoxelwiseMapOp::from_map("composed", composed)),
        ),
        // The same four maps as the row above, written as closures. The pair
        // exists because one composed row could not have told apart "composing
        // costs something" from "composing *these* costs something": a closure
        // holds its parameters as immediates and a `Threshold` holds them in
        // fields, and the gap between the two rows is that difference and
        // nothing else.
        (
            "four closures composed".to_string(),
            Box::new(VoxelwiseMapOp::from_map(
                "closures",
                Compose::new(
                    Compose::new(
                        |value: f64| if value > 500.0 { 1.0 } else { 0.0 },
                        |value: f64| if value > 0.5 { 700.0 } else { 300.0 },
                    ),
                    Compose::new(
                        |value: f64| if value > 500.0 { 1.0 } else { 0.0 },
                        |value: f64| if value > 0.5 { 700.0 } else { 300.0 },
                    ),
                ),
            )),
        ),
        // The same arithmetic as the first row, written as a closure. The row
        // exists to answer "is the escape hatch a slow path", and the answer has
        // to be re-askable rather than asserted in prose.
        (
            "threshold, as a closure".to_string(),
            Box::new(VoxelwiseMapOp::new("closure", |value: f64| {
                if value > 500.0 {
                    1.0
                } else {
                    0.0
                }
            })),
        ),
    ];

    let mut rows = Vec::new();
    for (name, op) in cases {
        let mut out = Voxels::zeros(op.produces(input.dtype()), op.output_shape(shape)).unwrap();
        // One untimed pass: a freshly allocated output pays a page fault per
        // page on first touch, and at a nanosecond a voxel that fault *is* the
        // measurement.
        op.apply(&input, &mut out, &anchor).unwrap();
        let mut best = f64::INFINITY;
        for _ in 0..repetitions.max(1) {
            let started = Instant::now();
            op.apply(&input, &mut out, &anchor).unwrap();
            let elapsed = started.elapsed().as_secs_f64() * 1e9;
            std::hint::black_box(out.view::<f64>().unwrap().iter().take(1).sum::<f64>());
            best = best.min(elapsed / voxels);
        }
        rows.push((name, best, op.cost_per_voxel()));
    }

    let unit = rows.first().map(|(_, nanos, _)| *nanos).unwrap_or(1.0);
    let mut out = format!(
        "voxelwise map cost, {}x{}x{}, best of {repetitions}\n{:<40} {:>10} {:>10} {:>10}\n",
        shape[0], shape[1], shape[2], "case", "ns/voxel", "relative", "stored"
    );
    for (name, nanos, stored) in &rows {
        out.push_str(&format!(
            "{name:<40} {nanos:>10.2} {:>10.2} {stored:>10.2}\n",
            nanos / unit
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    fn mask(values: [bool; 8]) -> Array3<bool> {
        Array3::from_shape_vec((2, 2, 2), values.to_vec()).unwrap()
    }

    #[test]
    fn the_connectives_are_the_connectives() {
        let left = mask([true, true, false, false, true, false, true, false]);
        let right = mask([true, false, true, false, false, true, true, true]);
        for (logic, want) in [
            (
                Logic::And,
                [true, false, false, false, false, false, true, false],
            ),
            (Logic::Or, [true, true, true, false, true, true, true, true]),
            (
                Logic::Xor,
                [false, true, true, false, true, true, false, true],
            ),
        ] {
            let mut out = Array3::from_elem((2, 2, 2), false);
            logic_into(left.view(), right.view(), out.view_mut(), logic).unwrap();
            assert_eq!(out, mask(want), "{logic:?}");
        }
    }

    #[test]
    fn not_is_the_complement() {
        let input = mask([true, false, true, false, true, false, true, false]);
        let mut out = Array3::from_elem((2, 2, 2), false);
        not_into(input.view(), out.view_mut()).unwrap();
        assert_eq!(
            out,
            mask([false, true, false, true, false, true, false, true])
        );
    }

    /// The one thing a two-input op may say about a constant, and the three it
    /// may not.
    #[test]
    fn only_an_absorbing_connective_declares_a_constant() {
        assert_eq!(Logic::And.absorbs(false), Some(false));
        assert_eq!(Logic::And.absorbs(true), None);
        assert_eq!(Logic::Or.absorbs(true), Some(true));
        assert_eq!(Logic::Or.absorbs(false), None);
        assert_eq!(Logic::Xor.absorbs(false), None);
        assert_eq!(Logic::Xor.absorbs(true), None);
    }

    #[test]
    fn the_combine_op_slices_its_operand_at_the_anchor() {
        let volume = [4usize, 2, 2];
        let mut operand = Array3::<f64>::zeros((4, 2, 2));
        // set exactly the second half of axis 0
        for i in 2..4 {
            for j in 0..2 {
                for k in 0..2 {
                    operand[[i, j, k]] = 1.0;
                }
            }
        }
        let op = CombineOp::new("and", Logic::And, Arc::new(operand.into()));
        let input: Voxels = Array3::from_elem((2, 2, 2), 1.0).into();

        let mut low = Voxels::zeros(Dtype::F64, [2, 2, 2]).unwrap();
        op.apply(&input, &mut low, &Anchor::new([0, 0, 0], volume))
            .unwrap();
        assert!(low.view::<f64>().unwrap().iter().all(|&value| value == 0.0));

        let mut high = Voxels::zeros(Dtype::F64, [2, 2, 2]).unwrap();
        op.apply(&input, &mut high, &Anchor::new([2, 0, 0], volume))
            .unwrap();
        assert!(high
            .view::<f64>()
            .unwrap()
            .iter()
            .all(|&value| value == 1.0));
    }

    #[test]
    fn a_combine_op_refuses_an_operand_of_the_wrong_shape() {
        let op = CombineOp::new(
            "and",
            Logic::And,
            Arc::new(Array3::<f64>::zeros((4, 2, 2)).into()),
        );
        let input: Voxels = Array3::from_elem((2, 2, 2), 1.0).into();
        let mut out = Voxels::zeros(Dtype::F64, [2, 2, 2]).unwrap();
        let err = op
            .apply(&input, &mut out, &Anchor::whole([2, 2, 2]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("the volume is"), "got: {err}");
    }

    /// The same connective over two `bool` operands, at an eighth of the bytes
    /// and with no mask conversion on either arm.
    #[test]
    fn the_combine_op_takes_two_bool_operands_directly() {
        let volume = [4usize, 2, 2];
        let operand = Array3::from_shape_fn((4, 2, 2), |(i, _, _)| i >= 2);
        let op = CombineOp::new("and", Logic::And, Arc::new(operand.into()));
        assert!(op.accepts(Dtype::Bool));
        assert!(!op.accepts(Dtype::F64));

        let input: Voxels = Array3::from_elem((2, 2, 2), true).into();
        let mut high = Voxels::zeros(Dtype::Bool, [2, 2, 2]).unwrap();
        op.apply(&input, &mut high, &Anchor::new([2, 0, 0], volume))
            .unwrap();
        assert!(high.view::<bool>().unwrap().iter().all(|&value| value));
        assert_eq!(high.bytes(), 8);
    }

    #[test]
    fn a_voxelwise_map_declares_exactly_what_it_computes() {
        let op = VoxelwiseMapOp::threshold("t", 0.5, 1.0, 0.0);
        assert_eq!(op.constant_maps_to(0.25), Some(0.0));
        assert_eq!(op.constant_maps_to(0.75), Some(1.0));
        assert_eq!(op.reach(0, 1000), 0);

        let op = VoxelwiseMapOp::not("not");
        assert_eq!(op.constant_maps_to(0.0), Some(1.0));
        assert_eq!(op.constant_maps_to(7.0), Some(0.0));
    }

    // ------------------------------------------------- the map plug-in --

    /// A ramp with the awkward values in it, because a representation change
    /// that is byte-identical on `[0, 1)` and not on a negative zero is not
    /// byte-identical.
    fn ramp(shape: (usize, usize, usize)) -> Array3<f64> {
        let mut array = Array3::zeros(shape);
        let odd = [-0.0f64, 0.0, -1.0, f64::MIN_POSITIVE, 1e300, -1e-300];
        for (flat, value) in array.iter_mut().enumerate() {
            *value = if flat % 17 < odd.len() {
                odd[flat % 17]
            } else {
                ((flat * 7919) % 1013) as f64 * 0.5 - 200.0
            };
        }
        array
    }

    fn applied<M: MapFn>(op: &VoxelwiseMapOp<M>, input: &Voxels) -> Voxels {
        one(op, input)
    }

    fn one(op: &dyn BlockOp, input: &Voxels) -> Voxels {
        let mut out = Voxels::zeros(Dtype::F64, input.shape()).unwrap();
        op.apply(input, &mut out, &Anchor::whole(input.shape()))
            .unwrap();
        out
    }

    /// Run the ops as a sequence of separate passes, materialising every
    /// intermediate — which is exactly what fusing them is supposed to avoid.
    fn in_sequence(ops: &[&dyn BlockOp], input: &Voxels) -> Voxels {
        let mut current = input.clone();
        for op in ops {
            current = one(*op, &current);
        }
        current
    }

    /// Equal **bit for bit**, so that `-0.0` and `0.0` are different answers and
    /// two `NaN`s of the same payload are the same one. `PartialEq` on `f64`
    /// says neither of those, and a representation change is exactly where the
    /// difference would show up.
    #[track_caller]
    fn identical(left: &Voxels, right: &Voxels, what: &str) {
        let left = left.view::<f64>().unwrap();
        let right = right.view::<f64>().unwrap();
        assert_eq!(left.shape(), right.shape(), "{what}: shapes");
        for (index, (a, b)) in left.iter().zip(right.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{what}: voxel {index} is {a} fused and {b} in sequence"
            );
        }
    }

    /// The equality the whole of [`Compose`] rests on: fusing two maps into one
    /// op gives, voxel for voxel and bit for bit, what running them as two ops
    /// gives.
    ///
    /// Bit for bit rather than approximately, and over a ramp that contains a
    /// negative zero and two denormals, because the reason to fuse is that it is
    /// the *same computation* with one buffer fewer. An equality that held only
    /// to a tolerance would be a different computation, and the planner would be
    /// choosing between two answers rather than two costs.
    #[test]
    fn a_composed_map_is_the_two_maps_run_in_sequence() {
        let input: Voxels = ramp((7, 5, 3)).into();

        // 1. threshold then complement
        let a = VoxelwiseMapOp::threshold("t", 0.0, 1.0, 0.0);
        let b = VoxelwiseMapOp::not("n");
        let fused =
            VoxelwiseMapOp::from_map("fused", Compose::new(Threshold::above(0.0, 1.0, 0.0), Not));
        identical(
            &applied(&fused, &input),
            &in_sequence(&[&a, &b], &input),
            "threshold then complement",
        );

        // 2. an identity on either side, which must change nothing at all
        let identity = VoxelwiseMapOp::identity("id");
        let before = VoxelwiseMapOp::from_map(
            "fused",
            Compose::new(Identity, Threshold::above(0.0, 1.0, 0.0)),
        );
        let after = VoxelwiseMapOp::from_map(
            "fused",
            Compose::new(Threshold::above(0.0, 1.0, 0.0), Identity),
        );
        identical(
            &applied(&before, &input),
            &in_sequence(&[&identity, &a], &input),
            "identity first",
        );
        identical(
            &applied(&after, &input),
            &in_sequence(&[&a, &identity], &input),
            "identity last",
        );

        // 3. a closure on both sides, since the escape hatch composes too
        let scale = |value: f64| value * 2.0 - 1.0;
        let floor = |value: f64| value.max(0.0);
        let c = VoxelwiseMapOp::new("scale", scale);
        let d = VoxelwiseMapOp::new("floor", floor);
        identical(
            &applied(
                &VoxelwiseMapOp::from_map("fused", Compose::new(scale, floor)),
                &input,
            ),
            &in_sequence(&[&c, &d], &input),
            "two closures",
        );

        // 4. three deep, and **both ways of associating** the same three maps.
        // Composition is associative here as a fact about function composition,
        // and the two nestings are different Rust types with different generated
        // code, so it is worth pinning that they are also the same answer.
        let e = VoxelwiseMapOp::at_or_above("t2", 1.0, 5.0, -5.0);
        let three = in_sequence(&[&c, &d, &e], &input);
        let left = VoxelwiseMapOp::from_map(
            "fused",
            Compose::new(
                Compose::new(scale, floor),
                Threshold::at_or_above(1.0, 5.0, -5.0),
            ),
        );
        let right = VoxelwiseMapOp::from_map(
            "fused",
            Compose::new(
                scale,
                Compose::new(floor, Threshold::at_or_above(1.0, 5.0, -5.0)),
            ),
        );
        identical(&applied(&left, &input), &three, "(a . b) . c");
        identical(&applied(&right, &input), &three, "a . (b . c)");
    }

    /// Both halves of the short-circuit contract, for every variant: what the op
    /// *declares* a constant block maps to, and what computing that block
    /// actually produces, must agree.
    ///
    /// This is the property the boxed closure got for free by calling itself.
    /// Deriving the answer from the expression is what puts it at risk, so it is
    /// pinned per variant rather than once.
    #[test]
    fn what_a_map_declares_for_a_constant_is_what_computing_it_gives() {
        let probes = [-1.0, -0.0, 0.0, 0.5, 1.0, 7.0, 500.0];

        fn agree<M: MapFn>(op: &VoxelwiseMapOp<M>, probes: &[f64]) {
            for &value in probes {
                let declared = op.constant_maps_to(value);
                let input = Voxels::filled(Dtype::F64, [2, 2, 2], value).unwrap();
                let computed = applied(op, &input);
                let computed = computed.view::<f64>().unwrap();
                let first = computed[[0, 0, 0]];
                assert!(
                    computed
                        .iter()
                        .all(|&other| other.to_bits() == first.to_bits()),
                    "{}: a constant block did not map to a constant block",
                    op.name()
                );
                assert_eq!(
                    declared.map(f64::to_bits),
                    Some(first.to_bits()),
                    "{} at {value}: declared {declared:?}, computed {first}",
                    op.name()
                );
            }
        }

        agree(&VoxelwiseMapOp::identity("identity"), &probes);
        agree(&VoxelwiseMapOp::not("not"), &probes);
        agree(&VoxelwiseMapOp::threshold("above", 0.5, 1.0, 0.0), &probes);
        agree(&VoxelwiseMapOp::at_or_above("at", 0.5, 1.0, 0.0), &probes);
        agree(
            &VoxelwiseMapOp::new("closure", |v: f64| v.max(0.0)),
            &probes,
        );
        agree(
            &VoxelwiseMapOp::from_map(
                "composed",
                Compose::new(Threshold::above(0.5, 1.0, 0.0), Not),
            ),
            &probes,
        );
        agree(
            &VoxelwiseMapOp::from_map(
                "deep",
                Compose::new(
                    Compose::new(Identity, Threshold::at_or_above(0.0, 2.0, -2.0)),
                    Compose::new(Not, Identity),
                ),
            ),
            &probes,
        );
    }

    /// The two tests are one voxel apart, and the voxel is the level itself.
    /// Stated because that voxel is why the non-strict test exists at all.
    #[test]
    fn the_two_threshold_tests_differ_exactly_at_the_level() {
        let strict = VoxelwiseMapOp::threshold("above", 0.0, 1.0, 0.0);
        let loose = VoxelwiseMapOp::at_or_above("at", 0.0, 1.0, 0.0);
        assert_eq!(strict.constant_maps_to(0.0), Some(0.0));
        assert_eq!(loose.constant_maps_to(0.0), Some(1.0));
        for probe in [-1.0, -0.5, 0.5, 1.0] {
            assert_eq!(
                strict.constant_maps_to(probe),
                loose.constant_maps_to(probe),
                "at {probe}"
            );
        }
    }

    /// Cost is derived from the expression, which is the whole reason it moved
    /// off the op and onto the map. No timing here — only the orderings the
    /// derivation must produce, which are assertions and not measurements.
    #[test]
    fn a_maps_cost_follows_its_expression() {
        let identity = VoxelwiseMapOp::identity("identity");
        let one = VoxelwiseMapOp::threshold("t", 0.5, 1.0, 0.0);
        assert!(identity.cost_per_voxel() < one.cost_per_voxel());
        assert!(identity.cost_per_voxel() > 0.0);

        // A composition costs its parts, so depth is visible to the planner.
        let pair = || Compose::new(Threshold::above(0.5, 1.0, 0.0), Not);
        let two = VoxelwiseMapOp::from_map("two", pair());
        let eight = VoxelwiseMapOp::from_map(
            "eight",
            Compose::new(Compose::new(pair(), pair()), Compose::new(pair(), pair())),
        );
        assert!(two.cost_per_voxel() > one.cost_per_voxel());
        assert!(eight.cost_per_voxel() > two.cost_per_voxel());
        // and it is the *sum*, which is what the flat constant could not say:
        // eight maps' arithmetic, not one map's.
        assert_eq!(eight.cost_per_voxel(), 8.0 * MAP_COST);

        // and `with_cost` still overrides, for a map that cannot price itself
        assert_eq!(one.with_cost(40.0).cost_per_voxel(), 40.0);
    }

    /// Retaking the measurement. Ignored because timing in a test suite is a
    /// measurement of the machine's mood, not of the code — but it is here, it
    /// runs, and it is one command:
    ///
    /// ```text
    /// cargo test --release -- --ignored --nocapture ops::voxelwise
    /// ```
    ///
    /// Forty repetitions where `ops::cost` takes five: these ops run at about a
    /// nanosecond a voxel, and five samples do not separate 0.60 from 0.71.
    #[test]
    #[ignore = "a measurement, not an assertion"]
    fn print_the_cost_table() {
        println!("{}", cost_report([96, 64, 64], 40));
    }
}

/// Widen any element type to `f64`, value for value.
///
/// **The op that was missing between a level and a kernel stated in `f64`.**
/// `Voxels::widened` has always existed; nothing exposed it as a step of a
/// chain, so a phase writing `u16` — a median run at the reference's own width,
/// say — could not be read by an op whose kernel is `f64`, and the chain simply
/// could not be assembled.
///
/// **Widening only, and the name says so.** Narrowing is not the inverse of
/// this: it has to decide rounding, saturation and what a negative value means
/// in an unsigned type, and each of those is a decision a caller should make
/// explicitly rather than inherit from an op called `cast`. Every type a buffer
/// can hold is exactly representable in `f64` except `u64` and `i64` beyond 2^53
/// — stated here because it is the one lossy corner, and it is the same corner
/// `Voxels::widened` already has.
pub struct WidenOp {
    name: &'static str,
}

impl WidenOp {
    pub fn new(name: &'static str) -> Self {
        Self { name }
    }
}

impl BlockOp for WidenOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// Nothing: a widening reads the voxel it writes and no other.
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    /// Every type a buffer holds, `f64` included — widening an `f64` is the
    /// identity and is a legitimate thing for a chain to contain rather than a
    /// case to refuse, since which arm of a parameterised chain is live is not
    /// this op's business.
    fn accepts(&self, dtype: Dtype) -> bool {
        dtype != Dtype::F16
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::F64
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        if input.dtype() == Dtype::F16 {
            return Err(Error::InvalidArgument(format!(
                "{}: no buffer holds half-precision; `accepts` refuses it before a run starts",
                self.name
            )));
        }
        let widened = input.widened();
        out.view_mut::<f64>()?.assign(&widened);
        Ok(())
    }

    /// A constant widens to itself. Exact for every type this accepts, because
    /// the short circuit's constant is already an `f64` and widening is the
    /// identity on it.
    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        Some(value)
    }

    fn cost_per_voxel(&self) -> f64 {
        MAP_COST
    }
}
