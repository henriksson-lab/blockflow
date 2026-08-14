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

/// `out = map(in)`, voxelwise.
///
/// The map must be **pure and position-independent**. That is a precondition of
/// the constructor rather than something checked, and it is what licenses
/// `constant_maps_to` to answer by simply calling the map: if the map consulted
/// anything but its argument, the answer for a short-circuited block and for a
/// computed one would differ.
pub struct VoxelwiseMapOp {
    name: &'static str,
    map: Box<dyn Fn(f64) -> f64 + Send + Sync>,
    cost: f64,
}

impl VoxelwiseMapOp {
    /// `map` must be pure: same argument, same answer, wherever it is called.
    pub fn new(name: &'static str, map: impl Fn(f64) -> f64 + Send + Sync + 'static) -> Self {
        Self {
            name,
            map: Box::new(map),
            cost: MAP_COST,
        }
    }

    /// The complement of a mask, under this module's mask convention.
    pub fn not(name: &'static str) -> Self {
        Self::new(name, |value| from_set(!is_set(value)))
    }

    /// Compare against a fixed level: `above` where `value > level`, `below`
    /// elsewhere. A *global* threshold, with no position dependence at all —
    /// [`super::AdaptiveThresholdOp`] is the one that varies with position.
    pub fn threshold(name: &'static str, level: f64, above: f64, below: f64) -> Self {
        Self::new(name, move |value| if value > level { above } else { below })
    }

    /// Override the measured cost. For a `map` that is much more expensive than
    /// the arithmetic the default was measured on.
    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }
}

impl BlockOp for VoxelwiseMapOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// Zero, on every axis and at every volume size. A voxelwise op reads the
    /// voxel it writes and nothing else.
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    /// `f64` only, because the map it holds is an `f64 -> f64` closure the
    /// **caller** supplied. Accepting a narrower type would mean choosing a
    /// conversion on the caller's behalf, and there is no honest choice: a map
    /// written for intensities is not a map on flags. A caller wanting a
    /// narrower map supplies a narrower op.
    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        map_into(input.view::<f64>()?, out.view_mut::<f64>()?, |&value| {
            (self.map)(value)
        })
    }

    /// Exactly the map, applied to the constant. True because the map is pure
    /// and the op reads nothing else.
    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        Some((self.map)(value))
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

/// Measured; see `super::COST_MEASUREMENT`.
pub(super) const MAP_COST: f64 = 1.0;
/// Measured; see `super::COST_MEASUREMENT`.
pub(super) const COMBINE_COST: f64 = 0.49;

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
}
