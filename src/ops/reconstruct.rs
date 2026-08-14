// SPDX-License-Identifier: MIT
//
// Original work for this crate. Written from the definitions of the operations,
// not adapted from any implementation of them.
//
// **Grey morphological reconstruction**, and the h-maxima transform built on it.
//
// Reconstruction by dilation takes a seed `g` and a mask `f` with `g <= f`
// everywhere and repeats
//
//     g <- min( dilate(g, element), f )
//
// until nothing changes. The seed floods upward and outward and can never climb
// above the mask, so it fills every region reachable without descending and stops
// at the ridges. Reconstruction by **erosion** is the dual — `g <- max(erode(g,
// element), f)` with `g >= f` — and is the same code with two comparisons turned
// round; see [`Reconstruction`], which is why there is one loop here rather than
// two.
//
// The h-maxima transform is the reconstruction whose seed is the mask lowered by
// a constant:
//
//     HMAX_h(f) = reconstruct_by_dilation( seed = f - h, mask = f )
//
// A peak rising less than `h` above its surroundings is flooded from its own base
// and flattened; one rising more survives, truncated by `h`. **`h` is a
// prominence threshold in intensity units**, and that is the property
// `tests/reconstruct.rs` asserts against peaks of known prominence, because every
// other test in this file would pass on an op that got `h` wrong by a factor.
// `HMIN_h` is the dual.
//
// Why this is an iteration and not a `BlockOp`
// --------------------------------------------
// Flooding is transitive along paths, which is the wall `ops::fill` and
// `ops::regional` hit and describe. A plateau at constant height can run the
// length of the volume, and whether its far end rises depends on a voxel at the
// other end; no halo of any size answers that. Where those two ops answer it
// with a fragment-and-join, this one answers it with **depth**: a substage
// carries the flood one element's radius, blocks exchange their *outputs*
// between substages, and the phase's external reach is one substage's whatever
// the count turns out to be. That is what `crate::iterate` is for, and this is
// its first real user — `probes::CappedSpreadOp` is `g <- min(spread(g), f)`
// written to prove the machinery, and this is the operation it was a model of.
//
// The kernel is one substage, not the loop
// -----------------------------------------
// [`reconstruct_step_into`] computes one step and nothing else, and it is
// generic over `T: Copy + PartialOrd` — a comparison is the whole of what the
// algorithm needs, and demanding `Ord` would exclude the floating-point volumes
// this is mostly used on. The loop belongs to the framework: `IterativeOp`
// declares the operands, the executor runs substages until a global convergence
// check says stop, and the substage count appears in no reach, no level and no
// plan.
//
// [`reconstruct_to_fixed_point`] is the same kernel in a plain loop over a whole
// array. It is not a second implementation and it is `pub` for two reasons: it
// is the oracle a decomposed run is checked against, and it is the **only** way
// to reconstruct from an externally supplied seed today — see below.
//
// The seed is derived inside substage 0, and that is a constraint rather than a
// preference
// ----------------------------------------------------------------------------
// [`HExtremaOp`] is a thin `IterativeOp` shell over the kernel, following this
// module's house rule. Substage 0 writes `f - h` from the operand it was handed;
// every later substage takes the `Running` estimate and the `Fixed` mask.
//
// **This works only because a reconstruction's seed and its mask are the same
// array.** A phase has one input level, so every `Fixed` operand is a view of
// the array substage 0 seeds from; `crate::iterate::Operand::Fixed` is where a
// level id will go when levels are a DAG, and until they are, *reconstruct A
// under a mask B computed elsewhere* cannot be a phase. It is not lost — it is
// [`reconstruct_to_fixed_point`], for a caller holding the whole volume — and
// when the level DAG exists the op for it is a shell over this same kernel and
// not a second implementation. That is what the split is for.
//
// The reach, and what it is not multiplied by
// --------------------------------------------
// **The element's radius, per substage.** One dilation per substage reads one
// radius; there is no composition here to double it as an opening doubles it,
// and no substage count to multiply it by, because the depth is paid in private
// round trips rather than in halo. Nothing configures it: it is
// `element.reach(axis)` and there is no field that could be set to something
// else, on `super`'s first rule.
//
// Termination
// -----------
// **It terminates, and the argument is short.** Under dilation `g` is
// non-decreasing — the step takes the extremum over a neighbourhood that
// includes the voxel's own value, so no voxel can fall — and it is bounded above
// by `f`, since the step's last act is to cap it there. Every value `g` takes is
// a value that was already present in `g` or in `f`, so it moves through a
// *finite* set, monotonically, inside a bound: it reaches a fixed point in
// finitely many steps. The erosion dual is the same argument upside down.
//
// **One obligation the shape puts on substage 0, discharged here and worth
// knowing about.** The executor asks after every substage whether what was
// written differs from what was read, and what substage 0 read is the input
// level — so an op whose seed *equals* its input is declared converged before a
// single step runs. For this op that is exactly right: `h = 0` gives the seed
// `f`, and `min(dilate(f), f)` is `f`, so the seed is already the fixed point and
// `HMAX_0` costs one pass rather than two. It is right by argument rather than by
// luck, and an op whose seed could equal its input without being a fixed point of
// its own step would be silently truncated by the same rule.
//
// The limit is nonetheless required, and the reason is not doubt about the proof.
// A caller cannot see the proof. `crate::iterate::SubstageLimit` is a guard
// against an iteration that does not converge or a step that has stopped being
// monotone, and exceeding it is an error naming the op rather than a partially
// flooded volume, which would be plausible, well-formed and wrong.
//
// The bound is a **flooding** bound and the peeling bound would be too small
// -------------------------------------------------------------------------
// `ops::skeleton::PassLimit::for_volume` gives half the shortest axis, which is
// exactly right for an op that peels from the surface inward: nothing thicker
// than half the shortest axis fits in the volume, so no correct run can need
// more passes. **It is wrong here by construction.** A flood does not eat inward
// from the faces; it travels along paths, and a value seeded at one corner of a
// 48 x 4 x 4 volume has 47 voxels of axis 0 to cross. `PassLimit::for_volume`
// offers four. A correct run would be refused, which is the worst behaviour for a
// guard — see `the_peeling_bound_would_refuse_a_run_this_op_needs` in
// `tests/reconstruct.rs`, which pins the two numbers against each other.
//
// [`flooding_bound`] derives its own: the volume's **L1 diameter over the
// element's reach**, per axis, plus two. Crossing the volume takes at most
// `ceil((n_a - 1) / r_a)` substages on axis `a` — the element contains its poles,
// so one substage advances `r_a` voxels along it — and a shortest path between
// any two voxels is monotone on each axis, so the sum of the three is the whole
// crossing. The two are the substage that derives the seed and the substage that
// observes that nothing moved.
//
// An axis the element cannot move along at all — `r_a == 0`, a flat element —
// contributes **zero** rather than its length, because the flood provably never
// changes that coordinate.
//
// **What this bound assumes, stated because it is an assumption.** It is the
// length of a *shortest* path, and a mask can force a detour: a serpentine
// corridor of constant height in a slab has a geodesic quadratic in the slab's
// side while its L1 diameter is linear in it. The sound universal ceiling is the
// voxel count — a geodesic visits distinct voxels — and it is useless as a
// runaway guard, since it would let a 512-cube iterate a hundred million times
// before complaining. So the geometric bound is the one offered, a mask that
// forces a detour makes the guard fire **by name** with a message that says to
// raise it, and `tests/reconstruct.rs` builds exactly that corridor so the
// behaviour is on record rather than discovered by a caller.
//
// Preconditions, and what the non-finite values do
// -------------------------------------------------
// Reconstruction by dilation requires `seed <= mask`, and a seed above the mask
// is refused where it can be seen: [`reconstruct_to_fixed_point`] checks it once,
// naming the first voxel that breaks it. The step would otherwise cap the
// violation away at the first substage and compute a well-formed answer to a
// different question. For the op the precondition is `h >= 0`, which is checked
// at construction — a negative `h` puts the seed above the mask — and `h` must
// also be **finite**, because `f - inf` is `-inf` where `f` is finite and `NaN`
// where `f` is `+inf`, which is the one way this op could manufacture a NaN out
// of data that had none.
//
// The infinities in the *data* need no rule: `+inf` in the mask caps nothing and
// `-inf` floods nothing, both of which are what the order says, and `inf - h` is
// `inf` for a finite `h`.
//
// **NaN needs a rule, and there are two of them, in two places.** The kernel
// takes `ops::regional`'s position verbatim: a NaN is unordered with everything,
// itself included, so it displaces no value and is displaced by none. The whole
// of that is the strict comparisons in [`Reconstruction::spreads`] and
// [`Reconstruction::caps`]; there is no NaN test anywhere in the loop. A NaN
// voxel therefore neither floods nor lowers its neighbours — it is a *hole* in
// the propagation graph, which the flood cannot pass through and which does not
// hold anything back — and it keeps its own value forever.
//
// The **shell** is stricter, and this is a limit of the framework rather than of
// the operation: the executor's convergence test is `==` on what a substage
// wrote against what it read, and `NaN != NaN`, so one NaN anywhere makes every
// substage report a change and the run ends at the limit with a message about
// convergence that is true and useless. [`HExtremaOp`] therefore **refuses a mask
// holding a NaN**, at substage 0, naming itself and saying why. It costs one pass
// over the block at substage 0 only. A caller who wants a reading gets it by
// replacing NaN before the run — one voxelwise map, `+inf` for "treat missing
// data as a ceiling" and `-inf` for "treat it as a floor" — which is
// `ops::regional`'s own advice and says exactly what it means.
//
// Costs
// -----
// Measured, per substage and per element voxel; see [`COST_MEASUREMENT`] and
// [`cost_report`]. The measurement lives in this file rather than in
// `ops::cost::measure` for `super`'s stated reason applied to a new case: that
// harness builds `Box<dyn BlockOp>` and this op is not a `BlockOp`, so it cannot
// be fed to it at all. `ridge` and `skeleton` are the precedent for a module that
// measures itself and says where.

use ndarray::{Array3, ArrayView3, ArrayViewMut3};

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::iterate::{IterativeOp, Substage, SubstageLimit, SubstageOperand};
use crate::voxels::Voxels;

use super::element::StructuringElement;
use super::shapes_agree;

// ------------------------------------------------------------- the kernel --

/// Which of the two dual reconstructions.
///
/// One type rather than two functions, because the dual differs from the
/// original in exactly two comparisons and a separate erosion implementation is
/// a second thing to drift from the first — `ops::morphology` writes its opening
/// and closing as compositions for the same reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Reconstruction {
    /// `g <- min(dilate(g), f)`, from a seed **below** the mask. The one the
    /// h-maxima transform is built on.
    ByDilation,
    /// `g <- max(erode(g), f)`, from a seed **above** the mask.
    ByErosion,
}

impl Reconstruction {
    /// Does `candidate` displace `held` in the element's reduction: is it
    /// greater (dilation) or lesser (erosion)?
    ///
    /// **Strict, and that is the whole of this file's NaN rule.** A value that
    /// compares with nothing displaces nothing, and — because the reduction
    /// starts from the voxel's own value rather than from an identity — is
    /// displaced by nothing either.
    fn spreads<T: PartialOrd>(self, candidate: &T, held: &T) -> bool {
        match self {
            Reconstruction::ByDilation => candidate > held,
            Reconstruction::ByErosion => candidate < held,
        }
    }

    /// Does the mask hold `extreme` back — is the `min` (dilation) or the `max`
    /// (erosion) of the two the mask's own value?
    fn caps<T: PartialOrd>(self, mask: &T, extreme: &T) -> bool {
        match self {
            Reconstruction::ByDilation => mask < extreme,
            Reconstruction::ByErosion => mask > extreme,
        }
    }

    /// The seed this reconstruction takes from a mask lowered (or raised) by `h`.
    pub fn seed_from(self, mask: f64, h: f64) -> f64 {
        match self {
            Reconstruction::ByDilation => mask - h,
            Reconstruction::ByErosion => mask + h,
        }
    }

    /// Does this pair satisfy the precondition — `seed <= mask` for a dilation,
    /// `seed >= mask` for an erosion?
    ///
    /// False for a NaN on either side, which is the honest answer: an unordered
    /// value satisfies no inequality, and a reconstruction whose precondition
    /// cannot be checked is one whose answer means nothing.
    pub fn admits<T: PartialOrd>(self, seed: &T, mask: &T) -> bool {
        match self {
            Reconstruction::ByDilation => seed <= mask,
            Reconstruction::ByErosion => seed >= mask,
        }
    }

    /// The transform this reconstruction implements when its seed is the mask
    /// offset by `h`, for a message that names the operation a caller asked for.
    fn transform(self) -> &'static str {
        match self {
            Reconstruction::ByDilation => "h-maxima",
            Reconstruction::ByErosion => "h-minima",
        }
    }
}

/// One step of a grey reconstruction: `min(dilate(running), mask)`, or the
/// erosion dual.
///
/// Generic over `T: Copy + PartialOrd`, which is the whole of what a
/// reconstruction needs — it selects a value that was read and hands it back,
/// never combining two, so no arithmetic bound appears here and none is asked
/// for.
///
/// **The reduction starts from the voxel's own value**, before any offset is
/// consulted. That is not an optimisation over an identity element: `dilate` in
/// a reconstruction must be *extensive* (`dilate(g) >= g`), and starting there
/// makes it so whatever the element is — including one that does not contain its
/// own centre, which every element this crate builds does but which nothing in
/// this signature promises.
///
/// **Edge behaviour is `super`'s convention.** The element is clamped to the
/// array handed in: offsets that fall outside it are skipped, so the reduction
/// is over the voxels that exist. At a real volume boundary that is the whole
/// story and the whole-volume reference clamps identically; at a block seam it
/// is deliberately *wrong*, which is what makes a short halo loud instead of
/// silent.
pub fn reconstruct_step_into<T: Copy + PartialOrd>(
    running: ArrayView3<'_, T>,
    mask: ArrayView3<'_, T>,
    element: &StructuringElement,
    method: Reconstruction,
    mut out: ArrayViewMut3<'_, T>,
) -> Result<()> {
    shapes_agree(running.shape(), mask.shape(), "reconstruct_step_into")?;
    shapes_agree(running.shape(), out.shape(), "reconstruct_step_into")?;
    if element.is_empty() {
        return Err(Error::InvalidArgument(
            "reconstruct_step_into: an empty element has nothing to reduce over, so the flood \
             would never leave the voxel it started on"
                .to_string(),
        ));
    }
    let extent = [
        running.shape()[0] as isize,
        running.shape()[1] as isize,
        running.shape()[2] as isize,
    ];
    for i in 0..running.shape()[0] {
        for j in 0..running.shape()[1] {
            for k in 0..running.shape()[2] {
                let mut extreme = running[[i, j, k]];
                for offset in element.offsets() {
                    let a = i as isize + offset[0];
                    let b = j as isize + offset[1];
                    let c = k as isize + offset[2];
                    if a < 0 || b < 0 || c < 0 || a >= extent[0] || b >= extent[1] || c >= extent[2]
                    {
                        continue;
                    }
                    let candidate = running[[a as usize, b as usize, c as usize]];
                    if method.spreads(&candidate, &extreme) {
                        extreme = candidate;
                    }
                }
                let held = mask[[i, j, k]];
                out[[i, j, k]] = if method.caps(&held, &extreme) {
                    held
                } else {
                    extreme
                };
            }
        }
    }
    Ok(())
}

/// Reconstruct `seed` under `mask` over the **whole volume**, and say how many
/// steps it took.
///
/// The same kernel in a plain loop, with no blocks and no halo. Two jobs, and
/// neither is a second implementation of anything:
///
/// * it is the oracle a block-decomposed run is checked against, which is the bar
///   every op in this crate meets;
/// * it is the only way to reconstruct from an **externally supplied** seed
///   today. A phase has one input level, so an op cannot be handed two arrays;
///   see the module header.
///
/// `limit` bounds the steps this loop takes, on
/// [`crate::iterate::SubstageLimit`]'s argument, and exceeding it is an error
/// rather than a partial answer. Note that a *phase* spends one substage
/// deriving its seed, so its count is one greater than this function's over the
/// same data; [`h_extrema`] does that arithmetic so that the two agree exactly.
///
/// The stop is `next == current`, which is the executor's own predicate — the
/// reference must converge where the real run converges or the counts cannot be
/// compared. It inherits the executor's NaN behaviour with it, which is why the
/// precondition check below refuses a NaN before the loop can spin on one.
pub fn reconstruct_to_fixed_point<T: Copy + PartialOrd>(
    seed: ArrayView3<'_, T>,
    mask: ArrayView3<'_, T>,
    element: &StructuringElement,
    method: Reconstruction,
    limit: SubstageLimit,
) -> Result<(Array3<T>, usize)> {
    shapes_agree(seed.shape(), mask.shape(), "reconstruct_to_fixed_point")?;
    for i in 0..seed.shape()[0] {
        for j in 0..seed.shape()[1] {
            for k in 0..seed.shape()[2] {
                if !method.admits(&seed[[i, j, k]], &mask[[i, j, k]]) {
                    return Err(Error::InvalidArgument(format!(
                        "reconstruct_to_fixed_point: the seed does not satisfy the \
                         precondition at [{i}, {j}, {k}] — a {:?} reconstruction needs the \
                         seed on the mask's own side of it everywhere, and an unordered value \
                         on either side satisfies nothing. The step would cap the violation \
                         away at the first substage and hand back a well-formed answer to a \
                         different question, so it is refused here instead.",
                        method
                    )));
                }
            }
        }
    }

    let mut current = seed.to_owned();
    let mut steps = 0usize;
    loop {
        let mut next = current.clone();
        reconstruct_step_into(current.view(), mask, element, method, next.view_mut())?;
        steps += 1;
        let changed = next != current;
        current = next;
        if !changed {
            return Ok((current, steps));
        }
        if steps >= limit.substages() {
            return Err(Error::InvalidArgument(format!(
                "a {:?} reconstruction did not reach a fixed point in {} step(s) over a {:?} \
                 volume. Either the limit is below what this data needs — raise it, or take it \
                 from `flooding_bound`, and note that a mask forcing a serpentine path needs \
                 more than the geometric bound gives — or the step has stopped being monotone, \
                 which would be a defect in the kernel rather than in the data. The partially \
                 flooded volume is deliberately not returned: it is a plausible, well-formed, \
                 wrong answer.",
                method,
                limit.substages(),
                [mask.shape()[0], mask.shape()[1], mask.shape()[2]]
            )));
        }
    }
}

/// The h-maxima (or h-minima) transform over the **whole volume**, and how many
/// substages it took **counted the way a phase counts them**.
///
/// `HMAX_h(f) = reconstruct_by_dilation(seed = f - h, mask = f)`, composed here
/// out of the same seed derivation and the same kernel [`HExtremaOp`] runs, so a
/// disagreement between this and a decomposed run is a decomposition bug rather
/// than a modelling difference. The returned count includes the seeding
/// substage, so it is directly comparable with `Stats::substages`.
///
/// **Substage 0 both seeds and answers the convergence question**, which is the
/// one place this has to model the executor rather than the mathematics: the
/// executor compares what a substage wrote against what it read, and what
/// substage 0 read is the input level. So a seed equal to the input — which is
/// what `h = 0` gives — converges in one substage without a step ever running,
/// and `HMAX_0` costs one pass rather than two. Modelled by the comparison below
/// rather than by a test on `h`, because it is the array the executor compares
/// and not the parameter.
pub fn h_extrema(
    values: ArrayView3<'_, f64>,
    element: &StructuringElement,
    method: Reconstruction,
    h: f64,
    limit: SubstageLimit,
) -> Result<(Array3<f64>, usize)> {
    check_h(h, method)?;
    let seed = values.mapv(|value| method.seed_from(value, h));
    if seed == values {
        return Ok((seed, 1));
    }
    // One of the phase's substages goes on the seed, so the loop below gets the
    // rest of the budget. Without this the two would refuse at different points
    // over the same data and the whole-volume run would stop being a model of
    // the phase.
    let steps = SubstageLimit::of(limit.substages().saturating_sub(1).max(1))?;
    let (out, ran) = reconstruct_to_fixed_point(seed.view(), values, element, method, steps)?;
    Ok((out, ran + 1))
}

/// The substage limit the **geometry** gives: the volume's L1 diameter over the
/// element's reach, plus two.
///
/// See the module header for the derivation and for the assumption it rests on.
/// In one line: a flood crosses axis `a` in `ceil((n_a - 1) / r_a)` substages
/// because the element contains its poles, a shortest path is monotone on each
/// axis so the three add, and the two extra are the substage that derives the
/// seed and the substage that observes that nothing moved.
///
/// **Not `PassLimit::for_volume`.** That is half the shortest axis, which bounds
/// an op that peels inward from the surface and is far *below* what an op that
/// floods along paths needs; a guard set from it would refuse correct runs.
pub fn flooding_bound(volume: [usize; 3], element: &StructuringElement) -> SubstageLimit {
    let mut crossing = 0usize;
    for axis in 0..3 {
        let reach = element.reach(axis);
        // A flat axis is not a short axis: with reach zero the flood provably
        // never changes that coordinate, so it contributes nothing rather than
        // its length.
        if reach == 0 || volume[axis] <= 1 {
            continue;
        }
        crossing += (volume[axis] - 1).div_ceil(reach);
    }
    SubstageLimit::of(crossing + 2).expect("a sum plus two is positive")
}

fn check_h(h: f64, method: Reconstruction) -> Result<()> {
    if !h.is_finite() || h < 0.0 {
        return Err(Error::InvalidArgument(format!(
            "the {} transform needs a finite, non-negative h and was given {h}. A negative h \
             puts the seed on the wrong side of the mask, which is the one precondition a \
             reconstruction has; an infinite one turns a finite mask into an infinite seed and \
             an infinite mask into a NaN, which is the only way this op could manufacture an \
             unordered value out of data that held none.",
            method.transform()
        )));
    }
    Ok(())
}

// -------------------------------------------------------------- the shell --

/// Position of each operand in [`HExtremaOp::operands`]. Named rather than
/// written as `0` and `1` at the four places they are used, because the two are
/// the same array at substage 0 and a transposition would be invisible there.
const RUNNING: usize = 0;
const MASK: usize = 1;

/// The h-maxima transform, or its dual, as one iterative phase.
///
/// A thin shell over [`reconstruct_step_into`]: substage 0 derives the seed
/// `f - h` from the mask, and every substage after it runs one reconstruction
/// step. See the module header for why the seed has to be derived *here* rather
/// than supplied, and for what that will look like when it need not be.
pub struct HExtremaOp {
    name: &'static str,
    element: StructuringElement,
    method: Reconstruction,
    h: f64,
    limit: SubstageLimit,
    cost: f64,
}

impl HExtremaOp {
    /// `h` is a prominence in the volume's own units and must be finite and
    /// non-negative; `limit` is the runaway guard and is the caller's to derive,
    /// from [`flooding_bound`] or from something it knows about its data that
    /// the geometry does not.
    pub fn new(
        name: &'static str,
        method: Reconstruction,
        element: StructuringElement,
        h: f64,
        limit: SubstageLimit,
    ) -> Result<Self> {
        check_h(h, method)?;
        if element.is_empty() {
            return Err(Error::InvalidArgument(format!(
                "the {} transform was given an empty element, which has nothing to reduce over: \
                 the flood would never leave the voxel it started on and every peak would \
                 survive whatever its prominence.",
                method.transform()
            )));
        }
        let cost = cost_for(&element);
        Ok(Self {
            name,
            element,
            method,
            h,
            limit,
            cost,
        })
    }

    /// `HMAX_h`: flatten every peak rising less than `h` above its surroundings.
    pub fn maxima(
        name: &'static str,
        element: StructuringElement,
        h: f64,
        limit: SubstageLimit,
    ) -> Result<Self> {
        Self::new(name, Reconstruction::ByDilation, element, h, limit)
    }

    /// `HMIN_h`: the dual, over the basins.
    pub fn minima(
        name: &'static str,
        element: StructuringElement,
        h: f64,
        limit: SubstageLimit,
    ) -> Result<Self> {
        Self::new(name, Reconstruction::ByErosion, element, h, limit)
    }

    /// The bound this op's own behaviour gives over `volume`. [`flooding_bound`],
    /// under the op's own element — offered here so that a caller who has the op
    /// does not have to re-derive it from the parts.
    pub fn bound_for(volume: [usize; 3], element: &StructuringElement) -> SubstageLimit {
        flooding_bound(volume, element)
    }

    pub fn element(&self) -> &StructuringElement {
        &self.element
    }

    pub fn method(&self) -> Reconstruction {
        self.method
    }

    pub fn h(&self) -> f64 {
        self.h
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }
}

impl IterativeOp for HExtremaOp {
    fn name(&self) -> &'static str {
        self.name
    }

    /// The running estimate over the element's radius, and the mask pointwise.
    ///
    /// **No factor anywhere.** One dilation per substage reads one radius; the
    /// substage count multiplies nothing, because the depth is paid in private
    /// round trips. The mask's reach is zero and is stated rather than left to
    /// the widest operand, because it is a fact about the algorithm — the cap is
    /// pointwise — and `crate::iterate::substage_reach` takes the max, so a
    /// wrong zero here would be invisible while a wrong zero on the running
    /// operand would not.
    fn operands(&self) -> Vec<SubstageOperand> {
        vec![
            SubstageOperand::running([
                self.element.reach(0),
                self.element.reach(1),
                self.element.reach(2),
            ]),
            SubstageOperand::fixed([0, 0, 0]),
        ]
    }

    fn limit(&self) -> SubstageLimit {
        self.limit
    }

    /// `f64` only, which is narrower than the kernel and is deliberate.
    ///
    /// The kernel is generic over `Copy + PartialOrd` and would take any of the
    /// element types. What the *shell* adds is `f - h`, and that subtraction is
    /// where the bridges stop being exactly true: on a narrow integer type it
    /// saturates at the bottom of the range, so `h` silently stops being `h`
    /// there; on `f32` an `h` outside its range rounds to an infinity, which is
    /// the one value `check_h` refuses. This module's rule is that a shell
    /// declares only what is exactly true, so it declares `f64` and a caller
    /// with a narrower volume either widens it or calls the kernel, which is
    /// generic and is `pub` for that.
    fn accepts(&self, dtype: Dtype) -> bool {
        dtype == Dtype::F64
    }

    fn substage(&self, at: &Substage<'_>, out: &mut Voxels) -> Result<()> {
        let mask = at.operand(MASK)?.view::<f64>()?;
        if at.index() == 0 {
            // **The seed is derived from the mask operand**, not from the running
            // one, and the two are the same array — that identity is what makes
            // this op expressible with one input level at all, and reading the
            // mask here says so in the code rather than only in the header.
            let mut target = out.view_mut::<f64>()?;
            shapes_agree(mask.shape(), target.shape(), "h-extrema seed")?;
            for (slot, &value) in target.iter_mut().zip(mask.iter()) {
                // The convergence test upstream is `==`, and `NaN != NaN`, so one
                // unordered voxel would report a change forever and the run would
                // end at the limit with a true and useless message. Caught here,
                // at substage 0, where it costs one pass and can be explained.
                if value.is_nan() {
                    return Err(Error::InvalidArgument(format!(
                        "iterative op {:?} was handed a mask holding a NaN. The kernel would \
                         treat it as this crate's order says — unordered with everything, so it \
                         neither floods nor holds anything back — but an iterative phase stops \
                         when a substage writes what it read, and a NaN is not equal to itself, \
                         so the run would never converge and would fail at the limit with a \
                         message about convergence instead of about the data. Replace it before \
                         the run: one voxelwise map, to +inf to treat missing data as a ceiling \
                         or to -inf to treat it as a floor.",
                        self.name
                    )));
                }
                *slot = self.method.seed_from(value, self.h);
            }
            return Ok(());
        }
        let running = at.operand(RUNNING)?.view::<f64>()?;
        reconstruct_step_into(
            running,
            mask,
            &self.element,
            self.method,
            out.view_mut::<f64>()?,
        )
    }

    fn cost_per_voxel(&self) -> f64 {
        self.cost
    }
}

// ---------------------------------------------------------------- costs --

/// Measured; see [`COST_MEASUREMENT`]. Per substage, and per **element voxel**,
/// because a 7-voxel element and a 343-voxel element are not one number — which
/// is `super::COST_MEASUREMENT`'s rule for every neighbourhood op here.
///
/// **Above `ops::morphology`'s 1.38, and the direction is the interesting part.**
/// The binary sweep reads one byte a voxel and stops at the first hit; this loop
/// reads eight and can stop at nothing, because a maximum is only known once
/// every offset has been seen. So a grey reconstruction step is a *dearer*
/// neighbourhood pass than a binary dilation over the same element, which is not
/// what one would guess from the two loops looking alike, and is the reason the
/// figure is measured rather than borrowed from next door.
///
/// **The spread is stated because the ratio is noisier than the op is.** Three
/// runs gave 1.79, 1.49 and 1.30 for the 27-voxel element and 1.73, 1.50, 1.32
/// for the 125-voxel one; `1.6` is stored. The reconstruction column itself moved
/// by under two percent across those runs — 143.3, 144.5, 145.7 nanoseconds a
/// voxel — and all the movement is in the *denominator*, since the voxelwise map
/// costs three to four nanoseconds a voxel and at that size a ratio inherits the
/// noise of memory bandwidth. `ops::skeleton`'s constant carries the same caveat
/// for the same reason.
///
/// A small element costs more per element voxel than a large one — 1.9 against
/// 1.5 over the three runs — because the per-voxel work that is not per offset
/// is amortised over fewer offsets. One rate is still the right model: it
/// under-prices a tiny element by about a quarter, and an op that reads three
/// voxels is not where a schedule is won or lost.
///
/// The seeding substage is cheaper than a stepping one — it is one subtraction
/// per voxel, measured at 0.6 to 0.8 of a voxelwise map — and is *not* priced
/// separately. One substage out of a count the planner cannot know is not worth a
/// second constant, and pricing it as a step over-prices the phase by one
/// substage's worth, which is the safe direction: `ops::skeleton`'s note applies
/// unchanged, that a schedule chosen with an op over-priced costs some redundancy
/// while one chosen with an op under-priced fuses something it should have cut.
pub const RECONSTRUCT_COST_PER_ELEMENT_VOXEL: f64 = 1.6;

/// Measured; see [`COST_MEASUREMENT`].
pub(super) fn cost_for(element: &StructuringElement) -> f64 {
    RECONSTRUCT_COST_PER_ELEMENT_VOXEL * element.len() as f64
}

/// The measurement the constant above came from, kept as text so that a re-run
/// somewhere else can be **compared** against it rather than merely replacing
/// it. `--release`, one thread, 96 x 64 x 64, best of 5, on the machine this was
/// written on:
///
/// ```text
/// case                                                       ns/voxel   relative    per elem
/// voxelwise map (the unit)                                      2.968       1.00
/// reconstruction step, 3-voxel element                         20.176       6.80       2.266
/// reconstruction step, 27-voxel element                       143.346      48.29       1.789
/// reconstruction step, 125-voxel element                      640.580     215.82       1.727
/// seed derivation (substage 0 only)                             2.407       0.81
/// ```
///
/// Three element sizes rather than one, because the whole question about a
/// neighbourhood op's cost is whether it scales with the element, and one sample
/// cannot answer it. It does: the 27-voxel and 125-voxel rows agree on the rate
/// to within four percent, and the row that disagrees is the 3-voxel one, which
/// is the amortisation and not the arithmetic. See
/// [`RECONSTRUCT_COST_PER_ELEMENT_VOXEL`] for what moved between runs and what
/// did not.
pub const COST_MEASUREMENT: &str = "ops::reconstruct::cost_report";

/// Retake the measurement, through the kernel the substage calls. Runnable;
/// `print_the_cost_table` below is the one command.
///
/// Not through `ops::cost::measure`: that harness builds `Box<dyn BlockOp>` and
/// an iterative op is not one, so it cannot be fed to it. `ridge` and `skeleton`
/// are the precedent, and the unit is the same voxelwise map so the numbers stay
/// comparable with the module's table.
pub fn cost_report(shape: [usize; 3], repetitions: usize) -> String {
    use std::time::Instant;

    use super::element::ElementShape;

    let voxels = (shape[0] * shape[1] * shape[2]) as f64;
    let repetitions = repetitions.max(1);

    let best_of = |mut run: Box<dyn FnMut()>| -> f64 {
        // One untimed pass first: a freshly allocated output pays a page fault
        // per page on first touch, and that fault is the measurement for the
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

    let ramp = Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
        ((i * 7919 + j * 104_729 + k * 1013) % 1013) as f64
    });

    let mut rows: Vec<(String, f64, f64)> = Vec::new();

    {
        let input: Voxels = ramp.clone().into();
        let op = super::voxelwise::VoxelwiseMapOp::threshold("map", 500.0, 1.0, 0.0);
        let mut out = Voxels::zeros(Dtype::F64, shape).unwrap();
        let anchor = crate::op::Anchor::whole(shape);
        rows.push((
            "voxelwise map (the unit)".to_string(),
            best_of(Box::new(move || {
                use crate::op::BlockOp;
                op.apply(&input, &mut out, &anchor).unwrap();
            })),
            1.0,
        ));
    }

    for radius in [[1, 0, 0], [1, 1, 1], [2, 2, 2]] {
        let element = StructuringElement::from_radius(ElementShape::Box, radius);
        let mask = ramp.clone();
        let seed = mask.mapv(|value| value - 100.0);
        let mut out = Array3::<f64>::zeros((shape[0], shape[1], shape[2]));
        let size = element.len();
        let named = element.clone();
        rows.push((
            format!("reconstruction step, {size}-voxel element"),
            best_of(Box::new(move || {
                reconstruct_step_into(
                    seed.view(),
                    mask.view(),
                    &named,
                    Reconstruction::ByDilation,
                    out.view_mut(),
                )
                .unwrap();
            })),
            size as f64,
        ));
    }

    {
        let mask = ramp.clone();
        let mut out = Array3::<f64>::zeros((shape[0], shape[1], shape[2]));
        rows.push((
            "seed derivation (substage 0 only)".to_string(),
            best_of(Box::new(move || {
                for (slot, &value) in out.iter_mut().zip(mask.iter()) {
                    *slot = value - 100.0;
                }
            })),
            1.0,
        ));
    }

    let unit = rows.first().map(|(_, nanos, _)| *nanos).unwrap_or(1.0);
    let mut out = format!(
        "reconstruct cost, {}x{}x{}, best of {repetitions}\n{:<56} {:>10} {:>10} {:>11}\n",
        shape[0], shape[1], shape[2], "case", "ns/voxel", "relative", "per elem"
    );
    for (name, nanos, divisor) in &rows {
        let relative = nanos / unit;
        if *divisor > 1.0 {
            out.push_str(&format!(
                "{name:<56} {nanos:>10.3} {relative:>10.2} {:>11.3}\n",
                relative / divisor
            ));
        } else {
            out.push_str(&format!("{name:<56} {nanos:>10.3} {relative:>10.2}\n"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::element::ElementShape;
    use super::*;

    fn box3() -> StructuringElement {
        StructuringElement::from_radius(ElementShape::Box, [1, 1, 1])
    }

    fn generous() -> SubstageLimit {
        SubstageLimit::of(10_000).expect("a positive limit")
    }

    /// A step is `min(dilate(g), f)` and nothing else. Written out here against a
    /// hand-computed case, because every property in `tests/reconstruct.rs` rests
    /// on this arithmetic being the arithmetic claimed.
    #[test]
    fn one_step_is_the_capped_dilation_and_the_dual_is_the_capped_erosion() {
        let element = StructuringElement::from_radius(ElementShape::Box, [1, 0, 0]);
        let mask = Array3::from_shape_vec((4, 1, 1), vec![9.0, 5.0, 5.0, 5.0]).unwrap();
        let seed = Array3::from_shape_vec((4, 1, 1), vec![9.0, 0.0, 0.0, 0.0]).unwrap();
        let mut out = Array3::<f64>::zeros((4, 1, 1));
        reconstruct_step_into(
            seed.view(),
            mask.view(),
            &element,
            Reconstruction::ByDilation,
            out.view_mut(),
        )
        .unwrap();
        // the 9 stays (capped by its own mask), its neighbour rises to the mask,
        // and nothing two voxels away has moved yet
        assert_eq!(
            out.iter().copied().collect::<Vec<_>>(),
            vec![9.0, 5.0, 0.0, 0.0]
        );

        // the dual, with the same numbers turned over
        let mask = Array3::from_shape_vec((4, 1, 1), vec![1.0, 5.0, 5.0, 5.0]).unwrap();
        let seed = Array3::from_shape_vec((4, 1, 1), vec![1.0, 9.0, 9.0, 9.0]).unwrap();
        let mut out = Array3::<f64>::zeros((4, 1, 1));
        reconstruct_step_into(
            seed.view(),
            mask.view(),
            &element,
            Reconstruction::ByErosion,
            out.view_mut(),
        )
        .unwrap();
        assert_eq!(
            out.iter().copied().collect::<Vec<_>>(),
            vec![1.0, 5.0, 9.0, 9.0]
        );
    }

    /// The kernel is generic, and an integer volume is the case that proves it:
    /// no `f64` appears anywhere in the algorithm.
    #[test]
    fn the_kernel_reconstructs_an_integer_volume_with_no_widening() {
        let element = StructuringElement::from_radius(ElementShape::Box, [1, 0, 0]);
        let mask = Array3::from_shape_vec((5, 1, 1), vec![9u16, 5, 5, 5, 0]).unwrap();
        let seed = Array3::from_shape_vec((5, 1, 1), vec![9u16, 0, 0, 0, 0]).unwrap();
        let (got, steps) = reconstruct_to_fixed_point(
            seed.view(),
            mask.view(),
            &element,
            Reconstruction::ByDilation,
            generous(),
        )
        .unwrap();
        assert_eq!(
            got.iter().copied().collect::<Vec<_>>(),
            vec![9u16, 5, 5, 5, 0]
        );
        // three voxels to travel, and one more step to see that nothing moved
        assert_eq!(steps, 4);
    }

    /// NaN, as the module header defines it: it neither floods nor holds anything
    /// back, and it keeps its own value. The property is the kernel's; the shell
    /// refuses such a mask for a different reason, which
    /// `tests/reconstruct.rs` pins.
    #[test]
    fn a_nan_neither_floods_nor_blocks_and_keeps_its_own_value() {
        let element = StructuringElement::from_radius(ElementShape::Box, [1, 0, 0]);
        let mask = Array3::from_shape_vec((3, 1, 1), vec![9.0, f64::NAN, 9.0]).unwrap();
        let seed = Array3::from_shape_vec((3, 1, 1), vec![9.0, f64::NAN, 0.0]).unwrap();
        let mut out = Array3::<f64>::zeros((3, 1, 1));
        reconstruct_step_into(
            seed.view(),
            mask.view(),
            &element,
            Reconstruction::ByDilation,
            out.view_mut(),
        )
        .unwrap();
        assert_eq!(out[[0, 0, 0]], 9.0);
        assert!(out[[1, 0, 0]].is_nan(), "a NaN voxel keeps its own value");
        assert_eq!(
            out[[2, 0, 0]],
            0.0,
            "the flood cannot pass through a NaN: it is a hole in the graph"
        );

        // and it holds nothing back either — a voxel whose only high neighbour is
        // a NaN is neither raised nor lowered by it
        let mask = Array3::from_shape_vec((3, 1, 1), vec![9.0, 9.0, 9.0]).unwrap();
        let seed = Array3::from_shape_vec((3, 1, 1), vec![f64::NAN, 4.0, 4.0]).unwrap();
        let mut out = Array3::<f64>::zeros((3, 1, 1));
        reconstruct_step_into(
            seed.view(),
            mask.view(),
            &element,
            Reconstruction::ByDilation,
            out.view_mut(),
        )
        .unwrap();
        assert_eq!(out[[1, 0, 0]], 4.0);
    }

    /// The precondition is checked where it can be seen, rather than capped away
    /// silently at the first step.
    #[test]
    fn a_seed_above_the_mask_is_refused_and_the_voxel_is_named() {
        let mask = Array3::from_elem((3, 3, 3), 1.0);
        let mut seed = Array3::from_elem((3, 3, 3), 0.0);
        seed[[1, 2, 0]] = 5.0;
        let message = reconstruct_to_fixed_point(
            seed.view(),
            mask.view(),
            &box3(),
            Reconstruction::ByDilation,
            generous(),
        )
        .unwrap_err()
        .to_string();
        assert!(message.contains("[1, 2, 0]"), "{message}");
        assert!(message.contains("precondition"), "{message}");

        // and a NaN satisfies nothing, so it is caught by the same check
        let mut seed = Array3::from_elem((3, 3, 3), 0.0);
        seed[[0, 0, 0]] = f64::NAN;
        assert!(reconstruct_to_fixed_point(
            seed.view(),
            mask.view(),
            &box3(),
            Reconstruction::ByDilation,
            generous(),
        )
        .is_err());
    }

    /// A zero radius is the degenerate element — one voxel, its own centre — and
    /// it is **not** empty. The flood then never leaves the voxel it started on,
    /// so `HMAX_h` becomes `f - h` clamped at `f`, which is `f - h`. Pinned
    /// because it is the boundary the emptiness guards sit next to, and because a
    /// reader would otherwise have to guess which side of it a zero radius falls
    /// on.
    #[test]
    fn a_zero_radius_is_one_voxel_rather_than_no_voxels() {
        let element = StructuringElement::from_radius(ElementShape::Box, [0, 0, 0]);
        assert_eq!(element.len(), 1);
        assert!(!element.is_empty());

        let values = Array3::from_shape_vec((3, 1, 1), vec![1.0, 9.0, 1.0]).unwrap();
        let (got, substages) = h_extrema(
            values.view(),
            &element,
            Reconstruction::ByDilation,
            2.0,
            generous(),
        )
        .unwrap();
        assert_eq!(
            got.iter().copied().collect::<Vec<_>>(),
            vec![-1.0, 7.0, -1.0]
        );
        assert_eq!(
            substages, 2,
            "the seed, and one substage that moves nothing"
        );
    }

    /// `h` is a prominence, and the two values that are not prominences are
    /// refused where they are stated rather than where they do damage.
    #[test]
    fn a_negative_or_non_finite_h_is_refused_at_construction() {
        for bad in [-1.0, -0.5, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let message = HExtremaOp::maxima("hmax", box3(), bad, generous())
                .map(|_| ())
                .unwrap_err()
                .to_string();
            assert!(message.contains("h-maxima"), "{bad}: {message}");
            assert!(message.contains("non-negative"), "{bad}: {message}");
        }
        assert!(HExtremaOp::maxima("hmax", box3(), 0.0, generous()).is_ok());
        assert!(HExtremaOp::minima("hmin", box3(), 3.5, generous()).is_ok());
    }

    /// The bound is the diameter over the reach, and it is **larger** than the
    /// peeling bound on any volume that is not a cube of side two — which is the
    /// whole reason it is derived here rather than taken from `PassLimit`.
    #[test]
    fn the_flooding_bound_is_the_diameter_over_the_reach_and_exceeds_the_peeling_one() {
        let unit = StructuringElement::from_radius(ElementShape::Box, [1, 1, 1]);
        // 47 + 3 + 3 crossings, plus the seed and the quiet substage
        assert_eq!(flooding_bound([48, 4, 4], &unit).substages(), 55);
        assert_eq!(
            super::super::skeleton::PassLimit::for_volume([48, 4, 4]).passes(),
            4
        );

        // a wider element crosses in fewer substages, and the bound follows it
        let wide = StructuringElement::from_radius(ElementShape::Box, [4, 4, 4]);
        // ceil(47/4) + ceil(3/4) + ceil(3/4), plus the two
        assert_eq!(flooding_bound([48, 4, 4], &wide).substages(), 16);

        // a flat axis contributes nothing, because the flood cannot move along it
        let flat = StructuringElement::from_radius(ElementShape::Box, [0, 1, 1]);
        assert_eq!(flooding_bound([48, 4, 4], &flat).substages(), 8);

        // and a single voxel needs the two substages nothing can avoid
        assert_eq!(flooding_bound([1, 1, 1], &unit).substages(), 2);
    }

    /// What can be asserted about a measured cost without measuring: the
    /// orderings the constant encodes.
    #[test]
    fn the_stored_cost_keeps_the_order_the_measurement_found() {
        use crate::op::BlockOp;

        let small = StructuringElement::from_radius(ElementShape::Box, [1, 1, 1]);
        let large = StructuringElement::from_radius(ElementShape::Box, [2, 2, 2]);
        let cheap = HExtremaOp::maxima("hmax", small.clone(), 1.0, generous()).unwrap();
        let dear = HExtremaOp::maxima("hmax", large, 1.0, generous()).unwrap();
        assert!(dear.cost_per_voxel() > cheap.cost_per_voxel());

        // per substage, and above a voxelwise map's 1.0 by roughly the element
        assert!(cheap.cost_per_voxel() > 1.0);

        // A grey step costs *more* per element voxel than the binary sweep over
        // the same element — eight bytes a voxel against one, and no hit to stop
        // at — which is the measurement's least obvious finding and the one a
        // future edit is most likely to invert by borrowing the neighbour's
        // constant.
        let sweep = super::super::morphology::MorphologyOp::new(
            "erode",
            super::super::morphology::Morphology::Erode,
            small,
        );
        assert!(cheap.cost_per_voxel() > sweep.cost_per_voxel());
    }

    /// Retaking the measurement. Ignored because timing in a test suite is a
    /// measurement of the machine's mood, not of the code — but it is here, it
    /// runs, and it is one command:
    ///
    /// ```text
    /// cargo test --release -- --ignored --nocapture ops::reconstruct
    /// ```
    #[test]
    #[ignore = "a measurement, not an assertion"]
    fn print_the_cost_table() {
        println!("{}", cost_report([96, 64, 64], 5));
    }
}
