// SPDX-License-Identifier: MIT
//
// **The acceptance bar for cutting a block into slabs: bit-identity, not a
// tolerance.**
//
// `slab::apply_sliced` claims that cutting a block into slabs, each grown by the
// chain's reach, and running them on separate threads produces *the same bits*
// as running the block uncut. This file is what holds it to that, and it runs in
// the default suite rather than behind `--ignored`, because it is a correctness
// property and not a measurement. `docs/design/intra-block.md` is the
// measurement.
//
// Why bit-identity is the right bar and a tolerance is not
// -------------------------------------------------------
// The property is exactly derivable, so anything weaker would be hiding
// something. A slab's core carries the full reach of margin, so every output
// voxel sums exactly the values it would have summed over the uncut block, in
// the same order; no floating-point sum is reassociated and only *which thread
// runs the loop* changes. A tolerance would pass a cut that was subtly wrong at
// the seams, which is the one failure this whole mechanism can produce and the
// one that would never be noticed — a complete, well-formed volume with a
// slightly wrong stripe every few planes.
//
// The four controls, and what each one catches
// --------------------------------------------
// A test that passes while measuring nothing is this project's most common
// failure, so each of these exists to make a specific one impossible:
//
// * `a_position_dependent_op_survives_the_cut` — a slab is told where it sits.
//   Without it, every test here would still pass on a position-*independent* op
//   while `slab_placement` was silently wrong, and every anchored op in the
//   crate would compute the block corner's answer for every slab.
// * `an_op_that_lies_about_being_a_stencil_is_caught_by_this_bar` — the bar can
//   **fail**. An op that declares `Stencil` and folds across its buffer is cut,
//   and the answer differs. Without this, a green file would be evidence that
//   the assertions ran, not that they could ever fire.
// * `every_shipped_op_refuses_to_be_sliced_today` — the default is doing its
//   job. `Slicing::UNDECLARED` is what an op that says nothing means, and no
//   shipped op has been declared yet.
// * `the_uncut_path_is_taken_at_one_thread` — the `threads <= 1` short circuit
//   is a different code path, so identity there is trivially true and proves
//   nothing about the cut. Every other assertion is at two threads or more.

use blockflow::decomposition::{Decomposition, PhaseDecomposition, SlabPolicy};
use blockflow::dtype::Dtype;
use blockflow::env::ArrayEnvironment;
use blockflow::error::Result;
use blockflow::geometry::BlockGrid;
use blockflow::log::Stats;
use blockflow::op::{Anchor, BlockOp, Chain, Combine, Placement, Slicing, SourceInputs};
use blockflow::slab::{apply_sliced, SlabCut};
use blockflow::strategy::{execute, Hints, Workflow};
use blockflow::voxels::Voxels;
use ndarray::Array3;

// ------------------------------------------------------------------- ops --

/// A second difference on each axis folded through a transcendental, so that a
/// reassociated sum would not survive rounding.
///
/// Deliberately **not** a sum of independent terms that a compiler could reorder
/// into the same bits by luck: the fold is a running accumulator that is halved
/// at each tap, so the answer depends on the order the taps were visited in and
/// a reassociation would show.
///
/// **The first version of this used `tanh` and was blind.** `tanh` saturates to
/// exactly `1.0` for any argument past about 20, and the fixture's values go to
/// 140 — so every accumulator pinned at `1.0` and the op returned the same
/// answer whatever it read. A mutant that dropped the halo to zero left this
/// test green while every other one in the file failed. The fold below cannot
/// saturate.
struct Curvature {
    reach: usize,
}

impl BlockOp for Curvature {
    fn name(&self) -> &'static str {
        "curvature"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        self.reach
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        dtype == Dtype::F64
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::F64
    }

    fn slicing(&self) -> Slicing {
        Slicing::Stencil
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        let source = input.view::<f64>()?;
        let mut sink = out.view_mut::<f64>()?;
        let [nx, ny, nz] = input.shape();
        let r = self.reach as isize;
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let mut acc = 0.0f64;
                    for (axis, extent) in [nx, ny, nz].into_iter().enumerate() {
                        for step in -r..=r {
                            let mut at = [i, j, k];
                            let moved =
                                (at[axis] as isize + step).clamp(0, extent as isize - 1) as usize;
                            at[axis] = moved;
                            let weight = if step == 0 { -2.0 } else { 1.0 };
                            acc = acc.mul_add(0.5, weight * source[at] * 1.000_000_1);
                        }
                    }
                    sink[[i, j, k]] = acc;
                }
            }
        }
        Ok(())
    }
}

/// A stencil whose answer depends on **where the voxel is in the volume**.
///
/// The whole point of the anchor a slab is given. If `slab_placement` shifted by
/// the wrong amount — or not at all — this op's answer would differ and nothing
/// else in the file would notice.
struct Positioned;

impl BlockOp for Positioned {
    fn name(&self) -> &'static str {
        "positioned"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        1
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        dtype == Dtype::F64
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::F64
    }

    fn slicing(&self) -> Slicing {
        Slicing::Stencil
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, at: &Anchor) -> Result<()> {
        let source = input.view::<f64>()?;
        let mut sink = out.view_mut::<f64>()?;
        let [nx, ny, nz] = input.shape();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let below = source[[i.saturating_sub(1), j, k]];
                    let here = source[[i, j, k]];
                    let global = [at.offset[0] + i, at.offset[1] + j, at.offset[2] + k];
                    let position = (global[0] * 7 + global[1] * 3 + global[2]) as f64;
                    sink[[i, j, k]] = (here - below) * 1.000_001 + position;
                }
            }
        }
        Ok(())
    }
}

/// Declares `Stencil` and **is not one**: every output voxel is a function of
/// the whole buffer it was handed.
///
/// A deliberate liar, here to prove this file's bar can fail. Nothing in the
/// framework can detect this from the declaration — that is exactly why the
/// declaration is a claim a test has to check.
struct LiesAboutBeingAStencil;

impl BlockOp for LiesAboutBeingAStencil {
    fn name(&self) -> &'static str {
        "lies about being a stencil"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        1
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        dtype == Dtype::F64
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::F64
    }

    fn slicing(&self) -> Slicing {
        Slicing::Stencil
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        let source = input.view::<f64>()?;
        let total: f64 = source.iter().sum();
        let mut sink = out.view_mut::<f64>()?;
        for value in sink.iter_mut() {
            *value = total;
        }
        Ok(())
    }
}

/// An op that says nothing about slicing, which is every op in `src/ops` today.
struct Undeclared;

impl BlockOp for Undeclared {
    fn name(&self) -> &'static str {
        "undeclared"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        1
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        dtype == Dtype::F64
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::F64
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        out.assign(input)
    }
}

/// A voxelwise join of two branches, declared sliceable.
struct Mean;

impl Combine for Mean {
    fn name(&self) -> &'static str {
        "mean"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn slicing(&self) -> Slicing {
        Slicing::Stencil
    }

    fn accepts(&self, inputs: &[Dtype]) -> bool {
        inputs.len() == 2 && inputs.iter().all(|dtype| *dtype == Dtype::F64)
    }

    fn produces(&self, _inputs: &[Dtype]) -> Dtype {
        Dtype::F64
    }

    fn output_shape(&self, inputs: &[[usize; 3]]) -> Result<[usize; 3]> {
        Ok(inputs[0])
    }

    fn apply(&self, inputs: &[Voxels], out: &mut Voxels, _at: &Anchor) -> Result<()> {
        let left = inputs[0].view::<f64>()?;
        let right = inputs[1].view::<f64>()?;
        let mut sink = out.view_mut::<f64>()?;
        for (value, (a, b)) in sink.iter_mut().zip(left.iter().zip(right.iter())) {
            *value = (a + b) * 0.5;
        }
        Ok(())
    }
}

/// A voxelwise join that says nothing, so a `Parallel` node over two stencil
/// branches is still refused.
struct UndeclaredCombine;

impl Combine for UndeclaredCombine {
    fn name(&self) -> &'static str {
        "undeclared combine"
    }
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }
    fn accepts(&self, inputs: &[Dtype]) -> bool {
        inputs.len() == 2
    }
    fn produces(&self, _inputs: &[Dtype]) -> Dtype {
        Dtype::F64
    }
    fn output_shape(&self, inputs: &[[usize; 3]]) -> Result<[usize; 3]> {
        Ok(inputs[0])
    }
    fn apply(&self, inputs: &[Voxels], out: &mut Voxels, _at: &Anchor) -> Result<()> {
        out.assign(&inputs[0])
    }
}

// --------------------------------------------------------------- fixtures --

fn structured(shape: [usize; 3]) -> Voxels {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut volume = Array3::<f64>::zeros((shape[0], shape[1], shape[2]));
    for value in volume.iter_mut() {
        *value = (next() % 1000) as f64 / 7.0;
    }
    Voxels::F64(volume)
}

/// Somewhere off the origin, so an anchor that is ignored and an anchor that is
/// shifted wrongly give different answers from an anchor that is right.
fn placement(shape: [usize; 3]) -> Placement {
    Placement::same(Anchor::new(
        [11, 5, 3],
        [shape[0] + 40, shape[1] + 40, shape[2] + 40],
    ))
}

fn uncut(chain: &Chain, input: &Voxels, at: &Placement) -> Voxels {
    let mut out = Voxels::zeros(
        chain.produces(input.dtype()).expect("produces"),
        chain.placed_output_shape(input.shape(), at).expect("shape"),
    )
    .expect("out");
    chain
        .apply_placed(input, SourceInputs::none(), &mut out, at)
        .expect("uncut apply");
    out
}

fn cut(chain: &Chain, input: &Voxels, at: &Placement, threads: usize) -> Result<Voxels> {
    let mut out = Voxels::zeros(
        chain.produces(input.dtype())?,
        chain.placed_output_shape(input.shape(), at)?,
    )?;
    apply_sliced(chain, input, SourceInputs::none(), &mut out, at, threads)?;
    Ok(out)
}

/// How a probe perturbs one voxel, for the liveness check on the fixtures.
///
/// **Three, because two were not enough and one was not enough.** A monotone
/// fold notices a move towards its own extreme; a set-valued op notices only a
/// change of membership.
#[derive(Debug, Clone, Copy)]
enum Perturbation {
    /// Far above anything in the fixture: what a maximum notices.
    Up,
    /// Far below: what a minimum notices.
    Down,
    /// Out of the set: what a binary morphology notices, and the only one of the
    /// three it does.
    Clear,
}

impl Perturbation {
    const ALL: [Perturbation; 3] = [Perturbation::Up, Perturbation::Down, Perturbation::Clear];

    fn applied(self, value: f64) -> f64 {
        match self {
            Perturbation::Up => value + 1234.5,
            Perturbation::Down => value - 1234.5,
            Perturbation::Clear => 0.0,
        }
    }
}

/// How many voxels of two answers differ, **by bits**, for the element types this
/// file's bars are stated over.
///
/// **Per element type and refusing the rest by name**, rather than widening
/// everything to `f64` and comparing that. `Voxels::widened` is infallible and
/// lossy; a bar that is bit-identity cannot be built on a lossy comparison, and
/// an integer past `2^53` would then agree with a different integer. Adding an
/// element type here is a deliberate act, which is what the refusal is for.
fn differing(left: &Voxels, right: &Voxels) -> usize {
    assert_eq!(
        left.dtype(),
        right.dtype(),
        "two answers of different element types are not comparable"
    );
    match left.dtype() {
        Dtype::F64 => left
            .view::<f64>()
            .expect("f64")
            .iter()
            .zip(right.view::<f64>().expect("f64").iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count(),
        Dtype::Bool => left
            .view::<bool>()
            .expect("bool")
            .iter()
            .zip(right.view::<bool>().expect("bool").iter())
            .filter(|(a, b)| a != b)
            .count(),
        other => panic!(
            "this file's bars are stated over float64 and bool; {} needs an arm here rather \
             than a widening",
            other.numpy_name()
        ),
    }
}

/// Whether an answer is anything other than one value repeated.
///
/// The vacuity guard: bit-identity between two constant volumes holds for a
/// reason that has nothing to do with the cut being right, and a fixture whose
/// op saturates produces exactly that. It used to be "not zero everywhere",
/// which a `bool` answer of `true` everywhere would pass.
fn varies(answer: &Voxels) -> bool {
    match answer.dtype() {
        Dtype::F64 => {
            let view = answer.view::<f64>().expect("f64");
            view.iter()
                .any(|value| value.to_bits() != view.iter().next().expect("a voxel").to_bits())
        }
        Dtype::Bool => {
            let view = answer.view::<bool>().expect("bool");
            let first = *view.iter().next().expect("a voxel");
            view.iter().any(|value| *value != first)
        }
        other => panic!(
            "this file's bars are stated over float64 and bool; {} needs an arm here",
            other.numpy_name()
        ),
    }
}

/// **The fixture can see its halo**, which is the check the `tanh` mutant taught
/// this file to make: an op with a real reach must answer differently when the
/// data outside a slab's core changes, or bit-identity across a cut holds for a
/// reason that has nothing to do with the cut being right.
///
/// **Three perturbations, and the first two versions of this probe were blind to
/// one op each.** Perturbing the voxel *upwards* moved **zero** voxels of an
/// erosion's answer — an erosion takes the minimum, so a value made larger is
/// never selected — and the check then reported a fixture that could not see its
/// halo when in fact the probe could not see the op. A monotone fold notices a
/// perturbation towards its own extreme and ignores the other, so the probe has
/// to try both and take the one that moved.
///
/// **And then `erode` moved for neither.** `MorphologyOp` is a *binary*
/// morphology — an `f64` buffer goes through `is_set` — so a value made larger
/// *or* smaller is set either way and the mask never changes. A set-valued op is
/// perturbed by flipping membership, not by moving a magnitude. Three probes,
/// and a probe that cannot move an op says nothing about that op's fixture.
fn assert_the_fixture_can_see_its_halo(chain: &Chain, shape: [usize; 3], what: &str) {
    let input = structured(shape);
    let at = placement(shape);
    let unperturbed = uncut(chain, &input, &at);
    let mut best = 0usize;
    for perturb in Perturbation::ALL {
        let mut moved_input = input.clone();
        {
            let mut view = moved_input.view_mut::<f64>().expect("f64");
            let cell = [shape[0] / 2, shape[1] / 2, shape[2] / 2];
            view[cell] = perturb.applied(view[cell]);
        }
        let after = uncut(chain, &moved_input, &at);
        best = best.max(differing(&unperturbed, &after));
    }
    assert!(
        best > 1,
        "{what}: perturbing one voxel up, down or clear moved at most {best} voxel(s) of the \
         answer, so this fixture cannot see past a voxel and would not notice a halo dropped to \
         zero"
    );
}

/// The bar itself, applied to one chain.
fn assert_identical_at_every_thread_count(chain: &Chain, shape: [usize; 3], what: &str) {
    let input = structured(shape);
    let at = placement(shape);
    let reference = uncut(chain, &input, &at);
    assert!(
        varies(&reference),
        "{what}: the reference answered one value everywhere, so identity below would hold \
         vacuously — the op cannot be seeing its input"
    );
    for threads in [2usize, 3, 4, 5, 8] {
        if threads > shape[0] {
            continue;
        }
        let sliced = cut(chain, &input, &at, threads).expect("sliced apply");
        assert_eq!(
            sliced, reference,
            "{what}: cutting into {threads} slabs changed the answer"
        );
    }
}

// ------------------------------------------------------------------ tests --

#[test]
fn a_stencil_survives_the_cut_bit_for_bit() {
    let chain = Chain::Op(Box::new(Curvature { reach: 1 }));
    assert_identical_at_every_thread_count(&chain, [24, 7, 6], "one op, reach 1");
    let wider = Chain::Op(Box::new(Curvature { reach: 3 }));
    assert_identical_at_every_thread_count(&wider, [24, 7, 6], "one op, reach 3");
}

/// The control without which every other test here would pass on a broken
/// `slab_placement`.
#[test]
fn a_position_dependent_op_survives_the_cut() {
    let chain = Chain::Op(Box::new(Positioned));
    assert_identical_at_every_thread_count(&chain, [24, 5, 4], "position dependent");

    // And it really is position dependent: the same data at a different anchor
    // must give a different answer, or the control above is checking nothing.
    let input = structured([24, 5, 4]);
    let here = uncut(&chain, &input, &placement([24, 5, 4]));
    let elsewhere = uncut(
        &chain,
        &input,
        &Placement::same(Anchor::new([12, 5, 3], [64, 45, 44])),
    );
    assert_ne!(
        here, elsewhere,
        "the position-dependent op ignored its anchor, so it cannot detect a wrong one"
    );
}

/// A sequence's reaches **add**, so a slab needs the whole chain's halo. Getting
/// this wrong would leave the second op reading the first op's edge.
#[test]
fn a_sequence_survives_the_cut_and_its_reaches_add() {
    let chain = Chain::Sequence(vec![
        Chain::Op(Box::new(Curvature { reach: 2 })),
        Chain::Op(Box::new(Curvature { reach: 1 })),
    ]);
    assert_eq!(chain.reach(0, 64), 3, "a sequence's reaches must add");
    assert_identical_at_every_thread_count(&chain, [28, 5, 4], "sequence of two stencils");
}

#[test]
fn a_fan_in_survives_the_cut_when_the_combine_is_declared_too() {
    let chain = Chain::Parallel {
        branches: vec![
            Chain::Op(Box::new(Curvature { reach: 1 })),
            Chain::Op(Box::new(Positioned)),
        ],
        combine: Box::new(Mean),
    };
    assert!(chain.slicing().is_stencil());
    assert_identical_at_every_thread_count(&chain, [24, 5, 4], "fan-in with a declared combine");
}

/// **The bar can fail.** Without this, a green file would say the assertions ran
/// rather than that they could ever fire.
#[test]
fn an_op_that_lies_about_being_a_stencil_is_caught_by_this_bar() {
    let chain = Chain::Op(Box::new(LiesAboutBeingAStencil));
    let shape = [24usize, 5, 4];
    let input = structured(shape);
    let at = placement(shape);
    let reference = uncut(&chain, &input, &at);
    let sliced = cut(&chain, &input, &at, 4).expect("the framework cannot detect the lie");
    assert_ne!(
        sliced, reference,
        "the liar was cut and produced the same answer, so this file's bar cannot detect a \
         non-stencil and every other assertion in it is worthless"
    );
}

#[test]
fn an_undeclared_op_is_refused_and_the_refusal_says_so() {
    let chain = Chain::Op(Box::new(Undeclared));
    assert!(!chain.slicing().is_stencil());
    let shape = [16usize, 4, 4];
    let error = cut(&chain, &structured(shape), &placement(shape), 4)
        .expect_err("an undeclared op must be refused");
    let message = format!("{error}");
    assert!(
        message.contains("not sliceable") && message.contains("did not declare"),
        "the refusal must name the cause: {message}"
    );
}

/// One undeclared op anywhere poisons the subtree, and the refusal carried out
/// is the one that refused.
#[test]
fn one_undeclared_op_refuses_the_whole_chain() {
    let shape = [16usize, 4, 4];
    let sequence = Chain::Sequence(vec![
        Chain::Op(Box::new(Curvature { reach: 1 })),
        Chain::Op(Box::new(Undeclared)),
    ]);
    assert!(!sequence.slicing().is_stencil());
    assert!(cut(&sequence, &structured(shape), &placement(shape), 4).is_err());

    // A `Parallel` is only as sliceable as its combine, and two stencil branches
    // do not speak for it.
    let fan_in = Chain::Parallel {
        branches: vec![
            Chain::Op(Box::new(Curvature { reach: 1 })),
            Chain::Op(Box::new(Curvature { reach: 1 })),
        ],
        combine: Box::new(UndeclaredCombine),
    };
    assert!(
        !fan_in.slicing().is_stencil(),
        "an undeclared combine must refuse the node its declared branches sit in"
    );

    // An `Alternative` folds over **every** branch, not the one taken.
    let alternative = Chain::Alternative {
        branches: vec![
            Chain::Op(Box::new(Curvature { reach: 1 })),
            Chain::Op(Box::new(Undeclared)),
        ],
        taken: 0,
    };
    assert!(
        !alternative.slicing().is_stencil(),
        "a branch that is not taken must still be able to refuse the node"
    );
}

/// The default is doing its job, and this test is the record of when that stops
/// being true: an op declared sliceable will appear here.
///
/// **Inverted in part, not deleted.** Four of the ops this test was written
/// against — `SmoothOp`, `MorphologyOp`, `RankFilterOp` and `ConvolveOp` — now
/// declare themselves stencils and are held to it by
/// `the_shipped_stencils_survive_the_cut_bit_for_bit`. What remains here is the
/// half that is still true and is the more interesting half: the ops that *look*
/// sliceable from outside and are not. `ops::sliding` computes the same
/// statistic as `ops::rank` with the same reach and the same output shape, by
/// carrying a histogram along the scan — so **a reach says what an op reads, it
/// does not say the answer is a function only of what was read**, and this is
/// the pair that demonstrates it.
/// **The four shipped ops that now declare themselves stencils, held to the bar
/// rather than believed.**
///
/// The declaration is the cheap half; this is the half that costs something. Each
/// op is run uncut and then cut at every thread count and must agree **bit for
/// bit** — and each is a case the reasoning could have got wrong:
///
/// * `ConvolveOp` sums a fixed tap list per voxel in the element's own order, so
///   a cut cannot reassociate it. The kernel here is **asymmetric**, so a slab
///   whose offsets were mirrored would show.
/// * `MorphologyOp` folds with a minimum, which is associative and commutative —
///   the one case where order genuinely cannot matter.
/// * `RankFilterOp` gathers a window per voxel and carries nothing between them.
///   **`ops::sliding` computes the same statistic and is not a stencil**, which
///   is why the declaration is per op.
/// * `SmoothOp` is the one that had to be measured rather than argued: a
///   separable Gaussian is three passes, and pass two reads pass one's output.
///
/// **The fixture is checked against the trap this file already paid for.** A
/// halo-to-zero mutant once left a test green because the fold saturated; here
/// the assertion below is that the ops' answers actually *move* when the halo
/// moves, so a fixture that could not see its halo would be caught.
#[test]
fn the_shipped_stencils_survive_the_cut_bit_for_bit() {
    use blockflow::ops::element::{ElementShape, StructuringElement};

    let shape = [24usize, 7, 6];
    let box3 = StructuringElement::from_radius(ElementShape::Box, [1, 1, 1]);
    let weights: Vec<f64> = (0..box3.len())
        .map(|which| (which as f64 + 1.0) / box3.len() as f64 - 0.37)
        .collect();
    let kernel = blockflow::ops::Kernel::new(box3.clone(), weights).expect("a kernel");

    let cases: Vec<(&str, Chain)> = vec![
        (
            "convolve",
            Chain::Op(Box::new(blockflow::ops::ConvolveOp::new(
                "convolve",
                kernel,
                blockflow::ops::Sense::Correlate,
                blockflow::ops::ridge::Boundary::Clamp,
            ))),
        ),
        (
            "erode",
            Chain::Op(Box::new(blockflow::ops::morphology::MorphologyOp::new(
                "erode",
                blockflow::ops::morphology::Morphology::Erode,
                box3.clone(),
            ))),
        ),
        (
            "median",
            Chain::Op(Box::new(blockflow::ops::rank::RankFilterOp::new(
                "median",
                box3.clone(),
                blockflow::ops::Rank::median(&box3),
            ))),
        ),
        (
            "smooth",
            Chain::Op(Box::new(blockflow::ops::smooth::SmoothOp::new(
                "smooth",
                blockflow::ops::smooth::Gaussian::new([1.0, 1.0, 1.0], 4.0).expect("gaussian"),
            ))),
        ),
    ];
    for (what, chain) in &cases {
        assert!(
            chain.slicing().is_stencil(),
            "{what} must declare itself a stencil for this test to be about anything"
        );
        assert_identical_at_every_thread_count(chain, shape, what);
        assert_the_fixture_can_see_its_halo(chain, shape, what);
    }
}

#[test]
fn every_shipped_op_refuses_to_be_sliced_today() {
    let element = blockflow::ops::element::StructuringElement::from_radius(
        blockflow::ops::element::ElementShape::Box,
        [1, 1, 1],
    );
    let ops: Vec<Box<dyn BlockOp>> = vec![
        // **The one that matters.** Bounded reach, identity output shape, and
        // the same answer as `ops::rank`'s median — computed by carrying a
        // histogram along the scan, so where the scan starts is in the answer.
        // It is the counter-example to inferring sliceability from a reach.
        Box::new(blockflow::ops::sliding::SlidingHistogramOp::rank(
            "sliding median",
            element.clone(),
            blockflow::ops::Rank::median(&element),
            blockflow::ops::sliding::Domain::of_size(256).expect("a domain"),
        )),
    ];
    for op in &ops {
        assert!(
            !op.slicing().is_stencil(),
            "{} declares itself sliceable; add it to `docs/design/intra-block.md`'s table and \
             give it a bit-identity case in this file",
            op.name()
        );
        assert!(
            op.slicing().refusal().is_some_and(|why| !why.is_empty()),
            "{} refuses without saying why",
            op.name()
        );
    }
}

#[test]
fn the_uncut_path_is_taken_at_one_thread() {
    // At one thread `apply_sliced` does not cut at all, so it must accept a
    // chain it would otherwise refuse. That is what makes the parameter safe to
    // leave at 1 everywhere, which is what today's behaviour is.
    let chain = Chain::Op(Box::new(Undeclared));
    let shape = [16usize, 4, 4];
    let input = structured(shape);
    let at = placement(shape);
    let reference = uncut(&chain, &input, &at);
    for threads in [0usize, 1] {
        let same = cut(&chain, &input, &at, threads)
            .expect("one thread must not cut, so it must not refuse");
        assert_eq!(same, reference);
    }
}

/// A chain that reads a stored image is refused by name rather than sliced with
/// a guessed halo.
#[test]
fn a_source_leaf_is_refused_by_name() {
    let chain = Chain::Parallel {
        branches: vec![
            Chain::Op(Box::new(Curvature { reach: 1 })),
            Chain::source(1usize, Dtype::F64),
        ],
        combine: Box::new(Mean),
    };
    let shape = [16usize, 4, 4];
    let mut out = Voxels::zeros(Dtype::F64, shape).expect("out");
    let error = apply_sliced(
        &chain,
        &structured(shape),
        SourceInputs::none(),
        &mut out,
        &placement(shape),
        4,
    )
    .expect_err("a source leaf must be refused");
    assert!(
        format!("{error}").contains("Refused rather than guessed"),
        "the refusal must say it is a scope boundary: {error}"
    );
}

/// The cut's arithmetic, against the primitive that computes it.
#[test]
fn the_cut_amplification_is_what_the_slabs_actually_read() {
    let block = [40usize, 6, 6];
    let reach = blockflow::reach::Reach::symmetric([2, 2, 2]);
    for pieces in [1usize, 2, 4, 8] {
        let plan = SlabCut::plan(block, 0, pieces, &reach, block).expect("cut");
        let counted: usize = plan.slabs().iter().map(|slab| slab.extent.voxels()).sum();
        let written: usize = block.iter().product();
        assert!(
            (plan.amplification() - counted as f64 / written as f64).abs() < 1e-12,
            "the reported amplification must be what the slabs read"
        );
    }
}

// ------------------------------------------------------ the shipped sinks --

/// A voxelwise threshold, and it is here for the connective's sake.
///
/// `LogicCombine` joins **masks**. Two arms that are non-zero almost everywhere
/// give an `And` that is one everywhere and an `Xor` that is zero everywhere —
/// an answer that does not depend on what was read, which is exactly the shape
/// of fixture this file was caught by once already. So the arms under a
/// connective end in a threshold and the mask actually varies.
///
/// A stencil, and trivially: it reads the voxel it writes and nothing else.
struct Threshold {
    at: f64,
}

impl BlockOp for Threshold {
    fn name(&self) -> &'static str {
        "threshold"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        dtype == Dtype::F64
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::F64
    }

    fn slicing(&self) -> Slicing {
        Slicing::Stencil
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        let source = input.view::<f64>()?;
        let mut sink = out.view_mut::<f64>()?;
        ndarray::Zip::from(&mut sink)
            .and(source)
            .for_each(|slot, value| *slot = f64::from(*value > self.at));
        Ok(())
    }
}

/// **The three shipped `Combine`s that now declare themselves stencils, held to
/// the bar rather than believed.**
///
/// A `Parallel` node is only as sliceable as its narrowest part, so a diamond
/// whose arms are declared stencils was still refused while its sink said
/// nothing — and both of the arms this framework's consumers spend their time
/// in are fan-ins. **The declaration sits on the op; the refusal sits on the
/// chain around it.** These three are what unblocks that, and each is a case the
/// reasoning could have got wrong:
///
/// * `DifferenceCombine` is `a - b` per voxel, and the order is load-bearing —
///   a cut that mirrored its operands would show here and nowhere else.
/// * `ArithmeticCombine` reaches this file twice, once through the arithmetic
///   kernel (`Subtract`) and once through the selection one (`Maximum`), because
///   `pair` dispatches on the element type and the two are different code.
/// * `LogicCombine` converts each operand through `MaskElement::set` before the
///   connective, so its arms end in a threshold here: two arms that are non-zero
///   everywhere would make the answer independent of what was read.
///
/// Each fan-in also goes through `assert_the_fixture_can_see_its_halo`, so a
/// combine that passed by joining two answers that could not see their own halos
/// is caught rather than counted.
#[test]
fn the_shipped_combines_survive_the_cut_bit_for_bit() {
    use blockflow::ops::background::DifferenceCombine;
    use blockflow::ops::{Arithmetic, ArithmeticCombine, Logic, LogicCombine};

    let shape = [24usize, 6, 5];
    let arms = || {
        vec![
            Chain::Op(Box::new(Curvature { reach: 3 })),
            Chain::Op(Box::new(Curvature { reach: 1 })),
        ]
    };
    let masked_arms = || {
        vec![
            Chain::Sequence(vec![
                Chain::Op(Box::new(Curvature { reach: 3 })),
                Chain::Op(Box::new(Threshold { at: 0.0 })),
            ]),
            Chain::Sequence(vec![
                Chain::Op(Box::new(Curvature { reach: 1 })),
                Chain::Op(Box::new(Threshold { at: 0.0 })),
            ]),
        ]
    };

    let cases: Vec<(&str, Chain)> = vec![
        (
            "difference",
            Chain::Parallel {
                branches: arms(),
                combine: Box::new(DifferenceCombine::new("difference")),
            },
        ),
        (
            "arithmetic subtract",
            Chain::Parallel {
                branches: arms(),
                combine: Box::new(ArithmeticCombine::new("subtract", Arithmetic::Subtract)),
            },
        ),
        (
            "arithmetic maximum",
            Chain::Parallel {
                branches: arms(),
                combine: Box::new(ArithmeticCombine::new("maximum", Arithmetic::Maximum)),
            },
        ),
        (
            "logic and",
            Chain::Parallel {
                branches: masked_arms(),
                combine: Box::new(LogicCombine::new("and", Logic::And)),
            },
        ),
        (
            "logic xor",
            Chain::Parallel {
                branches: masked_arms(),
                combine: Box::new(LogicCombine::new("xor", Logic::Xor)),
            },
        ),
    ];

    for (what, chain) in &cases {
        assert!(
            chain.slicing().is_stencil(),
            "{what}: the fan-in must be sliceable for this case to be about the sink at all — a \
             refusal here is the combine's, because both arms are declared"
        );
        assert_identical_at_every_thread_count(chain, shape, what);
        assert_the_fixture_can_see_its_halo(chain, shape, what);
    }
}

/// **The declaration is the combine's own, and the branches do not speak for
/// it.** Without this, the test above would pass on a `Chain::slicing` that had
/// quietly stopped consulting the sink.
#[test]
fn a_declared_sink_is_what_makes_the_fan_in_sliceable() {
    let arms = vec![
        Chain::Op(Box::new(Curvature { reach: 2 })),
        Chain::Op(Box::new(Curvature { reach: 1 })),
    ];
    let undeclared = Chain::Parallel {
        branches: arms,
        combine: Box::new(UndeclaredCombine),
    };
    assert!(
        !undeclared.slicing().is_stencil(),
        "two declared arms under an undeclared sink must still refuse, or the declarations added \
         to the shipped combines are not what is doing the work above"
    );
}

// ------------------------------------------------------------ the wiring --
//
// Everything above this line is about the primitive. **Nothing above it is
// reached by a run**, and for most of this feature's life nothing was: the
// policy, the cut and the declarations all existed and `strategy.rs` never
// called any of them, so setting the policy could not move a voxel or a
// nanosecond. These are the tests that say the executor consults it — and, just
// as importantly, that it does *not* on the plans where the measurement says
// cutting is a loss.

/// A one-phase plan over `volume`, cut into blocks of `block` along axis 0.
fn one_phase(workflow: &Workflow, volume: [usize; 3], block: usize) -> Decomposition {
    let slots = workflow.chain.slots();
    let names: Vec<String> = slots.iter().map(|slot| slot.display_name()).collect();
    let reach = workflow.chain.reach3(&volume);
    let grid = BlockGrid::along(volume, &[0], block).expect("a block grid");
    let phase = PhaseDecomposition::derive((0..slots.len()).collect(), names, reach, reach, grid);
    Decomposition {
        volume,
        dtype: workflow.dtype,
        phases: vec![phase],
        chain_reach: reach,
    }
}

/// Run a plan under a stated worker count and slab policy, and answer what it
/// produced beside what it cost.
fn run_plan(
    workflow: &Workflow,
    decomposition: &Decomposition,
    input: &Voxels,
    workers: usize,
    policy: SlabPolicy,
) -> (Voxels, Stats) {
    let env = ArrayEnvironment::new(input.clone(), decomposition.n_phases(), [4, 4, 4])
        .expect("an environment");
    let hints = Hints {
        concurrency: workers,
        slab_policy: policy,
        ..Hints::default()
    };
    let stats = execute("intra-block", workflow, decomposition, &hints, &env).expect("a run");
    (env.output(), stats)
}

/// **The wiring itself: a plan that leaves workers parked is cut, and the answer
/// does not move.**
///
/// One block and four workers is the regime the measurement found the whole case
/// for the feature in — `docs/design/intra-block.md` §7's one-block row, 4.6-5.4x
/// with thirty-nine workers otherwise idle. The rule is
/// `floor(workers / n_blocks)` capped at [`SlabPolicy::CAP`], so this asks for
/// four slabs and the assertion below is that it got exactly four rather than
/// "some".
#[test]
fn the_executor_cuts_a_block_when_the_lattice_leaves_workers_parked() {
    let volume = [24usize, 8, 6];
    let workflow = Workflow::new(
        Chain::Op(Box::new(Curvature { reach: 2 })),
        volume,
        Dtype::F64,
    );
    // One block: `block >= volume[0]`, so the phase has a single task and three
    // of the four workers have nothing else to do.
    let plan = one_phase(&workflow, volume, volume[0]);
    assert_eq!(
        plan.phases[0].blocks.len(),
        1,
        "the fixture must be one block"
    );
    let input = structured(volume);

    let (cut, cut_stats) = run_plan(&workflow, &plan, &input, 4, SlabPolicy::FillIdleWorkers);
    let (whole, whole_stats) = run_plan(&workflow, &plan, &input, 4, SlabPolicy::Off);

    // **The acceptance bar, through the executor rather than through the
    // primitive.** Byte-identical, not close.
    assert_eq!(
        cut, whole,
        "cutting the block inside the executor changed the run's answer"
    );

    // **And it really was cut**, which is the liveness control without which the
    // equality above would be two identical uncut runs agreeing with each other.
    assert_eq!(
        cut_stats.blocks_sliced, 1,
        "the one block of this plan was not cut, so the equality above is measuring nothing"
    );
    assert_eq!(
        cut_stats.slabs_run,
        SlabPolicy::CAP as u64,
        "four workers over one block is `floor(4 / 1)` capped at four"
    );
    assert_eq!(whole_stats.blocks_sliced, 0, "`Off` must not cut");
    assert_eq!(whole_stats.slabs_run, 1, "`Off` runs the block whole");

    // The plan is the same plan and the storage traffic is the same traffic: a
    // slab cut is arithmetic inside a block the executor already read.
    assert_eq!(
        cut_stats.decomposition_fingerprint,
        whole_stats.decomposition_fingerprint
    );
    assert_eq!(cut_stats.tasks, whole_stats.tasks);
    assert_eq!(cut_stats.reads, whole_stats.reads);
    assert_eq!(cut_stats.read_voxels, whole_stats.read_voxels);
    assert_eq!(cut_stats.write_voxels, whole_stats.write_voxels);
}

/// **The negative that matters: a well-cut plan is untouched.**
///
/// §6 of the design note measured this regime directly — at a fixed thread
/// budget, sixteen blocks on one thread each beat one block on sixteen threads,
/// because a block's read is a serial prefix and sixteen blocks read on sixteen
/// threads. So `floor(workers / n_blocks)` is `1` whenever `n_blocks >=
/// workers`, and this asserts that the run is then **the same run**: the same
/// plan, the same reads, the same answer, and no cut anywhere.
#[test]
fn a_well_cut_plan_is_untouched_by_the_policy() {
    let volume = [24usize, 8, 6];
    let workflow = Workflow::new(
        Chain::Op(Box::new(Curvature { reach: 2 })),
        volume,
        Dtype::F64,
    );
    // Six blocks against four workers: `floor(4 / 6)` is zero, clamped to one.
    let plan = one_phase(&workflow, volume, 4);
    let blocks = plan.phases[0].blocks.len();
    assert!(
        blocks >= 4,
        "the fixture must have at least as many blocks as workers; it has {blocks}"
    );
    let input = structured(volume);

    let (on, on_stats) = run_plan(&workflow, &plan, &input, 4, SlabPolicy::FillIdleWorkers);
    let (off, off_stats) = run_plan(&workflow, &plan, &input, 4, SlabPolicy::Off);

    assert_eq!(
        on, off,
        "a plan with work for every worker must answer the same"
    );
    assert_eq!(
        on_stats.blocks_sliced, 0,
        "the policy cut a block on a plan that already had work for every worker"
    );
    assert_eq!(
        on_stats.slabs_run, off_stats.slabs_run,
        "one slab per application either way"
    );
    assert_eq!(on_stats.slabs_run, blocks as u64);
    assert_eq!(on_stats.reads, off_stats.reads);
    assert_eq!(on_stats.read_voxels, off_stats.read_voxels);
    assert_eq!(on_stats.ops_applied, off_stats.ops_applied);
    assert_eq!(on_stats.estimated_work, off_stats.estimated_work);
    assert!(on_stats.same_work_as(&off_stats));

    // **And the rule, checked where the executor reads it rather than only in
    // `decomposition.rs`'s own unit test.** These two lines are what the
    // paragraph above is about; if they ever disagree the assertions above stop
    // being about the regime they name.
    assert_eq!(SlabPolicy::FillIdleWorkers.slabs_for(4, blocks), 1);
    assert_eq!(SlabPolicy::FillIdleWorkers.slabs_for(4, 1), 4);
}

/// **An undeclared chain declines the offer and runs uncut**, which is every
/// chain this crate shipped before the declarations existed.
///
/// The planner's slab count is derived from the worker count and the block count
/// and from nothing else — it has not looked at the chain. So the offer meets
/// chains that cannot take it constantly, and an offer that *failed* on those
/// would fail every plan in this crate. This is the fallback, seen working, and
/// it is asserted against the answer as well as against the counter: a fallback
/// that ran nothing would also report no cut.
#[test]
fn an_undeclared_chain_declines_the_planners_offer_and_runs_uncut() {
    let volume = [24usize, 8, 6];
    let workflow = Workflow::new(Chain::Op(Box::new(Undeclared)), volume, Dtype::F64);
    let plan = one_phase(&workflow, volume, volume[0]);
    let input = structured(volume);

    let (offered, offered_stats) =
        run_plan(&workflow, &plan, &input, 4, SlabPolicy::FillIdleWorkers);
    let (refused, refused_stats) = run_plan(&workflow, &plan, &input, 4, SlabPolicy::Off);

    assert_eq!(
        offered.view::<f64>().expect("f64"),
        refused.view::<f64>().expect("f64"),
        "declining the cut must not change the answer"
    );
    assert_eq!(
        offered_stats.blocks_sliced, 0,
        "an undeclared chain was cut, which is the one thing the default exists to prevent"
    );
    assert_eq!(offered_stats.slabs_run, 1);
    assert_eq!(
        offered_stats.slabs_run, refused_stats.slabs_run,
        "an offer that was declined must cost exactly what switching the policy off costs"
    );

    // The run answered something, so the equality above is not two empty
    // volumes agreeing.
    assert!(
        offered
            .view::<f64>()
            .expect("f64")
            .iter()
            .any(|value| *value != 0.0),
        "the run produced zero everywhere"
    );
}

// ----------------------------------------------------------- and does it pay --

/// One run, timed, with the CPU it spent beside the wall it took.
///
/// Nothing here allocates or formats inside the timed region: `execute_accounted`
/// takes both of its readings at the run boundary and the formatting happens in
/// the caller, between arms. `crate::cpu`'s header records what this project paid
/// the last time an instrument allocated inside the thing it was measuring.
#[cfg(test)]
fn timed(
    workflow: &Workflow,
    decomposition: &Decomposition,
    input: &Voxels,
    workers: usize,
    policy: SlabPolicy,
) -> (f64, f64, Voxels, Stats) {
    let env = ArrayEnvironment::new(input.clone(), decomposition.n_phases(), [4, 4, 4])
        .expect("an environment");
    let hints = Hints {
        concurrency: workers,
        slab_policy: policy,
        ..Hints::default()
    };
    let started = std::time::Instant::now();
    let (stats, ledger) = blockflow::strategy::execute_accounted(
        "intra-block",
        workflow,
        decomposition,
        &hints,
        &env,
    )
    .expect("a run");
    let wall = started.elapsed().as_secs_f64();
    let cores = ledger.mean_cores_busy().unwrap_or(0.0);
    (wall, cores, env.output(), stats)
}

/// **Does the wiring pay, and how many cores did it keep busy buying it.**
///
/// Ignored because it is a measurement and this box is shared and oversubscribed;
/// run it with
///
/// ```text
/// cargo test --release --test intra_block_slicing -- --ignored --nocapture \
///     --exact the_cut_pays_on_a_one_block_plan
/// ```
///
/// **Ratios are claimed and absolutes are not**, so the two arms are
/// *interleaved* round by round rather than run one after the other: two runs of
/// one configuration on this machine have differed by 1.5x while an interleaved
/// ratio moved by 5%. The reported figure is the **median** of the per-round
/// ratios, which is what survives one round landing next to somebody else's
/// build.
///
/// `mean_cores_busy` is printed beside every wall time and it is the column to
/// read. CPU-seconds against wall-seconds is what says whether the extra threads
/// worked or waited; wall time cannot, and a speedup with no rise in cores busy
/// is a scheduler artefact rather than a result.
///
/// `INTRA_SLAB_EDGE`, `INTRA_SLAB_WORKERS` and `INTRA_SLAB_ROUNDS` set the
/// geometry. The cut is capped at [`SlabPolicy::CAP`], so asking for more
/// workers than that on a one-block plan changes the *pool* and not the number
/// of slabs — which is itself worth being able to see.
#[test]
#[ignore]
fn the_cut_pays_on_a_one_block_plan() {
    fn setting(name: &str, fallback: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(fallback)
    }
    let edge = setting("INTRA_SLAB_EDGE", 128);
    let workers = setting("INTRA_SLAB_WORKERS", 4);
    let rounds = setting("INTRA_SLAB_ROUNDS", 5);

    // The window radius is a setting because it is what sets the **halo**, and
    // the halo is what ends the scaling: the cut's amplification is printed
    // below so the ratio can be read against the redundant arithmetic that
    // bought it rather than on its own.
    let radius = setting("INTRA_SLAB_RADIUS", 1);
    let element = blockflow::ops::element::StructuringElement::from_radius(
        blockflow::ops::element::ElementShape::Box,
        [radius, radius, radius],
    );
    let volume = [edge, edge, edge];
    // Two chains, because the one the consumer spends its time in is a **fan-in
    // over a sequence** and not a single filter: its reach is the opening's two
    // passes, so its slab halo is twice the filter's, and it is the composite
    // that says whether the declarations bought anything where they were needed.
    let named = std::env::var("INTRA_SLAB_CHAIN").unwrap_or_else(|_| "median".to_string());
    let chain = match named.as_str() {
        "diamond" => blockflow::ops::background::remove_background(&element).expect("a diamond"),
        // **The case the cut lost on**, and it is here to be measured rather than
        // assumed away. A reach-0 chain has an amplification of exactly `1.0` —
        // there is no redundant arithmetic at all — so anything it loses is
        // copies, and a voxelwise map is bound by that same memory and not by
        // what it computes (`ops::voxelwise::IDENTITY_COST` is 0.95 against a
        // threshold's 1.00). It is the arm that measured below one while the
        // cores were placed by one serial pass, and it is the arm that says
        // whether threading that pass was worth building.
        "voxelwise" => Chain::Sequence(vec![
            Chain::op(blockflow::ops::VoxelwiseMapOp::new("scale", |value| {
                value * 2.0
            })),
            Chain::op(blockflow::ops::VoxelwiseMapOp::new("floor", |value| {
                value.max(0.0)
            })),
            Chain::op(blockflow::ops::VoxelwiseMapOp::new("shift", |value| {
                value + 1.0
            })),
        ]),
        "one map" => Chain::op(blockflow::ops::VoxelwiseMapOp::new("scale", |value| {
            value * 2.0
        })),
        _ => Chain::Op(Box::new(blockflow::ops::rank::RankFilterOp::new(
            "median",
            element.clone(),
            blockflow::ops::Rank::median(&element),
        ))),
    };
    let what = named.as_str();
    let workflow = Workflow::new(chain, volume, Dtype::F64);
    // One block: the regime the whole feature is for. Thirty-nine parked
    // workers is what §7 of the design note measured 4.6-5.4x in.
    let plan = one_phase(&workflow, volume, edge);
    assert_eq!(plan.phases[0].blocks.len(), 1);
    let input = structured(volume);

    let slabs = SlabPolicy::FillIdleWorkers.slabs_for(workers, 1);
    let amplification = SlabCut::plan_longest(
        volume,
        slabs,
        &workflow.chain.reach_spec(volume).expect("a reach"),
        volume,
    )
    .expect("a cut")
    .amplification();
    println!(
        "{what} (cost/voxel {:.2}): volume {volume:?} f64 ({:.1} MB), radius {radius}, one block, {workers} workers, {slabs} \
         slabs (cap {}), computed/written {amplification:.3}, {rounds} interleaved rounds",
        workflow.chain.cost_per_voxel(),
        (edge * edge * edge * 8) as f64 / 1e6,
        SlabPolicy::CAP
    );
    // **The two ways to place the slabs' cores, measured against each other.**
    //
    // `run_cut` used to end with one serial pass over the written volume; it now
    // splits the output into one disjoint `VoxelsMut` per slab and lets each
    // thread place its own. The difference is Amdahl's term and nothing else, so
    // it is measured on its own rather than inferred from the arms below — the
    // slab answers are made **before** either clock starts, because in the real
    // thing they are produced inside the slab's own thread either way.
    //
    // Interleaved round by round for the same reason every other figure here is:
    // an absolute on this box moves by more than the effect.
    let (serial_seconds, threaded_seconds) = {
        let block = Voxels::zeros(Dtype::F64, volume).expect("a block");
        let cut = SlabCut::plan_longest(
            volume,
            slabs,
            &workflow.chain.reach_spec(volume).expect("a reach"),
            volume,
        )
        .expect("a cut");
        let axis = cut.axis();
        let answers: Vec<Voxels> = cut
            .slabs()
            .iter()
            .map(|slab| block.slice_region(&slab.extent).expect("a slab"))
            .collect();
        let mut serial = Vec::new();
        let mut threaded = Vec::new();
        let mut agreed = false;
        for _ in 0..rounds {
            let mut one = Voxels::zeros(Dtype::F64, volume).expect("a sink");
            let started = std::time::Instant::now();
            for (slab, answer) in cut.slabs().iter().zip(&answers) {
                one.assign_region_from(&slab.core, answer, &slab.core_within_extent)
                    .expect("a placement");
            }
            serial.push(started.elapsed().as_secs_f64());

            let mut two = Voxels::zeros(Dtype::F64, volume).expect("a sink");
            let started = std::time::Instant::now();
            {
                let mut cores: Vec<blockflow::voxels::VoxelsMut<'_>> = Vec::new();
                let mut rest = blockflow::voxels::VoxelsMut::of(&mut two);
                let mut peeled = 0usize;
                for slab in cut.slabs() {
                    let end = slab.core.start[axis] + slab.core.shape[axis];
                    let (mine, tail) = rest.split_at(axis, end - peeled).expect("a split");
                    cores.push(mine);
                    rest = tail;
                    peeled = end;
                }
                std::thread::scope(|scope| {
                    for ((slab, answer), mut core) in cut.slabs().iter().zip(&answers).zip(cores) {
                        scope.spawn(move || {
                            core.assign_from(answer, &slab.core_within_extent)
                                .expect("a placement")
                        });
                    }
                });
            }
            threaded.push(started.elapsed().as_secs_f64());
            // **The arms must agree**, or the faster one is faster for the wrong
            // reason. Checked once rather than every round, because it is a
            // whole-volume comparison and would otherwise be in the measurement.
            if !agreed {
                assert_eq!(one, two, "the two placements wrote different volumes");
                agreed = true;
            }
        }
        serial.sort_by(f64::total_cmp);
        threaded.sort_by(f64::total_cmp);
        (serial[serial.len() / 2], threaded[threaded.len() / 2])
    };
    println!(
        "placing the cores: serial {serial_seconds:.4} s | threaded {threaded_seconds:.4} s | \
         {:.2}x, medians of {rounds} interleaved rounds",
        serial_seconds / threaded_seconds
    );

    let mut ratios: Vec<f64> = Vec::with_capacity(rounds);
    let mut reference: Option<Voxels> = None;
    for round in 0..rounds {
        let (off_wall, off_cores, off_answer, off_stats) =
            timed(&workflow, &plan, &input, workers, SlabPolicy::Off);
        let (on_wall, on_cores, on_answer, on_stats) = timed(
            &workflow,
            &plan,
            &input,
            workers,
            SlabPolicy::FillIdleWorkers,
        );
        // **The measurement is void without this.** A slab that computed a
        // cheaper wrong answer would show a speedup.
        assert_eq!(
            on_answer, off_answer,
            "round {round}: the cut changed the answer"
        );
        assert_eq!(off_stats.blocks_sliced, 0);
        // **Every application was cut, and a fused phase has one per slot.**
        // `run_task` applies the phase's slots one at a time and offers the cut
        // to each, so a three-slot phase pays three cuts and three join passes
        // rather than one — which is worth seeing rather than netting away.
        assert_eq!(on_stats.blocks_sliced, on_stats.ops_applied as u64);
        assert!(on_stats.blocks_sliced >= 1);
        let ratio = off_wall / on_wall;
        ratios.push(ratio);
        println!(
            "round {round}: off {off_wall:.4} s ({off_cores:.2} cores busy) | on {on_wall:.4} s \
             ({on_cores:.2} cores busy, {} slabs) | {ratio:.3}x",
            on_stats.slabs_run
        );
        if reference.is_none() {
            reference = Some(off_answer);
        }
    }
    ratios.sort_by(f64::total_cmp);
    println!(
        "median of {rounds} interleaved ratios: {:.3}x",
        ratios[ratios.len() / 2]
    );
    println!(
        "load average now: {}",
        std::fs::read_to_string("/proc/loadavg")
            .unwrap_or_default()
            .trim()
    );
}

/// **The composite the declarations were for**, held to the same bar as its
/// parts.
///
/// `ops::background::remove_background` is a diamond: an identity map on one
/// arm, a grey opening — two rank filters — on the other, joined by a
/// difference. **Every one of its parts had to be declared before this chain
/// could be cut** — the two rank filters, the map, and the sink — and until the
/// last of them was, the node refused however many of the others were stencils. That is the shape of the blocker: *the declaration sits on the
/// op, the refusal sits on the chain around it.*
///
/// A composite rather than a fifth single op on purpose. Each part is already
/// held to bit-identity on its own; what this adds is the fold — a fan-in over a
/// sequence, whose reach is the opening's two passes and whose slab halo is
/// therefore the sum rather than one filter's.
#[test]
fn the_background_removal_diamond_survives_the_cut_bit_for_bit() {
    let element = blockflow::ops::element::StructuringElement::from_radius(
        blockflow::ops::element::ElementShape::Box,
        [1, 1, 1],
    );
    let chain =
        blockflow::ops::background::remove_background(&element).expect("a background diamond");
    assert!(
        chain.slicing().is_stencil(),
        "the shipped background diamond still refuses: {:?}",
        chain.slicing().refusal()
    );
    // The opening is two passes, so the fan-in's reach is twice the element's
    // and the slab halo is that. Asserted because a cut that used one pass's
    // reach would read short at every seam.
    assert_eq!(
        chain.reach(0, 64),
        2,
        "an opening is two passes of the element"
    );
    let shape = [24usize, 7, 6];
    assert_identical_at_every_thread_count(&chain, shape, "background removal");
    assert_the_fixture_can_see_its_halo(&chain, shape, "background removal");
}

/// **The one line that carries the caller's knob into the run.**
///
/// `Constraints::slab_policy` is what a caller sets; `Hints::slab_policy` is
/// what the executor reads; `Strategy::plan` is the single line between them,
/// because it is the only method holding both. One line with nothing asserting
/// it is how this feature came to be built, tested and connected to nothing —
/// so it is asserted, over [`SlabPolicy::ALL`] rather than over one variant, so
/// that a policy added later cannot quietly stop travelling.
#[test]
fn the_caller_s_policy_reaches_the_hints_through_plan() {
    use blockflow::decomposition::Constraints;
    use blockflow::strategy::{Strategy, Trivial};

    let workflow = Workflow::new(
        Chain::Op(Box::new(Curvature { reach: 1 })),
        [16, 8, 8],
        Dtype::F64,
    );
    for policy in SlabPolicy::ALL {
        let plan = Trivial
            .plan(&workflow, &Constraints::default().with_slab_policy(policy))
            .expect("a plan");
        assert_eq!(
            plan.hints.slab_policy, policy,
            "the policy stated on the constraints did not reach the hints the executor reads"
        );
    }
    // The sweep is only a sweep if it covers a policy that changes behaviour and
    // one that does not; a list of one identical variant would pass the loop
    // above while proving nothing.
    assert!(SlabPolicy::ALL.contains(&SlabPolicy::Off));
    assert!(SlabPolicy::ALL.contains(&SlabPolicy::FillIdleWorkers));
}

// ------------------------------------------- the distributed entry point --

/// **A fragment phase is never offered a cut, so a hoisted reduction cannot be
/// disturbed by one.**
///
/// This is the safety property behind `execute_task_with_reduction`'s `slabs`
/// argument, asserted rather than restated. A barrier phase's reduction is
/// **derived** from the fragment set rather than transported: every worker folds
/// the same bytes from the same storage with no election and no upload, and the
/// agreement is by construction. Anything that could make two workers fold
/// differently would break a distributed run in a way no single-node test would
/// notice, so a new per-worker thread count is exactly the kind of change that
/// has to be shown not to reach it.
///
/// It cannot: `strategy::run_task` dispatches a `PhaseWork::Fragments` phase
/// before the slab offer exists.
///
/// **The liveness control is the last assertion.** A fragment plan of many
/// blocks would be offered one slab anyway, and this test would then be green
/// for the wrong reason — so the plan is deliberately **one block**, and the
/// offer the executor computes for it is asserted to be greater than one.
#[test]
fn a_fragment_phase_is_never_offered_a_cut() {
    use blockflow::fragment::{fragment_phase, PhaseWork};
    use blockflow::ops::components::Merge;
    use blockflow::ops::fill::{FillHolesOp, LabelBackgroundOp};
    use blockflow::sidecar::Lifecycle;
    use blockflow::strategy::execute_phases;
    use ndarray::Array3;

    const STREAM: &str = "slab.fill.faces";
    let volume = [12usize, 16, 16];

    // A mask with a sealed cavity, so the phases have something to answer.
    let mut mask = Array3::from_elem((volume[0], volume[1], volume[2]), false);
    for i in 1..volume[0] - 1 {
        for j in 2..volume[1] - 2 {
            for k in 2..volume[2] - 2 {
                let inner = (3..volume[0] - 3).contains(&i)
                    && (4..volume[1] - 4).contains(&j)
                    && (4..volume[2] - 4).contains(&k);
                mask[[i, j, k]] = !inner;
            }
        }
    }

    let run = |policy: SlabPolicy| {
        // **One block**, which is what makes the offer non-trivial.
        let grid = BlockGrid::new(volume, volume).expect("a lattice");
        assert_eq!(grid.n_blocks(), 1);
        let label = LabelBackgroundOp::new("label", STREAM, Lifecycle::DeleteOnExit);
        // `OnceForThePhase` is the hoisted merge — the placement that computes a
        // reduction for the phase, which is the thing that must not move.
        let fill =
            FillHolesOp::new("fill", STREAM, 0, Dtype::Bool, &grid).merging(Merge::OnceForThePhase);
        let mut labelling = fragment_phase(&label, grid.clone()).expect("phase 0");
        labelling.dtype = Some(Dtype::U32);
        let mut filling = fragment_phase(&fill, grid).expect("phase 1");
        filling.dtype = Some(Dtype::Bool);
        let plan = Decomposition {
            volume,
            dtype: Dtype::Bool,
            phases: vec![labelling, filling],
            chain_reach: [0, 0, 0],
        };
        plan.check().expect("the plan tiles");
        let env = ArrayEnvironment::for_decomposition(Voxels::from(mask.clone()), &plan, volume)
            .expect("an environment");
        let workflow = Workflow::new(Chain::sequence(Vec::new()), volume, Dtype::Bool);
        let hints = Hints {
            concurrency: 4,
            slab_policy: policy,
            ..Hints::default()
        };
        let stats = execute_phases(
            "slab-fragment",
            &workflow,
            &plan,
            &hints,
            &env,
            &[],
            &[PhaseWork::Fragments(&label), PhaseWork::Fragments(&fill)],
        )
        .expect("a run");
        (
            env.output().view::<bool>().expect("a mask").to_owned(),
            stats,
        )
    };

    let (offered, offered_stats) = run(SlabPolicy::FillIdleWorkers);
    let (refused, refused_stats) = run(SlabPolicy::Off);

    assert_eq!(
        offered, refused,
        "offering a fragment phase a cut changed what it filled"
    );
    assert!(
        offered_stats.fragment_applications > 0,
        "no fragment op was applied, so this test is about nothing"
    );
    assert_eq!(
        offered_stats.fragment_applications, refused_stats.fragment_applications,
        "the fragment work must be identical under either policy"
    );
    assert_eq!(
        offered_stats.blocks_sliced, 0,
        "a fragment phase was cut; a hoisted reduction is derived rather than transported, so \
         two workers folding differently is a distributed wrong answer with no diagnostic"
    );
    assert_eq!(
        offered_stats.slabs_run, 0,
        "a fragment phase applies no chain slot, so it runs no slab"
    );
    assert_eq!(offered_stats.sidecar_writes, refused_stats.sidecar_writes);
    assert_eq!(offered_stats.sidecar_reads, refused_stats.sidecar_reads);

    // **The liveness control.** With one block and four workers the policy asks
    // for four slabs; if it asked for one, everything above would hold for a
    // reason that has nothing to do with fragment phases.
    assert_eq!(
        SlabPolicy::FillIdleWorkers.slabs_for(4, 1),
        SlabPolicy::CAP,
        "the offer this run was made must be greater than one, or nothing above is a control"
    );
}

/// **`execute_task_with_reduction` takes the offer, and `1` is the run it was.**
///
/// The entry point a distributed worker uses, driven directly: the same task,
/// the same environment, run once uncut and once at the cap, must produce the
/// same bytes and must differ only in that one of them cut. It is the same bar
/// as the executor's, applied to the entry point that has no executor around it.
#[test]
fn the_task_entry_point_takes_the_offer_and_one_is_the_run_it_was() {
    use blockflow::env::Environment;
    use blockflow::fragment::PhaseWork;
    use blockflow::graph::TaskGraph;
    use blockflow::strategy::execute_task_with_reduction;

    let volume = [24usize, 8, 6];
    let chain = Chain::Op(Box::new(Curvature { reach: 2 }));
    let workflow = Workflow::new(
        Chain::Op(Box::new(Curvature { reach: 2 })),
        volume,
        Dtype::F64,
    );
    let plan = one_phase(&workflow, volume, volume[0]);
    let graph = TaskGraph::build(&plan);
    assert_eq!(graph.tasks.len(), 1, "one block, one task");
    let input = structured(volume);

    let run = |slabs: usize| {
        let env = ArrayEnvironment::new(input.clone(), plan.n_phases(), [4, 4, 4])
            .expect("an environment");
        execute_task_with_reduction(
            &chain,
            &plan,
            &graph.tasks[0],
            &PhaseWork::Pixels,
            &[],
            &env,
            &[],
            slabs,
        )
        .expect("a task");
        let counters = env.counters();
        (
            env.output(),
            counters
                .blocks_sliced
                .load(std::sync::atomic::Ordering::SeqCst),
            counters.slabs_run.load(std::sync::atomic::Ordering::SeqCst),
        )
    };

    let (uncut_answer, uncut_sliced, uncut_slabs) = run(1);
    let (cut_answer, cut_sliced, cut_slabs) = run(SlabPolicy::CAP);

    assert_eq!(
        cut_answer, uncut_answer,
        "the entry point's answer moved when the block was cut"
    );
    assert_eq!(uncut_sliced, 0, "`1` must take the uncut path outright");
    assert_eq!(uncut_slabs, 1);
    assert_eq!(
        cut_sliced, 1,
        "the offer was not taken, so the equality above is measuring nothing"
    );
    assert_eq!(cut_slabs, SlabPolicy::CAP as u64);

    // The answer is not zero everywhere, so byte-identity above is not two empty
    // volumes agreeing.
    let answered: &ndarray::Array3<f64> = &uncut_answer.view::<f64>().expect("f64").to_owned();
    assert!(answered.iter().any(|value| *value != 0.0));
}

/// **A worker built today spends one thread inside a block**, which is exactly
/// what it spent before the option existed.
///
/// The one assertion that stands between this feature and every existing
/// deployment quietly changing how many cores it uses per node. It is a default
/// rather than `available_parallelism` on purpose: how many worker processes
/// share a box is a fact about the deployment, and a worker that guessed would
/// oversubscribe exactly the machines that packed the most work onto a node.
#[test]
fn a_worker_spends_one_thread_per_block_unless_it_is_told_otherwise() {
    use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};

    let options = blockflow::distributed::worker::WorkerOptions::new(SocketAddr::V4(
        SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0),
    ));
    assert_eq!(
        options.threads, 1,
        "a worker's default thread count is what every recorded distributed run was taken at"
    );
    // And what the default turns into is the uncut path, not merely a small cut.
    assert_eq!(SlabPolicy::FillIdleWorkers.slabs_for(options.threads, 1), 1);
    // Above one it is the cap that bounds it, not the core count: a node with
    // forty cores does not get forty slabs of one block.
    assert_eq!(
        SlabPolicy::FillIdleWorkers.slabs_for(40, 1),
        SlabPolicy::CAP
    );
}

/// **The two voxelwise shells, and the one that writes a narrower answer.**
///
/// `VoxelwiseMapOp` writes `f64` and `VoxelwiseMaskOp` writes `Bool` — the same
/// shape of op through a bar that, until this file's helpers were generalised,
/// could only read one of them. Declaring the mask op without a case beside it
/// would have been the thing the bar exists to prevent, and relaxing the bar to
/// admit it would have been worse; so the helpers now dispatch per element type
/// and refuse the rest by name.
///
/// Both cases put a **reaching** op in front of the voxelwise one, because a
/// reach-0 chain cut into slabs has no halo at all and its bit-identity would
/// hold whatever the halo arithmetic did. What is being tested is the shell in a
/// chain that has a seam, which is the only place it can be wrong.
#[test]
fn the_voxelwise_shells_survive_the_cut_bit_for_bit() {
    use blockflow::ops::{VoxelwiseMapOp, VoxelwiseMaskOp};

    let shape = [24usize, 7, 6];
    let cases: Vec<(&str, Chain)> = vec![
        (
            "map after a stencil",
            Chain::Sequence(vec![
                Chain::Op(Box::new(Curvature { reach: 2 })),
                Chain::op(VoxelwiseMapOp::new("scale", |value| value * 1.5 + 0.25)),
            ]),
        ),
        (
            "mask after a stencil",
            Chain::Sequence(vec![
                Chain::Op(Box::new(Curvature { reach: 2 })),
                Chain::op(VoxelwiseMaskOp::threshold("binarize", 0.0)),
            ]),
        ),
    ];
    for (what, chain) in &cases {
        assert!(
            chain.slicing().is_stencil(),
            "{what}: the chain refuses, so this case is about nothing"
        );
        assert_identical_at_every_thread_count(chain, shape, what);
        assert_the_fixture_can_see_its_halo(chain, shape, what);
    }

    // **The bar really did read the narrower answer.** Without this the mask
    // case could be passing through an `f64` arm by accident — which is exactly
    // the shape of the gap that kept this op undeclared.
    let mask = &cases[1].1;
    assert_eq!(
        mask.produces(Dtype::F64).expect("a width"),
        Dtype::Bool,
        "the mask case must actually be checked as bool, or generalising the helpers bought \
         nothing"
    );
}

/// **The local runner starts the workers it always started.**
///
/// `LocalOptions::threads` is what reaches a spawned worker's `--threads`, so it
/// is the second place a default could quietly change how many cores every
/// multi-node run uses — and the one a reader is less likely to check, because
/// it is a harness rather than a deployment.
#[test]
#[cfg(feature = "distributed")]
fn the_local_runner_starts_single_threaded_workers_unless_it_is_told_otherwise() {
    // `Binaries::beside_this_one` fails where the binaries are not built beside
    // the test, which is not what this is about; an absence is skipped rather
    // than failed, and the assertion is the whole content either way.
    let Ok(options) = blockflow::distributed::local::LocalOptions::new(
        std::env::temp_dir().join("blockflow-slab-default-probe"),
        3,
    ) else {
        return;
    };
    assert_eq!(
        options.threads, 1,
        "the local runner's default thread count is what every recorded multi-node run was taken \
         at"
    );
}
