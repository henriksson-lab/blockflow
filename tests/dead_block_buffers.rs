// SPDX-License-Identifier: MIT
//
// **The block buffers `Chain::apply_placed` held and did not read, and the proof
// that removing them moved no value.**
//
// Two of them, in two nodes, and they are the same defect: a buffer that has
// been written and is not being read by anything is bytes and nothing else, so
// giving it up costs no locality and no cache. Everything here is the identity
// check that each removal is only a residency change.
//
// 1. **A fan-in's collected branches.** `Chain::Parallel` held one block buffer
//    per branch until the combine had read them all; from the moment a branch
//    finished until the combine ran, its buffer was dead weight. A combine that
//    declares itself a left fold over pairs (`Combine::fold_carrier`) is now
//    folded as its branches are computed.
// 2. **A sequence's clone of its input.** `Chain::Sequence` began with
//    `input.clone()`, so a chain of `n` children held its input twice over
//    before computing anything — and every read of the copy could have gone to
//    the original, which the caller owns for the whole call. **It bound the
//    peak only at two children**; from three on, the two intermediates in
//    flight are the tall term and the clone was already freed. The test below
//    measures both cases rather than letting the headline stand for the shape
//    it does not describe.
//
// The fan-in half is the larger claim and takes most of this file, because it is
// the one that could have moved a value. At one block a block buffer is a whole
// volume, so a seven-arm join held seven whole volumes and six of them were idle;
// the fold holds three whatever the arity. That is a memory claim, and it is only
// allowed if it is not also a numeric claim.
//
// **Why byte equality and not a tolerance.** `f64` addition is not associative,
// so a fold that visited the branches in a different order would be a different
// volume — and a tolerance is exactly the instrument that would not notice. The
// comparison here is `differing_bits`, and there is a fixture below whose answer
// moves under a reordering so that the comparison is known to be able to see
// one.
//
// **The liveness control is a second combine.** `Collected` computes the same
// left fold as `LogicCombine`/`ArithmeticCombine` and declares no fold carrier,
// so the walk collects it. Same bits, different bytes: every equality here is
// checked against it, and the residency assertions are what say the two paths
// are actually different paths rather than one path measured twice.

use blockflow::assemble::ImageId;
use blockflow::op::{Anchor, Chain, Combine, SourceInputs};
use blockflow::ops::background::DifferenceCombine;
use blockflow::ops::{Arithmetic, ArithmeticCombine, Logic, LogicCombine, VoxelwiseMapOp};
use blockflow::voxels::{differing_bits, differing_elements, Voxels};
use blockflow::{Dtype, Error, Result};

use ndarray::Array3;

const BLOCK: [usize; 3] = [12, 9, 7];

fn anchor() -> Anchor {
    Anchor::new([4, 2, 1], [BLOCK[0] + 20, BLOCK[1] + 20, BLOCK[2] + 20])
}

/// A ramp with a value at every voxel, so a branch that ignored its input and a
/// branch that read it give different answers.
fn input() -> Voxels {
    let mut array = Array3::<f64>::zeros((BLOCK[0], BLOCK[1], BLOCK[2]));
    for (index, value) in array.iter_mut().enumerate() {
        *value = (index % 17) as f64 - 5.0;
    }
    Voxels::F64(array)
}

/// Branch `i`, producing `value * scale + offset` — distinct per branch, so a
/// fold that dropped one or took them out of order is visible in the answer.
fn arm(scale: f64, offset: f64) -> Chain {
    Chain::op(VoxelwiseMapOp::new("arm", move |value: f64| {
        value * scale + offset
    }))
}

/// **The magnitudes that make a sum order-sensitive.** At `1e16` the gap
/// between representable `f64`s is `2.0`, so `1e16 + 1.0` rounds back to `1e16`
/// and a left fold starting from the large value absorbs every `1.0` after it,
/// while the reverse order accumulates them first and keeps the total. Used by
/// the reordering control below.
fn absorbing_offsets(arity: usize) -> Vec<f64> {
    let mut offsets = vec![1.0; arity];
    offsets[0] = 1e16;
    offsets
}

/// Branches that read the input, so nothing here is a constant the walk could
/// have short-circuited.
fn arms(offsets: &[f64]) -> Vec<Chain> {
    offsets
        .iter()
        .enumerate()
        .map(|(index, &offset)| arm(1.0 + index as f64, offset))
        .collect()
}

/// One block through the chain, which is the path under test.
fn walked(chain: &Chain) -> Voxels {
    walked_with(chain, &[])
}

/// [`walked`], with stored images supplied for the chain's source leaves.
fn walked_with(chain: &Chain, stored: &[(ImageId, &Voxels)]) -> Voxels {
    let block = input();
    let mut out = Voxels::zeros(
        chain.produces(block.dtype()).expect("an element type"),
        chain.output_shape(block.shape()).expect("a shape"),
    )
    .expect("an output block");
    chain
        .apply_with(&block, SourceInputs::new(stored), &mut out, &anchor())
        .expect("one block through the chain");
    out
}

/// **The reference: every branch computed into its own buffer, all of them
/// alive, then one `Combine::apply` over the whole list.** This is what the walk
/// did before it learned to fold, written out here so the old path stays
/// available to compare against rather than only in the history.
fn collected(branches: &[Chain], combine: &dyn Combine) -> Voxels {
    collected_with(branches, combine, &[])
}

/// [`collected`], with stored images supplied.
///
/// **This is also the *copying* reference for a source arm**, and that is worth
/// saying twice. Every branch here is run through `apply_with` into a buffer of
/// its own, so a `Chain::Source` branch takes `Chain::apply_tallied`'s copying
/// arm — `Voxels::assign` into that buffer — which is what the walk did for a
/// source arm before it learned to borrow. So the comparison below is the
/// borrowed answer against the copied one, from two executions.
fn collected_with(
    branches: &[Chain],
    combine: &dyn Combine,
    stored: &[(ImageId, &Voxels)],
) -> Voxels {
    let block = input();
    let at = anchor();
    let results: Vec<Voxels> = branches
        .iter()
        .map(|branch| {
            let mut result = Voxels::zeros(
                branch.produces(block.dtype()).expect("an element type"),
                branch.output_shape(block.shape()).expect("a shape"),
            )
            .expect("a branch buffer");
            branch
                .apply_with(&block, SourceInputs::new(stored), &mut result, &at)
                .expect("one branch");
            result
        })
        .collect();
    let dtypes: Vec<Dtype> = results.iter().map(Voxels::dtype).collect();
    let shapes: Vec<[usize; 3]> = results.iter().map(Voxels::shape).collect();
    let mut out = Voxels::zeros(
        combine.produces(&dtypes),
        combine.output_shape(&shapes).expect("a joined shape"),
    )
    .expect("an output block");
    combine
        .apply(&results.iter().collect::<Vec<_>>(), &mut out, &at)
        .expect("the collected join");
    out
}

/// Zero differing voxels, compared as bit patterns for the floats and as values
/// for the one type that has no other pattern.
fn identical(left: &Voxels, right: &Voxels, what: &str) {
    assert_eq!(left.dtype(), right.dtype(), "{what}: element type");
    assert_eq!(left.shape(), right.shape(), "{what}: shape");
    let differing = match left.dtype() {
        Dtype::F64 => differing_bits(
            left.view::<f64>().expect("f64"),
            right.view::<f64>().expect("f64"),
        ),
        Dtype::F32 => differing_bits(
            left.view::<f32>().expect("f32"),
            right.view::<f32>().expect("f32"),
        ),
        Dtype::Bool => differing_elements(
            left.view::<bool>().expect("bool"),
            right.view::<bool>().expect("bool"),
        ),
        other => panic!("{what}: nothing here produces {}", other.numpy_name()),
    }
    .expect("comparable buffers");
    assert_eq!(
        differing, 0,
        "{what}: the folded walk and the collected reference differ in {differing} voxels. \
         Byte equality is the bar for this change: it is a residency change and is allowed to \
         move no value at all."
    );
}

// ------------------------------------------------- the liveness control --

/// **The same left fold, declaring no carrier**, so `Chain::Parallel` collects
/// its branches exactly as it did before `fold_carrier` existed.
///
/// It exists twice over. As a *numeric* control it is the old path, still
/// compiled and still run, so "the two paths agree" is an assertion about two
/// executions rather than about one execution and a memory. As a *residency*
/// control it is the shape that still grows with arity, which is what says the
/// flat figure beside it is the fold working and not the measurement failing to
/// see anything.
struct Collected;

impl Combine for Collected {
    fn name(&self) -> &'static str {
        "collected"
    }
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }
    fn accepts(&self, inputs: &[Dtype]) -> bool {
        inputs.len() >= 2 && inputs.iter().all(|dtype| *dtype == Dtype::F64)
    }
    fn produces(&self, _inputs: &[Dtype]) -> Dtype {
        Dtype::F64
    }
    fn output_shape(&self, inputs: &[[usize; 3]]) -> Result<[usize; 3]> {
        inputs
            .first()
            .copied()
            .ok_or_else(|| Error::InvalidArgument("collected: no branches".to_string()))
    }
    fn apply(&self, inputs: &[&Voxels], out: &mut Voxels, _at: &Anchor) -> Result<()> {
        let mut sink = out.view_mut::<f64>()?;
        for (index, buffer) in inputs.iter().enumerate() {
            let arm = buffer.view::<f64>()?;
            if index == 0 {
                sink.assign(&arm);
            } else {
                sink.zip_mut_with(&arm, |total, value| *total = *total + *value);
            }
        }
        Ok(())
    }
}

/// **Associative and *not* commutative**, which nothing this crate ships is.
///
/// It exists because of a mutant that survived. Swapping `fold_pair`'s two
/// operands is invisible to every combine in the crate — `and`, `or`, `xor`,
/// `add`, `multiply`, `minimum` and `maximum` are all commutative pairwise, so
/// `f(a, b)` and `f(b, a)` are the same volume and no fixture over them can tell
/// a fold that keeps its argument order from one that does not. The *association*
/// order is pinned by the arithmetic above; the *operand* order needs an
/// operation for which the two differ, and this is one: "the first operand
/// unless it is zero" is associative, so a left fold over it is well defined,
/// and it is plainly not commutative.
///
/// So this is not a contrived case admitted for coverage — it is the only
/// instrument that can see one half of what `fold_carrier` promises, and a
/// caller outside this crate may well have such a combine.
struct Leftmost;

impl Leftmost {
    fn join(left: f64, right: f64) -> f64 {
        if left != 0.0 {
            left
        } else {
            right
        }
    }
}

impl Combine for Leftmost {
    fn name(&self) -> &'static str {
        "leftmost"
    }
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }
    fn accepts(&self, inputs: &[Dtype]) -> bool {
        inputs.len() >= 2 && inputs.iter().all(|dtype| *dtype == Dtype::F64)
    }
    fn produces(&self, _inputs: &[Dtype]) -> Dtype {
        Dtype::F64
    }
    fn output_shape(&self, inputs: &[[usize; 3]]) -> Result<[usize; 3]> {
        Ok(inputs[0])
    }
    fn fold_carrier(&self, _inputs: &[Dtype]) -> Option<Dtype> {
        Some(Dtype::F64)
    }
    fn fold_pair(
        &self,
        left: &Voxels,
        right: &Voxels,
        out: &mut Voxels,
        _at: &Anchor,
    ) -> Result<()> {
        let one = left.view::<f64>()?;
        let other = right.view::<f64>()?;
        let mut sink = out.view_mut::<f64>()?;
        for (value, (a, b)) in sink.iter_mut().zip(one.iter().zip(other.iter())) {
            *value = Self::join(*a, *b);
        }
        Ok(())
    }
    fn apply(&self, inputs: &[&Voxels], out: &mut Voxels, _at: &Anchor) -> Result<()> {
        let mut sink = out.view_mut::<f64>()?;
        for (index, buffer) in inputs.iter().enumerate() {
            let arm = buffer.view::<f64>()?;
            if index == 0 {
                sink.assign(&arm);
            } else {
                sink.zip_mut_with(&arm, |folded, value| *folded = Self::join(*folded, *value));
            }
        }
        Ok(())
    }
}

/// **A combine that declares a carrier and does not implement the step.** The
/// default `fold_pair` must refuse by name rather than invent a join.
struct HalfDeclared;

impl Combine for HalfDeclared {
    fn name(&self) -> &'static str {
        "half declared"
    }
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }
    fn accepts(&self, inputs: &[Dtype]) -> bool {
        inputs.len() >= 2
    }
    fn produces(&self, _inputs: &[Dtype]) -> Dtype {
        Dtype::F64
    }
    fn output_shape(&self, inputs: &[[usize; 3]]) -> Result<[usize; 3]> {
        Ok(inputs[0])
    }
    fn fold_carrier(&self, _inputs: &[Dtype]) -> Option<Dtype> {
        Some(Dtype::F64)
    }
    fn apply(&self, inputs: &[&Voxels], out: &mut Voxels, _at: &Anchor) -> Result<()> {
        out.assign(&inputs[0])
    }
}

// ------------------------------------------------------------ the checks --

/// **The connective, at every arity from two to eight.**
///
/// `And`, `Or` and `Xor` over masks carried in `f64`: the folded walk against
/// the collected reference, and the collected reference against a hand-written
/// fold of the same list, so neither side is trusted on its own.
#[test]
fn a_connective_folds_to_the_same_bits_it_collects_to_at_every_arity() {
    for logic in [Logic::And, Logic::Or, Logic::Xor] {
        for arity in 2..=8usize {
            // Offsets that put a mix of set and clear voxels in every branch, so
            // an `And` is not uniformly clear and an `Or` not uniformly set.
            let offsets: Vec<f64> = (0..arity).map(|index| index as f64 - 3.0).collect();
            let branches = arms(&offsets);
            let combine = LogicCombine::new("join", logic);
            let chain = Chain::parallel(arms(&offsets), Box::new(LogicCombine::new("join", logic)))
                .expect("a fan-in");
            identical(
                &walked(&chain),
                &collected(&branches, &combine),
                &format!("{logic:?} at arity {arity}"),
            );
        }
    }
}

/// **The arithmetic, at every arity, for the four operations that are a fold —
/// and the refusal, by name, for the two that are not.**
#[test]
fn the_arithmetic_folds_where_it_says_it_does_and_refuses_where_it_says_it_does_not() {
    for op in [
        Arithmetic::Add,
        Arithmetic::Multiply,
        Arithmetic::Minimum,
        Arithmetic::Maximum,
    ] {
        for arity in 2..=8usize {
            let offsets = absorbing_offsets(arity);
            let branches = arms(&offsets);
            let combine = ArithmeticCombine::new("join", op);
            let chain =
                Chain::parallel(arms(&offsets), Box::new(ArithmeticCombine::new("join", op)))
                    .expect("a fan-in");
            identical(
                &walked(&chain),
                &collected(&branches, &combine),
                &format!("{} at arity {arity}", op.label()),
            );
        }
    }

    // `subtract` and `divide` have a left operand and a right operand, so there
    // is no fold of them over a longer list that is not a convention about
    // parentheses. They are refused above two branches — before this change and
    // after it — and the refusal is what `fold_carrier` declines to override.
    for op in [Arithmetic::Subtract, Arithmetic::Divide] {
        assert_eq!(
            ArithmeticCombine::new("join", op).fold_carrier(&[Dtype::F64; 3]),
            None,
            "{} is not a fold and must not declare a carrier",
            op.label()
        );
        let three = Chain::parallel(
            arms(&[1.0, 2.0, 3.0]),
            Box::new(ArithmeticCombine::new("join", op)),
        )
        .expect("a fan-in the constructor allows");
        assert!(
            three.produces(Dtype::F64).is_err(),
            "{} must still refuse three branches",
            op.label()
        );
        // At two it is unchanged, and that is checked rather than assumed.
        let branches = arms(&[1.0, 2.0]);
        let combine = ArithmeticCombine::new("join", op);
        let chain = Chain::parallel(
            arms(&[1.0, 2.0]),
            Box::new(ArithmeticCombine::new("join", op)),
        )
        .expect("a fan-in");
        identical(
            &walked(&chain),
            &collected(&branches, &combine),
            &format!("{} at arity 2", op.label()),
        );
    }

    // A difference declares no carrier for the same reason, and its two-branch
    // answer must not have moved either.
    assert_eq!(
        DifferenceCombine::new("difference").fold_carrier(&[Dtype::F64; 2]),
        None,
        "a difference is not a fold and must not declare a carrier"
    );
    let branches = arms(&[7.0, -3.0]);
    let combine = DifferenceCombine::new("difference");
    let chain = Chain::parallel(
        arms(&[7.0, -3.0]),
        Box::new(DifferenceCombine::new("difference")),
    )
    .expect("a fan-in");
    identical(
        &walked(&chain),
        &collected(&branches, &combine),
        "difference at arity 2",
    );
}

/// **The control on every equality above: the comparison can see a reordering.**
///
/// Without this, "the folded answer equals the collected answer" would be
/// consistent with a comparison that could not tell any two volumes apart. The
/// same branches summed in the reverse order must give a *different* volume, and
/// at these magnitudes `f64` addition guarantees it: `1e16 + 1.0` is `1e16`, so
/// a left fold that meets the large value first loses every `1.0` after it and
/// one that meets it last does not.
#[test]
fn the_comparison_sees_an_order_that_moves_the_answer() {
    for arity in 3..=8usize {
        let offsets = absorbing_offsets(arity);
        let forward = arms(&offsets);
        let mut backward = arms(&offsets);
        backward.reverse();

        let combine = ArithmeticCombine::new("join", Arithmetic::Add);
        let one = collected(&forward, &combine);
        let other = collected(&backward, &combine);
        let differing = differing_bits(
            one.view::<f64>().expect("f64"),
            other.view::<f64>().expect("f64"),
        )
        .expect("comparable");
        assert!(
            differing > 0,
            "at arity {arity} the sum did not move when the branches were reversed, so the \
             fixture is not order-sensitive and every equality in this file is being checked \
             by an instrument that cannot fail"
        );

        // And the walk agrees with the *forward* one, not merely with one of
        // them: the fold's order is branch order and nothing is licensed to
        // reorder it.
        let chain = Chain::parallel(
            arms(&offsets),
            Box::new(ArithmeticCombine::new("join", Arithmetic::Add)),
        )
        .expect("a fan-in");
        identical(&walked(&chain), &one, &format!("add at arity {arity}"));
    }
}

/// **What the walk holds, which is the point of the change.**
///
/// `Chain::apply_observing` reports the high-water mark of the buffers the walk
/// allocated. For a combine that declares a fold that figure must not grow with
/// the arity; for one that does not, it must — and the second half is what says
/// the first is a measurement rather than a constant.
#[test]
fn a_declared_fold_holds_a_figure_that_does_not_grow_with_the_arity() {
    let block = input();
    let at = anchor();
    let held = |chain: &Chain| -> u64 {
        let mut out = Voxels::zeros(
            chain.produces(block.dtype()).expect("an element type"),
            chain.output_shape(block.shape()).expect("a shape"),
        )
        .expect("an output block");
        chain
            .apply_observing(&block, SourceInputs::none(), &mut out, &at)
            .expect("one block")
            .chain_bytes()
    };

    let mut folded = Vec::new();
    let mut collecting = Vec::new();
    for arity in 2..=8usize {
        let offsets = absorbing_offsets(arity);
        folded.push(held(
            &Chain::parallel(
                arms(&offsets),
                Box::new(ArithmeticCombine::new("join", Arithmetic::Add)),
            )
            .expect("a fan-in"),
        ));
        collecting.push(held(
            &Chain::parallel(arms(&offsets), Box::new(Collected)).expect("a fan-in"),
        ));
    }
    eprintln!("\narity   folded   collected  (bytes the walk held, {BLOCK:?} f64 block)");
    for (index, (fold, collect)) in folded.iter().zip(collecting.iter()).enumerate() {
        eprintln!("{:>5}  {fold:>8}  {collect:>10}", index + 2);
    }

    // The collected shape grows by one whole branch buffer per arm, plus the one
    // slot the holding vector's spine grows by — the walk counts that spine and
    // this row is an equality, so it is named here rather than absorbed into a
    // tolerance. Without this row the assertion below would pass on a chain that
    // allocated nothing.
    let buffer = (BLOCK.iter().product::<usize>() * 8) as u64;
    let slot = std::mem::size_of::<Voxels>() as u64;
    for window in collecting.windows(2) {
        assert_eq!(
            window[1] - window[0],
            buffer + slot,
            "a collected fan-in must take one more whole branch buffer per arm; if it does not, \
             this measurement is not seeing branch buffers at all and the flat row beside it \
             means nothing"
        );
    }

    // The folded shape does not grow at all beyond three arms: a partial and
    // the branch just computed.
    //
    // **It used to be three, and the third is gone.** The join needed a buffer
    // of its own until `Combine::fold_in_place` let it be written over the left
    // operand; this combine's carrier is `fold_carrier == inputs[0]`, the
    // branches' own element type, so the partial qualifies from branch 0 and
    // every join accumulates. `tests/fold_in_place.rs` measures the change on
    // its own fixture — `5184` bytes against `3456` — and pins the condition,
    // which is sharper than it looks: a combine whose carrier is *not* the
    // branches' type, `LogicCombine` being the one that ships, still allocates
    // at its first join, and the first join is the peak.
    for (index, window) in folded.windows(2).enumerate() {
        if index + 2 >= 3 {
            assert_eq!(
                window[0],
                window[1],
                "a folded fan-in held {} at arity {} and {} at arity {}; the whole claim is that \
                 the figure is independent of the arity",
                window[0],
                index + 2,
                window[1],
                index + 3
            );
        }
    }
    assert_eq!(
        *folded.last().expect("a row"),
        2 * buffer,
        "the fold holds exactly two block buffers at its worst moment — the partial and the \
         branch just computed. It held three until the join stopped needing a buffer of its own; \
         a return to three means `Combine::fold_in_place` has stopped being reached"
    );
    assert!(
        *collecting.last().expect("a row") > *folded.last().expect("a row"),
        "at eight arms the collected path must hold more than the folded one, or there was \
         nothing here to fix"
    );
}

/// **The operand order inside one pair, which nothing this crate ships can see.**
///
/// Every folding combine here is commutative pairwise, so a `fold_pair` that
/// swapped its two arguments would pass every other test in this file — measured,
/// not supposed: that mutant was written and it survived. [`Leftmost`] is
/// associative and not commutative, so the swap moves its answer, and the
/// control below is that a deliberately reversed reference does differ.
#[test]
fn the_operand_order_inside_a_pair_is_pinned_by_a_combine_that_is_not_commutative() {
    for arity in 2..=8usize {
        // Zeros in some branches and not others, so "the first non-zero" picks a
        // different arm at different voxels.
        let offsets: Vec<f64> = (0..arity).map(|index| (index % 3) as f64).collect();
        let branches = arms(&offsets);
        let chain = Chain::parallel(arms(&offsets), Box::new(Leftmost)).expect("a fan-in");
        let forward = collected(&branches, &Leftmost);
        identical(
            &walked(&chain),
            &forward,
            &format!("leftmost at arity {arity}"),
        );

        // The control: reversing the branches moves the answer, so the equality
        // above is being checked by something that can fail.
        let mut backward = arms(&offsets);
        backward.reverse();
        let other = collected(&backward, &Leftmost);
        let differing = differing_bits(
            forward.view::<f64>().expect("f64"),
            other.view::<f64>().expect("f64"),
        )
        .expect("comparable");
        assert!(
            differing > 0,
            "at arity {arity} `leftmost` did not move when the branches were reversed, so \
             this fixture cannot see an operand order either"
        );
    }
}

/// A carrier declared without the step it names is refused by name, rather than
/// joined by whatever a default invented.
#[test]
fn declaring_a_carrier_without_the_step_is_refused_by_name() {
    let chain = Chain::parallel(arms(&[1.0, 2.0]), Box::new(HalfDeclared)).expect("a fan-in");
    let block = input();
    let mut out = Voxels::zeros(Dtype::F64, BLOCK).expect("an output block");
    let error = chain
        .apply_with(&block, SourceInputs::none(), &mut out, &anchor())
        .expect_err("a combine that declared a fold it does not implement");
    let text = error.to_string();
    assert!(
        text.contains("half declared") && text.contains("fold_pair"),
        "the refusal must name the combine and the missing step: {text}"
    );
}

// ------------------------------------------- the sequence's clone of input --

/// The children of the sequence below, as a list, so the chain and the
/// hand-composed reference are built from **one** statement of what the sequence
/// is. Two lists would let the test pass while the two ran different ops.
fn steps() -> Vec<(f64, f64)> {
    vec![(2.0, 1.0), (0.5, -3.0), (-1.0, 7.0), (3.0, 0.25)]
}

/// **A sequence writes what applying its children one at a time writes.**
///
/// `Chain::Sequence` used to start from `input.clone()`. The clone was only ever
/// read, and everything that read it could have read the caller's own block, so
/// removing it is a residency change — but only if the children are still handed
/// the same bytes in the same order, and that is what this asserts against a
/// composition built outside the walk.
#[test]
fn a_sequence_writes_what_its_children_write_one_at_a_time() {
    let at = anchor();
    let children: Vec<Chain> = steps()
        .into_iter()
        .map(|(scale, offset)| arm(scale, offset))
        .collect();

    // The reference: each child applied on its own, into its own buffer.
    let mut carried = input();
    for child in &children {
        let mut next = Voxels::zeros(
            child.produces(carried.dtype()).expect("an element type"),
            child.output_shape(carried.shape()).expect("a shape"),
        )
        .expect("a buffer");
        child
            .apply_with(&carried, SourceInputs::none(), &mut next, &at)
            .expect("one child");
        carried = next;
    }

    let chain = Chain::sequence(
        steps()
            .into_iter()
            .map(|(scale, offset)| arm(scale, offset))
            .collect(),
    );
    identical(&walked(&chain), &carried, "a sequence of four maps");

    // **The control on the reference.** A composition of four maps that ignored
    // one of them would still be a well-formed volume, so the reference is
    // checked against a shorter one: three of the four must give a different
    // answer, or this test would pass on a walk that dropped a child.
    let mut shorter = input();
    for child in &children[..3] {
        let mut next = Voxels::zeros(
            child.produces(shorter.dtype()).expect("an element type"),
            child.output_shape(shorter.shape()).expect("a shape"),
        )
        .expect("a buffer");
        child
            .apply_with(&shorter, SourceInputs::none(), &mut next, &at)
            .expect("one child");
        shorter = next;
    }
    let differing = differing_bits(
        carried.view::<f64>().expect("f64"),
        shorter.view::<f64>().expect("f64"),
    )
    .expect("comparable");
    assert!(
        differing > 0,
        "three of the four children gave the same volume as all four, so this fixture cannot \
         see a child being skipped"
    );
}

/// **What a sequence holds, and the honest size of what the clone was costing.**
///
/// The clone was a whole block buffer, written once and read only where the
/// caller's own block would have served — so it was dead by the test this file
/// applies. **It was not, however, the tall term at every length**, and that is
/// worth writing down because the first version of this test assumed it was:
///
/// * **two children** held the clone *and* the one intermediate. Removing it
///   halves the chain's own residency, from two block buffers to one.
/// * **three or more** hold two intermediates at once anyway — the one being
///   read and the one being written — and the clone was freed before the second
///   was allocated. The peak was already two, and removing the clone leaves it
///   at two.
///
/// So this is a whole block buffer off every two-child sequence and nothing off
/// a longer one. At one block that is a whole volume per such phase, which is
/// worth having and is not what a reader would have guessed from "a sequence
/// clones its input".
#[test]
fn a_sequence_holds_its_intermediates_and_no_copy_of_its_input() {
    let block = input();
    let at = anchor();
    let buffer = (BLOCK.iter().product::<usize>() * 8) as u64;
    let held = |chain: &Chain| -> u64 {
        let mut out = Voxels::zeros(
            chain.produces(block.dtype()).expect("an element type"),
            chain.output_shape(block.shape()).expect("a shape"),
        )
        .expect("an output block");
        chain
            .apply_observing(&block, SourceInputs::none(), &mut out, &at)
            .expect("one block")
            .chain_bytes()
    };

    // One child is not a sequence in any sense that allocates: it is applied
    // straight into `out`. Without this row the assertions below would pass on a
    // walk that allocated nothing at all.
    let one = held(&Chain::sequence(vec![arm(2.0, 1.0)]));
    assert_eq!(
        one, 0,
        "a one-child sequence has nothing between the input and the output, so it must hold \
         nothing; it held {one}"
    );

    eprintln!("\nchildren   held  buffers   ({BLOCK:?} f64 block, the walk's own tally)");
    for length in 2..=6usize {
        let chain = Chain::sequence(
            (0..length)
                .map(|index| arm(1.0 + index as f64, index as f64))
                .collect(),
        );
        let bytes = held(&chain);
        // The two bookkeeping vectors the walk counts — `parts` and `places`,
        // one slot each per child. Named rather than tolerated, for the reason
        // the equality in `tests/working_set_residency.rs` gives.
        let book = (length * (std::mem::size_of::<&Chain>() + PLACEMENT_SIZE)) as u64;
        let buffers = if length == 2 { 1 } else { 2 };
        eprintln!("{length:>8}  {bytes:>5}  {buffers:>7}");
        assert_eq!(
            bytes,
            buffers * buffer + book,
            "a sequence of {length} children must hold {buffers} block buffer(s) and its \
             bookkeeping. One more than that at any length is the input being copied again."
        );
    }
}

/// `size_of::<op::Placement>()`, which is not public. Taken from the walk's own
/// figure and then held constant, so a change to the type shows up here as a
/// failure rather than silently widening the window.
const PLACEMENT_SIZE: usize = 152;

// ------------------------------------------ the copy a source arm made --

/// A stored image, distinct per seed, holding a mix of set and clear voxels so
/// that a connective over it is neither uniformly set nor uniformly clear.
///
/// Not a constant volume, deliberately: an arm that answered from the wrong
/// buffer, or from no buffer at all, has to be visible in the result.
fn stored_volume(seed: u64) -> Voxels {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut array = Array3::<f64>::zeros((BLOCK[0], BLOCK[1], BLOCK[2]));
    for value in array.iter_mut() {
        *value = (next() % 3) as f64;
    }
    Voxels::F64(array)
}

/// The two image handles the tests below read.
fn one() -> ImageId {
    ImageId::from(7usize)
}

fn two() -> ImageId {
    ImageId::from(8usize)
}

/// A fan-in of `computed` map arms followed by one branch per entry of
/// `images`, joined by a connective — the shape `Chain::Source` exists for.
fn fan_in_over(computed: usize, images: &[ImageId], logic: Logic) -> (Vec<Chain>, LogicCombine) {
    let mut branches: Vec<Chain> = (0..computed)
        .map(|index| arm(1.0 + index as f64, index as f64))
        .collect();
    for &image in images {
        branches.push(Chain::source(image, Dtype::F64));
    }
    (branches, LogicCombine::new("join", logic))
}

/// **A borrowed source arm is the same volume as a copied one.**
///
/// Every mix of computed arms and source arms from two branches to six, over
/// both connectives that can disagree, against a reference that copies each
/// branch into a buffer of its own — including the source branches, which is
/// what makes it the *old* path and not a restatement of the new one.
#[test]
fn a_borrowed_source_arm_is_the_same_volume_as_a_copied_one() {
    let first = stored_volume(1);
    let second = stored_volume(2);
    let held = [(one(), &first), (two(), &second)];

    for logic in [Logic::And, Logic::Or, Logic::Xor] {
        for computed in 0..=3usize {
            for count in 0..=2usize {
                if computed + count < 2 {
                    continue;
                }
                let images = &[one(), two()][..count];
                let (branches, combine) = fan_in_over(computed, images, logic);
                let (again, _) = fan_in_over(computed, images, logic);
                let chain = Chain::parallel(again, Box::new(LogicCombine::new("join", logic)))
                    .expect("a fan-in");
                identical(
                    &walked_with(&chain, &held),
                    &collected_with(&branches, &combine, &held),
                    &format!("{logic:?} over {computed} computed and {count} source arms"),
                );
            }
        }
    }
}

/// **The control on every equality above: the stored buffer is being read.**
///
/// A borrowed arm that answered from the wrong buffer — or a walk that skipped
/// the branch entirely — would still write a well-formed volume, and every
/// comparison above would still pass if the reference made the same mistake.
/// So the same chain is run against two *different* stored images and required
/// to disagree.
#[test]
fn the_answer_depends_on_the_stored_buffer_it_borrows() {
    let first = stored_volume(1);
    let other = stored_volume(9);
    let differing_input = differing_bits(
        first.view::<f64>().expect("f64"),
        other.view::<f64>().expect("f64"),
    )
    .expect("comparable");
    assert!(
        differing_input > 0,
        "the two stored volumes are the same volume, so this control cannot fail"
    );

    let chain = || {
        let (branches, _) = fan_in_over(1, &[one()], Logic::Or);
        Chain::parallel(branches, Box::new(LogicCombine::new("join", Logic::Or))).expect("a fan-in")
    };
    let with_first = walked_with(&chain(), &[(one(), &first)]);
    let with_other = walked_with(&chain(), &[(one(), &other)]);
    let differing = differing_bits(
        with_first.view::<f64>().expect("f64"),
        with_other.view::<f64>().expect("f64"),
    )
    .expect("comparable");
    assert!(
        differing > 0,
        "the fan-in gave the same answer for two different stored images, so a borrowed arm \
         that read nothing at all would pass every comparison in this file"
    );
}

/// **Two arms naming one image are handed one buffer, twice.**
///
/// That is what those two arms mean — `SourceInputs` is keyed by image and not
/// by leaf, because an image is one thing — and it is what the two copies said
/// before, at twice the bytes. Aliasing is the case a borrow introduces and a
/// copy did not have, so it is checked rather than assumed: the combine reads
/// both operands through shared references and writes a third buffer, so there
/// is nothing for the aliasing to disturb.
#[test]
fn two_arms_naming_one_image_agree_with_two_arms_holding_two_copies() {
    let volume = stored_volume(3);
    let held = [(one(), &volume)];
    let branches = vec![
        Chain::source(one(), Dtype::F64),
        Chain::source(one(), Dtype::F64),
    ];
    let combine = LogicCombine::new("join", Logic::Xor);
    let chain = Chain::parallel(
        vec![
            Chain::source(one(), Dtype::F64),
            Chain::source(one(), Dtype::F64),
        ],
        Box::new(LogicCombine::new("join", Logic::Xor)),
    )
    .expect("a fan-in");
    identical(
        &walked_with(&chain, &held),
        &collected_with(&branches, &combine, &held),
        "two arms naming one image",
    );

    // The control: an `Xor` of a mask with itself is clear everywhere, so a
    // fixture that could not tell "read the same image twice" from "read two
    // different images" would pass this by accident. Against a *different*
    // second image it must differ.
    let other = stored_volume(4);
    let two_images = Chain::parallel(
        vec![
            Chain::source(one(), Dtype::F64),
            Chain::source(two(), Dtype::F64),
        ],
        Box::new(LogicCombine::new("join", Logic::Xor)),
    )
    .expect("a fan-in");
    let apart = walked_with(&two_images, &[(one(), &volume), (two(), &other)]);
    let together = walked_with(&chain, &held);
    let differing = differing_bits(
        apart.view::<f64>().expect("f64"),
        together.view::<f64>().expect("f64"),
    )
    .expect("comparable");
    assert!(
        differing > 0,
        "naming one image twice and naming two images gave the same volume, so this fixture \
         cannot see which buffer an arm was handed"
    );
}

/// **A sequence whose first child is a source leaf reads the stored image and
/// writes what it always wrote.**
///
/// The head is borrowed rather than computed into an intermediate, so the
/// reference is built by handing the stored buffer to the remaining children
/// one at a time — outside the walk, so it is not the walk restating itself.
#[test]
fn a_sequence_starting_at_a_source_writes_what_the_stored_buffer_maps_to() {
    let volume = stored_volume(5);
    let held = [(one(), &volume)];
    let at = anchor();
    let tail = || vec![arm(3.0, -1.0), arm(0.25, 2.0)];

    let chain = Chain::sequence(
        std::iter::once(Chain::source(one(), Dtype::F64))
            .chain(tail())
            .collect(),
    );

    let mut carried = volume.clone();
    for child in &tail() {
        let mut next = Voxels::zeros(
            child.produces(carried.dtype()).expect("an element type"),
            child.output_shape(carried.shape()).expect("a shape"),
        )
        .expect("a buffer");
        child
            .apply_with(&carried, SourceInputs::none(), &mut next, &at)
            .expect("one child");
        carried = next;
    }

    identical(
        &walked_with(&chain, &held),
        &carried,
        "a sequence starting at a source",
    );

    // The control: the sequence must not be answering from its *input* block,
    // which is what it would do if the head were skipped without the borrow
    // taking its place.
    let from_input = Chain::sequence(tail());
    let differing = differing_bits(
        walked_with(&chain, &held).view::<f64>().expect("f64"),
        walked(&from_input).view::<f64>().expect("f64"),
    )
    .expect("comparable");
    assert!(
        differing > 0,
        "the sequence gave the same answer whether it started at the stored image or at its \
         own input block, so this fixture cannot see which buffer the head read"
    );
}

/// **What a source arm costs the walk, which is now nothing on either path.**
///
/// # The control moved, and why it had to
///
/// This test used to read the collected path as its control: a combine
/// declaring no fold carrier was handed owned `Voxels`, so it copied every
/// source arm into a buffer of its own, and that growth was the proof that the
/// tally can see a branch buffer at all. Without such a control, "the folded
/// path is flat" is equally consistent with a measurement that sees nothing.
///
/// **`Combine::apply` now takes `&[&Voxels]`, so the collected path borrows
/// too** — the asymmetry that control depended on was the bug. Its growth per
/// source arm fell from one whole block buffer to the vector spine's one slot.
///
/// So the control moves to the axis that still allocates: a **computed** arm
/// must be bought and cannot be borrowed, and the collected path must grow by a
/// full buffer for each one. Both rows are asserted here, and the slot is
/// **measured from the source-arm row rather than named**, because the spine
/// holds a type this crate does not export and a hard-coded width would be a
/// second spelling of `size_of` that drifts.
#[test]
fn a_borrowed_source_arm_costs_the_walk_nothing() {
    let first = stored_volume(1);
    let second = stored_volume(2);
    let held = [(one(), &first), (two(), &second)];
    let block = input();
    let at = anchor();
    let buffer = (BLOCK.iter().product::<usize>() * 8) as u64;
    let chain_bytes = |chain: &Chain| -> u64 {
        let mut out = Voxels::zeros(
            chain.produces(block.dtype()).expect("an element type"),
            chain.output_shape(block.shape()).expect("a shape"),
        )
        .expect("an output block");
        chain
            .apply_observing(&block, SourceInputs::new(&held), &mut out, &at)
            .expect("one block")
            .chain_bytes()
    };

    eprintln!("\nsources  borrowed  collected   (bytes the walk held)");
    let mut borrowed = Vec::new();
    let mut collecting = Vec::new();
    for count in 1..=2usize {
        let images = &[one(), two()][..count];
        let folding = Chain::parallel(
            fan_in_over(1, images, Logic::Or).0,
            Box::new(LogicCombine::new("join", Logic::Or)),
        )
        .expect("a fan-in");
        // `Collected` declares no fold carrier, so this one takes the collect
        // path — which borrows its source arms exactly as the fold path does,
        // and differs from it only in holding every branch at once.
        let holding = Chain::parallel(fan_in_over(1, images, Logic::Or).0, Box::new(Collected))
            .expect("a fan-in");
        let (a, b) = (chain_bytes(&folding), chain_bytes(&holding));
        eprintln!("{count:>7}  {a:>8}  {b:>9}");
        borrowed.push(a);
        collecting.push(b);
    }

    // A source arm costs the collect path its spine slot and nothing else. The
    // slot is what a `Vec` grows by, three orders below a block buffer.
    let slot = collecting[1] - collecting[0];
    // Strictly less than a buffer, and that is the exact claim rather than a
    // loose one: the control below pins a real buffer's growth at `buffer +
    // slot`, so anything under `buffer` cannot be one. A ratio would be the
    // weaker test here — this fixture's block is deliberately tiny, so "a small
    // fraction of a buffer" says more about the fixture than about the walk.
    assert!(
        slot < buffer,
        "an extra source arm cost the collected path {slot} bytes, and a block buffer in this \
         fixture is {buffer}. A borrowed arm should cost only the vector slot that holds it; a \
         growth at or above a buffer means one is still being copied"
    );

    // **The control, on the axis that still allocates.** A computed arm cannot
    // be borrowed — the walk has to buy somewhere to put it — so the collected
    // path must grow by a whole buffer for each, plus the same slot measured
    // above. If this row were flat too, the flat rows above would be consistent
    // with a tally that sees no buffers at all and would mean nothing.
    let mut computing = Vec::new();
    for computed in 1..=3usize {
        let holding = Chain::parallel(fan_in_over(computed, &[one()], Logic::Or).0, Box::new(Collected))
            .expect("a fan-in");
        computing.push(chain_bytes(&holding));
    }
    eprintln!("computed arms (collected): {computing:?}");
    for window in computing.windows(2) {
        assert_eq!(
            window[1] - window[0],
            buffer + slot,
            "the collected path must take one more branch buffer per **computed** arm. This is \
             the control: it is what proves the tally can see a branch buffer, which is what \
             makes the flat source-arm rows above a finding rather than a blind instrument"
        );
    }

    // And the folded path takes none of them. What it does hold is the computed
    // arm's own buffer, and — from three branches on — one partial, which is a
    // `Bool` block and an eighth of the width; at two branches the join writes
    // `out` directly and there is no partial at all. Neither term is a source
    // arm's, which is the claim: **stated exactly per count rather than as an
    // inequality**, because "did not grow much" is what a window would say.
    let carrier = BLOCK.iter().product::<usize>() as u64;
    for (index, held_bytes) in borrowed.iter().enumerate() {
        let count = index + 1;
        let expected = if 1 + count > 2 {
            buffer + carrier
        } else {
            buffer
        };
        assert_eq!(
            *held_bytes, expected,
            "a fan-in of one computed arm and {count} source arms held {held_bytes} against \
             {expected}. The computed arm's buffer and, above two branches, one `Bool` partial \
             — and nothing for a source arm at any count."
        );
    }
}
