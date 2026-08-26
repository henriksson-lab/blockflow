//! **Accumulating over the partial, and which combine it is worth anything to.**
//!
//! `Combine::fold_in_place` lets a fold write its join over its left operand
//! instead of into a third buffer. Without it the fold holds **three** block
//! buffers at its worst moment: the partial, the branch just computed, and the
//! buffer their join goes into.
//!
//! # The condition, which is sharper than it first looks
//!
//! An in-place join needs an accumulator that is **owned by the walk** and
//! **already in the carrier's element type**, since a fold cannot change the
//! width of a buffer it writes over. That makes the saving a property of the
//! *combine*, not of this change:
//!
//! * **`ArithmeticCombine` declares `fold_carrier == inputs[0]`** — the
//!   branches' own type — so its partial is in the carrier from branch 0 and
//!   **every** join accumulates. The saving is real.
//! * **`LogicCombine` declares `Bool`** while its arms generally produce `f64`,
//!   so its *first* join must allocate. And the first join is the peak: it is
//!   the moment two branch buffers and a fresh join buffer are all live. Later
//!   joins accumulate and change nothing this can measure.
//!
//! Both are asserted. Reporting only the first would be reporting the fixture.
use blockflow::op::{Chain, Combine};
use blockflow::ops::voxelwise::{
    Arithmetic, ArithmeticCombine, Logic, LogicCombine, VoxelwiseMapOp,
};
use blockflow::{Anchor, SourceInputs, Voxels};
use ndarray::Array3;

const BLOCK: [usize; 3] = [6, 6, 6];

fn input() -> Voxels {
    Voxels::F64(Array3::from_shape_fn(
        (BLOCK[0], BLOCK[1], BLOCK[2]),
        |(z, y, x)| ((z * 36 + y * 6 + x) % 7) as f64,
    ))
}

fn arm(scale: f64) -> Chain {
    Chain::op(VoxelwiseMapOp::new("arm", move |value: f64| value * scale))
}

/// Run a fan-in and report what the walk held, and what it answered.
fn held(arity: usize, combine: Box<dyn Combine>) -> (u64, Voxels) {
    let branches: Vec<Chain> = (1..=arity).map(|i| arm(i as f64)).collect();
    let chain = Chain::parallel(branches, combine).expect("a fan-in");
    let block = input();
    let mut out = Voxels::zeros(
        chain.produces(block.dtype()).expect("an element type"),
        chain.output_shape(block.shape()).expect("a shape"),
    )
    .expect("an output block");
    let empty = [];
    let tally = chain
        .apply_observing(
            &block,
            SourceInputs::new(&empty),
            &mut out,
            &Anchor::whole(BLOCK),
        )
        .expect("one block");
    (tally.chain_bytes(), out)
}

fn maximum() -> Box<dyn Combine> {
    Box::new(ArithmeticCombine::new("max", Arithmetic::Maximum))
}

/// **Two buffers, not three, where the carrier is the branches' own type.**
///
/// One `f64` block is `6^3 * 8 = 1728` bytes. Three of them is `5184`; two is
/// `3456`. The figure must be flat across the arity — that property predates
/// this change and is what the fold exists for — **and** it must sit at the
/// two-buffer figure, which is what accumulating means.
#[test]
fn an_arithmetic_fan_in_holds_two_buffers_not_three() {
    let buffer = BLOCK.iter().product::<usize>() as u64 * 8;
    let mut figures = Vec::new();
    for arity in 2..=6usize {
        let (bytes, _) = held(arity, maximum());
        figures.push(bytes);
        eprintln!(
            "arithmetic arity {arity}: {bytes} bytes ({} buffers)",
            bytes / buffer
        );
    }
    for window in figures.windows(2) {
        assert_eq!(
            window[0], window[1],
            "the folded figure grew with the arity: {figures:?}"
        );
    }
    assert_eq!(
        figures[0],
        buffer * 2,
        "an arithmetic fan-in held {} bytes where two blocks are {}. Three would mean \
         `fold_in_place` is not being reached at all",
        figures[0],
        buffer * 2
    );
}

/// **And the control: a combine whose carrier is not the branches' type is
/// unchanged**, because its first join — the peak — cannot accumulate.
///
/// Without this the test above is consistent with a measurement that cannot see
/// a buffer, which is the trap this suite has already been caught by twice.
#[test]
fn a_logic_fan_in_still_holds_three_because_its_first_join_cannot_accumulate() {
    let float_block = BLOCK.iter().product::<usize>() as u64 * 8;
    let (two, _) = held(2, Box::new(LogicCombine::new("or", Logic::Or)));
    let (six, _) = held(6, Box::new(LogicCombine::new("or", Logic::Or)));
    eprintln!("logic arity 2: {two} bytes; arity 6: {six} bytes");
    assert!(
        six > two,
        "the logic fold held {six} at arity six against {two} at two; the partial it cannot \
         accumulate into should still cost a buffer"
    );
    assert!(
        six > float_block * 2,
        "the logic fold held {six}, at or below the two-buffer figure {}. If it has started \
         accumulating from branch 0, its carrier now matches its arms and this control is stale",
        float_block * 2
    );
}

/// **The answer does not move**, which is the only thing that must not.
///
/// The folded path against the collected one — a combine declaring no carrier
/// takes the latter — at every arity, bit for bit. `Maximum` is used precisely
/// because it is a *selection*: it goes through `selection_in_place`, the kernel
/// with the different element-type story, and it is order-insensitive so a
/// disagreement here is a real disagreement rather than a rounding.
#[test]
fn accumulating_in_place_is_the_same_answer() {
    for arity in 2..=6usize {
        let (_, folded) = held(arity, maximum());

        // The reference: every branch computed into its own buffer, then one
        // call to `Combine::apply` over the whole set.
        let block = input();
        let mut results = Vec::new();
        for i in 1..=arity {
            let branch = arm(i as f64);
            let mut buffer = Voxels::zeros(
                branch.produces(block.dtype()).expect("a type"),
                branch.output_shape(block.shape()).expect("a shape"),
            )
            .expect("a buffer");
            let empty = [];
            branch
                .apply_observing(
                    &block,
                    SourceInputs::new(&empty),
                    &mut buffer,
                    &Anchor::whole(BLOCK),
                )
                .expect("a branch");
            results.push(buffer);
        }
        let refs: Vec<&Voxels> = results.iter().collect();
        let mut expected = Voxels::zeros(folded.dtype(), BLOCK).expect("a buffer");
        maximum()
            .apply(&refs, &mut expected, &Anchor::whole(BLOCK))
            .expect("the collected path");

        assert!(
            folded == expected,
            "arity {arity}: the in-place fold and the collected path disagree. A fold that \
             changes a voxel is not a fold"
        );
    }
}

/// **The operand-order hazard cannot arise, and the guard is `fold_carrier`.**
///
/// `acc = op(acc, right)` and `acc = op(right, acc)` are different volumes for a
/// non-commutative operation, and an in-place kernel is exactly where that could
/// be got backwards with nothing else noticing.
///
/// It is unreachable, but **not for the reason I first wrote down**. My first
/// version of this test asserted that `Chain::parallel` refuses a subtraction
/// over four branches. It does not — the chain builds, and the refusal comes
/// later, from `produces`. The actual guard is one level down:
/// `ArithmeticCombine::fold_carrier` is `self.op.folds_over_many().then(...)`,
/// so a non-associative operation declares **no carrier at all**, takes the
/// collected path, and never reaches `fold_pair` or `fold_in_place`.
///
/// And every operation that does pass that gate — the selections and the
/// associative arithmetic — is commutative, so there is no fixture in which the
/// accumulator's side of a pair could be observed to matter. The guard is
/// asserted; the hazard has nothing to test.
#[test]
fn a_non_commutative_operation_declares_no_carrier_and_never_reaches_the_fold() {
    let four = [
        blockflow::Dtype::F64,
        blockflow::Dtype::F64,
        blockflow::Dtype::F64,
        blockflow::Dtype::F64,
    ];
    for op in [Arithmetic::Subtract, Arithmetic::Divide] {
        let combine = ArithmeticCombine::new("op", op);
        assert!(
            combine.fold_carrier(&four).is_none(),
            "{:?} declared a fold carrier. It folds left, so the accumulator's side of each pair \
             now matters — write the operand-order test this comment says is unreachable",
            op
        );
    }
    // And the ones that do fold, so the assertion above is about this operation
    // rather than about `fold_carrier` answering `None` to everything.
    for op in [Arithmetic::Maximum, Arithmetic::Minimum] {
        assert!(
            ArithmeticCombine::new("op", op)
                .fold_carrier(&four)
                .is_some(),
            "{op:?} declared no carrier, so the control above proves nothing"
        );
    }
}
