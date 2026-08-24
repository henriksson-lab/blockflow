// SPDX-License-Identifier: MIT
//
// **A mask is one bit, and this file is about which thing carries it.**
//
// `ops::voxelwise` has always had two carriers for a mask — `bool`, and `f64`
// under the `is_set` / `from_set` convention — and until now a chain could only
// *arrive* at the narrow one by computing the answer in the wide one first.
// `VoxelwiseMapOp` holds an `f64 -> f64` map by construction, so a threshold's
// verdict was `1.0` and `0.0` in an eight-byte buffer, and `NarrowOp::to_mask`
// could only narrow that buffer after it had already been allocated and filled.
//
// Three things landed, and this file is the evidence for all of them:
//
// * `VoxelwiseMaskOp` — a voxelwise op that **produces `Bool`**, holding a
//   `MaskFn` (`f64 -> bool`) where `VoxelwiseMapOp` holds a `MapFn`;
// * `LogicCombine` joining branches whose **carriers differ**, with the carrier
//   of the answer stated by the caller rather than inferred from a branch;
// * `CarryOp` — the block, unchanged, at whatever width it arrived in. The
//   third is not a decoration on the first two: the branch that carries a sink
//   forward through a chain of `OR`s was `VoxelwiseMapOp::identity`, which holds
//   a `MapFn` and is therefore `f64`-only, so a `Bool` sink had nothing to carry
//   it. Two of the three would not have been enough to narrow anything.
//
// Why the second is the shape it is
// ---------------------------------
// `LogicCombine` never reads a number: both of its paths have always gone
// through `is_set` and written through `from_set`, and its answer is one bit per
// voxel in every case. The rule it used to enforce — that every branch agree —
// was therefore not a statement about the connective. It was an artefact of
// `pair` reading one branch's tag and using it for all three buffers.
//
// What is *not* relaxed is the output. `Chain::produces` refuses an element type
// that depends on which branch is live, because "an image is allocated at one
// width and a decomposition is binding"; inferring the join's carrier from the
// branches would make an image's width a consequence of an arm's, so flipping
// one arm from `f64` to `Bool` would silently re-width an image other phases
// read. So branches that agree produce what they agreed on — which is exactly
// the old rule, and is why every plan already written still means what it meant —
// and branches that differ are joined only where the caller has said what to
// write. The refusal is not deleted: it is `LogicCombine::producing` that lifts
// it, and `a_mixed_fan_in_is_still_refused_when_nobody_says_what_to_write` below
// is that same absence, still asserted, in the case that is still absent.
//
// The alternative considered and rejected was "make the chain `Bool` end to
// end". It is not enough on its own: the arms of a real fan-in are not all one
// crate's ops — a local statistic's verdict, a `Chain::Source` reading an
// existing `f64` image, a branch carrying a sink an earlier phase wrote — and
// every one of them would have to change width in lockstep for the chain to
// typecheck. A chain that cannot be converted a phase at a time cannot be
// converted at all when its phases are built by different callers. Mixed
// branches with a stated output is what lets a sink become `Bool` at the *first*
// join and stay `Bool`, while the arms feeding it keep whatever width their own
// arithmetic has.
//
// What is measured here, and what is measured next door
// ----------------------------------------------------
// This file is about voxels: that the two carriers of one threshold agree
// exactly, that a fan-in's answer does not depend on which carrier its branches
// used, and that all of it is decomposition-invariant. The **bytes** are
// `tests/mask_carrier_residency.rs`, which runs the same two plans under a
// counting allocator and reports the peak the process actually held.

use std::sync::Arc;

use ndarray::Array3;

use blockflow::assemble::{Assembly, PlanBuilder};
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, BlockOp, Chain, Combine};
use blockflow::ops::{
    from_set, CarryOp, CombineOp, Logic, LogicCombine, MapFn, MaskFn, NarrowOp, Threshold,
    ThresholdMask, ThresholdTest, VoxelwiseMapOp, VoxelwiseMaskOp, MAP_COST, MASK_COST,
};
use blockflow::region::Region;
use blockflow::strategy::{execute_phases, Hints};
use blockflow::voxels::Voxels;
use blockflow::Dtype;

const VOLUME: [usize; 3] = [20, 16, 14];

/// The fixture's values are eleven exact tenths, so a band `[a, b)` selects
/// whole level sets and a comparison at `0.5` has voxels sitting **exactly on
/// it**. A ramp of arbitrary reals would not: `>` and `>=` differ only on the
/// level itself, so a fixture that never lands there cannot tell the two tests
/// apart and would pass against either.
fn intensities() -> Array3<f64> {
    Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(i, j, k)| {
        ((i * 7 + j * 5 + k * 3) % 11) as f64 / 10.0
    })
}

/// The values a mask convention can be got wrong on, and the ones a ramp of
/// ordinary numbers would never visit.
const AWKWARD: [f64; 11] = [
    0.0,
    -0.0,
    0.5,
    f64::MIN_POSITIVE,
    -f64::MIN_POSITIVE,
    1.0,
    -1.0,
    f64::NAN,
    f64::INFINITY,
    f64::NEG_INFINITY,
    0.49999999999999994,
];

// ------------------------------------------------- one threshold, two carriers --

fn mapped(op: &impl BlockOp, values: &[f64]) -> Vec<f64> {
    let input: Voxels = Array3::from_shape_vec((values.len(), 1, 1), values.to_vec())
        .unwrap()
        .into();
    let mut out = Voxels::zeros(op.produces(Dtype::F64), input.shape()).unwrap();
    op.apply(&input, &mut out, &Anchor::whole(input.shape()))
        .expect("a voxelwise op over a contiguous block");
    match out.dtype() {
        Dtype::Bool => out
            .view::<bool>()
            .unwrap()
            .iter()
            .map(|&set| from_set(set))
            .collect(),
        _ => out.view::<f64>().unwrap().iter().copied().collect(),
    }
}

/// **The two carriers of one threshold agree on every voxel, bit for bit.**
///
/// Both tests, over every awkward value and over the fixture's own tenths, with
/// the `Bool` op's answer read back through `from_set` — which is
/// `VoxelElement::into_f64` for `bool` and is the same convention `NarrowOp`
/// uses for its `Bool` target. `to_bits` rather than `==` so that a `-0.0`
/// against a `0.0` is a failure and a `NaN` against a `NaN` is not.
///
/// It would pass against a broken implementation if the fixture could not tell
/// `>` from `>=`, so the last block is the negative control: the same program
/// with one thing changed — the test — must disagree, and the values it
/// disagrees on must be exactly the ones sitting on the level.
#[test]
fn the_two_carriers_of_one_threshold_agree_on_every_voxel() {
    let mut values: Vec<f64> = AWKWARD.to_vec();
    values.extend((0..=10).map(|n| n as f64 / 10.0));

    for level in [0.0, 0.5, 1.0] {
        for test in [ThresholdTest::Above, ThresholdTest::AtOrAbove] {
            let mask = ThresholdMask { level, test };
            let wide = VoxelwiseMapOp::from_map("wide", Threshold::from_mask(mask, 1.0, 0.0));
            let narrow = VoxelwiseMaskOp::from_mask("narrow", mask);

            assert_eq!(wide.produces(Dtype::F64), Dtype::F64);
            assert_eq!(narrow.produces(Dtype::F64), Dtype::Bool);

            let from_wide = mapped(&wide, &values);
            let from_narrow = mapped(&narrow, &values);
            for (index, (&w, &n)) in from_wide.iter().zip(from_narrow.iter()).enumerate() {
                assert_eq!(
                    w.to_bits(),
                    n.to_bits(),
                    "at {} (level {level}, {test:?}): the f64 carrier says {w} and the bool one \
                     says {n}",
                    values[index]
                );
            }

            // and the same answer again through the pair this replaces — the
            // wide threshold followed by `NarrowOp::to_mask` — which is what
            // says the new op is the *same* function and not merely a
            // self-consistent one.
            let two_pass: Vec<f64> = {
                let wide_block: Voxels =
                    Array3::from_shape_vec((values.len(), 1, 1), from_wide.clone())
                        .unwrap()
                        .into();
                let to_mask = NarrowOp::to_mask("mask");
                let mut out = Voxels::zeros(Dtype::Bool, wide_block.shape()).unwrap();
                to_mask
                    .apply(&wide_block, &mut out, &Anchor::whole(wide_block.shape()))
                    .unwrap();
                out.view::<bool>()
                    .unwrap()
                    .iter()
                    .map(|&set| from_set(set))
                    .collect()
            };
            assert_eq!(two_pass, from_narrow, "level {level}, {test:?}");
        }
    }

    // The negative control. The two tests are a real axis, not two spellings of
    // one comparison, and the fixture can see the difference: at `0.5` they
    // disagree on exactly the voxels holding `0.5`.
    let strict = VoxelwiseMaskOp::from_mask("strict", ThresholdMask::above(0.5));
    let loose = VoxelwiseMaskOp::from_mask("loose", ThresholdMask::at_or_above(0.5));
    let strict = mapped(&strict, &values);
    let loose = mapped(&loose, &values);
    assert_ne!(
        strict, loose,
        "a fixture that cannot tell `>` from `>=` proves nothing about either"
    );
    let differing: Vec<f64> = values
        .iter()
        .zip(strict.iter().zip(loose.iter()))
        .filter(|(_, (s, l))| s != l)
        .map(|(&value, _)| value)
        .collect();
    assert!(
        differing.iter().all(|&value| value == 0.5),
        "the two tests differ somewhere other than on the level: {differing:?}"
    );
    assert!(!differing.is_empty());
}

/// **One boundary, four paths.**
///
/// A threshold's comparison is reached four ways, and each of them is the one
/// that runs in a different situation: `MapFn::map` is the loop body inside a
/// `Compose`, `MapFn::map_slice` is what a contiguous `f64` block takes,
/// `MaskFn::holds` is the composed case for the narrow carrier, and
/// `MaskFn::holds_slice` is its contiguous one. All four go through
/// `ThresholdMask::holds_with`, which is the only place `>` and `>=` are
/// written — and this is what says so, rather than the reader having to check.
///
/// The values include the level itself and the `f64` immediately below it, so a
/// path that had drifted by one comparison would be caught here rather than at
/// whichever call site happened to use it.
#[test]
fn every_path_to_a_thresholds_comparison_gives_the_same_answer() {
    let mut values: Vec<f64> = AWKWARD.to_vec();
    values.extend((0..=10).map(|n| n as f64 / 10.0));
    values.push(0.5000000000000001);

    for level in [0.0, 0.5, 1.0, f64::INFINITY] {
        for test in [ThresholdTest::Above, ThresholdTest::AtOrAbove] {
            let mask = ThresholdMask { level, test };
            let wide = Threshold::from_mask(mask, 1.0, 0.0);
            assert_eq!(wide.mask(), mask, "the round trip is a round trip");

            let mut wide_slice = vec![0.0; values.len()];
            wide.map_slice(&values, &mut wide_slice);
            let mut narrow_slice = vec![false; values.len()];
            mask.holds_slice(&values, &mut narrow_slice);

            for (index, &value) in values.iter().enumerate() {
                let want = mask.holds(value);
                assert_eq!(narrow_slice[index], want, "holds_slice at {value}");
                assert_eq!(
                    wide.map(value).to_bits(),
                    from_set(want).to_bits(),
                    "map at {value} (level {level}, {test:?})"
                );
                assert_eq!(
                    wide_slice[index].to_bits(),
                    from_set(want).to_bits(),
                    "map_slice at {value} (level {level}, {test:?})"
                );
            }
        }
    }

    // The control: the four paths agreeing would be worthless if the predicate
    // they agree on were constant. At `0.5` over these values it is neither
    // always true nor always false, and the strict and non-strict forms differ.
    let mask = ThresholdMask::above(0.5);
    let set = values.iter().filter(|&&value| mask.holds(value)).count();
    assert!(set > 0 && set < values.len(), "{set} of {}", values.len());
}

/// The narrow op writes an eighth of the bytes, and declares the same constant.
///
/// The second half is the part that would rot silently: `constant_maps_to` is
/// what the short circuit substitutes for computing a block, so a mask op whose
/// declaration disagreed with its kernel would produce different voxels for a
/// uniform block than for a block that merely happened to be uniform.
#[test]
fn a_mask_op_writes_a_bool_image_and_declares_what_it_would_have_computed() {
    let narrow = VoxelwiseMaskOp::threshold("narrow", 0.5);
    assert!(narrow.accepts(Dtype::F64));
    assert!(
        !narrow.accepts(Dtype::U16),
        "the way in is `WidenOp`, as it is for `NarrowOp`"
    );
    assert_eq!(narrow.produces(Dtype::U16), Dtype::Bool);
    assert_eq!(narrow.reach(0, 64), 0);

    let wide = VoxelwiseMapOp::threshold("wide", 0.5, 1.0, 0.0);
    let block: Voxels = intensities().into();
    let mut narrow_out = Voxels::zeros(Dtype::Bool, VOLUME).unwrap();
    let mut wide_out = Voxels::zeros(Dtype::F64, VOLUME).unwrap();
    narrow
        .apply(&block, &mut narrow_out, &Anchor::whole(VOLUME))
        .unwrap();
    wide.apply(&block, &mut wide_out, &Anchor::whole(VOLUME))
        .unwrap();
    assert_eq!(wide_out.bytes(), narrow_out.bytes() * 8);

    for &value in AWKWARD.iter() {
        assert_eq!(
            narrow.constant_maps_to(value).map(f64::to_bits),
            wide.constant_maps_to(value).map(f64::to_bits),
            "at {value}"
        );
        // and the declaration is what computing it gives
        let uniform = Voxels::filled(Dtype::F64, [2, 2, 2], value).unwrap();
        let mut out = Voxels::zeros(Dtype::Bool, [2, 2, 2]).unwrap();
        narrow
            .apply(&uniform, &mut out, &Anchor::whole([2, 2, 2]))
            .unwrap();
        let computed = from_set(out.view::<bool>().unwrap()[[0, 0, 0]]);
        assert_eq!(
            computed.to_bits(),
            narrow.constant_maps_to(value).unwrap().to_bits(),
            "at {value}"
        );
    }

    // The narrow carrier is cheaper to compute as well as to hold, and the
    // planner is told so: the comparison is one instruction either way and the
    // store is an eighth of the bytes. `MASK_COST`'s own documentation is the
    // measurement; this is the ordering, which is the part a planner uses.
    assert!(MASK_COST < MAP_COST);
    assert!(narrow.cost_per_voxel() < wide.cost_per_voxel());
    assert_eq!(narrow.cost_per_voxel(), MASK_COST);
    // and `with_cost` still overrides, for a predicate that cannot price itself
    assert_eq!(
        VoxelwiseMaskOp::threshold("t", 0.5)
            .with_cost(40.0)
            .cost_per_voxel(),
        40.0
    );

    // A closure is a `MaskFn` too, and gets the same shell.
    let band = VoxelwiseMaskOp::new("band", |value: f64| (0.25..0.35).contains(&value));
    assert_eq!(band.produces(Dtype::F64), Dtype::Bool);
    assert!(band.mask().holds(0.3) && !band.mask().holds(0.4));
}

// --------------------------------------------------- the connective, mixed --

/// A block of `shape` holding `set` in `carrier`.
fn mask_block(carrier: Dtype, values: &[bool]) -> Voxels {
    let shape = (values.len(), 1, 1);
    match carrier {
        Dtype::Bool => Array3::from_shape_vec(shape, values.to_vec())
            .unwrap()
            .into(),
        _ => Array3::from_shape_vec(shape, values.iter().map(|&set| from_set(set)).collect())
            .unwrap()
            .into(),
    }
}

fn joined(logic: Logic, carriers: &[Dtype], output: Dtype, operands: &[Vec<bool>]) -> Vec<bool> {
    let combine = LogicCombine::new("join", logic)
        .producing(output)
        .expect("a mask carrier");
    let inputs: Vec<Voxels> = carriers
        .iter()
        .zip(operands.iter())
        .map(|(&carrier, values)| mask_block(carrier, values))
        .collect();
    let shape = inputs[0].shape();
    assert!(combine.accepts(&carriers.to_vec()));
    assert_eq!(combine.produces(carriers), output);
    let mut out = Voxels::zeros(output, shape).unwrap();
    combine
        .apply(&inputs.iter().collect::<Vec<_>>(), &mut out, &Anchor::whole(shape))
        .expect("a join over mask carriers");
    match output {
        Dtype::Bool => out.view::<bool>().unwrap().iter().copied().collect(),
        _ => out
            .view::<f64>()
            .unwrap()
            .iter()
            .map(|&value| {
                assert!(value == 0.0 || value == 1.0, "a mask carrier holds {value}");
                value != 0.0
            })
            .collect(),
    }
}

/// **The answer is the connective's, whatever the branches and the sink are
/// carrying it in.**
///
/// Every assignment of carriers to two and to three branches, against every
/// carrier of the output, over every connective — and all sixteen answers are
/// one answer. The negative control is at the end: change the connective and
/// nothing else, and the answer must move, so this is not sixteen ways of
/// computing a constant.
#[test]
fn a_connective_joins_any_carriers_and_the_answer_does_not_depend_on_them() {
    let a: Vec<bool> = (0..8).map(|n| n & 1 == 1).collect();
    let b: Vec<bool> = (0..8).map(|n| n & 2 == 2).collect();
    let c: Vec<bool> = (0..8).map(|n| n & 4 == 4).collect();

    for logic in [Logic::And, Logic::Or, Logic::Xor] {
        let want_two: Vec<bool> = a
            .iter()
            .zip(b.iter())
            .map(|(&l, &r)| logic.apply(l, r))
            .collect();
        let want_three: Vec<bool> = want_two
            .iter()
            .zip(c.iter())
            .map(|(&l, &r)| logic.apply(l, r))
            .collect();

        for &left in &[Dtype::Bool, Dtype::F64] {
            for &right in &[Dtype::Bool, Dtype::F64] {
                for &output in &[Dtype::Bool, Dtype::F64] {
                    assert_eq!(
                        joined(logic, &[left, right], output, &[a.clone(), b.clone()]),
                        want_two,
                        "{logic:?} over {left:?}/{right:?} into {output:?}"
                    );
                    for &third in &[Dtype::Bool, Dtype::F64] {
                        assert_eq!(
                            joined(
                                logic,
                                &[left, right, third],
                                output,
                                &[a.clone(), b.clone(), c.clone()]
                            ),
                            want_three,
                            "{logic:?} over {left:?}/{right:?}/{third:?} into {output:?}"
                        );
                    }
                }
            }
        }
    }

    // The control: one thing changed, and the answer moves. Without this the
    // block above would pass against a join that ignored its operands.
    assert_ne!(
        joined(
            Logic::And,
            &[Dtype::Bool, Dtype::F64],
            Dtype::Bool,
            &[a.clone(), b.clone()]
        ),
        joined(
            Logic::Or,
            &[Dtype::Bool, Dtype::F64],
            Dtype::Bool,
            &[a.clone(), b.clone()]
        )
    );
}

/// The type rule, stated as a table — and the refusal that **survived** the
/// relaxation.
///
/// `tests/fan_in.rs`'s `a_combine_that_cannot_join_its_branches_is_refused_
/// before_anything_runs` and `op.rs`'s `a_combine_that_cannot_accept_a_branchs_
/// dtype_is_refused` both assert that a `float64` branch beside a `bool` one is
/// refused. Both still pass, unchanged, because both build the combine with
/// `LogicCombine::new` and never say what to write. That is the assertion
/// inverted rather than deleted: what landed is the *stated* case, and the
/// unstated one is still an absence.
#[test]
fn a_mixed_fan_in_is_still_refused_when_nobody_says_what_to_write() {
    let inferred = LogicCombine::new("or", Logic::Or);
    assert!(inferred.accepts(&[Dtype::Bool, Dtype::Bool]));
    assert!(inferred.accepts(&[Dtype::F64, Dtype::F64]));
    assert!(!inferred.accepts(&[Dtype::Bool, Dtype::F64]));
    assert!(!inferred.accepts(&[Dtype::F64, Dtype::Bool]));
    assert!(!inferred.accepts(&[Dtype::F64, Dtype::F64, Dtype::Bool]));
    assert!(!inferred.accepts(&[Dtype::F64]), "a join needs two");
    assert!(
        !inferred.accepts(&[Dtype::U16, Dtype::U16]),
        "not a carrier"
    );
    assert_eq!(inferred.produces(&[Dtype::Bool, Dtype::Bool]), Dtype::Bool);
    assert_eq!(inferred.produces(&[Dtype::F64, Dtype::F64]), Dtype::F64);

    let stated = LogicCombine::new("or", Logic::Or)
        .producing(Dtype::Bool)
        .unwrap();
    assert!(stated.accepts(&[Dtype::Bool, Dtype::F64]));
    assert!(stated.accepts(&[Dtype::F64, Dtype::Bool, Dtype::F64]));
    assert!(stated.accepts(&[Dtype::F64, Dtype::F64]));
    assert_eq!(stated.produces(&[Dtype::F64, Dtype::F64]), Dtype::Bool);
    assert!(
        !stated.accepts(&[Dtype::U16, Dtype::Bool]),
        "stating the output does not make a non-carrier joinable"
    );
    assert!(!stated.accepts(&[Dtype::Bool]), "a join still needs two");

    // A carrier this crate has stated no convention for cannot be the output.
    let refusal = match LogicCombine::new("or", Logic::Or).producing(Dtype::U8) {
        Ok(_) => panic!("uint8 was accepted as a mask carrier"),
        Err(err) => err.to_string(),
    };
    assert!(refusal.contains("bool and float64"), "got: {refusal}");

    // and the refusal reaches the plan, in the words `Chain::produces` uses.
    let mixed = || {
        vec![
            Chain::op(VoxelwiseMaskOp::threshold("narrow", 0.5)),
            Chain::op(VoxelwiseMapOp::threshold("wide", 0.5, 1.0, 0.0)),
        ]
    };
    let err = Chain::parallel(mixed(), Box::new(LogicCombine::new("or", Logic::Or)))
        .unwrap()
        .produces(Dtype::F64)
        .expect_err("a combine that was not told what to write must refuse")
        .to_string();
    assert!(
        err.contains("does not accept [bool, float64]"),
        "got: {err}"
    );

    let produced = Chain::parallel(
        mixed(),
        Box::new(
            LogicCombine::new("or", Logic::Or)
                .producing(Dtype::Bool)
                .unwrap(),
        ),
    )
    .unwrap()
    .produces(Dtype::F64)
    .expect("a combine that was told what to write must accept");
    assert_eq!(produced, Dtype::Bool);
}

// ------------------------------------------------- the held operand, mixed --

/// **`CombineOp` joins the two carriers too, and does not gain a stated
/// output.**
///
/// The held-operand form had the same rule `LogicCombine` had — its block's
/// element type must equal the operand's — for the same reason, which is to say
/// for none: `apply` read one tag and used it for all three buffers. The
/// connective is a function of two bits either way.
///
/// **The output is where the two ops genuinely differ, and it is why only one of
/// them takes `producing`.** `LogicCombine` has *n* branches and no canonical
/// one, so a carrier taken from a branch would make an image's width a
/// consequence of an arm's. This op has exactly one input and `BlockOp::produces`
/// hands it back — so the assertion below is that the output carrier is the
/// *input's*, unchanged, for every pair. Relaxing the input rule moved that
/// towards the plan, not away from it: the width of what it writes no longer
/// depends on an array the plan cannot see.
///
/// Blocked as well as whole, because this op is the one voxelwise op with a
/// position: it slices its operand at the anchor, so a crossed carrier that
/// sliced differently would show up here and nowhere else.
#[test]
fn the_held_operand_need_not_be_carried_the_way_the_block_is() {
    const VOL: [usize; 3] = [12, 8, 6];
    let count = VOL[0] * VOL[1] * VOL[2];
    // Coprime strides, so no pair of the three connectives agrees everywhere and
    // the operand is not a symmetry of the block.
    let block_bits: Vec<bool> = (0..count).map(|index| index % 3 == 0).collect();
    let operand_bits: Vec<bool> = (0..count).map(|index| index % 5 < 2).collect();

    let carried = |bits: &[bool], carrier: Dtype| -> Voxels {
        let shape = (VOL[0], VOL[1], VOL[2]);
        match carrier {
            Dtype::Bool => Array3::from_shape_vec(shape, bits.to_vec()).unwrap().into(),
            _ => Array3::from_shape_vec(shape, bits.iter().map(|&set| from_set(set)).collect())
                .unwrap()
                .into(),
        }
    };
    let read = |voxels: &Voxels| -> Vec<bool> {
        match voxels.dtype() {
            Dtype::Bool => voxels.view::<bool>().unwrap().iter().copied().collect(),
            _ => voxels
                .view::<f64>()
                .unwrap()
                .iter()
                .map(|&value| value != 0.0)
                .collect(),
        }
    };

    for logic in [Logic::And, Logic::Or, Logic::Xor] {
        let want: Vec<bool> = block_bits
            .iter()
            .zip(operand_bits.iter())
            .map(|(&left, &right)| logic.apply(left, right))
            .collect();
        for &block in &[Dtype::Bool, Dtype::F64] {
            for &held in &[Dtype::Bool, Dtype::F64] {
                let op = CombineOp::new("join", logic, Arc::new(carried(&operand_bits, held)));
                assert!(op.accepts(block), "{block:?} block, {held:?} operand");
                // The half that did **not** change: what it writes is what it
                // read, whatever the operand is carried in.
                assert_eq!(op.produces(block), block);

                let input = carried(&block_bits, block);
                let mut whole = Voxels::zeros(block, VOL).unwrap();
                op.apply(&input, &mut whole, &Anchor::whole(VOL)).unwrap();
                assert_eq!(read(&whole), want, "{logic:?} {block:?}/{held:?} whole");

                // The same answer a block at a time, at every anchor. Uneven
                // steps so the last block on each axis is a short one.
                let mut assembled = Voxels::zeros(block, VOL).unwrap();
                for i in (0..VOL[0]).step_by(5) {
                    for j in (0..VOL[1]).step_by(3) {
                        for k in (0..VOL[2]).step_by(4) {
                            let shape = [
                                (VOL[0] - i).min(5),
                                (VOL[1] - j).min(3),
                                (VOL[2] - k).min(4),
                            ];
                            let region = Region::new(&[i, j, k], &shape);
                            let piece = input.slice_region(&region).unwrap();
                            let mut out = Voxels::zeros(block, shape).unwrap();
                            op.apply(&piece, &mut out, &Anchor::new([i, j, k], VOL))
                                .unwrap();
                            assembled.assign_region(&region, &out).unwrap();
                        }
                    }
                }
                assert_eq!(
                    read(&assembled),
                    want,
                    "{logic:?} {block:?}/{held:?} blocked"
                );
            }
        }
    }

    // The control, on a crossed pair specifically: change the connective and
    // nothing else, and the answer moves. Without it the sweep above would pass
    // against a join that returned its block untouched — which is exactly what a
    // crossed pair read through the wrong conversion could look like.
    let held = Arc::new(carried(&operand_bits, Dtype::F64));
    let input = carried(&block_bits, Dtype::Bool);
    let mut and = Voxels::zeros(Dtype::Bool, VOL).unwrap();
    let mut or = Voxels::zeros(Dtype::Bool, VOL).unwrap();
    CombineOp::new("and", Logic::And, Arc::clone(&held))
        .apply(&input, &mut and, &Anchor::whole(VOL))
        .unwrap();
    CombineOp::new("or", Logic::Or, held)
        .apply(&input, &mut or, &Anchor::whole(VOL))
        .unwrap();
    assert_ne!(read(&and), read(&or));
    assert_ne!(
        read(&and),
        block_bits,
        "the AND returned its block untouched"
    );

    // An operand that is no carrier at all is refused at every input, when the
    // plan is made. The equality rule used to get this for free.
    let strange = CombineOp::new(
        "strange",
        Logic::And,
        Arc::new(Voxels::zeros(Dtype::U16, VOL).unwrap()),
    );
    assert!(!strange.accepts(Dtype::Bool) && !strange.accepts(Dtype::F64));
    let err = Chain::op(strange)
        .produces(Dtype::F64)
        .expect_err("an operand of a third type makes the op unusable")
        .to_string();
    assert!(err.contains("does not accept float64"), "got: {err}");
}

// ---------------------------------------------------- the sink, both ways --

/// The shape a consumer's segmentation actually has: a seeding predicate, then
/// one arm per criterion, each arm's verdict OR-ed into a sink the next arm
/// carries forward.
///
/// The arms select **disjoint** level sets of the fixture, so every one of them
/// contributes voxels no other does and dropping any changes the answer. A chain
/// of nested thresholds would not have that property — the lowest would subsume
/// the rest — and a liveness control over it would assert nothing.
fn sink_plan(carrier: Dtype, block: [usize; 3]) -> Assembly {
    let grid = BlockGrid::new(VOLUME, block).expect("a lattice");
    let mut plan = PlanBuilder::new(VOLUME, Dtype::F64, grid);

    // The seed: a band, in the carrier the sink is to be held in.
    let band = |low: f64, high: f64| move |value: f64| value >= low && value < high;
    let seed = match carrier {
        Dtype::Bool => Chain::op(VoxelwiseMaskOp::new("seed", band(0.25, 0.35))),
        _ => {
            let holds = band(0.25, 0.35);
            Chain::op(VoxelwiseMapOp::new("seed", move |value| {
                from_set(holds(value))
            }))
        }
    };
    let mut sink = plan.pixels(seed).expect("the seed");

    // Arm 1's verdict is `Bool`; arms 2 and 3's are `f64`. One of the two is
    // always the carrier the sink is *not* in, whichever carrier that is, so
    // both plans exercise a mixed join and they exercise it in opposite
    // directions.
    let arms: Vec<Chain> = vec![
        Chain::op(VoxelwiseMaskOp::at_or_above("arm 1", 0.9)),
        Chain::op(VoxelwiseMapOp::new("arm 2", |value| {
            from_set((0.45..0.55).contains(&value))
        })),
        Chain::op(VoxelwiseMapOp::new("arm 3", |value| {
            from_set((0.65..0.75).contains(&value))
        })),
    ];
    for arm in arms {
        let join = Chain::parallel(
            vec![
                Chain::op(CarryOp::new("sink so far")),
                Chain::sequence(vec![Chain::source(0usize, Dtype::F64), arm]),
            ],
            Box::new(
                LogicCombine::new("or", Logic::Or)
                    .producing(carrier)
                    .expect("a mask carrier"),
            ),
        )
        .expect("a fan-in");
        sink = plan.pixels(join).expect("an arm");
    }
    assert_eq!(sink.index(), 3);
    plan.finish().expect("a plan whose types check")
}

/// Run one, and read the sink back as bits whatever it was carried in.
fn run(assembly: &Assembly, input: &Array3<f64>) -> Array3<bool> {
    let env = ArrayEnvironment::for_decomposition(
        input.clone().into(),
        &assembly.decomposition,
        [4, 4, 4],
    )
    .expect("an environment typed by the plan");
    let listeners: Vec<Arc<dyn blockflow::listener::EventListener>> = Vec::new();
    execute_phases(
        "mask carrier",
        &assembly.workflow,
        &assembly.decomposition,
        &Hints::default(),
        &env,
        &listeners,
        &assembly.work(),
    )
    .expect("the run");
    let out = env.output();
    match out.dtype() {
        Dtype::Bool => out.view::<bool>().unwrap().to_owned(),
        _ => out.view::<f64>().unwrap().map(|&value| {
            assert!(value == 0.0 || value == 1.0, "the sink holds {value}");
            value != 0.0
        }),
    }
}

/// **Not one voxel moves**, across the carrier and across the block size.
///
/// Three claims in one run, because they are the same run: the `Bool` sink and
/// the `f64` sink agree exactly; each is byte-identical at every block size to
/// itself at one block; and the answer is not trivial, so the agreement is an
/// agreement about something.
///
/// What is claimed is **agreement, not parity**. The fixture is invented here and
/// there is no recorded answer for it anywhere, so the evidence supports "two
/// ways of computing this produce the same bytes, at every block size" and does
/// not support "these are the right bytes". Parity is a claim about a digest
/// somebody else recorded, and this file has none.
#[test]
fn the_sinks_two_carriers_agree_and_neither_depends_on_the_block_size() {
    let input = intensities();
    let whole = VOLUME;
    let blocks: [[usize; 3]; 4] = [whole, [8, 8, 8], [7, 5, 4], [20, 4, 14]];

    let mut references = Vec::new();
    for carrier in [Dtype::F64, Dtype::Bool] {
        let reference = run(&sink_plan(carrier, whole), &input);
        let set = reference.iter().filter(|&&bit| bit).count();
        assert!(
            set > 0 && set < reference.len(),
            "a sink that is all one value cannot distinguish anything: {set} of {}",
            reference.len()
        );
        for block in blocks {
            let got = run(&sink_plan(carrier, block), &input);
            assert_eq!(
                got, reference,
                "{carrier:?} sink: block {block:?} disagrees with one block"
            );
        }
        references.push(reference);
    }
    assert_eq!(
        references[0], references[1],
        "the f64 sink and the bool sink are not the same voxels"
    );

    // The liveness control: the same program with one arm dropped must give a
    // different answer, in both carriers. Without it, the agreement above would
    // be consistent with a chain whose arms never reached the sink at all.
    for carrier in [Dtype::F64, Dtype::Bool] {
        let grid = BlockGrid::new(VOLUME, whole).unwrap();
        let mut plan = PlanBuilder::new(VOLUME, Dtype::F64, grid);
        let holds = |value: f64| (0.25..0.35).contains(&value);
        let seed = match carrier {
            Dtype::Bool => Chain::op(VoxelwiseMaskOp::new("seed", holds)),
            _ => Chain::op(VoxelwiseMapOp::new("seed", move |value| {
                from_set(holds(value))
            })),
        };
        plan.pixels(seed).unwrap();
        let assembly = plan.finish().unwrap();
        let seed_only = run(&assembly, &input);
        assert_ne!(
            seed_only, references[0],
            "{carrier:?}: the seed alone equals the whole chain, so no arm contributed"
        );
    }
}

/// The `Bool` sink is a quarter of the plan's bytes and the `f64` one is more
/// than half — the same voxels, priced.
///
/// This is the plan-side figure only: `Decomposition::dtype_at` times the
/// volume, image by image. What a run *holds* is measured under an allocator in
/// `tests/mask_carrier_residency.rs`, because ops' own working buffers belong to
/// no image and no decomposition can see them.
#[test]
fn the_sinks_carrier_is_worth_seven_eighths_of_every_image_below_it() {
    let voxels: u64 = VOLUME.iter().product::<usize>() as u64;
    let priced = |carrier: Dtype| -> u64 {
        let assembly = sink_plan(carrier, VOLUME);
        let plan = &assembly.decomposition;
        (0..plan.n_images())
            .map(|image| {
                plan.volume_at(image).iter().product::<usize>() as u64
                    * plan.dtype_at(image).size_of() as u64
            })
            .sum()
    };
    let wide = priced(Dtype::F64);
    let narrow = priced(Dtype::Bool);
    // Image 0 is the `f64` source and is the same in both; the four the chain
    // writes are the seed and one per arm.
    assert_eq!(wide, voxels * 8 * 5);
    assert_eq!(narrow, voxels * 8 + voxels * 4);
    assert_eq!(wide - narrow, voxels * 7 * 4);
    eprintln!(
        "priced images over {VOLUME:?}: f64 sink {wide} bytes, bool sink {narrow} bytes, \
         saving {:.1}%",
        100.0 * (wide - narrow) as f64 / wide as f64
    );
}

// --------------------------------------------- what the refactor cost, A/B --

/// The comparison as it was written before [`ThresholdMask`] owned it: the
/// branchless expression inline, reading `Threshold`'s own fields.
///
/// Kept here rather than deleted from the crate, because "routing four paths
/// through one function did not cost anything" is a claim about generated code
/// and the only honest evidence for it is the two forms timed against each other
/// in one process on one machine. A figure quoted from a different run of a
/// loaded machine is not that.
#[derive(Debug, Clone, Copy)]
struct InlineThreshold {
    level: f64,
    test: ThresholdTest,
    above: f64,
    below: f64,
}

impl MapFn for InlineThreshold {
    fn map(&self, value: f64) -> f64 {
        let at_or_above = matches!(self.test, ThresholdTest::AtOrAbove);
        let set = (value > self.level) | (at_or_above & (value >= self.level));
        if set {
            self.above
        } else {
            self.below
        }
    }
}

/// The two forms of one comparison, four deep, alternated.
///
/// ```text
/// cargo test --release --test mask_carrier -- --ignored --nocapture
/// ```
///
/// Ignored because timing in a test suite is a measurement of the machine's
/// mood — this one was taken at a load average of 27, where the crate's own
/// recorded 3.15 ns/voxel for the composed threshold reads as 3.6 to 4.3. That
/// is exactly why the two are timed **beside each other, alternating**, rather
/// than against a number written down on a quieter day: the ratio survives the
/// load even though neither figure does.
///
/// What it said: **1.00x** — 3.66 ns/voxel through the type against 3.68
/// written inline, best of 40 alternating repetitions at `96 x 64 x 64`.
/// Routing the comparison through
/// `ThresholdMask::holds_with` costs nothing, which is what `#[inline(always)]`
/// on a method monomorphised over a `const bool` is for.
#[test]
#[ignore = "a measurement, not an assertion"]
fn routing_the_comparison_through_one_function_costs_nothing() {
    use std::time::Instant;

    const SHAPE: [usize; 3] = [96, 64, 64];
    let voxels = (SHAPE[0] * SHAPE[1] * SHAPE[2]) as f64;
    let mut source = Array3::<f64>::zeros((SHAPE[0], SHAPE[1], SHAPE[2]));
    for (flat, value) in source.iter_mut().enumerate() {
        *value = ((flat * 7919) % 1013) as f64;
    }
    let input: Voxels = source.into();
    let anchor = Anchor::whole(SHAPE);

    let pair = |low: f64, high: f64| {
        (
            Threshold::above(low, 1.0, 0.0),
            Threshold::above(high, 700.0, 300.0),
        )
    };
    let (a, b) = pair(500.0, 0.5);
    let owned = blockflow::ops::Compose::new(
        blockflow::ops::Compose::new(a, b),
        blockflow::ops::Compose::new(a, b),
    );
    let inline_of = |t: Threshold| InlineThreshold {
        level: t.level,
        test: t.test,
        above: t.above,
        below: t.below,
    };
    let (ia, ib) = (inline_of(a), inline_of(b));
    let inline = blockflow::ops::Compose::new(
        blockflow::ops::Compose::new(ia, ib),
        blockflow::ops::Compose::new(ia, ib),
    );

    let ops: [(&str, Box<dyn BlockOp>); 2] = [
        (
            "through ThresholdMask",
            Box::new(VoxelwiseMapOp::from_map("owned", owned)),
        ),
        (
            "written inline",
            Box::new(VoxelwiseMapOp::from_map("inline", inline)),
        ),
    ];
    let mut out = [
        Voxels::zeros(Dtype::F64, SHAPE).unwrap(),
        Voxels::zeros(Dtype::F64, SHAPE).unwrap(),
    ];
    let mut best = [f64::INFINITY; 2];
    for (index, (_, op)) in ops.iter().enumerate() {
        op.apply(&input, &mut out[index], &anchor).unwrap();
    }
    // Alternating, so a machine that gets busier partway through gets busier for
    // both. Interleaved rather than one loop after the other for the same
    // reason.
    for _ in 0..40 {
        for (index, (_, op)) in ops.iter().enumerate() {
            let started = Instant::now();
            op.apply(&input, &mut out[index], &anchor).unwrap();
            let nanos = started.elapsed().as_secs_f64() * 1e9 / voxels;
            std::hint::black_box(out[index].view::<f64>().unwrap()[[0, 0, 0]]);
            if nanos.total_cmp(&best[index]).is_lt() {
                best[index] = nanos;
            }
        }
    }
    // and the two agree on every voxel, which is what makes the timing a timing
    // of the same function.
    assert_eq!(out[0], out[1], "the two forms are not the same map");
    eprintln!(
        "\nfour composed thresholds, {SHAPE:?}, best of 40 alternating\n  {:<22} {:.2} \
         ns/voxel\n  {:<22} {:.2} ns/voxel  ({:.2}x)",
        ops[0].0,
        best[0],
        ops[1].0,
        best[1],
        best[0] / best[1]
    );
}
