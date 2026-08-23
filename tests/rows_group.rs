// SPDX-License-Identifier: MIT
//
// **The grouped reduction over rows**: `ops::rows`' fourth op and the first that
// is not a map.
//
// Why this file exists is a gap a consumer found. A grouped statistic —
// thousands of scattered rows reduced to one row per distinct value of a
// two-column key, with a count, a mean and two `first`s per group — had to carry
// **its own fold and its own block partitioning**, because `ops::rows` offered
// only `row -> Option<row>` and `ops::tabulate` grouped by a label read out of a
// *volume* rather than by a value the row already held. Its decomposition
// invariance was therefore *asserted* at six cuts by hand rather than **declared
// by a plan**, and nothing in the framework could cost, cache or fuse it.
// `ops::rows::GroupRowsOp` and `MergeGroupsOp` are that gap closed, and this
// file is what says they closed it.
//
// What each test is for
// ---------------------
// | | |
// |---|---|
// | ground truth | the fixture's answer written out by hand, not recomputed |
// | the two `First`s | the one place two references in the field disagree, on a fixture where they differ |
// | split invariance | the same rows cut every way give one map, and byte-identical blobs |
// | `Unordered`, honestly | every permutation of the partials folds to one map |
// | the duplication refusal | two partials claiming one least row is refused rather than folded |
// | the overflow refusal | a total past a signed 64-bit column is refused **by name**, with what `as i64` would have returned measured beside it |
// | ownership | exactly one block emits each group, and the union is the table once |
// | through a plan | the two phases executed at six cuts, byte-identical every time |
//
// The fixture, and what it is built to separate
// ---------------------------------------------
// Eight rows on a line, two groups interleaved, so **every cut of axis 0 splits
// both groups** and no cut splits them the same way. Two value columns, each
// with its own presence mask, absent at positions chosen so that:
//
// * `count` differs from `rows` in both groups — three of four and two of four;
// * `first_present` and `first_row` **differ**, in group 1 for `score` and in
//   group 2 for `mark`, so neither reading can pass for the other;
// * in group 1 the two columns' `first_present` come from **two different rows**
//   — `mark` from the row at 0 and `score` from the row at 2 — which is the
//   per-column property, and the thing a whole-row rule cannot produce;
// * `min` and `max` are not the first or the last row in either group, so a
//   selection that quietly took an end would be caught.
//
// An absent row's value column holds [`FILLER`], `-99.0`, and it is a number
// rather than a zero on purpose: it is smaller than every real value and larger
// than none, so a fold that ignored the mask would report it as the minimum, and
// [`Aggregate::FirstRow`] reports it *by design* — which is what makes the two
// `First`s visibly different answers rather than two spellings of one.
//
// Every real value is dyadic, so the fixed-point total is exact and the sums
// below can be written as themselves rather than as a tolerance.

use std::collections::BTreeMap;
use std::sync::Arc;

use blockflow::assemble::PlanBuilder;
use blockflow::decomposition::Decomposition;
use blockflow::dtype::Dtype;
use blockflow::env::{ArrayEnvironment, Environment};
use blockflow::error::Result;
use blockflow::fragment::fold_fragments;
use blockflow::geometry::BlockGrid;
use blockflow::ops::detect::owner_of;
use blockflow::ops::rows::{
    append_group_phases, collect_groups, decode_groups, encode_groups, group_values, Aggregate,
    GroupFold, GroupRowsOp, GroupValues, Grouping, MergeGroupsOp, Reduction, RowSourceOp,
    RowStreams, RowValues, GROUP_ROWS, MAX_PACKED_COORDINATE,
};
use blockflow::ops::tabulate::FixedPoint;
use blockflow::region::Region;
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints};
use blockflow::table::{Column, ColumnType, RowBuilder, Schema, Table, Value};
use blockflow::voxels::Voxels;

const VOLUME: [usize; 3] = [8, 4, 4];
const ROWS: &str = "rows.in";
const PARTIALS: &str = "rows.partials";
const GROUPED: &str = "rows.grouped";

// ------------------------------------------------------------- the fixture --

/// What an absent row holds in a value column. Nothing reads it except
/// [`Aggregate::FirstRow`], which is supposed to.
const FILLER: f64 = -99.0;

/// `key`, then each value column beside the `U64` mask that says which rows have
/// a value in it.
fn schema() -> Schema {
    Schema::new(vec![
        Column::u64("key"),
        Column::f64("score"),
        Column::u64("has_score"),
        Column::f64("mark"),
        Column::u64("has_mark"),
    ])
    .expect("five distinct names")
}

const KEY: usize = 0;
const SCORE: usize = 1;
const HAS_SCORE: usize = 2;
const MARK: usize = 3;
const HAS_MARK: usize = 4;

/// The eight rows, in the order the header describes them. `None` is an absent
/// value: the column holds [`FILLER`] and its mask holds zero.
fn fixture() -> Vec<([usize; 3], Vec<Value>)> {
    let row = |x: usize, key: u64, score: Option<f64>, mark: Option<f64>| {
        let held = |value: Option<f64>| {
            vec![
                Value::F64(value.unwrap_or(FILLER)),
                Value::U64(u64::from(value.is_some())),
            ]
        };
        let mut values = vec![Value::U64(key)];
        values.extend(held(score));
        values.extend(held(mark));
        ([x, 0, 0], values)
    };
    vec![
        row(0, 1, None, Some(5.0)),
        row(1, 2, Some(4.0), None),
        row(2, 1, Some(2.0), None),
        row(3, 2, Some(-1.0), Some(7.0)),
        row(4, 1, Some(8.0), Some(1.0)),
        row(5, 2, Some(0.5), None),
        row(6, 1, Some(-3.0), None),
        row(7, 2, None, Some(2.0)),
    ]
}

fn blob(rows: &[([usize; 3], Vec<Value>)]) -> Vec<u8> {
    let mut builder = RowBuilder::new(Arc::new(schema()));
    for (at, values) in rows {
        builder.push(*at, values).expect("the fixture matches");
    }
    builder.encode()
}

/// The grouping this file is about: keyed on `key`, with every aggregate the op
/// offers taken over one column or the other, so no test has to build a second
/// one to reach a variant.
fn grouping(fixed: FixedPoint) -> Grouping {
    Grouping::new(
        schema(),
        vec![KEY],
        vec![
            Reduction::masked(SCORE, Aggregate::Count, HAS_SCORE),
            Reduction::masked(SCORE, Aggregate::Sum, HAS_SCORE),
            Reduction::masked(SCORE, Aggregate::Min, HAS_SCORE),
            Reduction::masked(SCORE, Aggregate::Max, HAS_SCORE),
            Reduction::masked(SCORE, Aggregate::FirstPresent, HAS_SCORE),
            // No mask: `FirstRow` is about the row and not about presence, and
            // giving it one would suggest otherwise.
            Reduction::new(SCORE, Aggregate::FirstRow),
            Reduction::masked(MARK, Aggregate::Count, HAS_MARK),
            Reduction::masked(MARK, Aggregate::Sum, HAS_MARK),
            Reduction::masked(MARK, Aggregate::FirstPresent, HAS_MARK),
            Reduction::new(MARK, Aggregate::FirstRow),
        ],
        fixed,
    )
    .expect("the grouping is well formed")
}

/// Column indexes into the output, by the order [`grouping`] declares.
const SCORE_COUNT: usize = 0;
const SCORE_SUM: usize = 1;
const SCORE_MIN: usize = 2;
const SCORE_MAX: usize = 3;
const SCORE_FIRST_PRESENT: usize = 4;
const SCORE_FIRST_ROW: usize = 5;
const MARK_COUNT: usize = 6;
const MARK_SUM: usize = 7;
const MARK_FIRST_PRESENT: usize = 8;
const MARK_FIRST_ROW: usize = 9;

fn f64_of(value: Value) -> f64 {
    match value {
        Value::F64(number) => number,
        Value::U64(_) => panic!("a float column decoded as an integer"),
    }
}

fn u64_of(value: Value) -> u64 {
    match value {
        Value::U64(number) => number,
        other => panic!("expected an integer column, found {other:?}"),
    }
}

// ------------------------------------------------------------ ground truth --

/// **The answer, written out rather than recomputed.**
///
/// Every number below is read off the fixture table in this file's header by
/// hand. A test that recomputed them from the same loop the op runs would
/// certify that the loop is deterministic and nothing else.
#[test]
fn the_reduction_is_the_answer_the_fixture_has() {
    let fixed = FixedPoint::default();
    let grouping = grouping(fixed);
    let folded = grouping
        .fold_blob(VOLUME, &blob(&fixture()))
        .expect("the fold runs");
    assert_eq!(folded.len(), 2, "two keys");

    let rows = finished(&grouping, &folded);
    assert_eq!(rows.len(), 2);

    let one = &rows[0];
    assert_eq!(one.key, vec![1]);
    assert_eq!(one.at, [0, 0, 0], "the group's least row is where it sits");
    assert_eq!(one.rows, 4);
    assert_eq!(u64_of(one.values[SCORE_COUNT]), 3, "one score is absent");
    assert_eq!(one.sum(&grouping, SCORE_SUM).expect("a sum"), 7.0);
    assert_eq!(
        f64_of(one.values[SCORE_MIN]),
        -3.0,
        "the mask is respected: the group holds a filler of {FILLER}, which is smaller"
    );
    assert_eq!(f64_of(one.values[SCORE_MAX]), 8.0);
    assert_eq!(f64_of(one.values[SCORE_FIRST_PRESENT]), 2.0);
    assert_eq!(
        f64_of(one.values[SCORE_FIRST_ROW]),
        FILLER,
        "the least row of group 1 has no score, and the whole-row rule reports what that row \
         holds — which is the filler, and is the whole difference between the two readings"
    );
    assert_eq!(u64_of(one.values[MARK_COUNT]), 2);
    assert_eq!(one.sum(&grouping, MARK_SUM).expect("a sum"), 6.0);
    // **The per-column property**: `mark`'s first comes from the row at 0 and
    // `score`'s from the row at 2, out of one call.
    assert_eq!(f64_of(one.values[MARK_FIRST_PRESENT]), 5.0);
    assert_eq!(f64_of(one.values[MARK_FIRST_ROW]), 5.0);

    let two = &rows[1];
    assert_eq!(two.key, vec![2]);
    assert_eq!(two.at, [1, 0, 0]);
    assert_eq!(two.rows, 4);
    assert_eq!(u64_of(two.values[SCORE_COUNT]), 3);
    assert_eq!(two.sum(&grouping, SCORE_SUM).expect("a sum"), 3.5);
    assert_eq!(f64_of(two.values[SCORE_MIN]), -1.0);
    assert_eq!(f64_of(two.values[SCORE_MAX]), 4.0);
    assert_eq!(f64_of(two.values[SCORE_FIRST_PRESENT]), 4.0);
    assert_eq!(f64_of(two.values[SCORE_FIRST_ROW]), 4.0);
    assert_eq!(u64_of(two.values[MARK_COUNT]), 2);
    assert_eq!(two.sum(&grouping, MARK_SUM).expect("a sum"), 9.0);
    assert_eq!(f64_of(two.values[MARK_FIRST_PRESENT]), 7.0);
    assert_eq!(
        f64_of(two.values[MARK_FIRST_ROW]),
        FILLER,
        "the least row of group 2 has no mark"
    );
}

/// The two `First`s are different answers on this fixture, in both directions —
/// which is what makes offering both a statement rather than a courtesy.
#[test]
fn the_two_first_rules_disagree_in_both_groups_and_in_opposite_columns() {
    let grouping = grouping(FixedPoint::default());
    let folded = grouping
        .fold_blob(VOLUME, &blob(&fixture()))
        .expect("the fold runs");
    let rows = finished(&grouping, &folded);
    assert_ne!(
        rows[0].values[SCORE_FIRST_PRESENT], rows[0].values[SCORE_FIRST_ROW],
        "group 1 separates them on `score`"
    );
    assert_ne!(
        rows[1].values[MARK_FIRST_PRESENT], rows[1].values[MARK_FIRST_ROW],
        "group 2 separates them on `mark`, which is the other column"
    );
    // And where the least row does hold a value the two agree, which is the
    // condition under which the disagreement is invisible — stated here so that
    // a fixture that lost its absences would fail this rather than pass
    // everything.
    assert_eq!(
        rows[0].values[MARK_FIRST_PRESENT],
        rows[0].values[MARK_FIRST_ROW]
    );
    assert_eq!(
        rows[1].values[SCORE_FIRST_PRESENT],
        rows[1].values[SCORE_FIRST_ROW]
    );
}

/// `min` and `max` skip a `NaN` rather than returning it, which is what
/// `f64::total_cmp` buys and what `f64::min`/`max` would have got wrong in the
/// other direction — those return the *other* operand for a `NaN`, so a fold
/// over them depends on arrival order.
#[test]
fn the_selections_skip_the_absences_and_are_not_the_ends_of_the_group() {
    let grouping = grouping(FixedPoint::default());
    let folded = grouping
        .fold_blob(VOLUME, &blob(&fixture()))
        .expect("the fold runs");
    let rows = finished(&grouping, &folded);
    // Group 1's scores in row order are absent, 2, 8, -3: the min is the last
    // and the max is the third, so neither is an end of the group and neither is
    // the first present value.
    assert_eq!(f64_of(rows[0].values[SCORE_MIN]), -3.0);
    assert_eq!(f64_of(rows[0].values[SCORE_MAX]), 8.0);
    assert_ne!(
        f64_of(rows[0].values[SCORE_MIN]),
        f64_of(rows[0].values[SCORE_FIRST_PRESENT])
    );
    // **The mask is what makes that true.** The same grouping with the mask
    // taken off selects the filler, in both groups and in both directions — so
    // the assertion above is a measurement of the mask rather than of the
    // fixture happening not to contain a smaller number.
    let unmasked = Grouping::new(
        schema(),
        vec![KEY],
        vec![
            Reduction::new(SCORE, Aggregate::Min),
            Reduction::new(SCORE, Aggregate::Max),
        ],
        FixedPoint::default(),
    )
    .expect("a grouping");
    let folded = unmasked
        .fold_blob(VOLUME, &blob(&fixture()))
        .expect("the fold runs");
    let loose = finished_with(&unmasked, &folded);
    assert_eq!(
        f64_of(loose[0].values[0]),
        FILLER,
        "without the mask the filler is the least value, which is what the mask exists to stop"
    );
    assert_eq!(f64_of(loose[1].values[0]), FILLER);
}

// ----------------------------------------------------------- invariance --

/// **The same rows, cut every way, give one map — and one blob.**
///
/// `2^7` splits of eight rows into contiguous ranges, enumerated rather than
/// sampled: `n` is small, so "every way" is affordable and is a stronger claim
/// than any number of chosen cuts. Each range becomes a partial and the partials
/// are folded; the answer is compared **byte for byte** against the whole-table
/// fold, so a difference in the last bit of a sum would fail rather than round
/// away.
///
/// Contiguous ranges rather than block cores because this is a claim about the
/// **fold**, not about the lattice — the lattice's version is
/// `the_plan_gives_the_same_table_at_every_cut` below, and the two are different
/// claims that would be easy to conflate.
#[test]
fn folding_is_insensitive_to_how_the_rows_were_split() {
    let grouping = grouping(FixedPoint::default());
    let rows = fixture();
    let merge = merge_op(&grouping, [1, 1, 1]);
    let whole = merge
        .fold([partial(&grouping, &rows).as_slice()])
        .expect("the whole-table fold");

    let cuts = rows.len() - 1;
    let mut checked = 0usize;
    for mask in 0..(1usize << cuts) {
        let mut ranges: Vec<Vec<u8>> = Vec::new();
        let mut start = 0;
        for cut in 0..cuts {
            if mask & (1 << cut) != 0 {
                ranges.push(partial(&grouping, &rows[start..=cut]));
                start = cut + 1;
            }
        }
        ranges.push(partial(&grouping, &rows[start..]));
        let folded = merge
            .fold(ranges.iter().map(Vec::as_slice))
            .expect("every split folds");
        assert_eq!(folded, whole, "split {mask:b} gave a different answer");
        assert_eq!(
            encode_groups(&grouping, &folded).expect("an encode"),
            encode_groups(&grouping, &whole).expect("an encode"),
            "split {mask:b} gave a different blob"
        );
        checked += 1;
    }
    assert_eq!(checked, 1 << cuts, "every split was checked");
}

/// **`SeamFold::Unordered`, checked rather than declared.**
///
/// Four partials, every one of the 24 orders, one answer. The op declares
/// `Unordered` and the executor spot-checks it by reversing one neighbourhood;
/// this is the whole permutation group on a case the executor's check would only
/// sample.
#[test]
fn the_partials_fold_to_one_answer_in_every_order() {
    let grouping = grouping(FixedPoint::default());
    let rows = fixture();
    let merge = merge_op(&grouping, [1, 1, 1]);
    let partials: Vec<Vec<u8>> = rows
        .chunks(2)
        .map(|chunk| partial(&grouping, chunk))
        .collect();
    assert_eq!(partials.len(), 4);

    let mut answers = Vec::new();
    for order in permutations(partials.len()) {
        let folded = merge
            .fold(order.iter().map(|index| partials[*index].as_slice()))
            .expect("every order folds");
        answers.push(encode_groups(&grouping, &folded).expect("an encode"));
    }
    assert_eq!(answers.len(), 24);
    assert!(
        answers.windows(2).all(|pair| pair[0] == pair[1]),
        "the fold is not order-independent"
    );
}

fn permutations(n: usize) -> Vec<Vec<usize>> {
    if n <= 1 {
        return vec![(0..n).collect()];
    }
    let mut out = Vec::new();
    for first in 0..n {
        for rest in permutations(n - 1) {
            let mut order = vec![first];
            order.extend(
                rest.into_iter()
                    .map(|index| if index >= first { index + 1 } else { index }),
            );
            out.push(order);
        }
    }
    out
}

/// The blob round-trips, and a truncated one is refused by name rather than
/// decoded short.
#[test]
fn a_partial_round_trips_and_a_truncated_one_is_refused() {
    let grouping = grouping(FixedPoint::default());
    let folded = grouping
        .fold_blob(VOLUME, &blob(&fixture()))
        .expect("the fold runs");
    let bytes = encode_groups(&grouping, &folded).expect("an encode");
    let back = decode_groups(&grouping, &bytes).expect("a decode");
    let round: BTreeMap<Vec<u64>, GroupFold> = back.into_iter().collect();
    assert_eq!(round, folded);

    let short = &bytes[..bytes.len() - 8];
    let message = decode_groups(&grouping, short)
        .expect_err("a truncated partial is refused")
        .to_string();
    assert!(message.contains("whole number of"), "{message}");
}

// ------------------------------------------------------------- refusals --

/// **The duplication refusal**, and the case it does *not* catch, measured.
///
/// Two partials that agree on a group's least row can only come from one row
/// position reaching two blocks, and a duplicated row is indistinguishable from
/// a real one once it is in the sum — so it is refused rather than folded.
///
/// The second half of this test is the honest limit: a duplicate *above* the
/// group's least row leaves the two partials with different least rows and slips
/// through, doubling the group's `rows`. What rules that out is the producer
/// keying by `owner_of`, not this fold, and a test that only showed the refusal
/// firing would read as though the fold were a complete duplication check.
#[test]
fn two_partials_claiming_one_least_row_are_refused_by_name() {
    let grouping = grouping(FixedPoint::default());
    let rows = fixture();
    let merge = merge_op(&grouping, [1, 1, 1]);
    let one = partial(&grouping, &rows[0..4]);
    let message = merge
        .fold([one.as_slice(), one.as_slice()])
        .expect_err("the same partial twice is refused")
        .to_string();
    assert!(message.contains("least row"), "{message}");
    assert!(message.contains("disjoint"), "{message}");

    // And the control: two partials of *different* rows fold, because their
    // least rows differ. The refusal is about duplication and not about folding.
    let two = partial(&grouping, &rows[4..]);
    assert!(merge.fold([one.as_slice(), two.as_slice()]).is_ok());

    // **What it does not catch.** Group 1's rows are at 0, 2, 4 and 6; give one
    // partial the row at 0 and another the row at 4, then repeat the second. The
    // least rows are 0 and 4, so the refusal does not fire and the group's count
    // comes back with the duplicate in it — which is the producer's precondition
    // doing the work and not this fold, and is why the doc says so.
    let head = partial(&grouping, &rows[0..1]);
    let above = partial(&grouping, &rows[4..5]);
    let honest = merge
        .fold([head.as_slice(), above.as_slice()])
        .expect("the same rows once");
    let slipped = merge
        .fold([head.as_slice(), above.as_slice(), above.as_slice()])
        .expect("a duplicate above the least row is not caught here");
    let key = vec![1u64];
    assert_eq!(honest[&key].rows, 2, "one row of group 1 in each partial");
    assert_eq!(
        slipped[&key].rows, 3,
        "the duplicate is folded in silently, because its position is not the group's least"
    );
}

/// **The overflow refusal, with what wrapping would have returned measured
/// beside it.**
///
/// At 62 fraction bits the accumulator's range is `+/- 2`, so two rows of `1.0`
/// total exactly `2^63` — one past what a signed 64-bit column holds. The op
/// refuses by name. The number in the assertion is what `as i64` would have
/// produced instead: `-2.0`, a small negative mean in place of a large positive
/// total, with nothing failing.
#[test]
fn a_total_past_the_column_is_refused_rather_than_wrapped() {
    let fixed = FixedPoint::bits(62).expect("62 fraction bits");
    let grouping = Grouping::new(
        schema(),
        vec![KEY],
        vec![Reduction::new(SCORE, Aggregate::Sum)],
        fixed,
    )
    .expect("a grouping");
    let one = |x: usize| {
        (
            [x, 0usize, 0usize],
            vec![
                Value::U64(1),
                Value::F64(1.0),
                Value::U64(1),
                Value::F64(0.0),
                Value::U64(1),
            ],
        )
    };
    let rows = vec![one(0), one(1)];
    let folded = grouping
        .fold_blob(VOLUME, &blob(&rows))
        .expect("the fold itself is `i128` and has no range of its own");
    let (key, fold) = folded.iter().next().expect("one group");
    assert_eq!(fold.columns[0].total, 1i128 << 63, "the exact total");

    let message = grouping
        .finish(key, fold)
        .expect_err("the total does not fit the column")
        .to_string();
    assert!(message.contains("range"), "{message}");

    // What the refusal is instead of. `as i64` on `2^63` wraps to `-2^63`, which
    // `value_of` reads back as `-2.0` — a plausible number, of the wrong sign,
    // for a total of `+2.0`.
    let wrapped = fixed.value_of((fold.columns[0].total as u64) as i64);
    assert_eq!(wrapped, -2.0);
    assert_eq!(fixed.limit(), 2.0);

    // One row alone fits, so the refusal is about the total rather than about
    // the scale being unusable.
    let smaller = grouping
        .fold_blob(VOLUME, &blob(&rows[..1]))
        .expect("the fold runs");
    let (key, fold) = smaller.iter().next().expect("one group");
    assert!(grouping.finish(key, fold).is_ok());
}

/// Every refusal a `Grouping` makes at construction, each on the input that
/// makes it.
#[test]
fn a_grouping_refuses_what_would_make_the_output_meaningless() {
    let fixed = FixedPoint::default();
    let refusal = |key: Vec<usize>, reductions: Vec<Reduction>| {
        Grouping::new(schema(), key, reductions, fixed)
            .expect_err("this grouping must be refused")
            .to_string()
    };

    let empty = refusal(vec![], vec![]);
    assert!(empty.contains("no key columns"), "{empty}");

    let float_key = refusal(vec![SCORE], vec![]);
    assert!(float_key.contains("holds floats"), "{float_key}");

    let repeated = refusal(vec![KEY, KEY], vec![]);
    assert!(repeated.contains("twice in the key"), "{repeated}");

    let missing = refusal(vec![9], vec![]);
    assert!(missing.contains("the rows have"), "{missing}");

    let self_reduced = refusal(vec![KEY], vec![Reduction::new(KEY, Aggregate::FirstRow)]);
    assert!(self_reduced.contains("the key back"), "{self_reduced}");

    // A count with no mask is `rows` under another name.
    let bare_count = refusal(vec![KEY], vec![Reduction::new(SCORE, Aggregate::Count)]);
    assert!(bare_count.contains(GROUP_ROWS), "{bare_count}");

    // A mask that is not a `U64` column.
    let float_mask = refusal(
        vec![KEY],
        vec![Reduction::masked(SCORE, Aggregate::Count, MARK)],
    );
    assert!(float_mask.contains("presence mask"), "{float_mask}");

    // The three statistics over a column of names.
    let names = Schema::new(vec![Column::u64("key"), Column::u64("name")]).expect("two names");
    for aggregate in [Aggregate::Sum, Aggregate::Min, Aggregate::Max] {
        let message = Grouping::new(
            names.clone(),
            vec![0],
            vec![Reduction::new(1, aggregate)],
            fixed,
        )
        .expect_err("a name column admits no order statistic and no total")
        .to_string();
        assert!(message.contains("category error"), "{message}");
    }
    // And the three that are defined over it are accepted, with a `First`'s
    // output column keeping the input's type.
    let masked = Schema::new(vec![
        Column::u64("key"),
        Column::u64("name"),
        Column::u64("has_name"),
    ])
    .expect("three names");
    for aggregate in [Aggregate::FirstPresent, Aggregate::FirstRow] {
        let grouping = Grouping::new(
            masked.clone(),
            vec![0],
            vec![Reduction::new(1, aggregate)],
            fixed,
        )
        .expect("the first of a name is a name");
        assert_eq!(grouping.output().columns()[2].kind(), ColumnType::U64);
    }
    assert!(
        Grouping::new(
            masked,
            vec![0],
            vec![Reduction::masked(1, Aggregate::Count, 2)],
            fixed,
        )
        .is_ok(),
        "counting the rows that have a name is meaningful"
    );

    // A key column named after the group size collides with it, and says so
    // rather than producing a schema with two columns of one name.
    let clashing = Schema::new(vec![Column::u64(GROUP_ROWS), Column::f64("v")]).expect("two names");
    let message = Grouping::new(clashing, vec![0], vec![], fixed)
        .expect_err("the name is taken")
        .to_string();
    assert!(message.contains(GROUP_ROWS), "{message}");
}

/// A coordinate the packed position cannot hold is refused rather than
/// truncated, because a truncated position selects a different row as the
/// group's first and does it silently.
#[test]
fn a_coordinate_past_the_packed_position_is_refused() {
    let grouping = grouping(FixedPoint::default());
    let mut folded = grouping
        .fold_blob(VOLUME, &blob(&fixture()))
        .expect("the fold runs");
    let (_, fold) = folded.iter_mut().next().expect("a group");
    fold.columns[0].first_present = Some(([MAX_PACKED_COORDINATE + 1, 0, 0], 0));
    let message = encode_groups(&grouping, &folded)
        .expect_err("the coordinate does not fit")
        .to_string();
    assert!(message.contains("packed position"), "{message}");
    assert!(message.contains("silently"), "{message}");
}

/// The op refuses a stream whose schema is not the one it was built to reduce,
/// at construction — because a blob of the wrong schema decodes perfectly and
/// answers differently.
#[test]
fn the_op_refuses_a_stream_of_the_wrong_schema() {
    let grouping = grouping(FixedPoint::default());
    let other = Schema::new(vec![Column::u64("key"), Column::f64("elsewhere")]).expect("two names");
    let streams = RowStreams::new(ROWS, 0, PARTIALS, Lifecycle::DeleteOnExit, other)
        .expect("two distinct streams");
    let message = match GroupRowsOp::new("group", streams, grouping) {
        Err(error) => error.to_string(),
        Ok(_) => panic!("a grouping whose input schema is not the stream's must be refused"),
    };
    assert!(message.contains("decodes perfectly"), "{message}");
}

// ------------------------------------------------------------- the plan --

/// **The acceptance bar: the answer is declared by a plan, not asserted by
/// hand.**
///
/// The two phases run over six lattices, from one block to one voxel per block
/// on the populated axis. Every run's table is compared **byte for byte** with
/// the one-block run's, and the group count is checked so that a cut which lost
/// a group could not pass by having lost it everywhere.
#[test]
fn the_plan_gives_the_same_table_at_every_cut() {
    let grouping = grouping(FixedPoint::default());
    let reference = run_plan(&grouping, VOLUME).expect("the one-block run");
    assert_eq!(reference.len(), 2, "two groups");

    for block in [[4usize, 4, 4], [2, 4, 4], [1, 4, 4], [4, 2, 2], [1, 1, 1]] {
        let rows = run_plan(&grouping, block).expect("the blocked run");
        assert_eq!(rows.len(), reference.len(), "cut {block:?} lost a group");
        assert_eq!(rows, reference, "cut {block:?} moved the table");
    }
}

/// **Ownership**: exactly one block writes each group, so the union over the
/// lattice is the table once and not once per block.
///
/// Checked at the blob level rather than through the assembled table, because a
/// `Table` that was handed a row twice would hold it twice and the count is what
/// would say so.
#[test]
fn exactly_one_block_emits_each_group() {
    let grouping = grouping(FixedPoint::default());
    let block = [1usize, 4, 4];
    let grid = BlockGrid::new(VOLUME, block).expect("a grid");
    let merge = merge_op(&grouping, grid.blocks_per_axis());
    let totals = merge
        .fold([partial(&grouping, &fixture()).as_slice()])
        .expect("the whole fold");

    let mut written = 0usize;
    let mut owners = Vec::new();
    let counts = grid.blocks_per_axis();
    for index in (0..counts[0])
        .flat_map(|x| (0..counts[1]).flat_map(move |y| (0..counts[2]).map(move |z| [x, y, z])))
    {
        let bytes = merge
            .encode_owned(&totals, &grid, index)
            .expect("an encode");
        let mut table = Table::new(VOLUME, grouping.output().clone()).expect("a table");
        table.write(index, &bytes).expect("a write");
        table.seal().expect("a seal");
        if table.len() > 0 {
            owners.push((index, table.len()));
        }
        written += table.len();
    }
    assert_eq!(written, 2, "each group is written exactly once");
    // And the owners are the blocks holding the groups' least rows, which are
    // at 0 and 1 on axis 0 — so on this lattice they are two different blocks.
    assert_eq!(owners, vec![([0, 0, 0], 1), ([1, 0, 0], 1)]);
    for (index, _) in &owners {
        assert_eq!(owner_of(&grid, [index[0], 0, 0]), *index);
    }
}

/// `append_group_phases` puts the two phases on the lattice the plan already
/// has, and answers with the phase the grouped rows are keyed under.
#[test]
fn the_phases_can_be_appended_to_an_existing_plan() {
    let grouping = grouping(FixedPoint::default());
    let grid = BlockGrid::new(VOLUME, [4, 4, 4]).expect("a grid");
    let mut builder = PlanBuilder::new(VOLUME, Dtype::F64, grid.clone());
    let source = builder
        .fragments(RowSourceOp::new(
            "rows in",
            ROWS,
            Lifecycle::DeleteOnExit,
            schema(),
            source_rows(),
        ))
        .expect("a producer phase");
    let base: Decomposition = builder.finish().expect("a plan").decomposition;
    assert_eq!(base.phases.len(), 1);

    let streams = RowStreams::new(
        ROWS,
        source.index(),
        PARTIALS,
        Lifecycle::DeleteOnExit,
        schema(),
    )
    .expect("two distinct streams");
    let group = GroupRowsOp::new("group", streams, grouping.clone()).expect("an op");
    let merge = MergeGroupsOp::new(
        "merge",
        PARTIALS,
        1,
        grid.blocks_per_axis(),
        grouping,
        GROUPED,
        Lifecycle::Persistent,
    )
    .expect("an op");
    let (plan, rows_phase) = append_group_phases(base, &group, &merge).expect("the append");
    assert_eq!(plan.phases.len(), 3);
    assert_eq!(rows_phase, 2);

    // The merge reads and writes different streams, and says so if asked not to.
    let message = match MergeGroupsOp::new(
        "merge",
        PARTIALS,
        1,
        grid.blocks_per_axis(),
        grouping_again(),
        PARTIALS,
        Lifecycle::Persistent,
    ) {
        Err(error) => error.to_string(),
        Ok(_) => panic!("one stream cannot be both the merge's input and its output"),
    };
    assert!(
        message.contains("not even the same wire format"),
        "{message}"
    );
}

fn grouping_again() -> Grouping {
    grouping(FixedPoint::default())
}

// ------------------------------------------------------------- the harness --

/// One group folded and finished, as the row it becomes.
fn finished(grouping: &Grouping, folded: &BTreeMap<Vec<u64>, GroupFold>) -> Vec<GroupValues> {
    finished_with(grouping, folded)
}

fn finished_with(grouping: &Grouping, folded: &BTreeMap<Vec<u64>, GroupFold>) -> Vec<GroupValues> {
    let mut builder = RowBuilder::new(Arc::new(grouping.output().clone()));
    for (key, fold) in folded {
        let (at, values) = grouping.finish(key, fold).expect("a finished row");
        builder.push(at, &values).expect("a push");
    }
    let mut table = Table::new(VOLUME, grouping.output().clone()).expect("a table");
    table.write([0, 0, 0], &builder.encode()).expect("a write");
    table.seal().expect("a seal");
    table
        .scan(&Region::whole(&VOLUME))
        .expect("a scan")
        .map(|row| group_values(grouping, &row).expect("a decode"))
        .collect()
}

fn partial(grouping: &Grouping, rows: &[([usize; 3], Vec<Value>)]) -> Vec<u8> {
    let folded = grouping
        .fold_blob(VOLUME, &blob(rows))
        .expect("the fold runs");
    encode_groups(grouping, &folded).expect("an encode")
}

fn merge_op(grouping: &Grouping, lattice: [usize; 3]) -> MergeGroupsOp {
    MergeGroupsOp::new(
        "merge",
        PARTIALS,
        1,
        lattice,
        grouping.clone(),
        GROUPED,
        Lifecycle::Persistent,
    )
    .expect("two distinct streams")
}

/// The fixture as the producer takes it.
///
/// This file used to carry its own producer — *"the producer the row world has
/// no general op for, and the smallest honest one"* — and it was the third
/// writing of the same fifteen lines, after two in a consumer of this crate.
/// `ops::rows::RowSourceOp` is now that op, and this is what is left of the
/// copy: a conversion, and no rule of its own. The keying
/// it used to state — [`owner_of`], so the producer and the merge agree by
/// construction rather than by two matching expressions — is the library's now,
/// and `every_group_is_emitted_by_exactly_one_block` below still checks it from
/// this side.
fn source_rows() -> Vec<RowValues> {
    fixture()
        .into_iter()
        .map(|(at, values)| RowValues::new(at, values))
        .collect()
}

/// The three phases run, and the grouped table read back out.
fn run_plan(grouping: &Grouping, block: [usize; 3]) -> Result<Vec<GroupValues>> {
    let grid = BlockGrid::new(VOLUME, block)?;
    let lattice = grid.blocks_per_axis();
    let mut builder = PlanBuilder::new(VOLUME, Dtype::F64, grid);
    let source = builder.fragments(RowSourceOp::new(
        "rows in",
        ROWS,
        Lifecycle::DeleteOnExit,
        schema(),
        source_rows(),
    ))?;
    let streams = RowStreams::new(
        ROWS,
        source.index(),
        PARTIALS,
        Lifecycle::DeleteOnExit,
        schema(),
    )?;
    let group = builder.fragments(GroupRowsOp::new("group", streams, grouping.clone())?)?;
    builder.fragments(MergeGroupsOp::new(
        "merge",
        PARTIALS,
        group.index(),
        lattice,
        grouping.clone(),
        GROUPED,
        Lifecycle::Persistent,
    )?)?;
    let assembly = builder.finish()?;
    let phases = assembly.decomposition.n_phases();
    let env = ArrayEnvironment::new(Voxels::zeros(Dtype::F64, VOLUME)?, phases, [4, 4, 4])?;
    execute_phases(
        "group",
        &assembly.workflow,
        &assembly.decomposition,
        &Hints::default(),
        &env,
        &[],
        &assembly.work(),
    )?;
    let rows_phase = assembly.decomposition.phases.len() - 1;
    let mut seen = 0usize;
    fold_fragments(&env, GROUPED, &mut |key, _| {
        if key.phase == rows_phase {
            seen += 1;
        }
        Ok(())
    })?;
    assert!(seen > 0, "the merge wrote nothing at all");
    collect_groups(&env, GROUPED, rows_phase, VOLUME, grouping)
}
