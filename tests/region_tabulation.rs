// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **A per-region reduction over a second array, and the seam that comes with
// it.** `ops::tabulate` is the op; this is what it is accepted on.
//
// The hazard, restated because it is the content
// ----------------------------------------------
// A region of a label volume straddles block seams. Its voxels are visited in
// pieces, one per block, and the pieces are combined in a merge. Combine them
// with `f64` addition and the total depends on the order the blocks merged in —
// so the same data cut two ways gives two numbers, and neither looks wrong.
// `ops/detect.rs` deferred a weighted centroid on exactly this and named the way
// out: an accumulator whose combine associates.
//
// `ops::tabulate`'s **sum** takes that way out — a fixed-point integer
// accumulator — and the op therefore claims `SeamFold::Unordered`, which the
// executor **checks** by applying each block a second time with its
// neighbourhood reversed.
//
// The **selection** needs no way out and does not take one. `min` and `max`
// choose one of the values they are handed rather than computing a new one, and
// under a total order that is associative, commutative and idempotent in `f64`
// itself — so they are `F64` columns holding the voxel's own bits, and
// `SeamFold::Unordered` holds for them on their own account. Sections 5 and 6
// are that split: 5 is the selection being exact where a quantised one could
// not have been, 6 is the sum not having moved while it happened.
//
// What each fixture is arranged so that it *can* fail
// ---------------------------------------------------
// * the block sizes put regions across seams, and the test **asserts that they
//   did** — byte-identity across five runs that merged nothing would prove
//   nothing at all;
// * the value array is signed, so a sign error is visible, and holds a `NaN` and
//   an infinity, so the stated exclusion is exercised rather than described;
// * the label volume has a **disconnected** region, a **single-voxel** region, a
//   region spanning **every block**, and a **gap in the numbering** — the four
//   degenerate cases, in the same fixture as the ordinary ones;
// * the negative control is the same plan with an `f64` fold in the merge's
//   place, which must be refused. Without it, "the reversal check passed" could
//   mean the check never ran;
// * and the selection's fixture holds a value the fixed point cannot represent
//   at **any** scale the run admits — 50 accepted and 13 refused across the 63,
//   and every accepted one quantises it inexactly — so "the column is exact" is
//   a statement that could have come back false rather than one the scale was
//   chosen to make true.
//
// The first moment, and the fixture built to separate it
// ------------------------------------------------------
// `moment_0..2_q{n}` is `sum(value * coordinate)` per axis, and its quotient by
// `sum_q{n}` is the **weighted centroid** — the one per-region quantity the
// other columns do not determine. It is the same kind of accumulator as the sum,
// on the same fixed point, so it is accepted on the same terms; section 5 is
// what it takes for those terms to mean something for *this* column.
//
// A symmetric arrangement cannot tell a weighted centre from an unweighted one,
// so a fixture built out of one would pass an implementation that never read the
// value array. `separating_labels` and `separating_value_at` are built the other
// way: a weight that rises steeply and independently on all three axes, from a
// base that is not a dyadic rational, over two regions that every cut puts
// across a seam. The two centres then differ by at least a quarter of a voxel on
// every axis of every region, which is the assertion the byte-identity rests on.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::dtype::Dtype;
use blockflow::env::{ArrayEnvironment, Environment};
use blockflow::error::Result;
use blockflow::fragment::{
    fold_fragments, BlockOutput, BlockView, Coverage, FragmentInput, FragmentOp, FragmentOutput,
    PhaseWork, SeamFold,
};
use blockflow::geometry::BlockGrid;
use blockflow::log::{Event, ExecutionLog};
use blockflow::op::{Anchor, BlockOp, Chain};
use blockflow::ops::tabulate::{
    append_tabulate_phases, collect_tabulation, decode_partial, region_values, tabulation_schema,
    FixedPoint, MergeTabulationOp, RegionValues, TabulateValuesOp, Tally, DEFAULT_FRACTION_BITS,
    MAX as MAX_COLUMN, MAX_FRACTION_BITS, MIN as MIN_COLUMN, MOMENT as MOMENT_COLUMN,
    SUM as SUM_COLUMN,
};
use blockflow::region::Region;
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints, Workflow};
use blockflow::synthetic::{Scene, SceneSpec};
use blockflow::table::Table;
use blockflow::voxels::Voxels;
use ndarray::Array3;

// ------------------------------------------------------------- the fixture --

const VOLUME: [usize; 3] = [12, 4, 4];
/// Twenty fraction bits: a resolution of about `9.5e-7` and a range of about
/// `+/- 8.8e12`. Every value below is a few tens, so neither end binds — which
/// is the point of stating them.
fn fixed() -> FixedPoint {
    FixedPoint::default()
}

/// The label volume, as a function of the global coordinate.
///
/// Deliberately **not** a connected-component labelling, because that is the
/// distinction this op exists for:
///
/// * `1` is a full-length line on axis 0, so it touches every block of every
///   lattice tried below;
/// * `2` is a box from 3 to 9 on axis 0, whose faces sit at coordinates that no
///   block edge of 12, 6, 4, 3 or 2 lines up with on both sides;
/// * `3` is one voxel;
/// * `5` is **two disconnected pieces**, one at each end of axis 0, which
///   `ops::detect` would report as two components and this op reports as one
///   region — the whole reason it is not that op;
/// * `4` is never written, so the numbering has a gap.
fn label_at(at: [usize; 3]) -> u64 {
    let [z, y, x] = at;
    if y == 0 && x == 0 {
        return 1;
    }
    if (3..9).contains(&z) && (1..3).contains(&y) && (1..3).contains(&x) {
        return 2;
    }
    if at == [7, 3, 3] {
        return 3;
    }
    if y == 3 && x == 0 && !(2..10).contains(&z) {
        return 5;
    }
    0
}

/// The value array, as a function of the global coordinate.
///
/// **Signed**, so a sign error cannot hide, and with a `NaN` and an infinity in
/// it, so the stated exclusion is exercised. The `NaN` at `[9, 0, 0]` is in the
/// region that spans every block and the two at `[5, 2, 2]` and `[6, 1, 1]` are
/// in the one that straddles seams, so both cases meet the merge rather than
/// staying inside one block.
fn value_at(at: [usize; 3]) -> f64 {
    if at == [9, 0, 0] || at == [5, 2, 2] {
        return f64::NAN;
    }
    if at == [6, 1, 1] {
        return f64::INFINITY;
    }
    at[0] as f64 * 3.0 - at[1] as f64 * 11.5 + at[2] as f64 * 0.125 - 7.0
}

fn labels(volume: [usize; 3], of: fn([usize; 3]) -> u64) -> Voxels {
    let mut array = Array3::<f64>::zeros((volume[0], volume[1], volume[2]));
    for (index, slot) in array.indexed_iter_mut() {
        *slot = of([index.0, index.1, index.2]) as f64;
    }
    array.into()
}

/// Writes the value array from the **global** coordinate, so that a block's
/// output is a function of where it is and not of how big it is. Level 1 has to
/// be produced by a phase — an `ArrayEnvironment` holds one input array — and
/// this is the smallest honest producer of a second one.
struct CoordinateValuesOp {
    of: fn([usize; 3]) -> f64,
}

impl BlockOp for CoordinateValuesOp {
    fn name(&self) -> &'static str {
        "values"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn apply(&self, _input: &Voxels, out: &mut Voxels, at: &Anchor) -> Result<()> {
        let mut view = out.view_mut::<f64>()?;
        for (index, slot) in view.indexed_iter_mut() {
            *slot = (self.of)([
                at.offset[0] + index.0,
                at.offset[1] + index.1,
                at.offset[2] + index.2,
            ]);
        }
        Ok(())
    }
}

fn values_workflow(volume: [usize; 3], of: fn([usize; 3]) -> f64) -> Workflow {
    Workflow::new(Chain::op(CoordinateValuesOp { of }), volume, Dtype::F64)
}

/// One pixel phase over `block`, which writes the value array into level 1. The
/// two fragment phases are appended after it.
fn values_phase(volume: [usize; 3], block: [usize; 3]) -> Decomposition {
    Decomposition {
        volume,
        dtype: Dtype::F64,
        phases: vec![PhaseDecomposition::derive(
            vec![0],
            vec!["values".to_string()],
            [0, 0, 0],
            [0, 0, 0],
            BlockGrid::new(volume, block).expect("a lattice"),
        )],
        chain_reach: [0, 0, 0],
    }
}

/// The three-phase plan: the value array, the per-block partials read out of
/// **level 0 and level 1**, and the merge.
fn tabulation_plan(
    volume: [usize; 3],
    block: [usize; 3],
    point: FixedPoint,
) -> (Decomposition, TabulateValuesOp, MergeTabulationOp, usize) {
    let base = values_phase(volume, block);
    let lattice = base.phases[0].grid.blocks_per_axis();
    let tabulate =
        TabulateValuesOp::new("tabulate", 0, 1, point, "partials", Lifecycle::DeleteOnExit)
            .expect("two different levels");
    let merge = MergeTabulationOp::new(
        "merge",
        "partials",
        1,
        lattice,
        point,
        "rows",
        Lifecycle::Persistent,
    );
    let (plan, rows_phase) =
        append_tabulate_phases(base, &tabulate, &merge).expect("a tabulation plan");
    (plan, tabulate, merge, rows_phase)
}

/// What one run left behind: the canonical table as **words** — which is the
/// byte-level form, since a row is its words — the decoded rows, the per-block
/// partials, the raw row blobs and the event log.
#[derive(Debug)]
struct Run {
    words: Vec<u64>,
    rows: Vec<RegionValues>,
    partials: BTreeMap<[usize; 3], Vec<u8>>,
    row_blobs: BTreeMap<[usize; 3], Vec<u8>>,
    lattice_blocks: usize,
    log: Arc<ExecutionLog>,
}

fn run_at(volume: [usize; 3], block: [usize; 3], point: FixedPoint) -> Run {
    try_run(volume, block, point, label_at, value_at).expect("a run")
}

/// [`run_at`] over a named fixture, and as a `Result`.
///
/// A `Result` because the scale sweeps below need to *count* the refusals: `n`
/// is a range decision before it is a resolution one, and a sweep in which
/// nothing was ever refused would not have reached the edge it is about.
fn try_run(
    volume: [usize; 3],
    block: [usize; 3],
    point: FixedPoint,
    of_label: fn([usize; 3]) -> u64,
    of_value: fn([usize; 3]) -> f64,
) -> Result<Run> {
    let (plan, tabulate, merge, rows_phase) = tabulation_plan(volume, block, point);
    let env = ArrayEnvironment::new(labels(volume, of_label), plan.n_phases(), [4, 4, 4])
        .expect("an environment");
    let log = Arc::new(ExecutionLog::new());
    execute_phases(
        "tabulation",
        &values_workflow(volume, of_value),
        &plan,
        &Hints::default(),
        &env,
        &[log.clone()],
        &[
            PhaseWork::Pixels,
            PhaseWork::Fragments(&tabulate),
            PhaseWork::Fragments(&merge),
        ],
    )?;

    let mut partials = BTreeMap::new();
    for key in env.sidecar_keys("partials").expect("the partial keys") {
        if key.phase != 1 {
            continue;
        }
        let bytes = env
            .read_sidecar("partials", 1, key.block)
            .expect("a read")
            .expect("a listed fragment");
        partials.insert(key.block, bytes);
    }

    // The canonical table, assembled the way a caller assembles it: every
    // block's blob into one `Table`, which orders the rows from the rows.
    let mut row_blobs = BTreeMap::new();
    let mut table = Table::new(volume, tabulation_schema(point)).expect("a table");
    fold_fragments(&env, "rows", &mut |key, bytes| {
        if key.phase != rows_phase {
            return Ok(());
        }
        row_blobs.insert(key.block, bytes.to_vec());
        table.write(key.block, bytes)
    })
    .expect("the row blobs");
    table.seal().expect("a sealed table");
    let mut words = Vec::new();
    let mut rows = Vec::new();
    for row in table.scan(&Region::whole(&volume)).expect("a scan") {
        words.extend_from_slice(row.words());
        rows.push(region_values(&row, point).expect("a decoded row"));
    }

    // And the caller-facing path answers the same thing, so that a consumer
    // using `collect_tabulation` is not reading a different table from the one
    // this test pins.
    let collected =
        collect_tabulation(&env, "rows", rows_phase, volume, point).expect("the collected rows");
    assert_eq!(collected, rows, "the two read paths disagree");

    Ok(Run {
        words,
        rows,
        partials,
        row_blobs,
        lattice_blocks: plan.phases[rows_phase].grid.cores().len(),
        log,
    })
}

/// The answer computed over the whole volume with no blocking anywhere: the
/// oracle, so that "every block size agrees" is not "every block size is wrong
/// the same way".
fn reference(volume: [usize; 3], point: FixedPoint) -> Vec<Tally> {
    reference_with(volume, point, label_at, value_at)
        .into_values()
        .collect()
}

/// [`reference`] over a named fixture, keyed by label — which is how the oracle
/// is looked up, since the table's canonical order is on the row's *position*.
fn reference_with(
    volume: [usize; 3],
    point: FixedPoint,
    of_label: fn([usize; 3]) -> u64,
    of_value: fn([usize; 3]) -> f64,
) -> BTreeMap<u64, Tally> {
    let mut totals: BTreeMap<u64, Tally> = BTreeMap::new();
    for z in 0..volume[0] {
        for y in 0..volume[1] {
            for x in 0..volume[2] {
                let at = [z, y, x];
                let label = of_label(at);
                if label == 0 {
                    continue;
                }
                totals
                    .entry(label)
                    .or_insert_with(|| Tally::new(label))
                    .add(at, of_value(at), point)
                    .expect("no overflow");
            }
        }
    }
    totals
}

fn row_for(rows: &[RegionValues], label: u64) -> RegionValues {
    *rows
        .iter()
        .find(|row| row.label == label)
        .unwrap_or_else(|| panic!("no row for label {label}"))
}

// ---------------------------------------- 1. decomposition invariance --

/// The acceptance criterion, and the assertion that the fixture could have
/// failed it: regions really are cut, in every run but the single-block one.
#[test]
fn a_per_region_reduction_is_byte_identical_across_block_sizes() {
    let sizes: [[usize; 3]; 5] = [[12, 4, 4], [6, 4, 4], [4, 2, 4], [3, 4, 2], [2, 2, 2]];
    let mut answers: Vec<(String, Vec<u64>)> = Vec::new();
    let point = fixed();

    for block in sizes {
        let run = run_at(VOLUME, block, point);

        // Which labels were cut. Without this the byte-identity below could be
        // the identity of five runs that never merged anything.
        let mut contributors: BTreeMap<u64, BTreeSet<[usize; 3]>> = BTreeMap::new();
        for (index, bytes) in &run.partials {
            for tally in decode_partial(bytes).expect("a partial") {
                contributors.entry(tally.label).or_default().insert(*index);
            }
        }
        let straddled: Vec<u64> = contributors
            .iter()
            .filter(|(_, who)| who.len() > 1)
            .map(|(label, _)| *label)
            .collect();
        if block == [12, 4, 4] {
            assert!(
                straddled.is_empty(),
                "the one-block lattice cannot straddle anything"
            );
        } else {
            assert!(
                !straddled.is_empty(),
                "block {block:?} put no region across a seam, so this run merges nothing"
            );
            // and specifically the region that spans the volume, whatever the cut
            assert!(
                straddled.contains(&1),
                "block {block:?} did not cut label 1, which spans axis 0"
            );
        }

        // The answer is the unblocked one, not merely a stable one. Keyed by
        // label rather than zipped, because the table's canonical order is on
        // the *position* — a row is ordered by where it is — and the oracle's
        // is on the label.
        let expected: BTreeMap<u64, Tally> = reference(VOLUME, point)
            .into_iter()
            .map(|tally| (tally.label, tally))
            .collect();
        assert_eq!(run.rows.len(), expected.len(), "block {block:?}");
        for row in &run.rows {
            let tally = expected
                .get(&row.label)
                .unwrap_or_else(|| panic!("block {block:?} invented label {}", row.label));
            assert_eq!(
                row.count, tally.count,
                "block {block:?}: label {}",
                row.label
            );
            assert_eq!(row.nonfinite, tally.nonfinite, "block {block:?}");
            assert_eq!(
                i128::from(row.sum_fixed),
                tally.sum,
                "block {block:?}: label {}",
                row.label
            );
            // The selection on its bits, which is the only comparison that
            // could catch a `-0.0` moving or a value being carried through a
            // scale on the way.
            assert_eq!(
                row.min.to_bits(),
                tally.min.unwrap_or(0.0).to_bits(),
                "block {block:?}: label {}",
                row.label
            );
            assert_eq!(
                row.max.to_bits(),
                tally.max.unwrap_or(0.0).to_bits(),
                "block {block:?}: label {}",
                row.label
            );
            assert_eq!(Some(row.at), tally.centroid());
            // The first moments, on the integers, which is the form the
            // invariance claim is about — and the quotient they derive, on its
            // bits, so that a division that moved with the cut would show.
            for axis in 0..3 {
                assert_eq!(
                    i128::from(row.moment_fixed[axis]),
                    tally.moment[axis],
                    "block {block:?}: label {} on axis {axis}",
                    row.label
                );
            }
            assert_eq!(
                row.weighted_centroid.map(|centre| centre.map(f64::to_bits)),
                tally
                    .weighted_centroid()
                    .map(|centre| centre.map(f64::to_bits)),
                "block {block:?}: label {}",
                row.label
            );
        }

        answers.push((format!("{block:?}"), run.words));
    }

    let (first_name, first) = &answers[0];
    for (name, words) in &answers[1..] {
        assert_eq!(
            words, first,
            "the tabulation differs between block {first_name} and block {name}; a per-region \
             reduction over a second array must not be a function of the plan"
        );
    }
}

// --------------------------------- 2. the accumulator's claim is checked --

/// The same fold as [`MergeTabulationOp`], **in `f64`**, claiming
/// `SeamFold::Unordered`.
///
/// The negative control. It is here so that "the reversal check passed" is a
/// fact about the accumulator rather than about the check never running: this op
/// sits in exactly the same place in exactly the same plan and must be refused.
struct DriftingMergeOp {
    lattice: [usize; 3],
    fold: SeamFold,
    /// Which accumulator to drift: the sum, or the **first moment**. Two
    /// controls rather than one, because they are two accumulators and "the
    /// check catches an `f64` sum" does not say it catches an `f64` moment — the
    /// moment's terms are the sum's multiplied by a coordinate, so its
    /// order-dependence is a different arrangement of bits.
    moment: bool,
}

impl FragmentOp for DriftingMergeOp {
    fn name(&self) -> &'static str {
        "drifting-merge"
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own("partials", 1).with_reach(self.lattice)]
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(self.fold)
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            "drifted",
            Lifecycle::DeleteOnExit,
            Coverage::EveryBlock,
        )]
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let mut total = 0.0f64;
        for (_, bytes) in at.fragments("partials") {
            for tally in decode_partial(bytes)? {
                total += if self.moment {
                    tally.moment[0] as f64
                } else {
                    tally.sum as f64
                };
            }
        }
        Ok(BlockOutput::fragment(
            "drifted",
            blockflow::fragment::pack_u64(&[total.to_bits()]),
        ))
    }
}

/// Four voxels of one label, whose values are `1 + 1 + 2^60 - 2^60`: `0` added
/// one at a time and `2` when the large pair goes first. One block per voxel, so
/// the merge sees four fragments and the order is a real thing.
///
/// Its **first moment** is `1*0 + 1*1 + 2^60*2 - 2^60*3`, whose exact total is
/// `1 - 2^60` — a number that needs 61 significant bits and so is not an `f64` at
/// all. The integer accumulator reports it; an `f64` one cannot.
fn wide_value_at(at: [usize; 3]) -> f64 {
    match at[0] {
        0 | 1 => 1.0,
        2 => (1u64 << 60) as f64,
        _ => -((1u64 << 60) as f64),
    }
}

/// The same four blocks, arranged so that the **first moment** is the
/// accumulator whose `f64` fold depends on the order.
///
/// A separate fixture and not a reuse of [`wide_value_at`], because that one
/// does not do this job: its moments are `0, 1, 2^61, -3*2^60`, and an `f64`
/// fold loses the `1` at the same step whichever end it starts from, so both
/// orders agree on `-2^60` and the reversal check has nothing to catch. That is
/// the whole reason the moment needs a control of its own — it is a *different*
/// arrangement of bits from the sum's, and a fixture that breaks one need not
/// break the other.
///
/// Here the moments are `0, 1, 3*2^59, -3*2^59`. Forwards the `1` is absorbed
/// into `3*2^59` and the pair then cancels to `0`; backwards the pair cancels
/// first and the `1` survives. `0` against `1`, and the exact answer is `1`.
fn drifting_moment_value_at(at: [usize; 3]) -> f64 {
    match at[0] {
        0 => 0.0,
        1 => 1.0,
        // 3 * 2^58 at coordinate 2, so the moment term is 3 * 2^59
        2 => 3.0 * (1u64 << 58) as f64,
        // -2^59 at coordinate 3, so the moment term is -3 * 2^59
        _ => -((1u64 << 59) as f64),
    }
}

fn one_label(_at: [usize; 3]) -> u64 {
    1
}

fn run_wide(
    fold: Option<(SeamFold, bool)>,
    of_value: fn([usize; 3]) -> f64,
) -> Result<Vec<([usize; 3], Vec<u8>)>> {
    // Zero fraction bits, so the range is +/- 2^63 and `2^60` fits. The
    // resolution is then 1.0, which is exactly what a caller trades away to
    // reduce values this large — and it is `FixedPoint`'s whole point that the
    // trade is theirs and is readable.
    let point = FixedPoint::bits(0).expect("zero fraction bits");
    let volume = [4usize, 1, 1];
    let base = values_phase(volume, [1, 1, 1]);
    let lattice = base.phases[0].grid.blocks_per_axis();
    let tabulate =
        TabulateValuesOp::new("tabulate", 0, 1, point, "partials", Lifecycle::DeleteOnExit)?;
    let honest = MergeTabulationOp::new(
        "merge",
        "partials",
        1,
        lattice,
        point,
        "rows",
        Lifecycle::Persistent,
    );
    let drifting = fold.map(|(fold, moment)| DriftingMergeOp {
        lattice,
        fold,
        moment,
    });

    let (plan, rows_phase) = append_tabulate_phases(base, &tabulate, &honest)?;
    let env = ArrayEnvironment::new(labels(volume, one_label), plan.n_phases(), [1, 1, 1])?;
    let merge: &dyn FragmentOp = match &drifting {
        Some(op) => op,
        None => &honest,
    };
    execute_phases(
        "wide",
        &values_workflow(volume, of_value),
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[
            PhaseWork::Pixels,
            PhaseWork::Fragments(&tabulate),
            PhaseWork::Fragments(merge),
        ],
    )?;
    let stream = if drifting.is_some() {
        "drifted"
    } else {
        "rows"
    };
    let mut blobs = Vec::new();
    fold_fragments(&env, stream, &mut |key, bytes| {
        if key.phase == rows_phase {
            blobs.push((key.block, bytes.to_vec()));
        }
        Ok(())
    })?;
    Ok(blobs)
}

#[test]
fn the_fixed_point_merge_passes_the_reversal_check_on_the_fixture_that_breaks_f64() {
    // The honest merge, over four fragments whose `f64` sum is order-dependent.
    // It runs, and the executor applied every block twice to establish that.
    let blobs = run_wide(None, wide_value_at).expect("an integer fold is order-independent");
    let point = FixedPoint::bits(0).unwrap();
    let mut table = Table::new([4, 1, 1], tabulation_schema(point)).unwrap();
    assert_eq!(blobs.len(), 4, "one blob per block, even the empty ones");
    for (block, bytes) in &blobs {
        table.write(*block, bytes).unwrap();
    }
    table.seal().unwrap();
    let rows: Vec<RegionValues> = table
        .scan(&Region::whole(&[4usize, 1, 1]))
        .unwrap()
        .map(|row| region_values(&row, point).unwrap())
        .collect();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].count, 4);
    // `1 + 1 + 2^60 - 2^60` is 2, exactly, in every order.
    assert_eq!(rows[0].sum_fixed, 2);
    // And the selection is the voxels' own values, whose bits the reversal
    // check has just compared four times over.
    assert_eq!(rows[0].max.to_bits(), ((1u64 << 60) as f64).to_bits());
    assert_eq!(rows[0].min.to_bits(), (-((1u64 << 60) as f64)).to_bits());

    // **The first moment came through the same check.** `1*0 + 1*1 + 2^60*2 -
    // 2^60*3` is `1 - 2^60` exactly, which is not an `f64`: the accumulator is
    // integer, so the answer is the integer and the reversal check just compared
    // its bytes four times over. At zero fraction bits the scale is 1, so the
    // column word is the moment itself.
    assert_eq!(rows[0].moment_fixed[0], 1 - (1i64 << 60));
    assert_eq!(rows[0].moment_fixed[1], 0, "the volume is one voxel wide");
    assert_eq!(rows[0].moment_fixed[2], 0);

    // And the quotient, which is where the signed-value rule becomes visible:
    // `(1 - 2^60) / 2` is nowhere near the four voxels it was measured over.
    let centre = rows[0]
        .weighted_centroid
        .expect("the denominator is 2, not 0");
    assert_eq!(centre[0], (1.0 - (1u64 << 60) as f64) / 2.0);
    assert!(
        centre[0] < 0.0,
        "the weighted centre is {} and the region is z in 0..4 — a signed value array makes this \
         a ratio and not a point, which is the stated behaviour",
        centre[0]
    );
    // while the geometric centre stayed where the voxels are
    assert_eq!(rows[0].at, [2, 0, 0]);
}

/// The same claim on the fixture built to break an `f64` **moment** — see
/// [`drifting_moment_value_at`] — so that the moment's passing the reversal
/// check is a fact about a fold that had somewhere to drift to.
#[test]
fn the_fixed_point_moment_passes_the_reversal_check_on_the_fixture_that_breaks_f64() {
    let blobs = run_wide(None, drifting_moment_value_at)
        .expect("an integer moment fold is order-independent");
    let point = FixedPoint::bits(0).unwrap();
    let mut table = Table::new([4, 1, 1], tabulation_schema(point)).unwrap();
    for (block, bytes) in &blobs {
        table.write(*block, bytes).unwrap();
    }
    table.seal().unwrap();
    let rows: Vec<RegionValues> = table
        .scan(&Region::whole(&[4usize, 1, 1]))
        .unwrap()
        .map(|row| region_values(&row, point).unwrap())
        .collect();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].count, 4);
    // `0 + 1 + 3*2^59 - 3*2^59` is 1, exactly, in every order. In `f64` it is
    // `0` one way round and `1` the other.
    assert_eq!(rows[0].moment_fixed[0], 1);
    assert_eq!(
        1.0f64 + 3.0 * (1u64 << 59) as f64 - 3.0 * (1u64 << 59) as f64,
        0.0,
        "the fixture was supposed to lose the one when the large pair goes last"
    );
    assert_eq!(
        -3.0 * (1u64 << 59) as f64 + 3.0 * (1u64 << 59) as f64 + 1.0,
        1.0,
        "and to keep it when the large pair goes first"
    );
    // The sum is `1 + 2^58` and the quotient is a tiny fraction of a voxel — a
    // weighted centre pulled almost all the way to the origin by the one voxel
    // whose value did not cancel.
    assert_eq!(rows[0].sum_fixed, 1 + (1i64 << 58));
    assert_eq!(
        rows[0].weighted_centroid,
        Some([1.0 / (1.0 + (1u64 << 58) as f64), 0.0, 0.0])
    );
}

#[test]
fn an_f64_fold_in_the_same_place_claiming_unordered_is_refused_by_name() {
    let failed = run_wide(Some((SeamFold::Unordered, false)), wide_value_at)
        .expect_err("an `f64` sum over four fragments is not order-independent")
        .to_string();
    assert!(failed.contains("drifting-merge"), "{failed}");
    assert!(failed.contains("SeamFold::Unordered"), "{failed}");
    assert!(failed.contains("opposite order"), "{failed}");
    assert!(failed.contains("fixed-point"), "{failed}");

    // The same control on the **first moment**, which is a different arrangement
    // of bits and therefore a different fact: an `f64` moment over these four
    // fragments is order-dependent too, and the check catches that as well.
    // Without this, "the reversal check passed for the moment" would rest on the
    // check having been observed to run for the sum.
    let moment_failed = run_wide(Some((SeamFold::Unordered, true)), drifting_moment_value_at)
        .expect_err("an `f64` first moment over four fragments is not order-independent")
        .to_string();
    assert!(moment_failed.contains("drifting-merge"), "{moment_failed}");
    assert!(
        moment_failed.contains("SeamFold::Unordered"),
        "{moment_failed}"
    );

    // And declared honestly they run — which is what makes the refusals above a
    // statement about the accumulators rather than about the plan shape.
    run_wide(Some((SeamFold::OrderDependent, false)), wide_value_at)
        .expect("an order-dependent sum fold is admitted");
    run_wide(
        Some((SeamFold::OrderDependent, true)),
        drifting_moment_value_at,
    )
    .expect("an order-dependent moment fold is admitted");
}

// ------------------------------------------------------ 3. ground truth --

/// Region sizes against `Scene::object_table`, which is a **ground truth**
/// rather than a second opinion: the scene knows how many voxels it gave each
/// object, and the count in the table has to be that number.
#[test]
fn region_sizes_equal_the_scene_object_voxel_counts() {
    let volume = [24usize, 24, 24];
    let scene = Scene::new(SceneSpec::clean(volume, 20250817).with_objects(12)).expect("a scene");
    let truth = scene.object_table();
    let rendered = scene.render();
    let point = fixed();

    let block = [8usize, 12, 24];
    let base = values_phase(volume, block);
    let lattice = base.phases[0].grid.blocks_per_axis();
    let tabulate =
        TabulateValuesOp::new("tabulate", 0, 1, point, "partials", Lifecycle::DeleteOnExit)
            .expect("two levels");
    let merge = MergeTabulationOp::new(
        "merge",
        "partials",
        1,
        lattice,
        point,
        "rows",
        Lifecycle::Persistent,
    );
    let (plan, rows_phase) = append_tabulate_phases(base, &tabulate, &merge).expect("a plan");

    let mut as_f64 = Array3::<f64>::zeros((volume[0], volume[1], volume[2]));
    for (index, slot) in as_f64.indexed_iter_mut() {
        *slot = rendered.labels[[index.0, index.1, index.2]] as f64;
    }
    let env = ArrayEnvironment::new(as_f64.into(), plan.n_phases(), [8, 8, 8]).expect("an env");
    execute_phases(
        "ground-truth",
        &values_workflow(volume, value_at),
        &plan,
        &Hints::default(),
        &env,
        &[],
        &[
            PhaseWork::Pixels,
            PhaseWork::Fragments(&tabulate),
            PhaseWork::Fragments(&merge),
        ],
    )
    .expect("a run");
    let rows = collect_tabulation(&env, "rows", rows_phase, volume, point).expect("the rows");

    let expected: BTreeMap<u64, &blockflow::synthetic::ObjectRecord> = truth
        .iter()
        .filter(|record| record.voxels > 0)
        .map(|record| (record.id as u64, record))
        .collect();
    assert!(expected.len() > 1, "a one-object scene tests nothing");
    assert_eq!(
        rows.len(),
        expected.len(),
        "the table has a row per object that has voxels and no others"
    );
    for row in &rows {
        let record = expected.get(&row.label).unwrap_or_else(|| {
            panic!(
                "the table has a row for label {}, which the scene does not have",
                row.label
            )
        });
        assert_eq!(
            row.count, record.voxels,
            "label {} is {} voxels in the table and {} in the scene",
            row.label, row.count, record.voxels
        );
        // The centroid too, exactly: the scene measures voxel *centres* and the
        // table measures voxel *indices*, so the two differ by half a voxel and
        // by nothing else.
        for axis in 0..3 {
            assert!(
                (row.centroid[axis] + 0.5 - record.centroid[axis]).abs() < 1e-9,
                "label {} centroid on axis {axis}: {} against {}",
                row.label,
                row.centroid[axis] + 0.5,
                record.centroid[axis]
            );
        }
    }
}

// ----------------------------------------------- 4. the degenerate cases --

/// All four, in one fixture, against one run — and each is a *stated* outcome
/// rather than one discovered by whoever hits it first.
#[test]
fn the_degenerate_cases_answer_the_way_the_op_says_they_do() {
    let point = fixed();
    // Cut on axis 0 only, so that "spans every block" is literally every block
    // of the lattice rather than every block the line passes through.
    let run = run_at(VOLUME, [3, 4, 4], point);
    let labels: Vec<u64> = run.rows.iter().map(|row| row.label).collect();

    // A label in no voxel gets no row. There is no way for the op to know 4 was
    // expected: it reads voxels, not a caller's numbering. The order is the
    // table's canonical one, which is on the row's *position* — label 5 sits at
    // [6, 3, 0] and label 3 at [7, 3, 3].
    assert_eq!(labels, vec![1, 2, 5, 3], "label 4 is in no voxel");

    // A label in exactly one voxel.
    let single = row_for(&run.rows, 3);
    assert_eq!(single.count, 1);
    assert_eq!(single.at, [7, 3, 3]);
    // The selection is that voxel's value **exactly**, and the sum is its
    // quantisation — which is the same number only when the value was already on
    // the fixed point's lattice, and is why the two are asserted differently.
    assert_eq!(single.min.to_bits(), value_at([7, 3, 3]).to_bits());
    assert_eq!(single.max.to_bits(), single.min.to_bits());
    assert!(
        (single.sum - value_at([7, 3, 3])).abs() <= point.resolution(),
        "one voxel's sum is that voxel's value, to the resolution"
    );

    // A region spanning every block: twelve voxels on axis 0, one of them a
    // `NaN`, folded from every block of the lattice.
    let spanning = row_for(&run.rows, 1);
    assert_eq!(spanning.count, 12);
    assert_eq!(spanning.nonfinite, 1, "the `NaN` at [9, 0, 0]");
    let mut blocks = BTreeSet::new();
    for (index, bytes) in &run.partials {
        if decode_partial(bytes)
            .expect("a partial")
            .iter()
            .any(|tally| tally.label == 1)
        {
            blocks.insert(*index);
        }
    }
    assert_eq!(
        blocks.len(),
        run.partials.len(),
        "label 1 was supposed to touch every block"
    );

    // Negatives are exact, and a disconnected region is one region.
    let split = row_for(&run.rows, 5);
    assert_eq!(split.count, 4, "two pieces of two voxels each");
    let expected: f64 = [[0, 3, 0], [1, 3, 0], [10, 3, 0], [11, 3, 0]]
        .into_iter()
        .map(value_at)
        .sum();
    assert!(expected < 0.0, "the fixture was supposed to go negative");
    assert!((split.sum - expected).abs() <= 4.0 * point.resolution());
    assert!(split.min < 0.0 && split.min <= split.max);

    // `NaN` and infinity are set aside and counted, not folded in.
    let straddling = row_for(&run.rows, 2);
    assert_eq!(straddling.nonfinite, 2, "one `NaN` and one infinity");
    assert!(
        straddling.sum.is_finite() && straddling.max.is_finite(),
        "a non-finite value reached the reductions"
    );
    assert!(straddling.count > straddling.nonfinite);
    assert!(!straddling.all_nonfinite());
    // And precisely: the infinity at [6, 1, 1] is not the maximum. A `max` that
    // came back `+inf` because one voxel was broken is the failure the exclusion
    // rule exists to prevent, and a total order over the values alone would
    // otherwise have ranked it above every real one.
    let mut widest = f64::NEG_INFINITY;
    let mut narrowest = f64::INFINITY;
    for z in 0..VOLUME[0] {
        for y in 0..VOLUME[1] {
            for x in 0..VOLUME[2] {
                let at = [z, y, x];
                let value = value_at(at);
                if label_at(at) == 2 && value.is_finite() {
                    widest = widest.max(value);
                    narrowest = narrowest.min(value);
                }
            }
        }
    }
    assert_eq!(straddling.max.to_bits(), widest.to_bits());
    assert_eq!(straddling.min.to_bits(), narrowest.to_bits());

    // A region with no finite value at all reports `0.0` for both, and the
    // count is what says why — the same statement, from the other side.
    let mut nothing = Tally::new(11);
    nothing
        .add([0, 0, 0], f64::NAN, point)
        .expect("a non-finite value is not a refusal");
    nothing
        .add([1, 0, 0], f64::NEG_INFINITY, point)
        .expect("nor is an infinity");
    assert_eq!(nothing.nonfinite, nothing.count);
    assert_eq!(nothing.min, None);
    assert_eq!(nothing.max, None);
}

/// The rows partition: every block emits only the rows it owns, so a table
/// assembled from all of them holds each region exactly once. Without the
/// ownership rule every block would write the whole answer and the table would
/// hold every row once per block — `Table::write` has no ownership rule of its
/// own and would be right not to.
#[test]
fn every_region_is_written_by_exactly_one_block() {
    let point = fixed();
    let run = run_at(VOLUME, [3, 2, 2], point);

    let mut seen: BTreeMap<u64, Vec<[usize; 3]>> = BTreeMap::new();
    for (block, bytes) in &run.row_blobs {
        let mut table = Table::new(VOLUME, tabulation_schema(point)).expect("a table");
        table.write(*block, bytes).expect("a blob");
        table.seal().expect("a sealed table");
        for row in table.scan(&Region::whole(&VOLUME)).expect("a scan") {
            seen.entry(region_values(&row, point).expect("a row").label)
                .or_default()
                .push(*block);
        }
    }

    assert_eq!(
        run.row_blobs.len(),
        run.lattice_blocks,
        "every block writes a blob, even one with no rows — present and empty is a fact"
    );
    assert_eq!(seen.keys().copied().collect::<Vec<u64>>(), vec![1, 2, 3, 5]);
    for (label, who) in &seen {
        assert_eq!(who.len(), 1, "label {label} was written by {who:?}");
    }
}

/// The second array is read, and charged for, as a read of the level it names.
/// A second array that cost nothing in the counters would make every measurement
/// of this op a measurement of a different plan.
#[test]
fn both_arrays_are_read_and_counted_as_reads_of_their_own_levels() {
    let block = [3usize, 4, 4];
    let (plan, _, _, _) = tabulation_plan(VOLUME, block, fixed());
    // Level 0 and level 1 are both the tabulating phase's operands; nothing else
    // reads level 1, and the merge reads no pixels at all.
    assert_eq!(plan.phases[1].source_levels, vec![0, 1]);
    assert_eq!(plan.phases[0].source_levels, Vec::<usize>::new());
    assert_eq!(plan.phases[2].source_levels, Vec::<usize>::new());
    // The tabulating phase's operands are voxelwise, so it is granted no halo:
    // every voxel is read by the one block whose core holds it.
    assert_eq!(plan.phases[1].halo, [0, 0, 0]);
    assert_eq!(plan.phases[1].reach, [0, 0, 0]);
    // The merge reaches the whole lattice, which is what makes every block's
    // partial a dependency of every block's merge.
    assert_eq!(plan.phases[2].halo, VOLUME);

    // And the counters agree with the plan. Level 0 is read once per block by
    // the pixel phase and once per block as the tabulating phase's first
    // operand; level 1 once per block as its second. Nothing reads level 2 —
    // the tabulating phase declares `reads_pixels() == false` and the merge is
    // fragments to fragments.
    let run = run_at(VOLUME, block, fixed());
    let blocks = VOLUME[0] / block[0];
    assert_eq!(run.partials.len(), blocks, "every block wrote a partial");
    let mut per_level: BTreeMap<usize, usize> = BTreeMap::new();
    for event in run.log.events() {
        if let Event::RegionRead { level, .. } = event {
            *per_level.entry(level).or_default() += 1;
        }
    }
    assert_eq!(per_level.get(&0).copied().unwrap_or(0), 2 * blocks);
    assert_eq!(per_level.get(&1).copied().unwrap_or(0), blocks);
    assert_eq!(per_level.get(&2).copied().unwrap_or(0), 0);
}

/// A tabulation at one scale cannot be read as one at another. The scale is in
/// the column names, so the schema in the blob and the schema of the table
/// disagree and the write is refused rather than silently rescaled.
#[test]
fn two_fixed_point_scales_are_two_schemas_and_do_not_mix() {
    let coarse = FixedPoint::bits(4).expect("four bits");
    let run = run_at(VOLUME, [4, 4, 4], coarse);
    assert!(!run.words.is_empty(), "the coarse run produced no rows");

    let mut table = Table::new(VOLUME, tabulation_schema(fixed())).expect("a table at 20 bits");
    let (block, bytes) = run
        .row_blobs
        .iter()
        .next()
        .expect("some block wrote a blob");
    let failed = table
        .write(*block, bytes)
        .expect_err("a 4-bit blob is not a 20-bit table")
        .to_string();
    assert!(failed.contains("q4") || failed.contains("q20"), "{failed}");
}

/// The producer-side arithmetic, driven without a plan at all: the shell is thin
/// and the reduction is the part worth checking on its own.
#[test]
fn a_block_can_be_tallied_without_a_run() {
    let point = fixed();
    let op = TabulateValuesOp::new("tabulate", 0, 1, point, "partials", Lifecycle::DeleteOnExit)
        .expect("two levels");
    let read = Region::new(&[2, 0, 0], &[2, 4, 4]);
    let mut label_block = Array3::<f64>::zeros((2, 4, 4));
    let mut value_block = Array3::<f64>::zeros((2, 4, 4));
    for (index, slot) in label_block.indexed_iter_mut() {
        let at = [2 + index.0, index.1, index.2];
        *slot = label_at(at) as f64;
        value_block[[index.0, index.1, index.2]] = value_at(at);
    }
    let tallies = op
        .tally_block(
            &blockflow::env::BlockBuf::Array(label_block.into()),
            &blockflow::env::BlockBuf::Array(value_block.into()),
            &read,
            &read,
        )
        .expect("a tally");
    // z in 2..4: label 1 at y=0,x=0 for both slices, label 2 at z=3 only.
    assert_eq!(tallies[&1].count, 2);
    assert_eq!(tallies[&2].count, 4);
    assert!(!tallies.contains_key(&3));
    assert_eq!(
        tallies[&1].sum,
        point.quantise(value_at([2, 0, 0])).unwrap().unwrap()
            + point.quantise(value_at([3, 0, 0])).unwrap().unwrap()
    );
}

// ------------------------------------------- 5. the first moment, which is
// ---------------------------------- the one column the others do not give --

/// **The separating fixture, built for this and for nothing else.**
///
/// A symmetric arrangement cannot tell a weighted centroid from an unweighted
/// one — put the same weight on every voxel, or arrange the weights evenly about
/// the centre, and the two coincide, so an implementation that ignored the value
/// array entirely would pass. Every choice below is there to stop that:
///
/// * the weight is a **product of a per-axis exponential**, so it rises steeply
///   and *independently* on all three axes and the two centres separate on each
///   of them. An implementation that dropped an axis, permuted them, or divided
///   by `count` rather than by `sum` moves the answer on at least one;
/// * `1.7` is **not a dyadic rational**, so `1.7^z` is not on the fixed point's
///   lattice and the moment is a sum of genuinely quantised terms rather than of
///   integers that happened to be exact;
/// * every value is **positive**, so each region's weighted centre is a convex
///   combination of its own coordinates and therefore a point inside its
///   bounding box — which is what makes "it differs from the unweighted centre
///   by 0.25 or more on every axis" a statement about the weighting rather than
///   about a denominator near zero;
/// * the two regions split axis 1 and are whole on axes 0 and 2, so **every cut
///   on axis 0 or axis 2 puts both of them across a seam** and the merge is
///   doing real work in every run but the single-block one.
///
/// The separation it buys is a quarter of a voxel on the narrowest axis and four
/// voxels on the widest; the assertion asks for a fifth, which is under the
/// former by enough that the quantisation cannot close it and over zero by
/// enough that it is not asking whether two floats happen to be unequal.
const SEPARATING_VOLUME: [usize; 3] = [12, 4, 4];

fn separating_labels(at: [usize; 3]) -> u64 {
    if at[1] < 2 {
        1
    } else {
        2
    }
}

fn separating_value_at(at: [usize; 3]) -> f64 {
    let [z, y, x] = at;
    1.7f64.powi(z as i32) * 3.0f64.powi(y as i32) * 2.5f64.powi(x as i32)
}

/// **The acceptance criterion for the new column**, on a fixture where a broken
/// implementation has somewhere to be wrong.
///
/// Byte-identical against a whole-volume reference across five block sizes, with
/// the weighted and unweighted centres required to *differ* on every axis of
/// every region — so five runs that all reported the geometric centroid would
/// agree with each other perfectly and fail here on the first assertion.
#[test]
fn the_weighted_centroid_is_byte_identical_across_block_sizes_where_it_differs_from_the_unweighted()
{
    let point = fixed();
    let sizes: [[usize; 3]; 5] = [[12, 4, 4], [6, 4, 4], [4, 2, 4], [3, 4, 2], [2, 2, 2]];
    let mut answers: Vec<(String, Vec<u64>)> = Vec::new();

    for block in sizes {
        let run = try_run(
            SEPARATING_VOLUME,
            block,
            point,
            separating_labels,
            separating_value_at,
        )
        .expect("twenty fraction bits holds this fixture");

        // Which labels were cut. Without this the byte-identity below could be
        // the identity of five runs that never merged a moment.
        let mut contributors: BTreeMap<u64, BTreeSet<[usize; 3]>> = BTreeMap::new();
        for (index, bytes) in &run.partials {
            for tally in decode_partial(bytes).expect("a partial") {
                contributors.entry(tally.label).or_default().insert(*index);
            }
        }
        let straddled: Vec<u64> = contributors
            .iter()
            .filter(|(_, who)| who.len() > 1)
            .map(|(label, _)| *label)
            .collect();
        if block == [12, 4, 4] {
            assert!(
                straddled.is_empty(),
                "the one-block lattice cannot straddle anything"
            );
        } else {
            assert_eq!(
                straddled,
                vec![1, 2],
                "block {block:?} left a region inside one block, so this run merges less than the \
                 fixture was built to make it merge"
            );
        }

        let expected = reference_with(
            SEPARATING_VOLUME,
            point,
            separating_labels,
            separating_value_at,
        );
        assert_eq!(run.rows.len(), 2, "block {block:?}");
        for row in &run.rows {
            let tally = expected
                .get(&row.label)
                .unwrap_or_else(|| panic!("block {block:?} invented label {}", row.label));

            // The moments, on the integers — the form the invariance claim is
            // about, since the quotient is taken from them and not the reverse.
            for axis in 0..3 {
                assert_eq!(
                    i128::from(row.moment_fixed[axis]),
                    tally.moment[axis],
                    "block {block:?}: label {} on axis {axis}",
                    row.label
                );
            }
            // And the quotient, on its bits.
            let weighted = row
                .weighted_centroid
                .expect("every value in this fixture is positive, so the denominator is not zero");
            assert_eq!(
                weighted.map(f64::to_bits),
                tally
                    .weighted_centroid()
                    .expect("the oracle agrees there is one")
                    .map(f64::to_bits),
                "block {block:?}: label {}",
                row.label
            );

            // The unweighted centre is the box centre, exactly — the regions are
            // solid boxes — so the two quantities are here side by side and the
            // separation below is between two numbers a reader can check.
            let box_centre = [5.5, if row.label == 1 { 0.5 } else { 2.5 }, 1.5];
            assert_eq!(row.centroid, box_centre, "label {}", row.label);
            for axis in 0..3 {
                assert!(
                    (weighted[axis] - box_centre[axis]).abs() > 0.2,
                    "block {block:?}: label {} has weighted centre {} and box centre {} on axis \
                     {axis}. They have to differ, or this fixture cannot tell the two apart and \
                     would pass an implementation that never read the value array.",
                    row.label,
                    weighted[axis],
                    box_centre[axis]
                );
                // and the weight being positive keeps it a point in the region
                assert!(
                    weighted[axis] >= 0.0 && weighted[axis] <= (SEPARATING_VOLUME[axis] - 1) as f64,
                    "label {} left the volume on axis {axis} under a positive weight",
                    row.label
                );
            }
            // The weight rises with every coordinate, so the weighted centre is
            // on the far side of the box centre on every axis and not merely a
            // different number from it.
            for axis in 0..3 {
                assert!(
                    weighted[axis] > box_centre[axis],
                    "label {} on axis {axis}: a weight increasing in the coordinate has to pull \
                     the centre up",
                    row.label
                );
            }
        }

        answers.push((format!("{block:?}"), run.words));
    }

    let (first_name, first) = &answers[0];
    for (name, words) in &answers[1..] {
        assert_eq!(
            words, first,
            "the weighted centroid differs between block {first_name} and block {name}; a cross \
             moment of value against position must not be a function of the plan"
        );
    }
}

/// One region of four voxels, through a real run of four blocks, with **the
/// arithmetic written out** rather than taken from a helper.
///
/// Values `1, 1, 1, 5` at `z = 0, 1, 2, 3`, all dyadic and so exact at twenty
/// fraction bits:
///
/// * `sum = 1 + 1 + 1 + 5 = 8`;
/// * `moment_0 = 1*0 + 1*1 + 1*2 + 5*3 = 18`;
/// * weighted centre `= 18 / 8 = 2.25`;
/// * unweighted centre `= (0 + 1 + 2 + 3) / 4 = 1.5`, and the row sits at `2`,
///   which is that rounded half up.
///
/// The two differ by `0.75`, which is the whole point of writing them both down.
fn hand_value_at(at: [usize; 3]) -> f64 {
    if at[0] == 3 {
        5.0
    } else {
        1.0
    }
}

#[test]
fn a_hand_computed_region_reports_the_weighted_centre_the_arithmetic_says() {
    let point = fixed();
    let run = try_run(AWKWARD_VOLUME, [1, 1, 1], point, one_label, hand_value_at)
        .expect("a run of four blocks");
    assert_eq!(
        run.partials.len(),
        4,
        "four fragments, so the merge is real"
    );
    let region = row_for(&run.rows, 1);

    let one = 1i64 << 20; // one unit of value, in fixed-point steps
    assert_eq!(region.count, 4);
    assert_eq!(region.sum_fixed, 8 * one);
    assert_eq!(region.sum, 8.0);
    assert_eq!(region.moment_fixed[0], 18 * one);
    assert_eq!(region.moment[0], 18.0);
    assert_eq!(region.moment_fixed[1], 0);
    assert_eq!(region.moment_fixed[2], 0);

    assert_eq!(region.weighted_centroid, Some([2.25, 0.0, 0.0]));
    assert_eq!(region.centroid, [1.5, 0.0, 0.0]);
    assert_eq!(region.at, [2, 0, 0]);
    // 18/8 against 6/4: the two are different numbers over the same four voxels,
    // and no arrangement of `count`, `sum` and `sum_0` gives the first.
    assert_eq!(
        region.weighted_centroid.unwrap()[0] - region.centroid[0],
        0.75
    );
}

/// The degenerate denominator, through a run rather than on a tally: a region
/// whose finite values cancel has **no** weighted centroid, and the row says so
/// by carrying `None` — not a `NaN`, not the unweighted centre.
///
/// Two regions in one run, so the answer is a property of the region and not of
/// the table: the second has a perfectly good weighted centre in the same rows.
fn cancelling_value_at(at: [usize; 3]) -> f64 {
    match at[0] {
        0 => 3.0,
        1 => -3.0,
        _ => 1.0,
    }
}

#[test]
fn a_region_whose_values_cancel_reports_no_weighted_centroid_at_all() {
    let point = fixed();
    let run = try_run(
        AWKWARD_VOLUME,
        [1, 1, 1],
        point,
        two_region_labels,
        cancelling_value_at,
    )
    .expect("a run of four blocks");
    let one = 1i64 << 20;

    // Region 1 is `+3` at z = 0 and `-3` at z = 1.
    let cancelled = row_for(&run.rows, 1);
    assert_eq!(cancelled.count, 2);
    assert_eq!(cancelled.nonfinite, 0, "both values are finite");
    assert_eq!(cancelled.sum_fixed, 0, "the denominator is exactly zero");
    // The numerator is still exact, still written, and is not zero: `3*0 - 3*1`.
    assert_eq!(cancelled.moment_fixed[0], -3 * one);
    assert_eq!(cancelled.moment[0], -3.0);
    // So the quotient does not exist, and that is what the row carries.
    assert_eq!(
        cancelled.weighted_centroid, None,
        "a zero denominator has no quotient; reporting one would be inventing it"
    );
    // The geometric centre is unaffected — the absence is of one quantity, not
    // of the row.
    assert_eq!(cancelled.centroid, [0.5, 0.0, 0.0]);
    assert_eq!(cancelled.count, 2);

    // Region 2 is `1` at z = 2 and z = 3, in the same table, and has one.
    let ordinary = row_for(&run.rows, 2);
    assert_eq!(ordinary.sum_fixed, 2 * one);
    assert_eq!(ordinary.moment_fixed[0], 5 * one, "1*2 + 1*3");
    assert_eq!(ordinary.weighted_centroid, Some([2.5, 0.0, 0.0]));

    // And a region of nothing but zeros is the same fact by the other road:
    // `0/0` rather than `k/0`, and the same `None`.
    let zeros = try_run(AWKWARD_VOLUME, [2, 1, 1], point, one_label, |_| 0.0)
        .expect("a run over a zero array");
    let flat = row_for(&zeros.rows, 1);
    assert_eq!(flat.count, 4);
    assert_eq!(flat.sum_fixed, 0);
    assert_eq!(flat.moment_fixed, [0, 0, 0]);
    assert_eq!(flat.weighted_centroid, None);
    assert_eq!(flat.centroid, [1.5, 0.0, 0.0]);
}

// ------------------------------- 6. the selection carries no fixed point --

/// A value the fixed point **cannot hold at any scale this fixture admits**.
///
/// One unit in the last place above `1.0`, so `AWKWARD * 2^n` is a whole number
/// only for `n >= 52`. [`WIDE`] is `2^11`, which the op refuses at every `n >=
/// 52` because a value's range is `+/- 2^(63-n)`. The two windows do not
/// overlap, so there is no `n` at which a quantised selection would have come
/// back as the voxel's own value — which is what makes the byte-identity
/// assertion below discriminating rather than lucky.
const AWKWARD: f64 = 1.0 + f64::EPSILON;
/// See [`AWKWARD`]. Also large enough that the two voxels holding it total
/// `2^63` at 51 fraction bits, so the *sum*'s range closes that scale too.
const WIDE: f64 = 2048.0;

/// Two regions on `[4, 1, 1]`: the awkward pair, then the wide pair.
fn two_region_labels(at: [usize; 3]) -> u64 {
    if at[0] < 2 {
        1
    } else {
        2
    }
}

fn awkward_value_at(at: [usize; 3]) -> f64 {
    match at[0] {
        0 => AWKWARD,
        1 => -AWKWARD,
        _ => WIDE,
    }
}

const AWKWARD_VOLUME: [usize; 3] = [4, 1, 1];

/// **The selection is the voxel's own value, bit for bit, at every scale the run
/// admits** — and a quantised column would have differed at every one of them.
///
/// The second half is what makes this a test rather than a tautology. It is not
/// enough that `min` and `max` come back exact; the fixture has to be one where
/// carrying them through the fixed point *visibly* could not have. So the same
/// value is quantised and unquantised at the same scale in the same loop, and
/// the two are required to disagree.
#[test]
fn the_selection_is_the_voxels_own_value_at_every_scale_the_run_admits() {
    let mut accepted = 0usize;
    let mut refused = 0usize;
    for bits in 0..=MAX_FRACTION_BITS {
        let point = FixedPoint::bits(bits).expect("in range");
        // One block per voxel, so the awkward region is cut and its selection
        // reaches the merge rather than staying inside one block.
        let run = match try_run(
            AWKWARD_VOLUME,
            [1, 1, 1],
            point,
            two_region_labels,
            awkward_value_at,
        ) {
            Ok(run) => run,
            Err(_) => {
                refused += 1;
                continue;
            }
        };
        accepted += 1;
        let region = row_for(&run.rows, 1);
        assert_eq!(
            region.max.to_bits(),
            AWKWARD.to_bits(),
            "at {bits} fraction bits the maximum came back as {} and the voxel holds {AWKWARD}",
            region.max
        );
        assert_eq!(
            region.min.to_bits(),
            (-AWKWARD).to_bits(),
            "at {bits} fraction bits the minimum came back as {}",
            region.min
        );
        // and the discrimination: the same value through the fixed point is a
        // different number, at this very scale
        let quantised = point.value_of(
            point
                .quantise(AWKWARD)
                .expect("in range, or the run would have refused")
                .expect("finite") as i64,
        );
        assert_ne!(
            quantised.to_bits(),
            AWKWARD.to_bits(),
            "at {bits} fraction bits the fixed point holds {AWKWARD} exactly, so this scale does \
             not discriminate a quantised selection from an unquantised one"
        );
    }
    println!(
        "{accepted} scales accepted, {refused} refused, and every accepted one selects the \
              voxel's own bits"
    );
    assert!(accepted > 1, "one scale is not a sweep");
    assert!(
        refused > 0,
        "no scale in `0..={MAX_FRACTION_BITS}` was refused, so this fixture never reached the \
         range edge that closes the window a quantised selection would have needed"
    );
}

/// The same fixture, cut three ways: the selection is a function of the region's
/// voxels and not of the plan, which is the acceptance criterion the whole file
/// is about, restated for the column that stopped being an accumulator.
#[test]
fn the_selection_is_byte_identical_across_block_sizes() {
    let point = FixedPoint::default();
    let mut answers: Vec<(String, Vec<u64>)> = Vec::new();
    let mut straddled = 0usize;
    for block in [[4usize, 1, 1], [2, 1, 1], [1, 1, 1]] {
        let run = try_run(
            AWKWARD_VOLUME,
            block,
            point,
            two_region_labels,
            awkward_value_at,
        )
        .expect("twenty fraction bits holds this fixture");
        let region = row_for(&run.rows, 1);
        assert_eq!(region.max.to_bits(), AWKWARD.to_bits(), "block {block:?}");
        assert_eq!(
            region.min.to_bits(),
            (-AWKWARD).to_bits(),
            "block {block:?}"
        );
        // and the fixture could have failed: one of these cuts has to put the
        // region across a seam, or the byte-identity below is the identity of
        // three runs that merged nothing.
        let cut = run
            .partials
            .values()
            .filter(|bytes| {
                decode_partial(bytes)
                    .expect("a partial")
                    .iter()
                    .any(|tally| tally.label == 1)
            })
            .count();
        if cut > 1 {
            straddled += 1;
        }
        answers.push((format!("{block:?}"), run.words));
    }
    assert!(
        straddled > 0,
        "no cut put the awkward region across a seam, so none of these runs merged a selection"
    );
    let (first_name, first) = &answers[0];
    for (name, words) in &answers[1..] {
        assert_eq!(
            words, first,
            "the table differs between block {first_name} and block {name}"
        );
    }
}

/// `-0.0` against `0.0`, **through the executor's reversal check**.
///
/// This is acceptance criterion 1 for the selection, and it is deliberately not
/// an assertion about `least`: the merge here has four fragments, the phase
/// declares `SeamFold::Unordered`, so the executor applies every block a second
/// time with its neighbourhood reversed and requires byte-identical output.
/// A run that returns at all is that check having passed on a column that is now
/// `f64`. The fixture is the one pair that could have broken it — two values
/// that compare equal and are different bits, where `f64::min` is free to return
/// either operand — and the row that comes back says which one the total order
/// picks.
fn signed_zero_at(at: [usize; 3]) -> f64 {
    if at[0] == 0 {
        -0.0
    } else {
        0.0
    }
}

#[test]
fn a_signed_zero_survives_the_reversal_check_and_keeps_its_sign() {
    let run = try_run(
        AWKWARD_VOLUME,
        [1, 1, 1],
        fixed(),
        one_label,
        signed_zero_at,
    )
    .expect("a selection in `f64` is order-independent, and the executor just checked it");
    assert_eq!(
        run.partials.len(),
        4,
        "four fragments, so the order is real"
    );
    let region = row_for(&run.rows, 1);
    assert_eq!(region.count, 4);
    assert_eq!(
        region.min.to_bits(),
        (-0.0f64).to_bits(),
        "the minimum came back as {} and `-0.0` is the smaller of the two under the total order",
        region.min
    );
    assert_eq!(region.max.to_bits(), 0.0f64.to_bits());
    // The sum is unaffected either way: `-0.0` quantises to the integer zero.
    assert_eq!(region.sum_fixed, 0);
}

/// The schema says which columns carry a scale and which do not, and it says it
/// in the names — which is the fact a consumer reads to know what it is holding.
///
/// **The scale is on the accumulated *values* and on nothing else**: the sum and
/// the three first moments. Not on the selections, which are values that were
/// never scaled, and not on the coordinate sums, which are integers that never
/// needed to be.
#[test]
fn the_scale_is_on_the_accumulated_values_and_nowhere_else() {
    for bits in [0u32, 4, DEFAULT_FRACTION_BITS, MAX_FRACTION_BITS] {
        let point = FixedPoint::bits(bits).expect("in range");
        let schema = tabulation_schema(point);
        let suffix = format!("_q{bits}");
        let scaled: Vec<&str> = schema
            .columns()
            .iter()
            .map(|column| column.name())
            .filter(|name| name.ends_with(&suffix))
            .collect();
        let expected: Vec<String> = std::iter::once(format!("{SUM_COLUMN}{suffix}"))
            .chain(MOMENT_COLUMN.iter().map(|stem| format!("{stem}{suffix}")))
            .collect();
        assert_eq!(
            scaled,
            expected.iter().map(String::as_str).collect::<Vec<&str>>(),
            "at {bits} fraction bits the scale is on {scaled:?}"
        );
        assert!(schema.index_of(MIN_COLUMN).is_some());
        assert!(schema.index_of(MAX_COLUMN).is_some());
    }
}

// ---------------------------------------- 7. the sum, which did not move --

/// Every finite voxel of `label`, quantised at `point` and added as integers:
/// the sum column's definition, written out, and not the op's code.
fn quantised_total(volume: [usize; 3], point: FixedPoint, label: u64) -> i128 {
    let mut total = 0i128;
    for z in 0..volume[0] {
        for y in 0..volume[1] {
            for x in 0..volume[2] {
                let at = [z, y, x];
                if label_at(at) != label {
                    continue;
                }
                if let Some(value) = point.quantise(value_at(at)).expect("in range") {
                    total += value;
                }
            }
        }
    }
    total
}

/// **The sum is exactly what it was**, asserted two ways: against its own
/// definition for every region, and against a written-down integer for the one
/// that spans the volume — so a change to the quantiser that moved the reference
/// with it would still be caught here.
#[test]
fn the_fixed_point_sum_is_unchanged_and_is_still_the_quantised_total() {
    let point = fixed();
    let run = run_at(VOLUME, [3, 2, 2], point);
    let schema = tabulation_schema(point);
    assert_eq!(
        schema.index_of(&format!("{SUM_COLUMN}{}", point.suffix())),
        Some(3),
        "the sum is the fourth column and carries the scale, as it always did"
    );
    for row in &run.rows {
        assert_eq!(
            i128::from(row.sum_fixed),
            quantised_total(VOLUME, point, row.label),
            "label {}",
            row.label
        );
    }
    // Label 1 is `3z - 7` for z in 0..12 with the voxel at z = 9 non-finite:
    // 114 - 20 = 94, and 94 * 2^20 is the fixed-point word's value.
    let spanning = row_for(&run.rows, 1);
    assert_eq!(spanning.sum_fixed, 98_566_144);
    assert_eq!(spanning.sum, 94.0);
    assert_eq!(spanning.count, 12);
    assert_eq!(spanning.nonfinite, 1);
}

/// The scale a caller with no scale in mind can **derive** rather than pick, run
/// rather than described: a region holds at most every voxel, each finite value
/// is at most the array's peak magnitude, and quantising moves each at most half
/// a step further from zero, so no *sum* exceeds `voxels * (magnitude + 0.5)`.
///
/// **And the first moment multiplies that by the largest coordinate**, since
/// each term is a value times a coordinate — so the bound the range has to cover
/// is the sum's times `max(extent) - 1`, and the derived scale is coarser than
/// the sum alone would have asked for. That factor is the whole of what the new
/// column costs a caller who derives rather than picks, and it is stated here
/// because a derivation that covered only the sum would be refused on a moment.
fn derived_scale(volume: [usize; 3], of: fn([usize; 3]) -> f64) -> (FixedPoint, f64) {
    let mut magnitude = 0.0f64;
    for z in 0..volume[0] {
        for y in 0..volume[1] {
            for x in 0..volume[2] {
                let value = of([z, y, x]);
                if value.is_finite() {
                    magnitude = magnitude.max(value.abs());
                }
            }
        }
    }
    let voxels = (volume[0] * volume[1] * volume[2]) as f64;
    let reach = volume
        .iter()
        .map(|extent| extent.saturating_sub(1))
        .max()
        .expect("three axes")
        .max(1) as f64;
    let bound = voxels * (magnitude + 0.5) * reach;
    let mut best = FixedPoint::bits(0).expect("zero bits");
    for bits in 0..=MAX_FRACTION_BITS {
        let candidate = FixedPoint::bits(bits).expect("in range");
        if candidate.limit() > bound {
            best = candidate;
        }
    }
    (best, bound)
}

/// **The derived-scale path still works**, and the selection is the same bits at
/// the derived scale as at the default — which is the difference between the two
/// columns, made visible in one assertion.
#[test]
fn the_sum_runs_at_a_derived_scale_and_the_selection_does_not_care_which() {
    let (point, bound) = derived_scale(VOLUME, value_at);
    assert!(
        point.limit() > bound,
        "the derived scale's range {:e} does not cover the bound {bound:e}",
        point.limit()
    );
    assert!(
        point.fraction_bits() == MAX_FRACTION_BITS
            || FixedPoint::bits(point.fraction_bits() + 1)
                .expect("one more bit")
                .limit()
                <= bound,
        "one more fraction bit would still have covered the bound, so this is not the finest \
         scale the range admits"
    );
    assert_ne!(
        point.fraction_bits(),
        DEFAULT_FRACTION_BITS,
        "the derived scale equals the default, which would make this indistinguishable from \
         taking it"
    );

    let derived = run_at(VOLUME, [3, 2, 2], point);
    let oracle = reference_with(VOLUME, point, label_at, value_at);
    for row in &derived.rows {
        assert_eq!(
            i128::from(row.sum_fixed),
            quantised_total(VOLUME, point, row.label),
            "label {} at the derived {} fraction bits",
            row.label,
            point.fraction_bits()
        );
        // The moments fit at the derived scale too, which is the half of the
        // derivation the trailing factor exists for.
        for axis in 0..3 {
            assert_eq!(
                i128::from(row.moment_fixed[axis]),
                oracle[&row.label].moment[axis],
                "label {} on axis {axis} at the derived {} fraction bits",
                row.label,
                point.fraction_bits()
            );
        }
    }

    // The sum moved with the scale — it is a different integer in a differently
    // named column — and the selection did not move at all.
    let default = run_at(VOLUME, [3, 2, 2], fixed());
    assert_ne!(
        derived.rows[0].sum_fixed, default.rows[0].sum_fixed,
        "two scales gave the same fixed-point word, so the scale is not doing anything here"
    );
    for (one, other) in derived.rows.iter().zip(&default.rows) {
        assert_eq!(one.label, other.label);
        assert_eq!(
            one.min.to_bits(),
            other.min.to_bits(),
            "label {} moved its minimum with the scale",
            one.label
        );
        assert_eq!(
            one.max.to_bits(),
            other.max.to_bits(),
            "label {} moved its maximum with the scale",
            one.label
        );
    }
}

/// The sum's two refusals, both still by name: a **value** outside the column's
/// range, refused at the voxel it was read at, and a **total** outside it,
/// refused where the answer becomes a column.
#[test]
fn the_sums_out_of_range_refusals_are_both_still_named() {
    // 2048 needs 2^(63 - n) > 2048, so 52 fraction bits refuses the value.
    let by_value = try_run(
        AWKWARD_VOLUME,
        [1, 1, 1],
        FixedPoint::bits(52).expect("52 bits"),
        two_region_labels,
        awkward_value_at,
    )
    .expect_err("2048 does not fit 2^11 of range")
    .to_string();
    assert!(by_value.contains("52 fraction bit"), "{by_value}");
    assert!(by_value.contains("cannot hold"), "{by_value}");

    // At 51 the value fits and the *total* does not: two voxels of 2048 scale to
    // 2^63 exactly, which is one past what a signed column holds.
    let by_total = try_run(
        AWKWARD_VOLUME,
        [1, 1, 1],
        FixedPoint::bits(51).expect("51 bits"),
        two_region_labels,
        awkward_value_at,
    )
    .expect_err("4096 at 51 fraction bits is 2^63")
    .to_string();
    assert!(by_total.contains("fixed-point total"), "{by_total}");
    assert!(by_total.contains("above"), "{by_total}");
    assert!(by_total.contains("fewer fraction bits"), "{by_total}");

    // And at 50 the **moment** is the one that does not fit, which is the range
    // edge the new column brings with it: the two voxels of 2048 sit at z = 2 and
    // z = 3, so the sum is `2 * 2^61 = 2^62` and fits, while the moment is
    // `5 * 2^61` and does not. A run refused here on the sum's message would be
    // sending a caller to the wrong number.
    let by_moment = try_run(
        AWKWARD_VOLUME,
        [1, 1, 1],
        FixedPoint::bits(50).expect("50 bits"),
        two_region_labels,
        awkward_value_at,
    )
    .expect_err("5 * 2^61 is above what a signed 64-bit column holds")
    .to_string();
    assert!(by_moment.contains("first moment on axis 0"), "{by_moment}");
    assert!(by_moment.contains("binds first"), "{by_moment}");
    // and one bit coarser everything fits, so 50 really is the edge and not a
    // scale that was failing for some other reason
    try_run(
        AWKWARD_VOLUME,
        [1, 1, 1],
        FixedPoint::bits(49).expect("49 bits"),
        two_region_labels,
        awkward_value_at,
    )
    .expect("5 * 2^60 fits");
}

/// The one thing `Arc` is doing here: an op is shared across workers, so it is
/// `Send + Sync`, and a test that holds one behind an `Arc` is the cheapest
/// statement of that which the compiler checks.
#[test]
fn the_ops_are_shareable_across_workers() {
    let point = fixed();
    let tabulate: Arc<dyn FragmentOp> = Arc::new(
        TabulateValuesOp::new("tabulate", 0, 1, point, "partials", Lifecycle::DeleteOnExit)
            .expect("two levels"),
    );
    let merge: Arc<dyn FragmentOp> = Arc::new(MergeTabulationOp::new(
        "merge",
        "partials",
        1,
        [2, 1, 1],
        point,
        "rows",
        Lifecycle::Persistent,
    ));
    assert_eq!(tabulate.seam_fold(), Some(SeamFold::PerBlock));
    assert_eq!(merge.seam_fold(), Some(SeamFold::Unordered));
    assert!(!merge.gathers(), "the whole-lattice merge streams");
    assert!(!tabulate.reads_pixels() && !merge.reads_pixels());
}
